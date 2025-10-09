import logging
import cohere
import re
import xml.etree.ElementTree as ET

from typing import Any, Dict, List, Optional, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


from adaptiq.core.entities import (
    Feedback,
    FeedbackEvent,
    SentimentAnalysisResults,
    FeedbackDocument,
    RerankResult,
    FeedbackResult,
    FeedbackMap,
    ProcessedLogs,
    FeedbackRewards
)


class FeedbackEngine:
    """
    Engine for processing and analyzing feedback from AI agent runs.

    The FeedbackEngine provides capabilities for handling feedback events, including:
    - Creating queries from feedback events (human or event-based)
    - Reranking processed logs based on feedback relevance using Cohere's rerank API
    - Performing sentiment analysis on human feedback using LLM

    This engine integrates with both Cohere's reranking service and LangChain's LLM
    interface to provide comprehensive feedback analysis capabilities.

    Attributes:
        cohere_client: Cohere API client for reranking operations
        rerank_model: Model identifier for the reranking service
        llm: LangChain BaseChatModel for sentiment analysis and other LLM tasks
        processed_logs: ProcessedLogs object containing agent execution logs
        feedback: Feedback object containing feedback events and metadata
        logger: Logger instance for tracking operations
    """

    def __init__(self,
            rerank_provider: str,
            rerank_api_key: str,
            rerank_model: str,
            llm: BaseChatModel,
            processed_logs: ProcessedLogs,
        ):
        """
        Initialize the FeedbackEngine with reranking and LLM capabilities.

        Args:
            rerank_provider: Provider name for reranking service (currently only "cohere" supported)
            rerank_api_key: API key for the reranking service
            rerank_model: Model identifier for reranking (e.g., "rerank-v3.5")
            llm: LangChain BaseChatModel instance for LLM operations
            processed_logs: ProcessedLogs object containing agent execution logs

        Raises:
            ValueError: If rerank_provider is not "cohere"
        """
        if rerank_provider.lower() != "cohere":
            raise ValueError(f"Unsupported rerank provider: {rerank_provider}")
        else:
            self.cohere_client = cohere.ClientV2(api_key=rerank_api_key)

        self.rerank_model = rerank_model
        self.llm = llm
        self.processed_logs = processed_logs
        self.feedback = None  # To be set via method or externally

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-FeedbackEngine")

    def fetch_feedback_events(self) -> Feedback:
        """
        Retrieve the list of feedback events from the Feedback object.

        Returns:
            List[FeedbackEvent]: List of feedback events to process
        """
        return self.feedback

    def create_query(self, feedback_event: FeedbackEvent) -> str:
        """
        Create a query string from a feedback event based on its type.

        For HUMAN feedback type, returns the new_value directly as the query.
        For EVENT feedback type, constructs a query combining the event name,
        original value, and new value.

        Args:
            feedback_event: FeedbackEvent object containing the feedback data

        Returns:
            str: The constructed query string
        """
        payload = feedback_event.payload

        if feedback_event.feedback_type == "HUMAN":
            return str(payload.details.new_value)

        # For EVENT feedback type
        event_name = payload.event_name
        original_value = payload.details.original_value
        new_value = payload.details.new_value

        query = f"{event_name}: {original_value} -> {new_value} for item {payload.context.item_id} on page {payload.context.ui_location}"
        return query

    def rerank_feedback(
        self,
        processed_logs: ProcessedLogs,
        query: str,
        top_n: int = 3
    ) -> RerankResult:
        """
        Rerank processed logs based on their relevance to a feedback query.

        Converts processed logs into FeedbackDocument objects and uses Cohere's rerank model
        to identify the most relevant log entries for the given query. Returns results
        as structured FeedbackDocument objects with relevance scores.

        Args:
            processed_logs: ProcessedLogs object containing the log items to rerank
            query: The feedback query string to rank documents against
            top_n: Number of top-ranked results to return (default: 3)

        Returns:
            RerankResult: Object containing list of FeedbackDocuments with relevance scores
        """
        # Convert ProcessedLogs to document strings and FeedbackDocument objects
        documents: List[str] = []
        feedback_docs: List[FeedbackDocument] = []

        for log_item in processed_logs.processed_logs:
            state = log_item.key.state
            action = log_item.key.agent_action

            # Create document string for reranking (without reward)
            doc_str = (
                f"Task: {state.current_sub_task_or_thought}, "
                f"Action: {action}, "
            )
            documents.append(doc_str)

            # Create FeedbackDocument object (without relevance_score initially)
            feedback_doc = FeedbackDocument(
                agent_task=state.current_sub_task_or_thought,
                agent_action=action,
                relevance_score=None
            )
            feedback_docs.append(feedback_doc)

        # Perform reranking using Cohere
        rerank_response = self.cohere_client.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=top_n
        )

        # Map relevance scores back to FeedbackDocuments
        ranked_feedback_docs: List[FeedbackDocument] = []
        for result in rerank_response.results:
            # Get the original FeedbackDocument by index
            doc = feedback_docs[result.index]
            # Update with relevance score
            doc.relevance_score = result.relevance_score
            ranked_feedback_docs.append(doc)

        return RerankResult(feedback_documents=ranked_feedback_docs)

    def step_selection(self, rerank_result: RerankResult) -> Optional[FeedbackDocument]:
        """
        Select the first feedback document with relevance score above threshold.

        Iterates through reranked feedback documents and returns the first document
        that has a relevance score greater than 0.75, indicating high confidence
        in the match between the feedback and the log entry.

        Args:
            rerank_result: RerankResult object containing ranked feedback documents

        Returns:
            Optional[FeedbackDocument]: The first document with relevance_score > 0.75,
                or None if no document meets the threshold=
        """
        for doc in rerank_result.feedback_documents:
            if doc.relevance_score is not None and doc.relevance_score > 0.4:
                return doc

        return None

    def assign_feedback_reward(
            self,
            feedback_document: FeedbackDocument,
            feedback_event: FeedbackEvent,
            processed_logs: ProcessedLogs
        ) -> Tuple[FeedbackResult, ProcessedLogs]:
            """
            Assign a reward to a feedback document based on event type and sentiment analysis.

            Calculates reward values based on FeedbackRewards enum constants. For EVENT type
            feedback (VALUE_INCREASED, VALUE_DECREASED, TEXT_EDITED), applies fixed rewards.
            For HUMAN type feedback, performs sentiment analysis and scales reward by polarity score.
            Optionally scales by relevance score if available.

            Updates the matching log item's reward_exec by averaging it with the reward_feedback.

            Args:
                feedback_document: FeedbackDocument with relevance score from reranking
                feedback_event: FeedbackEvent containing feedback type and payload
                processed_logs: ProcessedLogs object containing log items to update

            Returns:
                tuple[FeedbackResult, ProcessedLogs]: FeedbackResult containing the event, document,
                    and calculated reward, along with the updated ProcessedLogs with modified reward_exec
            """
            reward = 0.0
            event_name = feedback_event.payload.event_name

            # Handle EVENT type feedback
            if feedback_event.feedback_type == "EVENT":
                if event_name == "VALUE_INCREASED":
                    reward = FeedbackRewards.REWARD_VALUE_INCREASED.value
                elif event_name == "VALUE_DECREASED":
                    reward = FeedbackRewards.PENALTY_VALUE_DECREASED.value
                elif event_name == "TEXT_EDITED":
                    reward = FeedbackRewards.REWARD_TEXT_EDITED.value

                self.logger.info(
                    f"EVENT feedback: {event_name} -> base reward: {reward}"
                )

            # Handle HUMAN type feedback with sentiment analysis
            elif feedback_event.feedback_type == "HUMAN":
                sentiment_result = self.analyze_sentiment(feedback_event)

                if sentiment_result:
                    polarity_score = sentiment_result.polarity_score
                    class_label = sentiment_result.class_label

                    # Calculate sentiment-based reward
                    if polarity_score > FeedbackRewards.POLARITY_THRESHOLD_POSITIVE.value:
                        # Positive sentiment: scale by polarity score
                        reward = (
                            FeedbackRewards.SENTIMENT_BASE_REWARD_POSITIVE.value
                            * polarity_score
                        )
                    elif polarity_score < FeedbackRewards.POLARITY_THRESHOLD_NEGATIVE.value:
                        # Negative sentiment: scale by polarity score (will be negative)
                        reward = (
                            FeedbackRewards.SENTIMENT_BASE_PENALTY_NEGATIVE.value
                            * abs(polarity_score)
                        )
                    else:
                        # Neutral sentiment
                        reward = FeedbackRewards.SENTIMENT_BASE_REWARD_NEUTRAL.value

                    self.logger.info(
                        f"HUMAN feedback: sentiment={class_label}, polarity={polarity_score} -> base reward: {reward}"
                    )
                else:
                    self.logger.warning("Sentiment analysis returned None for HUMAN feedback")

            # Scale by relevance score if available and above threshold
            if feedback_document.relevance_score is not None:
                relevance = feedback_document.relevance_score

                if relevance >= FeedbackRewards.MIN_RELEVANCE_THRESHOLD.value:
                    scaled_reward = reward * relevance * FeedbackRewards.RELEVANCE_SCALE_FACTOR.value
                    self.logger.info(
                        f"Scaling reward by relevance: {reward} * {relevance} = {scaled_reward}"
                    )
                    reward = scaled_reward
                else:
                    self.logger.info(
                        f"Relevance score {relevance} below threshold {FeedbackRewards.MIN_RELEVANCE_THRESHOLD.value}, no reward applied"
                    )
                    reward = 0.0

            # Create FeedbackResult
            feedback_result = FeedbackResult(
                event=feedback_event,
                document=feedback_document,
                reward_feedback=reward
            )

            # Update processed logs: find matching log item and update its reward_exec
            updated_logs = []
            for log_item in processed_logs.processed_logs:
                # Check if the log item's current_sub_task_or_thought matches the feedback document's agent_task
                if log_item.key.state.current_sub_task_or_thought == feedback_document.agent_task:
                    # Calculate new reward as simple average of reward_exec and reward_feedback
                    old_reward = log_item.reward_exec
                    new_reward = (old_reward + reward) / 2.0

                    # Create updated log item with new reward
                    updated_log_item = log_item.model_copy(deep=True)
                    updated_log_item.reward_exec = new_reward

                    self.logger.info(
                        f"Updated reward for task {feedback_document.agent_task}: {old_reward} -> {new_reward} (avg with feedback reward {reward})"
                    )

                    updated_logs.append(updated_log_item)
                else:
                    # Keep the log item unchanged
                    updated_logs.append(log_item)

            # Create updated ProcessedLogs
            updated_processed_logs = ProcessedLogs(processed_logs=updated_logs)

            return feedback_result, updated_processed_logs

    def analyze_sentiment(
        self,
        feedback_event: FeedbackEvent
    ) -> Optional[SentimentAnalysisResults]:
        """
        Perform sentiment analysis on HUMAN feedback events.

        Analyzes the sentiment of user feedback from AI agent outputs using LLM.
        Returns None for EVENT type feedback as sentiment analysis is only applicable
        to human-generated feedback.

        Args:
            feedback_event: FeedbackEvent object containing the feedback data

        Returns:
            Optional[SentimentAnalysisResults]: Sentiment analysis results with class_label
                (positive/negative/neutral) and polarity_score (-1.0 to 1.0), or None if
                feedback_type is EVENT
        """
        # Only analyze HUMAN feedback
        if feedback_event.feedback_type != "HUMAN":
            return None

        feedback_text = str(feedback_event.payload.details.new_value)

        # Create prompt template for sentiment analysis
        prompt_template = ChatPromptTemplate.from_template(
            """
        You are a sentiment analysis system analyzing user feedback about AI agent outputs.

        TASK:
        Analyze the sentiment of the provided feedback text and determine:
        1. The overall sentiment class (positive, negative, or neutral)
        2. A polarity score from -1.0 (most negative) to +1.0 (most positive)

        CONTEXT:
        This feedback is from a user commenting on an AI agent's output or behavior.
        Consider the emotional tone, satisfaction level, and overall sentiment expressed.

        OUTPUT FORMAT:
        Return your response as XML with this exact structure:

        <sentiment_analysis>
            <class_label>positive|negative|neutral</class_label>
            <polarity_score>SCORE_BETWEEN_-1.0_AND_1.0</polarity_score>
        </sentiment_analysis>

        GUIDELINES:
        - positive: User is satisfied, pleased, or approving (score: 0.1 to 1.0)
        - negative: User is dissatisfied, displeased, or disapproving (score: -1.0 to -0.1)
        - neutral: User is neither positive nor negative, or mixed (score: -0.1 to 0.1)
        - Be precise with the polarity score to reflect intensity

        Analyze this feedback: {feedback_text}
        """
        )

        # Invoke LLM
        prompt = prompt_template.format_messages(feedback_text=feedback_text)
        response = self.llm.invoke(prompt)

        # Extract and parse XML content
        xml_content = self._extract_xml_content(response.content)
        sentiment_data = self._parse_sentiment_xml(xml_content)

        return SentimentAnalysisResults(**sentiment_data)

    def parse_feedbacks(self) -> Tuple[ProcessedLogs, FeedbackMap]:
        """
        Main feedback processing pipeline that processes all feedback events.

        This function orchestrates the entire feedback attribution workflow:
        1. Fetches feedback events (returns early if None)
        2. Iterates through each feedback event
        3. Creates a query from the feedback
        4. Reranks processed logs based on relevance to the query
        5. Selects the most relevant step (threshold > 0.75)
        6. For HUMAN feedback: performs sentiment analysis
        7. Assigns rewards based on feedback type and sentiment
        8. Updates processed logs with new reward values

        Returns:
            Tuple[ProcessedLogs, FeedbackMap]: Updated processed logs with adjusted rewards
                and a map of all feedback results for tracking and analysis
        """
        # Fetch feedback events first
        feedback = self.fetch_feedback_events()

        # If no feedback content, skip all processing
        if feedback is None:
            self.logger.info("No feedback events to process. Skipping feedback processing.")
            return self.processed_logs, FeedbackMap(batch_id="", results=[])

        feedback_results: List[FeedbackResult] = []
        current_processed_logs = self.processed_logs

        self.logger.info(f"Starting feedback processing for batch: {feedback.metadata.batch_id}")
        self.logger.info(f"Processing {len(feedback.feedback_events)} feedback events")
        
        # Iterate through each feedback event
        for idx, feedback_event in enumerate(self.feedback.feedback_events):
            self.logger.info(f"Processing feedback event {idx + 1}/{len(self.feedback.feedback_events)}: {feedback_event.event_id}")
            
            try:
                # Step 1: Create query from feedback event
                query = self.create_query(feedback_event)
                self.logger.info(f"Created query: {query}")
                
                # Step 2: Rerank processed logs based on query relevance
                rerank_result = self.rerank_feedback(
                    processed_logs=current_processed_logs,
                    query=query,
                    top_n=3
                )
                self.logger.info(f"Reranking complete, got {len(rerank_result.feedback_documents)} ranked documents")
                
                # Step 3: Select the most relevant step (threshold > 0.75)
                selected_document = self.step_selection(rerank_result)
                
                if selected_document is None:
                    self.logger.warning(
                        f"No document met relevance threshold (>0.75) for feedback event {feedback_event.event_id}. Skipping."
                    )
                    continue
                
                self.logger.info(
                    f"Selected document with relevance score: {selected_document.relevance_score}"
                )
                
                # Step 4: Assign feedback reward and update logs
                # This function handles both HUMAN (with sentiment analysis) and EVENT feedback types
                feedback_result, current_processed_logs = self.assign_feedback_reward(
                    feedback_document=selected_document,
                    feedback_event=feedback_event,
                    processed_logs=current_processed_logs
                )
                
                feedback_results.append(feedback_result)
                
                self.logger.info(
                    f"Feedback reward assigned: {feedback_result.reward_feedback} for event {feedback_event.event_id}"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error processing feedback event {feedback_event.event_id}: {str(e)}",
                    exc_info=True
                )
                continue
        
        # Create FeedbackMap with all results
        feedback_map = FeedbackMap(
            batch_id=self.feedback.metadata.batch_id,
            results=feedback_results
        )
        
        self.logger.info(
            f"Feedback processing complete. Processed {len(feedback_results)} events successfully."
        )
        
        return current_processed_logs, feedback_map

    def _extract_xml_content(self, content: str) -> str:
        """
        Extract XML content from LLM response, handling markdown wrapping.

        Args:
            content: Raw LLM response content

        Returns:
            Clean XML content
        """
        # Remove markdown code blocks if present
        if "```xml" in content:
            xml_match = re.search(r"```xml\s*(.*?)\s*```", content, re.DOTALL)
            if xml_match:
                content = xml_match.group(1)
        elif "```" in content:
            xml_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if xml_match:
                content = xml_match.group(1)

        # Look for XML content between <sentiment_analysis> tags
        xml_pattern = r"<sentiment_analysis>.*?</sentiment_analysis>"
        xml_match = re.search(xml_pattern, content, re.DOTALL)

        if xml_match:
            return xml_match.group(0)
        else:
            return content.strip()

    def _parse_sentiment_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse XML response and extract sentiment analysis data.

        Args:
            xml_content: XML string containing sentiment analysis

        Returns:
            Dictionary with class_label and polarity_score
        """
        try:
            root = ET.fromstring(xml_content)

            class_label = self._get_xml_text(root, "class_label")
            polarity_score_str = self._get_xml_text(root, "polarity_score")

            # Validate and convert polarity score
            polarity_score = float(polarity_score_str)
            polarity_score = max(-1.0, min(1.0, polarity_score))  # Clamp to [-1, 1]

            # Validate class label
            if class_label not in ["positive", "negative", "neutral"]:
                # Default to neutral if invalid
                class_label = "neutral"

            return {
                "class_label": class_label,
                "polarity_score": polarity_score
            }

        except (ET.ParseError, ValueError) as e:
            self.logger.warning(f"XML parsing error: {e}. Using regex fallback.")
            return self._parse_sentiment_xml_with_regex(xml_content)

    def _parse_sentiment_xml_with_regex(self, xml_content: str) -> Dict[str, Any]:
        """
        Fallback regex-based XML parsing for sentiment analysis.

        Args:
            xml_content: XML string to parse

        Returns:
            Dictionary with class_label and polarity_score
        """
        class_label = self._extract_tag_content(xml_content, "class_label")
        polarity_score_str = self._extract_tag_content(xml_content, "polarity_score")

        # Default values
        if not class_label or class_label not in ["positive", "negative", "neutral"]:
            class_label = "neutral"

        try:
            polarity_score = float(polarity_score_str)
            polarity_score = max(-1.0, min(1.0, polarity_score))
        except (ValueError, TypeError):
            polarity_score = 0.0

        return {
            "class_label": class_label,
            "polarity_score": polarity_score
        }

    def _get_xml_text(self, element: ET.Element, tag_name: str) -> str:
        """
        Safely extract text from XML element.

        Args:
            element: XML element to search in
            tag_name: Tag name to find

        Returns:
            Text content or empty string if not found
        """
        child = element.find(tag_name)
        return child.text.strip() if child is not None and child.text else ""

    def _extract_tag_content(self, xml_string: str, tag_name: str) -> str:
        """
        Extract content from a specific XML tag using regex.

        Args:
            xml_string: XML string to search in
            tag_name: Name of the tag to extract

        Returns:
            Content of the tag or empty string if not found
        """
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, xml_string, re.DOTALL)
        return match.group(1).strip() if match else ""



