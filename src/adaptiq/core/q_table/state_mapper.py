import json
from langchain_core.prompts import ChatPromptTemplate
from adaptiq.core.abstract.q_table.base_state_mapper import BaseStateMapper
from adaptiq.core.entities import StateActionMapping, ClassificationResponse, Classification


class StateMapper(BaseStateMapper):
    """
    AdaptiqStateMapper - Matches execution trace states with Q-table states.

    Takes the "Warmed Q-table" (from previous runs) and matches input states to see
    if they correspond to any known state from the Q-table, ignoring actions completely.
    """


    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for state classification."""
        classification_template = """You are an AI Agent State Classifier specializing in semantic matching.

        # Input State to Classify:
        ```
        {input_state}
        ```

        # KNOWN STATES (Find the closest semantic match):
        ```
        {known_states}
        ```

        # Your Task:
        1. Analyze the provided input state components WITHOUT considering any action values.
        2. Find the semantically closest match from the known states list.
        3. Focus ONLY on matching the core state concepts (ignore syntax differences between arrays/tuples).
        4. Pay attention to the semantic meaning of the state components:
           - First component: Task/phase name (e.g., "RetrieveCompanyInfo" vs "InformationRetrieval_Company")
           - Second component: Previous tool used
           - Third component: Status/outcome
           - Fourth component: Context/data description

        # Examples of semantic matches:
        - "RetrieveCompanyInfo" could match with "InformationRetrieval_Company" (both about company info retrieval)
        - "CompanyData" could match with "company background" (both about company information)
        - "None" should match with "None" (both indicate no previous state)

        Output a JSON object with these fields:
        ```
        {{
        "classification": {{
            "is_known_state": true/false,
            "matched_state": "The exact matching state from the known states if found, null if not found",
            "reasoning": "Explanation of why this state matches or doesn't match a known state, with component-by-component comparison"
        }}
        }}
        ```

        CRITICAL REQUIREMENTS:
        - IGNORE any "action" field in the input state - ONLY match on the state components
        - If the input is a dictionary with a "state" key, extract and use ONLY that state field
        - Find the CLOSEST semantic match, not just exact string matches
        - You must return the EXACT matching state string from the known states without modification
        - Only return is_known_state: true if there's a clear semantic match
        - Be thorough in your reasoning, explaining similarities and differences by component

        Output ONLY the JSON object, no additional text."""

        return ChatPromptTemplate.from_template(classification_template)

    def _invoke_llm_for_classification(self, input_state: StateActionMapping) -> ClassificationResponse:
        """
        Invoke the LLM to classify a state.

        Args:
            input_state: State to classify (can be string, list, or dict)

        Returns:
            Dict containing the LLM's classification output
        """


        # Create formatted known states for better comparison
        formatted_known_states = []
        for original, parsed in self.parsed_states:
            formatted_known_states.append({"original": original, "components": parsed})

        # Create inputs for the LLM
        inputs = {
            "input_state": input_state.state,
            "known_states": json.dumps(formatted_known_states, indent=2),
        }

        # Create and invoke the prompt
        prompt = self.classification_prompt_template.format_messages(**inputs)
        response = self.llm.invoke(prompt)

        # Parse the response content as JSON
        try:
            classification_output = json.loads(response.content)
            return classification_output
        except json.JSONDecodeError:
            # If JSON parsing fails, extract just the JSON portion
            content = response.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_content = content[start_idx:end_idx]
                try:
                    return ClassificationResponse(**json.loads(json_content))
                except json.JSONDecodeError:
                    pass
        
            return ClassificationResponse(
                classification= Classification(
                    is_known_state=False,
                    matched_state=None,
                    reasoning="Error parsing LLM output",
                )
            )
