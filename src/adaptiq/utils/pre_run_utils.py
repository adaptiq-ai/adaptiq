import datetime
import json
import os
import re
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adaptiq.instrumental.instrumental import (
    capture_llm_response,
    instrumental_track_tokens,
)


class AdaptiqHypotheticalStateGenerator:
    """
    Generator that transforms a parsed plan into hypothetical state-action pairs
    using a single LLM invocation. The LLM returns structured state-action mappings.
    """

    def __init__(
        self,
        prompt_parsed_plan: List[Dict],
        model_name: str,
        api_key: str,
        provider: str,
    ):
        """
        Initialize with a parsed plan.

        Args:
            prompt_parsed_plan: List of dictionaries containing intended steps.
        """
        self.prompt_parsed_plan = prompt_parsed_plan
        self.api_key = api_key
        self.model = model_name
        self.provider = provider

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        # Updated prompt to prevent JSON formatting issues with escaped quotes
        self.prompt_template = ChatPromptTemplate.from_template(
            """
        You are an RL state-action pair generator for agent training.

        TASK:
        For EACH step in the provided plan, create a corresponding state-action pair.

        STATE FORMAT (as tuple string):
        ('Current_SubTask_Category', 'Last_Action_Taken', 'Last_Outcome_Category', 'Key_Context')

        Where:
        - Current_SubTask_Category: Categorize from: [InitialQuery, InformationRetrieval_Company, InformationRetrieval_Lead, ContentDrafting, ContentReview, ActionExecution_SendEmail, ResultFinalization, PlanningNextStep, UnknownSubTask]
        - Last_Action_Taken: The Intended_Action from previous step (use "None" for first step)
        - Last_Outcome_Category: Categorize from: [Success_DataFound, Success_ActionCompleted, Success_NoDataFound, Failure_PreconditionNotMet, Outcome_Unknown, None]
        - Key_Context: 1-3 keywords (max 3 words) summarizing information up to this point

        ACTION:
        The Intended_Action from the current step.

        OUTPUT:
        A JSON list where each item has:
        1. "hypothetical_state_representation": The 4-element tuple string
        2. "hypothetical_action": The current step's Intended_Action
        3. "source_prompt_step_details": Copy of the original step object

        IMPORTANT: Use single quotes inside JSON strings to avoid escaping issues.
        For example, write "Expected_Ideal_Outcome_Mentioned_In_Prompt": "Lead's name is retrieved."
        instead of using escaped double quotes.

        Parse this plan: {parsed_plan}
        """
        )

    def configure_llm(self, llm_instance: ChatOpenAI, prompt_template: str = None):
        """
        Configure the LLM and prompt template.

        Args:
            llm_instance: LLM to use for state representation
            prompt_template: Optional custom prompt template
        """
        self.llm = llm_instance
        if prompt_template:
            self.prompt_template = ChatPromptTemplate.from_template(prompt_template)

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def generate_hypothetical_state_action_pairs(self) -> List[Dict]:
        """
        Generate all hypothetical state-action pairs in a single LLM call.

        Returns:
            List of state-action pairs with detailed step context.
        """
        # Prepare full context for LLM
        context = {"parsed_plan": self.prompt_parsed_plan}

        prompt = self.prompt_template.format_messages(**context)
        response = self.llm.invoke(prompt)
        capture_llm_response(response)

        # Get the content from the response
        content = response.content

        # Check if response is wrapped in markdown code blocks (```json ... ```)
        if "```" in content:
            # Extract content between code blocks
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(code_block_pattern, content)
            if matches:
                # Use the first matched code block
                content = matches[0].strip()

        # Fix JSON before parsing
        try:
            # Apply comprehensive JSON fixing
            fixed_content = self._comprehensive_json_fix(content)
            result = json.loads(fixed_content)
        except Exception as e:
            # If parsing still fails, try a more aggressive approach
            try:
                result = self._parse_json_manually(content)
            except Exception as inner_e:
                # If all parsing attempts fail, provide detailed error info
                error_msg = f"Failed to parse LLM response as JSON: {e}\n\n"
                error_msg += f"Processed content:\n{content}\n\n"
                error_msg += f"Original response:\n{response.content}"
                raise ValueError(error_msg) from inner_e

        return result

    def _comprehensive_json_fix(self, json_str: str) -> str:
        """
        Comprehensive JSON fixing that handles various issues.

        Args:
            json_str: JSON string with potential issues

        Returns:
            Fixed JSON string
        """
        # Step 1: Normalize line endings
        fixed_str = json_str.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: Handle escaped quotes properly
        # Replace \" with ' when it appears within strings (between quotes)
        fixed_str = re.sub(r"(\"[^\"]*?)\\\"([^\"]*?\")", r"\1\'\2", fixed_str)

        # Step 3: Fix control characters
        control_chars = ["\b", "\f", "\n", "\r", "\t"]
        for char in control_chars:
            # Replace unescaped control characters within strings
            fixed_str = re.sub(f'(?<="[^"]*){char}(?=[^"]*")', " ", fixed_str)

        # Step 4: Fix quotes around JSON keys
        # This finds JSON keys that are not properly quoted
        fixed_str = re.sub(
            r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', fixed_str
        )

        # Step 5: Fix issues with apostrophes
        # Convert apostrophes in words to avoid JSON parsing issues
        fixed_str = re.sub(r'(\w)"(\w)', r"\1'\2", fixed_str)

        # Step 6: Fix missing quotes around string values
        # This is more complex and might need refinement for specific cases

        return fixed_str

    def _parse_json_manually(self, content: str) -> List[Dict]:
        """
        Manual JSON parsing for cases where automatic parsing fails.
        Implements a simplified parser for the specific structure we expect.

        Args:
            content: The JSON-like string to parse

        Returns:
            List of dictionaries representing the parsed JSON
        """
        result = []

        # Pattern to match each JSON object in the array
        object_pattern = r'{\s*"hypothetical_state_representation":\s*"([^"]*)",\s*"hypothetical_action":\s*"([^"]*)",\s*"source_prompt_step_details":\s*{([^}]*)}\s*}'

        # Find all objects in the content
        objects = re.findall(object_pattern, content, re.DOTALL)

        for obj in objects:
            state_repr = obj[0]
            action = obj[1]
            details_str = obj[2]

            # Parse the details dictionary
            details = {}
            detail_pattern = r'"([^"]*)"\s*:\s*"([^"]*)"'
            detail_matches = re.findall(detail_pattern, details_str)

            for key, value in detail_matches:
                details[key] = value

            # Create the result dictionary
            result.append(
                {
                    "hypothetical_state_representation": state_repr,
                    "hypothetical_action": action,
                    "source_prompt_step_details": details,
                }
            )

        return result

    def clean_representation(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Cleaner for LLM output. Handles escaped quotes and properly formats JSON.

        Args:
            raw_data: Raw state-action pairs

        Returns:
            Cleaned list of dictionaries
        """
        cleaned = []

        for item in raw_data:
            # Process any potentially problematic fields
            details = item.get("source_prompt_step_details", {})
            if isinstance(details, dict):
                # Deep clone to avoid modifying the original
                cleaned_details = {}
                for key, value in details.items():
                    # Handle escaped quotes in string values
                    if isinstance(value, str):
                        # Normalize the string value to avoid escaped quote issues
                        cleaned_details[key] = value.replace('\\"', '"').replace(
                            '\\"', '"'
                        )
                    else:
                        cleaned_details[key] = value
            else:
                cleaned_details = details

            cleaned.append(
                {
                    "state": item["hypothetical_state_representation"],
                    "action": item["hypothetical_action"],
                    "details": cleaned_details,
                }
            )

        return cleaned


class AdaptiqPromptConsulting:
    """
    Consultant that analyzes a given prompt and provides structured feedback
    using a single LLM invocation. The LLM returns structured analysis and recommendations.
    """

    def __init__(self, agent_prompt: str, model_name: str, api_key: str, provider: str):
        """
        Initialize with an agent prompt to analyze.

        Args:
            agent_prompt: The prompt text to be analyzed.
        """
        self.agent_prompt = agent_prompt
        self.api_key = api_key
        self.model = model_name
        self.provider = provider

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        self.analysis_template = ChatPromptTemplate.from_template(
            """
        You are an expert Prompt Consultant for AI Agents.
        You will be given a prompt intended to guide the behavior of an AI agent.
        Your tasks:
        1. Summarize what the prompt is trying to accomplish.
        2. Identify any potential weaknesses, ambiguities, or inconsistencies.
        3. Suggest specific improvements or edits to make it more effective.
        4. Provide 3 best practices for structuring prompts like this.
        5. Highlight any missing components (e.g., role definition, output format, input expectations).

        Here is the prompt to analyze:
        {agent_prompt}

        Respond in the following structured format:
        {{
        "summary": "...",
        "weaknesses": ["...", "..."],
        "suggested_modifications": ["...", "..."],
        "best_practices": ["...", "...", "..."],
        "missing_components": ["...", "..."]
        }}
        """
        )

    def configure_llm(self, llm_instance: ChatOpenAI, prompt_template: str = None):
        """
        Configure the LLM and prompt template.

        Args:
            llm_instance: LLM to use for prompt analysis
            prompt_template: Optional custom prompt template
        """
        self.llm = llm_instance
        if prompt_template:
            self.analysis_template = ChatPromptTemplate.from_template(prompt_template)

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def analyze_prompt(self) -> Dict:
        """
        Analyze the prompt and generate structured feedback in a single LLM call.

        Returns:
            Dictionary containing analysis and recommendations.
        """
        # Prepare context for LLM
        context = {"agent_prompt": self.agent_prompt}

        prompt = self.analysis_template.format_messages(**context)
        response = self.llm.invoke(prompt)
        capture_llm_response(response)

        # Try parsing JSON from response
        try:
            result = json.loads(response.content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\n\nResponse:\n{response.content}"
            )

        return result

    def get_formatted_analysis(self, raw_analysis: Dict = None) -> Dict:
        """
        Format the analysis results or get a new analysis if none provided.

        Args:
            raw_analysis: Optional pre-generated analysis results

        Returns:
            Formatted analysis dictionary
        """
        if raw_analysis is None:
            raw_analysis = self.analyze_prompt()

        # Ensure all expected keys exist
        formatted_analysis = {
            "summary": raw_analysis.get("summary", ""),
            "weaknesses": raw_analysis.get("weaknesses", []),
            "suggested_modifications": raw_analysis.get("suggested_modifications", []),
            "best_practices": raw_analysis.get("best_practices", []),
            "missing_components": raw_analysis.get("missing_components", []),
        }

        return formatted_analysis


class AdaptiqScenarioSimulator:
    """
    AdaptiqScenarioSimulator takes the output from AdaptiqHypotheticalStateGenerator
    and generates multiple plausible execution scenarios for each step using an LLM.

    Each scenario includes:
    - A description of the simulated outcome
    - An estimated reward for that outcome
    - The key features of the hypothetical next state resulting from that outcome

    The output is a list of simulated (s_hypothetical_features, a_intended, r_simulated, s_prime_hypothetical_features)
    data structures for Q-table warm-up.
    """

    def __init__(
        self,
        hypothetical_states: List[Dict],
        model_name: str,
        api_key: str,
        provider: str,
        output_path: str,
    ):
        """
        Initialize the AdaptiqScenarioSimulator with the hypothetical states and OpenAI credentials.

        Args:
            hypothetical_states: Output from AdaptiqHypotheticalStateGenerator
            model_name: OpenAI model name to use (e.g., "gpt-4-turbo")
            api_key: OpenAI API key
            output_path: The save path of the results
        """
        self.hypothetical_states = hypothetical_states
        self.output_path = output_path
        self.provider = provider

        if self.provider == "openai":
            self.scenario_generation_llm = ChatOpenAI(
                temperature=0.2, model=model_name, api_key=api_key
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        # Set up the prompt template
        self.scenario_generation_prompt_template = ChatPromptTemplate.from_template(
            """
            You are an AI Agent Scenario Simulator.

            The agent is currently in a hypothetical state described by:
            {current_state_description}

            The agent intends to perform the action: "{intended_action}".

            Additional details about this step:
            {step_details}

            IMPORTANT: The next subtask after this action will be: {next_subtask}

            Generate 3 plausible and distinct outcome scenarios:

            1. Ideal Success Scenario  
            - The agent uses the original intended action: "{intended_action}"  
            - The expected outcome is fully achieved.

            2. Common Failure Scenario  
            - The agent uses a different action than "{intended_action}" (e.g., wrong tool or method), leading to failure.  
            - ⚠️ The simulated action for this scenario must NOT be "{intended_action}".

            3. Partial Success or Unexpected Outcome Scenario  
            - The agent uses the original intended action: "{intended_action}"  
            - The outcome is partially successful or unexpected but plausible.

            For EACH scenario, provide ONLY the following JSON fields:
            - "scenario_type": (One of: "ideal_success", "common_failure", "partial_success")
            - "simulated_action": (The action/tool actually used in this scenario — must be the same as used in the outcome and next state)
            - "simulated_outcome_description": (Brief explanation of what happened)
            - "reward_sim": (A float indicating how beneficial this outcome is for the overall task — nuanced, not binary)
            - "next_state_components": A 4-element tuple in this format:  
            ({next_subtask}, simulated_action, outcome_type, context)
            - "key_context_changes": (Dictionary describing key changes in context after the outcome)

            Notes:
            - The "simulated_action" must match the action mentioned in "next_state_components".
            - For "common_failure", "simulated_action" MUST NOT be "{intended_action}".
            - For "ideal_success" and "partial_success", "simulated_action" MUST be "{intended_action}".

            Output a JSON list of exactly 3 scenario objects as described above.
            """
        )

        # Initialize storage for simulated traces
        self.simulated_traces = []

    def _parse_state_tuple(self, state_str: str) -> Dict:
        """
        Parse a state tuple string into its components.

        Args:
            state_str: String representation of state tuple, e.g., "('InformationRetrieval_Company', 'None', 'None', 'company background')"

        Returns:
            Dictionary with the parsed components
        """
        # Remove parentheses and split by commas
        clean_str = state_str.strip("()").replace("'", "")
        components = clean_str.split(", ")

        if len(components) >= 4:
            return {
                "task_type": components[0],
                "last_action": components[1],
                "outcome_type": components[2],
                "context": components[3],
            }
        else:
            return {
                "task_type": "Unknown",
                "last_action": "Unknown",
                "outcome_type": "Unknown",
                "context": "Unknown",
            }

    def _format_state_description(self, state_str: str) -> str:
        """
        Format a state tuple string into a human-readable description.

        Args:
            state_str: String representation of state tuple

        Returns:
            Human-readable description of the state
        """
        components = self._parse_state_tuple(state_str)

        description = f"""
        Task Type: {components['task_type']}
        Last Action: {components['last_action']}
        Last Outcome: {components['outcome_type']}
        Current Context: {components['context']}
        """

        return description

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def _invoke_llm_for_scenario_generation(
        self, state_str: str, intended_action: str, step_details: Dict
    ) -> List[Dict]:
        """
        Invoke the LLM to generate scenarios for the current state and intended action.

        Args:
            state_str: String representation of the current state
            intended_action: The intended action to be performed
            step_details: Additional details about this step

        Returns:
            List of scenarios generated by the LLM
        """
        try:
            # Format the state description
            state_description = self._format_state_description(state_str)

            # Format the step details for the prompt
            formatted_step_details = json.dumps(step_details, indent=2)

            # Determine the next subtask from the step details or provide a default
            # Extract from Intended_SubTask if available
            next_subtask = "Unknown"

            # If this isn't the last state in our list, try to get the next subtask
            current_index = -1
            for i, state in enumerate(self.hypothetical_states):
                if state["state"] == state_str and state["action"] == intended_action:
                    current_index = i
                    break

            if current_index >= 0 and current_index < len(self.hypothetical_states) - 1:
                # Get the next state's task type from its state tuple
                next_state_str = self.hypothetical_states[current_index + 1]["state"]
                next_state_components = self._parse_state_tuple(next_state_str)
                next_subtask = next_state_components["task_type"]
            else:
                # If this is the last state or we couldn't find it, use "GenerateFinalAnswer"
                next_subtask = "GenerateFinalAnswer"

            # Prepare the prompt inputs
            prompt_inputs = {
                "current_state_description": state_description,
                "intended_action": intended_action,
                "step_details": formatted_step_details,
                "next_subtask": next_subtask,
            }

            # Create the prompt
            scenario_prompt = self.scenario_generation_prompt_template.format_messages(
                **prompt_inputs
            )

            # Call the LLM
            llm_response = self.scenario_generation_llm.invoke(scenario_prompt)
            capture_llm_response(llm_response)

            # Parse the JSON response
            response_content = llm_response.content

            # Extract JSON content from the response if it's embedded in markdown or other text
            json_start = response_content.find("[")
            json_end = response_content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_content = response_content[json_start:json_end]
                scenarios = json.loads(json_content)
            else:
                # Fallback parsing approach
                scenarios = json.loads(response_content)

            return scenarios

        except Exception as e:
            print(f"Error in scenario generation: {e}")
            # Return a default scenario set in case of error
            state_components = self._parse_state_tuple(state_str)

            return [
                {
                    "scenario_type": "ideal_success",
                    "simulated_action": intended_action,  # Same as intended for success
                    "simulated_outcome_description": "Default success scenario due to error in LLM response parsing.",
                    "reward_sim": 0.5,
                    "next_state_components": f"('{next_subtask}', '{intended_action}', 'Success', '{state_components['context']}')",
                    "key_context_changes": {"success": True},
                },
                {
                    "scenario_type": "common_failure",
                    "simulated_action": f"Wrong{intended_action}",  # Different from intended for failure
                    "simulated_outcome_description": "Default failure scenario due to error in LLM response parsing.",
                    "reward_sim": -0.5,
                    "next_state_components": f"('{next_subtask}', 'Wrong{intended_action}', 'Failure', '{state_components['context']}')",
                    "key_context_changes": {"error": True},
                },
                {
                    "scenario_type": "partial_success",
                    "simulated_action": intended_action,  # Same as intended for partial success
                    "simulated_outcome_description": "Default partial success scenario due to error in LLM response parsing.",
                    "reward_sim": 0.0,
                    "next_state_components": f"('{next_subtask}', '{intended_action}', 'PartialSuccess', '{state_components['context']}')",
                    "key_context_changes": {"partial_success": True},
                },
            ]

    def generate_simulated_scenarios(self) -> List[Dict]:
        """
        Generate simulated scenarios for each step in the hypothetical states
        and save the results to the specified output path.

        Returns:
            List of dictionaries representing simulated scenario steps
        """
        all_simulated_steps = []

        # Iterate through each hypothetical state
        for state_data in self.hypothetical_states:
            # Extract the state, action, and details
            state_str = state_data["state"]
            intended_action = state_data["action"]
            step_details = state_data.get("details", {})

            # Generate scenarios for this step
            generated_scenarios = self._invoke_llm_for_scenario_generation(
                state_str, intended_action, step_details
            )

            # Process each generated scenario
            for scenario in generated_scenarios:
                # Create the simulated step
                simulated_step = {
                    "original_state": state_str,
                    "intended_action": intended_action,
                    # Add this line to include the simulated action
                    "simulated_action": scenario.get(
                        "simulated_action", intended_action
                    ),
                    "scenario_type": scenario.get("scenario_type", "unknown"),
                    "simulated_outcome": scenario.get(
                        "simulated_outcome_description", "Unknown outcome"
                    ),
                    "reward_sim": scenario.get("reward_sim", 0.0),
                    "next_state": scenario.get("next_state_components", "Unknown"),
                    "key_context_changes": scenario.get("key_context_changes", {}),
                    "source_details": step_details,
                }

                # Add the simulated step to the list
                all_simulated_steps.append(simulated_step)

        # Store the simulated traces
        self.simulated_traces = all_simulated_steps

        # Save the simulated traces to the specified output path if provided
        if hasattr(self, "output_path") and self.output_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

                # Save the scenarios to a JSON file
                with open(self.output_path, "w") as f:
                    json.dump(
                        {
                            "simulated_scenarios": all_simulated_steps,
                            "generation_timestamp": datetime.datetime.now().isoformat(),
                            "total_scenarios": len(all_simulated_steps),
                        },
                        f,
                        indent=2,
                    )

                print(
                    f"Successfully saved {len(all_simulated_steps)} simulated scenarios to {self.output_path}"
                )
            except Exception as e:
                print(f"Error saving scenarios to {self.output_path}: {e}")

        return all_simulated_steps
