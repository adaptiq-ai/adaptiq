import datetime
import json
import os
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ScenarioSimulator:
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
        Task Type: {components["task_type"]}
        Last Action: {components["last_action"]}
        Last Outcome: {components["outcome_type"]}
        Current Context: {components["context"]}
        """

        return description

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
