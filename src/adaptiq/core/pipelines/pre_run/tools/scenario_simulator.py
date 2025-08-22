import datetime
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from adaptiq.core.entities import HypotheticalStateRepresentation, ScenarioModel


class ScenarioSimulator:
    """
    AdaptiqScenarioSimulator takes the output from AdaptiqHypotheticalStateGenerator
    and generates multiple plausible execution scenarios for each step using an LLM with XML output.

    Each scenario includes:
    - A description of the simulated outcome
    - An estimated reward for that outcome
    - The key features of the hypothetical next state resulting from that outcome

    The output is a list of simulated (s_hypothetical_features, a_intended, r_simulated, s_prime_hypothetical_features)
    data structures for Q-table warm-up.
    """

    def __init__(
        self,
        hypothetical_states: List[HypotheticalStateRepresentation],
        llm: BaseChatModel,
        output_path: str,
    ):
        """
        Initialize the AdaptiqScenarioSimulator with the hypothetical states and OpenAI credentials.

        Args:
            hypothetical_states: List of HypotheticalStateRepresentation objects representing the hypothetical states
            llm: BaseChatModel instance for LLM interactions
            output_path: Path to save the generated scenarios
        """
        self.hypothetical_states = hypothetical_states
        self.output_path = output_path
        self.scenario_generation_llm=llm

        # XML-based prompt template
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

            OUTPUT FORMAT:
            Return your response as XML with this exact structure:

            <scenarios>
                <scenario>
                    <scenario_type>TYPE_HERE</scenario_type>
                    <simulated_action>ACTION_HERE</simulated_action>
                    <simulated_outcome_description>DESCRIPTION_HERE</simulated_outcome_description>
                    <reward_sim>FLOAT_VALUE</reward_sim>
                    <next_state_components>TUPLE_STRING_HERE</next_state_components>
                    <key_context_changes>
                        <change key="KEY">VALUE</change>
                        <!-- Repeat for each context change -->
                    </key_context_changes>
                </scenario>
                <!-- Repeat <scenario> for each of the 3 scenarios -->
            </scenarios>

            CRITICAL FORMAT REQUIREMENTS:
            - scenario_type must be one of: "ideal_success", "common_failure", "partial_success"
            - The "next_state_components" must be a STRING representation of a tuple in format: "('next_subtask', 'simulated_action', 'outcome_type', 'context')"
            - ALL 4 elements in the tuple must be simple strings, never dictionaries or complex objects
            - reward_sim must be a float value indicating benefit for the overall task
            - Use clear, concise text and avoid special characters that might break XML
            - For "common_failure", "simulated_action" MUST NOT be "{intended_action}"
            - For "ideal_success" and "partial_success", "simulated_action" MUST be "{intended_action}"

            Example next_state_components: "('InformationRetrieval_Company', 'search_web', 'Success', 'company background research')"
            """
        )

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

    def _parse_tuple_string(self, tuple_str: str) -> tuple:
        """
        Safely parse a string representation of a tuple into an actual tuple.
        
        Args:
            tuple_str: String representation like "('a', 'b', 'c', 'd')"
            
        Returns:
            Tuple with 4 string elements
        """
        try:
            # Remove outer parentheses and quotes, then split by commas
            clean_str = tuple_str.strip().strip("()")
            
            # Handle quoted elements
            elements = []
            current_element = ""
            in_quotes = False
            quote_char = None
            
            i = 0
            while i < len(clean_str):
                char = clean_str[i]
                
                if not in_quotes and (char == "'" or char == '"'):
                    in_quotes = True
                    quote_char = char
                elif in_quotes and char == quote_char:
                    # Check if it's an escaped quote
                    if i + 1 < len(clean_str) and clean_str[i + 1] == quote_char:
                        current_element += char
                        i += 1  # Skip the next quote
                    else:
                        in_quotes = False
                        quote_char = None
                elif not in_quotes and char == ',':
                    elements.append(current_element.strip())
                    current_element = ""
                else:
                    current_element += char
                
                i += 1
            
            # Add the last element
            if current_element.strip():
                elements.append(current_element.strip())
            
            # Ensure we have exactly 4 elements
            while len(elements) < 4:
                elements.append("Unknown")
            
            # Truncate if more than 4 elements
            elements = elements[:4]
            
            return tuple(elements)
            
        except Exception as e:
            print(f"Error parsing tuple string '{tuple_str}': {e}")
            return ("Unknown", "Unknown", "Unknown", "Unknown")

    def _extract_xml_content(self, content: str) -> str:
        """
        Extract XML content from LLM response, handling potential markdown wrapping.
        
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
            # Handle generic code blocks
            xml_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if xml_match:
                content = xml_match.group(1)

        # Look for XML content between <scenarios> tags
        xml_pattern = r"<scenarios>.*?</scenarios>"
        xml_match = re.search(xml_pattern, content, re.DOTALL)
        
        if xml_match:
            return xml_match.group(0)
        else:
            # If no wrapper found, assume the entire content is XML
            return content.strip()

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

    def _parse_xml_response(self, xml_content: str) -> List[Dict]:
        """
        Parse XML response and extract scenario data.
        
        Args:
            xml_content: XML string containing scenarios
            
        Returns:
            List of dictionaries with parsed scenario data
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            scenarios = []
            
            for scenario_elem in root.findall('scenario'):
                # Extract basic scenario information
                scenario_type = self._get_xml_text(scenario_elem, 'scenario_type')
                simulated_action = self._get_xml_text(scenario_elem, 'simulated_action')
                outcome_description = self._get_xml_text(scenario_elem, 'simulated_outcome_description')
                reward_sim_str = self._get_xml_text(scenario_elem, 'reward_sim')
                next_state_components = self._get_xml_text(scenario_elem, 'next_state_components')
                
                # Parse reward as float
                try:
                    reward_sim = float(reward_sim_str) if reward_sim_str else 0.0
                except (ValueError, TypeError):
                    reward_sim = 0.0
                
                # Parse key context changes
                key_context_changes = {}
                context_changes_elem = scenario_elem.find('key_context_changes')
                if context_changes_elem is not None:
                    for change_elem in context_changes_elem.findall('change'):
                        key = change_elem.get('key', '')
                        value = change_elem.text.strip() if change_elem.text else ''
                        if key:
                            key_context_changes[key] = value
                
                scenarios.append({
                    'scenario_type': scenario_type,
                    'simulated_action': simulated_action,
                    'simulated_outcome_description': outcome_description,
                    'reward_sim': reward_sim,
                    'next_state_components': next_state_components,
                    'key_context_changes': key_context_changes
                })
            
            return scenarios
            
        except ET.ParseError as e:
            # If XML parsing fails, try regex fallback
            print(f"XML parsing error: {e}. Attempting regex fallback.")
            return self._parse_xml_with_regex(xml_content)

    def _parse_xml_with_regex(self, xml_content: str) -> List[Dict]:
        """
        Fallback regex-based XML parsing for malformed XML.
        
        Args:
            xml_content: XML string to parse
            
        Returns:
            List of dictionaries with extracted scenario data
        """
        scenarios = []
        
        # Pattern to match each scenario block
        scenario_pattern = r'<scenario>(.*?)</scenario>'
        scenario_matches = re.findall(scenario_pattern, xml_content, re.DOTALL)
        
        for scenario_content in scenario_matches:
            # Extract individual components using regex
            scenario_type = self._extract_tag_content(scenario_content, 'scenario_type')
            simulated_action = self._extract_tag_content(scenario_content, 'simulated_action')
            outcome_description = self._extract_tag_content(scenario_content, 'simulated_outcome_description')
            reward_sim_str = self._extract_tag_content(scenario_content, 'reward_sim')
            next_state_components = self._extract_tag_content(scenario_content, 'next_state_components')
            
            # Parse reward as float
            try:
                reward_sim = float(reward_sim_str) if reward_sim_str else 0.0
            except (ValueError, TypeError):
                reward_sim = 0.0
            
            # Parse key context changes
            key_context_changes = {}
            context_changes_match = re.search(r'<key_context_changes>(.*?)</key_context_changes>', scenario_content, re.DOTALL)
            if context_changes_match:
                changes_content = context_changes_match.group(1)
                change_pattern = r'<change key="([^"]*)">(.*?)</change>'
                change_matches = re.findall(change_pattern, changes_content, re.DOTALL)
                for key, value in change_matches:
                    key_context_changes[key.strip()] = value.strip()
            
            scenarios.append({
                'scenario_type': scenario_type,
                'simulated_action': simulated_action,
                'simulated_outcome_description': outcome_description,
                'reward_sim': reward_sim,
                'next_state_components': next_state_components,
                'key_context_changes': key_context_changes
            })
        
        return scenarios

    def _extract_tag_content(self, xml_string: str, tag_name: str) -> str:
        """
        Extract content from a specific XML tag using regex.
        
        Args:
            xml_string: XML string to search in
            tag_name: Name of the tag to extract
            
        Returns:
            Content of the tag or empty string if not found
        """
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        match = re.search(pattern, xml_string, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _validate_and_fix_scenario(self, scenario: Dict, intended_action: str, next_subtask: str, context: str) -> Dict:
        """
        Validate and fix a scenario to ensure it meets the ScenarioModel requirements.
        
        Args:
            scenario: The scenario dictionary from LLM
            intended_action: The original intended action
            next_subtask: The next subtask name
            context: The context string
            
        Returns:
            Fixed scenario dictionary
        """
        # Ensure next_state_components is a string representation of a tuple
        next_state_components = scenario.get("next_state_components", "")
        
        # If it's not a string or doesn't look like a tuple, reconstruct it
        if not isinstance(next_state_components, str) or not next_state_components.startswith("("):
            simulated_action = scenario.get("simulated_action", intended_action)
            scenario_type = scenario.get("scenario_type", "unknown")
            
            # Map scenario type to outcome type
            outcome_type_map = {
                "ideal_success": "Success",
                "common_failure": "Failure", 
                "partial_success": "PartialSuccess"
            }
            outcome_type = outcome_type_map.get(scenario_type, "Unknown")
            
            # Ensure context is a simple string
            if isinstance(context, dict):
                context = str(context)
            
            next_state_components = f"('{next_subtask}', '{simulated_action}', '{outcome_type}', '{context}')"
            scenario["next_state_components"] = next_state_components
        
        # Ensure all required fields exist with defaults
        scenario.setdefault("scenario_type", "unknown")
        scenario.setdefault("simulated_action", intended_action)
        scenario.setdefault("simulated_outcome_description", "Unknown outcome")
        scenario.setdefault("reward_sim", 0.0)
        scenario.setdefault("key_context_changes", {})
        
        return scenario

    def _invoke_llm_for_scenario_generation(
        self, state_str: str, intended_action: str, step_details: Dict
    ) -> List[Dict]:
        """
        Invoke the LLM to generate scenarios for the current state and intended action using XML output.

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
            next_subtask = "Unknown"
            state_components = self._parse_state_tuple(state_str)
            context = state_components.get("context", "Unknown")

            # If this isn't the last state in our list, try to get the next subtask
            current_index = -1
            for i, state in enumerate(self.hypothetical_states):
                if state.state == state_str and state.action == intended_action:
                    current_index = i
                    break

            if current_index >= 0 and current_index < len(self.hypothetical_states) - 1:
                # Get the next state's task type from its state tuple
                next_state_str = self.hypothetical_states[current_index + 1].state
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

            # Extract XML content from response
            xml_content = self._extract_xml_content(llm_response.content)
            
            # Parse XML and extract scenarios
            scenarios = self._parse_xml_response(xml_content)

            # Validate and fix each scenario
            validated_scenarios = []
            for scenario in scenarios:
                fixed_scenario = self._validate_and_fix_scenario(
                    scenario, intended_action, next_subtask, context
                )
                validated_scenarios.append(fixed_scenario)

            return validated_scenarios

        except Exception as e:
            print(f"Error in scenario generation: {e}")
            # Return a default scenario set in case of error
            state_components = self._parse_state_tuple(state_str)
            context = state_components.get("context", "Unknown")

            # Determine next_subtask for fallback
            next_subtask = "Unknown"
            current_index = -1
            for i, state in enumerate(self.hypothetical_states):
                if state.state == state_str and state.action == intended_action:
                    current_index = i
                    break

            if current_index >= 0 and current_index < len(self.hypothetical_states) - 1:
                next_state_str = self.hypothetical_states[current_index + 1].state
                next_state_components = self._parse_state_tuple(next_state_str)
                next_subtask = next_state_components["task_type"]
            else:
                next_subtask = "GenerateFinalAnswer"

            return [
                {
                    "scenario_type": "ideal_success",
                    "simulated_action": intended_action,  # Same as intended for success
                    "simulated_outcome_description": "Default success scenario due to error in LLM response parsing.",
                    "reward_sim": 0.5,
                    "next_state_components": f"('{next_subtask}', '{intended_action}', 'Success', '{context}')",
                    "key_context_changes": {"success": "true"},
                },
                {
                    "scenario_type": "common_failure",
                    "simulated_action": f"Wrong{intended_action}",  # Different from intended for failure
                    "simulated_outcome_description": "Default failure scenario due to error in LLM response parsing.",
                    "reward_sim": -0.5,
                    "next_state_components": f"('{next_subtask}', 'Wrong{intended_action}', 'Failure', '{context}')",
                    "key_context_changes": {"error": "true"},
                },
                {
                    "scenario_type": "partial_success",
                    "simulated_action": intended_action,  # Same as intended for partial success
                    "simulated_outcome_description": "Default partial success scenario due to error in LLM response parsing.",
                    "reward_sim": 0.0,
                    "next_state_components": f"('{next_subtask}', '{intended_action}', 'PartialSuccess', '{context}')",
                    "key_context_changes": {"partial_success": "true"},
                },
            ]

    def generate_simulated_scenarios(self) -> List[ScenarioModel]:
        """
        Generate simulated scenarios for each step in the hypothetical states
        and save the results to the specified output path.

        Returns:
            List of ScenarioModel objects representing simulated scenario steps
        """
        all_simulated_steps: List[ScenarioModel] = []

        # Iterate through each hypothetical state
        for state_data in self.hypothetical_states:
            # Extract the state, action, and details
            state_str = state_data.state
            intended_action = state_data.action
            step_details = state_data.details

            # Generate scenarios for this step using XML parsing
            generated_scenarios = self._invoke_llm_for_scenario_generation(
                state_str, intended_action, step_details
            )

            # Process each generated scenario
            for scenario in generated_scenarios:
                try:
                    # Parse the next_state_components string into an actual tuple
                    next_state_str = scenario.get("next_state_components", "")
                    next_state_tuple = self._parse_tuple_string(next_state_str)
                    
                    # Validate reward_sim is a float
                    reward_sim = scenario.get("reward_sim", 0.0)
                    if not isinstance(reward_sim, (int, float)):
                        try:
                            reward_sim = float(reward_sim)
                        except (ValueError, TypeError):
                            reward_sim = 0.0
                    
                    # Ensure scenario_type is valid
                    scenario_type = scenario.get("scenario_type", "unknown")
                    valid_types = ["ideal_success", "common_failure", "partial_success"]
                    if scenario_type not in valid_types:
                        scenario_type = "unknown"
                    
                    # Create the simulated step with proper tuple conversion
                    simulated_step = {
                        "original_state": state_str,
                        "intended_action": intended_action,
                        "simulated_action": scenario.get("simulated_action", intended_action),
                        "scenario_type": scenario_type,
                        "simulated_outcome": scenario.get("simulated_outcome_description", "Unknown outcome"),
                        "reward_sim": reward_sim,
                        "next_state": next_state_tuple,  # Now a proper tuple
                        "key_context_changes": scenario.get("key_context_changes", {}),
                        "source_details": step_details,
                    }

                    # Add the simulated step to the list with additional error handling
                    try:
                        scenario_model = ScenarioModel(**simulated_step)
                        all_simulated_steps.append(scenario_model)
                    except Exception as validation_error:
                        print(f"Validation error for scenario: {validation_error}")
                        print(f"Problematic data: {simulated_step}")
                        
                        # Create a fallback scenario with safe defaults
                        fallback_step = {
                            "original_state": state_str,
                            "intended_action": intended_action,
                            "simulated_action": intended_action,
                            "scenario_type": "unknown",
                            "simulated_outcome": "Fallback scenario due to validation error",
                            "reward_sim": 0.0,
                            "next_state": ("Unknown", intended_action, "Unknown", "Unknown"),
                            "key_context_changes": {"error": "validation_failed"},
                            "source_details": step_details,
                        }
                        all_simulated_steps.append(ScenarioModel(**fallback_step))
                        
                except Exception as scenario_error:
                    print(f"Error processing scenario: {scenario_error}")
                    print(f"Scenario data: {scenario}")
                    
                    # Create a fallback scenario
                    fallback_step = {
                        "original_state": state_str,
                        "intended_action": intended_action,
                        "simulated_action": intended_action,
                        "scenario_type": "unknown",
                        "simulated_outcome": "Fallback scenario due to processing error",
                        "reward_sim": 0.0,
                        "next_state": ("Unknown", intended_action, "Unknown", "Unknown"),
                        "key_context_changes": {"error": "processing_failed"},
                        "source_details": step_details,
                    }
                    all_simulated_steps.append(ScenarioModel(**fallback_step))

        # Save the simulated traces to the specified output path if provided
        if hasattr(self, "output_path") and self.output_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

                # Save the scenarios to a JSON file
                with open(self.output_path, "w") as f:
                    json.dump(
                        {
                            "simulated_scenarios": [item.model_dump() for item in all_simulated_steps],
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