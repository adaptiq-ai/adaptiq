from typing import Dict, List, Any, Union, Tuple
import json
import re
import ast
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class AdaptiqStateActionExtractor:
    """
    Class for extracting state and action information from execution data and 
    transforming it into a standardized format.
    """
    
    def __init__(self, provider: str, model: str , api_key = None):
        """
        Initialize the extractor with OpenAI API key.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If not provided, it will
                                           use the environment variable.
        """

        if provider == "openai":
            self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Only 'openai' is currently supported.")
      
        self.prompt_template = PromptTemplate(
            input_variables=["input_data"],
            template = """
            You are an AI assistant helping to extract and transform state and action information.

            Given the following execution data:
            {input_data}

            Extract and transform the information into a state-action mapping, using the following rules:

            - From the "state" field:
                - Extract the **core meaning** of "current_sub_task_or_thought" → this becomes the first element of the state tuple
                - Use "last_action_taken" → second element of the state tuple
                - Use "last_outcome" → third element of the state tuple
                - Extract the **main role or situation** from "agent_context" → fourth element of the state tuple

            - From "agent_action":
                - Extract only the tool name (keep it concise, e.g. "FileReadTool")

            Format the output as a valid JSON object **exactly** like this:
            {{ "state": ["InformationRetrieval_Company", "None", "None", "company background"], "action": "FileReadTool" }}

            Rules:
            - Focus on capturing the **important ideas** in each tuple element; summarize clearly and concisely (2–3 words max per element)
            - Use 'None' if no relevant info exists
            - The action must be a **clean tool name only**, no extra description
            - Output must be valid JSON that can be parsed by Python's json.loads()
            - Return ONLY the JSON object, nothing else.
            """
        )
    
    def _extract_raw_state_and_action(self, input_data):
        """
        Extract raw state and action from the input data.
        
        Args:
            input_data (dict): The input data containing state and action information.
            
        Returns:
            tuple: (state_dict, action_str)
        """
        try:
            # If input is a string, try to parse it as JSON
            if isinstance(input_data, str):
                input_data = json.loads(input_data)
            
            # Extract state and action from the input data
            state_dict = input_data.get("key", {}).get("state", {})
            action_str = input_data.get("key", {}).get("agent_action", "")
            
            return state_dict, action_str
        except Exception as e:
            raise ValueError(f"Failed to extract state and action: {str(e)}")
    
    def _transform_with_llm(self, state_dict, action_str):
        """
        Use LangChain and OpenAI to transform the extracted state and action.
        
        Args:
            state_dict (dict): The extracted state dictionary.
            action_str (str): The extracted action string.
            
        Returns:
            dict: Transformed state and action.
        """
        input_for_llm = {
            "state": state_dict,
            "action": action_str
        }
        
        chain = self.prompt_template | self.llm
        result = chain.invoke({"input_data": json.dumps(input_for_llm)})
        
        # Extract the JSON from the result
        try:
            # Try to find JSON in the content
            content = result.content
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                return json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")
    
    def extract(self, input_data):
        """
        Extract and transform state and action from the input data.
        
        Args:
            input_data (dict or str): The input data containing state and action information.
            
        Returns:
            dict: Transformed state and action.
        """
        state_dict, action_str = self._extract_raw_state_and_action(input_data)
        transformed_data = self._transform_with_llm(state_dict, action_str)
        
        return transformed_data
    
    def process_batch(self, input_data_list):
        """
        Process a batch of input data.
        
        Args:
            input_data_list (list): List of input data dictionaries or strings.
            
        Returns:
            list: List of transformed state and action dictionaries.
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.extract(input_data)
                results.append(result)
            except Exception as e:
                print(f"Error processing input: {str(e)}")
                results.append({"error": str(e)})
        
        return results

class AdaptiqStateMapper:
    """
    AdaptiqStateMapper - Matches execution trace states with Q-table states.
    
    Takes the "Warmed Q-table" (from previous runs) and matches input states to see
    if they correspond to any known state from the Q-table, ignoring actions completely.
    """
    
    def __init__(self, warmed_qtable_data: Dict[str, Any], provider: str, llm_model_name_for_reconciliation: str, llm_api_key: str):
        """
        Initialize the AdaptiqStateMapper.
        
        Args:
            warmed_qtable_data: Q-table data containing Q_table and seen_states
            llm_model_name_for_reconciliation: OpenAI model name to use for reconciliation
            llm_api_key: API key for OpenAI
        """
        # Store the Q-table data
        self.qtable = warmed_qtable_data.get("Q_table", {})
        
        # Combine states from Q-table and seen_states, ensuring uniqueness
        self.known_states = set(self.qtable.keys())
        for state in warmed_qtable_data.get("seen_states", []):
            self.known_states.add(state)
        
        # Convert to a list for easier processing
        self.known_states = list(self.known_states)
        
        # Parse states for better matching
        self.parsed_states = self._parse_known_states()
        
        # Initialize the LLM for reconciliation
        if provider == "openai":
            self.reconciliation_llm = ChatOpenAI(
                model=llm_model_name_for_reconciliation,
                api_key=llm_api_key
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Only 'openai' is currently supported.")
        
        # Define the state classification prompt template
        self.classification_template = """You are an AI Agent State Classifier specializing in semantic matching.

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
        
        self.classification_prompt_template = ChatPromptTemplate.from_template(self.classification_template)
    
    def _parse_known_states(self) -> List[Tuple[str, List]]:
        """
        Parse known states into a more comparable format.
        
        Returns:
            List of tuples containing (original_state_string, parsed_components)
        """
        parsed_states = []
        
        for state_str in self.known_states:
            try:
                # Handle tuple-like strings
                if state_str.startswith("(") and state_str.endswith(")"):
                    # Use ast.literal_eval to safely parse tuple strings
                    components = ast.literal_eval(state_str)
                    parsed_states.append((state_str, list(components)))
                # Handle list-like strings
                elif state_str.startswith("[") and state_str.endswith("]"):
                    # Use ast.literal_eval to safely parse list strings
                    components = ast.literal_eval(state_str)
                    parsed_states.append((state_str, components))
                else:
                    # For any other format, store as is
                    parsed_states.append((state_str, [state_str]))
            except (SyntaxError, ValueError):
                # If parsing fails, store original string
                parsed_states.append((state_str, [state_str]))
        
        return parsed_states
    
    def _extract_state_from_input(self, input_data: Union[str, List, Dict]) -> Union[List, str]:
        """
        Extract the state from input data.
        
        Args:
            input_data: Input data (string, list, or dictionary)
            
        Returns:
            Extracted state (list or string)
        """
        if isinstance(input_data, dict) and "state" in input_data:
            return input_data["state"]
        return input_data
    
    def _invoke_llm_for_classification(self, input_state: Union[str, List, Dict]) -> Dict:
        """
        Invoke the LLM to classify a state.
        
        Args:
            input_state: State to classify (can be string, list, or dict)
            
        Returns:
            Dict containing the LLM's classification output
        """
        # Extract just the state part if input is a dict with state key
        state_to_classify = self._extract_state_from_input(input_state)
        
        # Convert to string for LLM input
        if not isinstance(state_to_classify, str):
            state_str = json.dumps(state_to_classify)
        else:
            state_str = state_to_classify
        
        # Create formatted known states for better comparison
        formatted_known_states = []
        for original, parsed in self.parsed_states:
            formatted_known_states.append({
                "original": original,
                "components": parsed
            })
        
        # Create inputs for the LLM
        inputs = {
            "input_state": state_str,
            "known_states": json.dumps(formatted_known_states, indent=2)
        }
        
        # Create and invoke the prompt
        prompt = self.classification_prompt_template.format_messages(**inputs)
        response = self.reconciliation_llm.invoke(prompt)
        
        # Parse the response content as JSON
        try:
            classification_output = json.loads(response.content)
            return classification_output
        except json.JSONDecodeError:
            # If JSON parsing fails, extract just the JSON portion
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_content = content[start_idx:end_idx]
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
            
            # If we can't parse JSON, return a default structure
            return {
                "classification": {
                    "is_known_state": False,
                    "matched_state": None,
                    "reasoning": "Error parsing LLM output"
                }
            }
    
    def _validate_classification(self, classification_output: Dict) -> Dict:
        """
        Validate the classification output from the LLM.
        
        Args:
            classification_output: The LLM's classification output
            
        Returns:
            Validated classification output
        """
        classification = classification_output.get("classification", {})
        
        # If LLM says it's a known state, verify the matched state is actually in our known states
        if classification.get("is_known_state", False):
            matched_state = classification.get("matched_state")
            
            if matched_state not in self.known_states:
                # If matched state not in known states, invalidate the classification
                classification["is_known_state"] = False
                classification["matched_state"] = None
                classification["reasoning"] = "State validation: matched state not found in known states"
        
        classification_output["classification"] = classification
        return classification_output
    
    def classify_states(self, input_states: List[Union[str, List, Dict]]) -> List[Dict]:
        """
        Classify input states against the known states.
        
        Args:
            input_states: List of states to classify
                Each state can be a string, list, or dictionary with a "state" key
            
        Returns:
            List of classification results
        """
        classification_results = []
        
        # Process each state
        for index, input_state in enumerate(input_states):
            # Invoke the LLM for classification
            classification_output = self._invoke_llm_for_classification(input_state)
            
            # Validate the classification output
            validated_output = self._validate_classification(classification_output)
            
            # Create the classification entry
            classification_entry = {
                "index": index,
                "input_state": input_state,
                "classification": validated_output.get("classification", {})
            }
            
            # Add to the classification results list
            classification_results.append(classification_entry)
        
        return classification_results
    
    def classify_single_state(self, input_state: Union[str, List, Dict]) -> Dict:
        """
        Classify a single input state against known states.
        
        Args:
            input_state: State to classify (can be string, list, or dict with a "state" key)
            
        Returns:
            Classification result for the input state
        """
        results = self.classify_states([input_state])
        if results:
            return results[0]
        return {
            "index": 0,
            "input_state": input_state,
            "classification": {
                "is_known_state": False,
                "matched_state": None,
                "reasoning": "Classification failed"
            }
        }
    
    def save_classification_json(self, classification_results: List[Dict], output_path: str) -> None:
        """
        Save the classification results to a JSON file.
        
        Args:
            classification_results: Classification results to save
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(classification_results, f, indent=2)
    
    @staticmethod
    def from_qtable_file(qtable_file_path: str, llm_model_name: str, llm_api_key: str, provider: str) -> 'AdaptiqStateMapper':
        """
        Create an AdaptiqStateMapper instance from a Q-table file.
        
        Args:
            qtable_file_path: Path to the Q-table JSON file
            llm_model_name: OpenAI model name to use for reconciliation
            llm_api_key: API key for OpenAI
            provider: Provider for the LLM (currently only "openai" is supported)
            
        Returns:
            AdaptiqStateMapper instance
        """
        with open(qtable_file_path, 'r') as f:
            qtable_data = json.load(f)
        
        return AdaptiqStateMapper(qtable_data, provider, llm_model_name, llm_api_key)
