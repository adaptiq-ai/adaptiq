import json
import re
from typing import Dict, Tuple
from langchain_core.prompts import  PromptTemplate
from adaptiq.core.abstract.q_table.base_state_action_extractor import BaseStateActionExtractor


class StateActionExtractor(BaseStateActionExtractor):
    """
    Class for extracting state and action information from execution data and
    transforming it into a standardized format.
    """

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for state-action extraction."""
        return PromptTemplate(
            input_variables=["input_data"],
            template="""
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
            """,
        )

    def _extract_raw_state_and_action(self, input_data) -> Tuple[Dict, str]:
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
                input_data:Dict = json.loads(input_data)

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
        input_for_llm = {"state": state_dict, "action": action_str}

        chain = self.prompt_template | self.llm
        result = chain.invoke({"input_data": json.dumps(input_for_llm)})

        # Extract the JSON from the result
        try:
            # Try to find JSON in the content
            content = result.content
            json_match = re.search(r"({.*})", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                return json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")