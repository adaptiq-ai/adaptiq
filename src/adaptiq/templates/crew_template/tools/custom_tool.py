from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# 🧰 Tool 1: Example tool with no inputs
class GenericRetrievalTool(BaseTool):
    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"

    def _run(self) -> str:
        return "{{ mocked_return_value }}"


# 🧰 Tool 2: Example tool with inputs
class GenericInputSchema(BaseModel):
    param1: str = Field(..., description="Describe param1")
    param2: str = Field(..., description="Describe param2")


class GenericActionTool(BaseTool):
    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"
    args_schema: Type[BaseModel] = GenericInputSchema

    def _run(self, param1: str, param2: str) -> str:
        return f"Executed action with param1={param1}, param2={param2}"
