from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# --- Enums ---
class ProviderEnum(str, Enum):
    openai = "openai"


class ModelNameEnum(str, Enum):
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1 = "gpt-4.1"  # Full model, not mini
    # add more here as they’re supported later

class EmbeddingModelNameEnum(str, Enum):
    text_embedding_ada_002 = "text-embedding-ada-002"


class FrameworkEnum(str, Enum):
    crewai = "crewai"


# --- LLM Config ---
class LLMConfig(BaseModel):
    provider: ProviderEnum = ProviderEnum.openai
    model_name: ModelNameEnum = ModelNameEnum.gpt_4_1_mini
    api_key: str

class EmbeddingConfig(BaseModel):
    provider: ProviderEnum = ProviderEnum.openai
    model_name: EmbeddingModelNameEnum = EmbeddingModelNameEnum.text_embedding_ada_002
    api_key: str

# --- Log Source Config ---
class LogSourceConfig(BaseModel):
    type: str = "file_path"
    path: str = "log.json"


# --- Framework Adapter Config ---
class FrameworkAdapterSettings(BaseModel):
    execution_mode: str = Field("prod", description="Execution mode: dev or prod")
    log_source: LogSourceConfig


class FrameworkAdapter(BaseModel):
    name: FrameworkEnum = FrameworkEnum.crewai
    settings: FrameworkAdapterSettings


# --- Agent Config ---
class AgentTool(BaseModel):
    name: str
    description: str


class AgentModifiableConfig(BaseModel):
    prompt_configuration_file_path: str = "./config/tasks.yaml"
    agent_definition_file_path: str = "./config/agents.yaml"
    agent_name: str = "generic_agent"
    agent_tools: List[AgentTool] = []


# --- Report Config ---
class ReportConfig(BaseModel):
    output_path: str = "./reports/{project_name}.md"
    prompts_path: str = "./reports/prompts.json"


# --- Main Config ---
class AdaptiQConfig(BaseModel):
    project_name: str
    email: Optional[str] = ""
    llm_config: LLMConfig
    embedding_config: EmbeddingConfig
    framework_adapter: FrameworkAdapter
    agent_modifiable_config: AgentModifiableConfig
    report_config: ReportConfig
