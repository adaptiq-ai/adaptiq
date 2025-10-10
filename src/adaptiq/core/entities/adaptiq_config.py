from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# --- Enums ---
class ProviderEnum(str, Enum):
    openai = "openai"
    cohere = "cohere"


class ModelNameEnum(str, Enum):
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1 = "gpt-4.1"  # Full model, not mini
    # add more here as they’re supported later

class EmbeddingModelNameEnum(str, Enum):
    text_embedding_3_small = "text-embedding-3-small"
    text_embedding_ada_002 = "text-embedding-ada-002"


class RerankModelNameEnum(str, Enum):
    rerank_v3_5 = "rerank-v3.5"


class FrameworkEnum(str, Enum):
    crewai = "crewai"


class DatabaseEnum(str, Enum):
    redis = "redis"


# --- LLM Config ---
class LLMConfig(BaseModel):
    provider: ProviderEnum = ProviderEnum.openai
    model_name: ModelNameEnum = ModelNameEnum.gpt_4_1_mini
    api_key: str

# --- Embedding Config ---
class EmbeddingConfig(BaseModel):
    provider: ProviderEnum = ProviderEnum.openai
    model_name: EmbeddingModelNameEnum = EmbeddingModelNameEnum.text_embedding_3_small
    api_key: str

# --- Rerank Config ---
class RerankConfig(BaseModel):
    provider: ProviderEnum = ProviderEnum.cohere
    model_name: RerankModelNameEnum = RerankModelNameEnum.rerank_v3_5
    api_key: str


# --- Database Config ---
class DatabaseConfig(BaseModel):
    provider: DatabaseEnum = DatabaseEnum.redis
    host: Optional[str] = None
    port: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


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
    rerank_config: RerankConfig
    database_config: Optional[DatabaseConfig] = None
    framework_adapter: FrameworkAdapter
    agent_modifiable_config: AgentModifiableConfig
    report_config: ReportConfig
