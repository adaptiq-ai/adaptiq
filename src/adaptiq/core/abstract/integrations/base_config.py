import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import yaml
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from adaptiq.core.entities import AdaptiQConfig, AgentTool, FrameworkEnum, ProviderEnum

# Set up logger
logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    """
    Base class for managing configuration files in JSON or YAML format.
    Provides common functionality for loading, saving, and accessing configuration data.

    This class can be extended to create specialized configuration managers
    for specific applications or services.
    """

    _shared_config: AdaptiQConfig = None
    _current_prompt: str = None

    @staticmethod
    def get_config() -> AdaptiQConfig:
        """
        Retrieve the shared configuration instance.
        Raises an error if no config was preloaded.
        """
        if BaseConfig._shared_config is None:
            raise RuntimeError("No configuration has been loaded yet.")
        return BaseConfig._shared_config

    @staticmethod
    def set_active_prompt(new_prompt: str):
        BaseConfig._current_prompt = new_prompt

    @staticmethod
    def update_instructions(prompt: str = None, params: Dict[str, Any] = None) -> str:
        """
        Replace placeholders of the form {{key}} in the prompt with values from params.

        Args:
            prompt (str): The prompt string containing placeholders. If None, uses _current_prompt.
            params (Dict[str, Any]): Dictionary of key-value pairs for replacement.

        Returns:
            str: Updated prompt with all placeholders replaced.
        """
        if BaseConfig._current_prompt is None and prompt is None:
            raise RuntimeError("No prompt has been loaded or provided.")

        if params is None:
            params = {}

        # Use the provided prompt or fall back to stored one
        template = prompt if prompt is not None else BaseConfig._current_prompt

        for key, value in params.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))

        return template

    @staticmethod
    def update_instructions_within_file(file_path: str, key: str):
        """
        Update the first matching key in a YAML file with BaseConfig._current_prompt,
        searching recursively through nested structures.

        Args:
            file_path (str): Path to the YAML file.
            key (str): The key to search for and update.

        Raises:
            RuntimeError: If _current_prompt is not set.
            FileNotFoundError: If the YAML file doesn't exist.
            KeyError: If the key is not found in the file.
        """
        if BaseConfig._current_prompt is None:
            raise RuntimeError("No prompt has been loaded to update the file.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        # Load YAML
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        found = False

        def recursive_update(obj):
            nonlocal found
            if found:
                return  # Stop searching after first match
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == key:
                        obj[k] = BaseConfig._current_prompt
                        found = True
                        return
                    recursive_update(v)
                    if found:
                        return
            elif isinstance(obj, list):
                for item in obj:
                    recursive_update(item)
                    if found:
                        return

        recursive_update(data)

        if not found:
            raise KeyError(f"Key '{key}' not found in YAML file: {file_path}")

        # Save YAML back
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def __init__(self, config_path: str = None, preload: bool = False):
        """
        Initialize the configuration manager with the path to the configuration file.

        Args:
            config_path (str): Path to the configuration file (JSON or YAML).
            auto_create (bool): If True, creates a default config file if it doesn't exist.

        Raises:
            FileNotFoundError: If the configuration file does not exist and auto_create is False.
            ValueError: If the configuration file cannot be parsed.
        """
        if preload:
            if not config_path:
                raise ValueError("Configuration path must be provided for preloading.")
            self.config: AdaptiQConfig = self._load_config(config_path)
            BaseConfig._shared_config = self.config

            return

        self.config_path = None
        self.config = None

    def _load_config(self, config_path: str) -> AdaptiQConfig:
        """
        Load the configuration file from the given path.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            AdaptiQConfig: The loaded configuration.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:

            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file) or {}

            return AdaptiQConfig(**raw_config)

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")

    def get_prompt(self, get_newest: bool = False) -> str:
        if self.config.framework_adapter.name == FrameworkEnum.crewai:
            prompt_path = (
                self.config.agent_modifiable_config.prompt_configuration_file_path
            )
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, prompt_path)

            # Read YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)

            task_name = list(yaml_content.keys())[0]
            description = yaml_content[task_name]["description"]
            expected_output = yaml_content[task_name]["expected_output"]

            # Combine both sections
            return f"{description}\nExpected output:\n{expected_output}"

        else:
            prompt_path = self.config.report_config.prompts_path
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, prompt_path)

            with open(file_path, "r", encoding="utf-8") as f:
                prompt_content = json.load(f)  # list of dicts

            if not prompt_content:
                return ""

            if get_newest:
                # sort by timestamp and take the latest
                newest_entry = max(
                    prompt_content,
                    key=lambda x: datetime.strptime(
                        x["timestamp"], "%Y-%m-%d %H:%M:%S"
                    ),
                )
                return newest_entry["prompt"]
            else:
                # get the first entry with type=default
                for entry in prompt_content:
                    if entry.get("type") == "default":
                        return entry["prompt"]

                # fallback: return first prompt if no default found
                return prompt_content[0]["prompt"]

    def get_config(self) -> AdaptiQConfig:
        """
        Get the entire configuration dictionary.

        Returns:
            AdaptiQConfig: The complete configuration.
        """
        return self.config

    def get_tools(self) -> List[AgentTool]:
        """
        Get the list of agent tools from the configuration.

        Returns:
            List[AgentTool]: The list of agent tools.
        """
        return self.config.agent_modifiable_config.agent_tools

    def reload_config(self) -> None:
        """
        Reload the configuration from the file.
        This will overwrite any unsaved changes.
        """
        self.config = self._load_config(self.config_path)
        logger.info(f"Configuration reloaded from: {self.config_path}")

    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate the current configuration.
        This method should be implemented by subclasses to provide specific validation logic.

        Returns:
            bool: True if configuration is valid, False otherwise.
            str: Validation message.
        """
        return self._validate_config()

    @abstractmethod
    def _validate_config(self) -> Tuple[bool, str]:
        """
        Abstract method for configuration validation.
        Must be implemented by subclasses.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating validity and a validation message.
        """
        pass

    @abstractmethod
    def create_project_template(project_name=None, base_path=".") -> Tuple[bool, str]:
        """
        Creates a repository template structure for an agent example.

        Args:
            project_name (str): Name of the project (replaces 'agent_example')
            base_path (str): The base directory where the template will be created

        Returns:
            str: Success message or error message
        """

        pass

    def get_agent_trace(self) -> str:
        """
        Access agent trace based on execution mode configuration.

        Returns:
            str: The execution trace as text.
        """
        framework_settings = self.config.framework_adapter.settings
        execution_mode = framework_settings.execution_mode or "dev"
        log_source_type = framework_settings.log_source.type or "stdout_capture"
        log_file_path = framework_settings.log_source.path

        trace_output = ""

        # Log the execution mode
        if execution_mode == "prod":
            logger.info("Running in PROD mode - accessing log file directly")
        else:
            logger.info("Running in DEV mode - accessing log file directly")

        # Read the log file if type is file_path
        if log_source_type == "file_path":
            if not log_file_path:
                logger.error(
                    "Log source type is 'file_path' but no path is specified in config."
                )
                return ""
            try:
                with open(log_file_path, "r", encoding="utf-8") as f:
                    trace_output = f.read()
                logger.info(f"Successfully read trace from log file: {log_file_path}")
            except FileNotFoundError:
                logger.error(f"Log file not found: {log_file_path}")
                return ""
            except Exception as e:
                logger.error(f"Error reading log file {log_file_path}: {str(e)}")
                return ""
        else:
            logger.warning(
                f"Log source type '{log_source_type}' is not 'file_path'. Cannot access logs without execution."
            )
            return ""

        return trace_output

    def get_llm_instance(self) -> BaseChatModel:
        """
        Get the LLM instance based on the configuration.

        Returns:
            BaseChatModel: The LLM instance configured for the project.
        Raises:
            ValueError: If provider is unsupported.
            Exception: For any runtime errors during initialization.
        """
        try:
            llm_config = self.config.llm_config
            model = llm_config.model_name.value
            api_key = llm_config.api_key

            if llm_config.provider == ProviderEnum.openai:
                return ChatOpenAI(model=model, api_key=api_key)

            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

        except Exception as e:
            logger.error(f"Failed to create LLM instance: {e}", exc_info=True)
            raise

    def get_embeddings_instance(self) -> Embeddings:
        """
        Get the Embeddings instance based on the configuration.

        Returns:
            Embeddings: The Embeddings instance configured for the project.
        Raises:
            ValueError: If provider is unsupported.
            Exception: For any runtime errors during initialization.
        """
        try:
            llm_config = self.config.llm_config
            api_key = llm_config.api_key

            if llm_config.provider == ProviderEnum.openai:
                return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

            raise ValueError(f"Unsupported Embeddings provider: {llm_config.provider}")

        except Exception as e:
            logger.error(f"Failed to create Embeddings instance: {e}", exc_info=True)
            raise
