import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import yaml

# Set up logger
logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    """
    Base class for managing configuration files in JSON or YAML format.
    Provides common functionality for loading, saving, and accessing configuration data.
    
    This class can be extended to create specialized configuration managers
    for specific applications or services.
    """
    
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
            self.config = self._load_config(config_path)

            return

        self.config_path = None
        self.config = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the configuration file from the given path.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            Dict[str, Any]: The loaded configuration.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            output_path (Optional[str]): Path to save the configuration. 
                                       If None, saves to the original config_path.
        
        Raises:
            ValueError: If the configuration cannot be saved.
        """
        save_path = output_path or self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, 
                                 allow_unicode=True, indent=2)
            
            logger.info(f"Configuration saved successfully to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise ValueError(f"Failed to save configuration: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        
        Returns:
            Dict[str, Any]: The complete configuration.
        """
        return self.config.copy()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value by key.
        Supports nested keys using dot notation (e.g., 'database.host').
        
        Args:
            key (str): The configuration key to retrieve.
            default (Any): Default value if key is not found.
            
        Returns:
            Any: The configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Set a specific configuration value by key.
        Supports nested keys using dot notation (e.g., 'database.host').
        
        Args:
            key (str): The configuration key to set.
            value (Any): The value to set.
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the final value
        config_ref[keys[-1]] = value
        logger.debug(f"Set configuration value: {key} = {value}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.
        
        Args:
            updates (Dict[str, Any]): Dictionary of key-value pairs to update.
        """
        for key, value in updates.items():
            self.set_value(key, value)
        
        logger.info(f"Updated {len(updates)} configuration values")
    
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
    def get_agent_trace(self) -> str:
        """
        Retrieve the trace or log of actions performed by the agent.

        Returns:
            str: A string representation of the agent's execution trace, 
                which may include decisions, actions, or reasoning steps.
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
    
    @abstractmethod 
    def get_prompt() -> str:
        pass
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"{self.__class__.__name__}(config_path='{self.config_path}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration manager."""
        return f"{self.__class__.__name__}(config_path='{self.config_path}', keys={list(self.config.keys())})"


