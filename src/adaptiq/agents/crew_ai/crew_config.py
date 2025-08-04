
import os
import shutil
from typing import Dict, Any, Optional, Tuple
import json
from adaptiq.core.abstract.integrations.base_config import BaseConfig
import logging


# Set up logger
logger = logging.getLogger(__name__)

class CrewConfig(BaseConfig):
    """
    AdaptiqAgentTracer extends BaseConfigManager for managing agent execution traces.
    It reads configuration from a JSON/YAML file to determine how to access the agent's execution logs.
    The class supports both development and production modes, allowing for flexible log access based on the execution context.
    """
    
    def __init__(self, config_path: str, auto_create: bool = True):
        """
        Initialize the AdaptiqAgentTracer with the path to the configuration file.
        
        Args:
            config_path (str): Path to the configuration file in JSON or YAML format.
            auto_create (bool): If True, creates a default config file if it doesn't exist.
        """
        super().__init__(config_path, auto_create)
    
    def create_project_template(project_name=None, base_path=".") -> Tuple[bool, str]:
        # Validate project name
        if not project_name:
            return False, "❌ Error: Project name not provided. Please specify a project name."

        # Clean project name (remove spaces, special characters, etc.)
        project_name = project_name.replace(" ", "_").replace("-", "_")
        project_name = "".join(c for c in project_name if c.isalnum() or c == "_")

        # Ensure src directory exists
        src_path = os.path.join(base_path, "src")
        if not os.path.exists(src_path):
            os.makedirs(src_path, exist_ok=True)
            print(f"Created src directory: {src_path}")

        # Check if project directory already exists
        project_path = os.path.join(src_path, project_name)
        if os.path.exists(project_path):
            return False, f"❌ Error: Folder template already exists at '{project_path}'"

        # Define template source path
        template_source = os.path.join("adaptiq.template", "crew_template")
        
        # Check if template source exists
        if not os.path.exists(template_source):
            return False, f"❌ Error: Template source not found at '{template_source}'"

        try:
            # Copy the entire crew_template directory to the new project location
            # The destination is already named with the project_name, so this effectively renames the folder
            shutil.copytree(template_source, project_path)
            print(f"Copied template from '{template_source}' to '{project_path}'")

            return True, f"✅ Repository template created successfully!\n📁 Structure created under: {project_path}"

        except Exception as e:
            # Clean up partial creation if error occurs
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
            return False, f"❌ Error creating template: {str(e)}"

    def _create_default_config(self) -> None:
        """Create a default configuration for the AdaptiqAgentTracer."""
        default_config = {
            "agent": {
                "name": "default_agent",
                "mode": "development",
                "trace_enabled": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": "logs/agent_trace.log"
            },
            "output": {
                "trace_format": "json",
                "include_timestamps": True,
                "max_trace_size": 1048576  # 1MB
            }
        }
        
        self.config = default_config
        self.save_config()
        logger.info(f"Created default configuration at: {self.config_path}")
    
    def _validate_config(self) -> bool:
        """
        Validate the AdaptiqAgentTracer configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        required_sections = ["agent", "logging", "output"]
        required_agent_keys = ["name", "mode", "trace_enabled"]
        required_logging_keys = ["level", "format", "file_path"]
        required_output_keys = ["trace_format", "include_timestamps"]
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Check required agent keys
        for key in required_agent_keys:
            if key not in self.config["agent"]:
                logger.error(f"Missing required agent configuration key: {key}")
                return False
        
        # Check required logging keys
        for key in required_logging_keys:
            if key not in self.config["logging"]:
                logger.error(f"Missing required logging configuration key: {key}")
                return False
        
        # Check required output keys
        for key in required_output_keys:
            if key not in self.config["output"]:
                logger.error(f"Missing required output configuration key: {key}")
                return False
        
        # Validate specific values
        valid_modes = ["development", "production"]
        if self.config["agent"]["mode"] not in valid_modes:
            logger.error(f"Invalid agent mode. Must be one of: {valid_modes}")
            return False
        
        valid_trace_formats = ["json", "yaml", "text"]
        if self.config["output"]["trace_format"] not in valid_trace_formats:
            logger.error(f"Invalid trace format. Must be one of: {valid_trace_formats}")
            return False
        
        return True
    
    def get_agent_trace(self) -> str:
        """
        Get the agent trace based on the current configuration.
        
        Returns:
            str: The agent trace output.
        """
        # Implementation would depend on your specific trace collection logic
        trace_output = "Agent trace data would be collected here based on configuration"
        logger.debug("Retrieved agent trace")
        return trace_output
    
    def get_agent_config(self) -> Dict[str, Any]:
        """
        Get the agent-specific configuration.
        
        Returns:
            Dict[str, Any]: The agent configuration section.
        """
        return self.get_value("agent", {})
    
    def update_log_source(self, log_type: str, log_path: Optional[str] = None) -> None:
        """
        Update the log source configuration.
        
        Args:
            log_type (str): Type of log source.
            log_path (Optional[str]): Path to the log file.
        """
        log_source_config = {
            "type": log_type,
            "path": log_path
        }
        
        self.set_value("logging.source", log_source_config)
        logger.info(f"Updated log source configuration to: {log_source_config}")