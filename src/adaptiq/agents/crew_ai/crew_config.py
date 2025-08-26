import logging
import os
import shutil
from importlib.resources import files
from typing import Any, Dict, Optional, Tuple

import yaml

from adaptiq.core.abstract.integrations.base_config import BaseConfig

# Set up logger
logger = logging.getLogger(__name__)


class CrewConfig(BaseConfig):
    """
    AdaptiqAgentTracer extends BaseConfigManager for managing agent execution traces.
    It reads configuration from a JSON/YAML file to determine how to access the agent's execution logs.
    The class supports both development and production modes, allowing for flexible log access based on the execution context.
    """

    def __init__(self, config_path: str = None, preload: bool = False):
        super().__init__(config_path=config_path, preload=preload)

    def create_project_template(
        self, base_path: str, project_name: str = None
    ) -> Tuple[bool, str]:
        if not project_name:
            return (
                False,
                "❌ Error: Project name not provided. Please specify a project name.",
            )
        if not base_path:
            return (
                False,
                "❌ Error: Base path not provided. Please specify a base path.",
            )

        # Clean project name
        project_name = project_name.replace(" ", "_").replace("-", "_")
        project_name = "".join(c for c in project_name if c.isalnum() or c == "_")

        # Ensure src directory exists
        src_path = os.path.join(base_path, "src")
        os.makedirs(src_path, exist_ok=True)

        project_path = os.path.join(src_path, project_name)
        if os.path.exists(project_path):
            return (
                False,
                f"❌ Error: Folder template already exists at '{project_path}'",
            )

        # Locate the template directory
        template_source = files("adaptiq.templates.crew_template")

        try:
            # Copy the template to the new location
            shutil.copytree(template_source, project_path)

            # 🔧 Modify the copied config YAML
            config_path = os.path.join(project_path, "config", "adaptiq_config.yml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace placeholder
                content = content.replace("{project_name}", project_name)

                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return (
                True,
                f"✅ Repository template created successfully!\n📁 Structure: {project_path}",
            )

        except Exception as e:
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
            return False, f"❌ Error creating template: {str(e)}"

    def _validate_config(self) -> Tuple[bool, str]:
        """
        Validate the AdaptiqAgentTracer configuration.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating validity and a validation message.
        """
        try:
            current_dir = os.getcwd()
            config_data = self.config.model_dump()

            # Check required top-level keys
            required_keys = [
                "project_name",
                "email",
                "llm_config",
                "framework_adapter",
                "agent_modifiable_config",
                "report_config"
            ]

            missing_keys = [key for key in required_keys if key not in config_data]
            if missing_keys:
                return (
                    False,
                    f"❌ Missing required configuration keys: {', '.join(missing_keys)}",
                )

            # Check required nested keys
            llm_required = ["model_name", "api_key", "provider"]
            llm_missing = [
                key
                for key in llm_required
                if key not in config_data.get("llm_config", {})
            ]
            if llm_missing:
                return (
                    False,
                    f"❌ Missing required llm_config keys: {', '.join(llm_missing)}",
                )

            framework_required = ["name", "settings"]
            framework_missing = [
                key
                for key in framework_required
                if key not in config_data.get("framework_adapter", {})
            ]
            if framework_missing:
                return (
                    False,
                    f"❌ Missing required framework_adapter keys: {', '.join(framework_missing)}",
                )

            agent_config_required = [
                "prompt_configuration_file_path",
                "agent_definition_file_path",
                "agent_name",
                "agent_tools",
            ]
            agent_config_missing = [
                key
                for key in agent_config_required
                if key not in config_data.get("agent_modifiable_config", {})
            ]
            if agent_config_missing:
                return (
                    False,
                    f"❌ Missing required agent_modifiable_config keys: {', '.join(agent_config_missing)}",
                )

            report_required = ["output_path", "prompts_path"]
            report_missing = [
                key
                for key in report_required
                if key not in config_data.get("report_config", {})
            ]
            if report_missing:
                return (
                    False,
                    f"❌ Missing required report_config keys: {', '.join(report_missing)}",
                )


            # Helper function to resolve relative paths
            def resolve_path(path_value):
                """Resolve relative paths relative to the current working directory."""
                if os.path.isabs(path_value):
                    return path_value
                return os.path.join(current_dir, path_value)

            # Check if file paths exist (with relative path support)
            paths_to_check = []
            agent_config = config_data.get("agent_modifiable_config", {})
            if "prompt_configuration_file_path" in agent_config:
                path_value = agent_config["prompt_configuration_file_path"]
                resolved_path = resolve_path(path_value)
                paths_to_check.append(
                    ("prompt_configuration_file_path", path_value, resolved_path)
                )
            if "agent_definition_file_path" in agent_config:
                path_value = agent_config["agent_definition_file_path"]
                resolved_path = resolve_path(path_value)
                paths_to_check.append(
                    ("agent_definition_file_path", path_value, resolved_path)
                )

            missing_paths = []
            for path_name, original_path, resolved_path in paths_to_check:
                if not os.path.exists(resolved_path):
                    missing_paths.append(
                        f"{path_name}: {original_path} (resolved to: {resolved_path})"
                    )

            if missing_paths:
                return False, "❌ Required file paths not found:\n" + "\n".join(
                    [f"  • {path}" for path in missing_paths]
                )

            # --- Placeholder Value Checks ---
            # Define placeholder values from adaptiq_init (including both .yml and .yaml)
            placeholders = [
                "your_project_name",
                "your_email@example.com",
                "your_openai_api_key",
                "your_agent_name",
                "list_of_your_tools",
                "path/to/your/log.txt",
                "path/to/your/config/tasks.yaml",
                "path/to/your/config/agents.yaml",
                "path/to/your/config/tasks.yml",
                "path/to/your/config/agents.yml",
                "reports/your_agent_name.md",
            ]

            # Recursively check all string values in config_data for placeholders
            def contains_placeholder(d):
                if isinstance(d, dict):
                    for v in d.values():
                        result = contains_placeholder(v)
                        if result:
                            return result
                elif isinstance(d, list):
                    for v in d:
                        result = contains_placeholder(v)
                        if result:
                            return result
                elif isinstance(d, str):
                    for ph in placeholders:
                        if ph in d:
                            return ph
                return False

            found_placeholder = contains_placeholder(config_data)
            if found_placeholder:
                return (
                    False,
                    f"❌ Placeholder value '{found_placeholder}' found in your config. Please update all example/template values with your actual project information.",
                )

            return (
                True,
                "✅ Config validation successful. All required keys, paths, and user-specific values are present and valid.",
            )

        except Exception as e:
            return False, f"❌ Error reading config file: {str(e)}"
