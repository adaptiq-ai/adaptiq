import asyncio
import logging
import os
from typing import AsyncGenerator

import yaml
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from adaptiq.wizard.assistant_utils import create_agent_repo_template
from adaptiq.wizard.chat_animation import (start_thinking_animation,
                                           stop_thinking_animation)
from adaptiq.wizard.logo_animation import display_logo_animated

# Configure logging to suppress API call logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.agents").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)


# Suppress the specific API key error message
class APIKeyErrorFilter(logging.Filter):
    def filter(self, record):
        return "Incorrect API key provided" not in record.getMessage()


logging.getLogger("openai.agents").addFilter(APIKeyErrorFilter())


class AdaptiqWizardAssistant:
    """
    AdaptiQ Wizard Assistant - An AI-powered CLI chatbot that helps users
    navigate and use the AdaptiQ optimization toolkit.
    """

    def __init__(self, llm_provider: str = "openai", api_key: str = None):
        """
        Initialize the AdaptiQ Wizard Assistant with OpenAI integration.

        Args:
            llm_provider: LLM provider type openai or groq
            api_key: LLM provier key, if not provided will look for api_key env var
        """
        self.memory = ""
        self.running_processes = []  # Track running processes
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "api_key environment variable or api_key parameter is required"
            )

        self.llm_provider = llm_provider

        if self.llm_provider == "groq":
            # Initialize Groq client with OpenAI-compatible interface
            self.client = AsyncOpenAI(
                api_key=self.api_key, base_url="https://api.groq.com/openai/v1"
            )

            # Initialize the model format api
            self.model = OpenAIChatCompletionsModel(
                model="qwen/qwen3-32b", openai_client=self.client
            )
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key
                # No base_url needed for OpenAI - uses default
            )

            # Initialize the model format api
            self.model = OpenAIChatCompletionsModel(
                model="gpt-4.1", openai_client=self.client
            )

        # Define the system prompt for the wizard
        self.system_prompt = """
        You are AdaptiQ Wizard, a command-driven AI assistant for the AdaptiQ CLI tool. 
        Your **ONLY** function is to process specific `wizard` commands to help users validate their configuration and run the optimization pipeline.

        You MUST NOT engage in general conversation or attempt to answer questions outside of the defined commands. Your goal is to be a precise and helpful command interpreter.

        ---
        ## 🧠 Memory Context
        {memory_context}

        **IMPORTANT**: Use the memory context to remember the `config_file_path` from a successful validation. When the user runs the pipeline, you MUST use this stored path.

        ---
        ## ⚙️ Valid Commands & Workflow

        Your entire interaction model is based on the following commands. You will route all user input to one of these actions.

        **1. `wizard init <name_of_project>`**
        - **User's Goal:** Create a complete agent repository template with all necessary files and configuration.
        - **Your Action:** Call the `adaptiq_init` tool with the provided project name to generate the agent repo template with all required files and provide next steps for validation and setup.

        **2. `wizard validate config <path_to_config_file>`**
        - **User's Goal:** Validate their `adaptiq_config.yml` file.
        - **Your Action:** Call the `adaptiq_check_config` tool with the provided `<path_to_config_file>`.
        - **On Success:** Confirm to the user that the configuration is valid and that they can now proceed with `wizard run pipeline`. Remember the valid path for the next step.
        - **On Failure:** Clearly state the errors returned by the tool and tell the user they MUST fix them before proceeding.

        **3. `wizard start`**
        - **User's Goal:** Execute the ADAPTIQ optimization process.
        - **Your Action:**
            a. Check your memory context for a previously validated config path from a successful `wizard validate config` command.
            b. If a validated config path exists in memory, ALWAYS call the `run_adaptiq_command` tool using that stored path.
            c. If no validated config path exists in memory, you MUST instruct the user to first run `wizard validate config <path>` before they can use `wizard start`.
            d. Once a config has been successfully validated in the current session, subsequent uses of `wizard start` will automatically use the same validated config file and call `run_adaptiq_command` without requiring re-validation.
        - **Critical Execution Rule:** EVERY time the user types `wizard start`, you MUST execute the optimization process by calling `run_adaptiq_command` (assuming a valid config exists). This is true even if:
            - The same command was just executed moments ago
            - The optimization just completed successfully
            - There are no changes to the configuration
            - The user runs it repeatedly after the initial config validation
        - **After Running:** Interpret and clearly communicate the results from the tool—success, errors, or other outcomes.

        **4. `wizard execute <path_to_config_file>`**
        - **User's Goal:** Validate configuration and execute the ADAPTIQ optimization process in a single command (for non-interactive mode).
        - **Your Action:** 
            a. FIRST call the `adaptiq_check_config` tool with the provided `<path_to_config_file>`.
            b. If validation succeeds, IMMEDIATELY call the `run_adaptiq_command_detached` tool with the same config path.
            c. If validation fails, stop and report the validation errors without proceeding to run.
        - **Sequential Execution:** You MUST execute these tools in order: validation first, then optimization only if validation passes.
        - **After Running:** Interpret and clearly communicate the results from both validation and optimization steps.

        **5. `wizard info`**
        - **User's Goal:** Get general help about ADAPTIQ's functionality and how to setup the agent of the user.
        - **Your Action:** Call the `get_adaptiq_help` tool and display its output to the user.

        ---
        ## 📜 Rules of Engagement

        1.  **Strict Command Parsing:** Your first step is ALWAYS to check if the user's input exactly matches one of the four valid command structures.
        2.  **Invalid Command Handling:** If the user's input does not match a valid command (e.g., "hello", "can you help me?", "run the process on my agent"), you MUST respond with:
            `"I can only process specific commands. Please use one of the following:
            - 'wizard init'
            - 'wizard validate config <path>'
            - 'wizard start'
            - 'wizard execute <path_to_config_file>'
            - 'wizard info'"`
            ... and then stop. Do not try to interpret their intent.
        3.  **Mandatory Workflow:** The user **MUST** successfully run `wizard validate config` before they are allowed to use `wizard start`. Enforce this sequence.
        4.  **Tool Interpretation:** Always wait for the tool's output. Present the information to the user in a clear, friendly, and actionable way. Never show raw function calls or syntax.
        5.  **Sequential Tool Execution:** For `wizard execute`, you MUST execute `adaptiq_check_config` first, then only proceed to `run_adaptiq_command_detached` if validation succeeds. Do not skip validation.
        6.  **Init Redundancy Memory:** Once 'wizard init' has been successfully executed, memorize the project name. If the user attempts to use the same project name again with 'wizard init', remind them that this project already exists and ask them to provide a new project name.
        7.  **Emoji Restriction:** Do not use emojis, emoticons, or any Unicode symbols in your responses. Keep all communication in plain text format using only standard alphanumeric characters and basic punctuation.

        Be a helpful guide, but a strict one. Your primary job is to keep the user on the correct and safe path to running their optimization."""

        self._setup_agent()

    def _setup_agent(self):
        """Setup the agent with tools and prompt"""

        @function_tool
        async def adaptiq_init(project_name: str = None) -> str:
            """
            Creates a complete agent repository template with all necessary files and configuration.

            This tool generates a comprehensive project structure including:
            - AdaptiQ configuration file (adaptiq_config.yml)
            - Agent and task configuration files
            - Custom tool templates
            - Crew setup with proper instrumentation
            - Main execution file with decorators

            Args:
                project_name (str): Name of the project (required). This will be used as the folder name
                                and will replace 'agent_example' throughout the structure.

            Returns:
                str: Success message confirming template creation or error message.
            """
            try:
                # Validate project name
                if not project_name:
                    return "❌ Error: Project name is required. Please provide a project name for your agent template."

                # Create the complete repository template with the provided project name
                result = create_agent_repo_template(project_name=project_name)

                return result

            except Exception as e:
                return f"❌ Error creating agent template: {str(e)}"

        @function_tool
        async def get_adaptiq_help() -> str:
            """
            Get detailed information about AdaptiQ, its pipeline, goals, and supported frameworks and how to set up the configuration file.

            Returns:
                str: Essential AdaptiQ information and supported frameworks.
            """
            help_text = """🧠 **ADAPTIQ - Agent Development & Prompt Tuning with Q-Learning**

            WHAT IS ADAPTIQ?
            AdaptiQ is a framework designed for the iterative improvement of AI agent performance through offline Reinforcement Learning (RL). Its primary goal is to systematically enhance an agent's configuration—especially its Task Description (Prompt)—by learning from the agent's past execution behaviors and incorporating user validation through an interactive Wizard process. AdaptiQ provides a structured, data-driven alternative to manual prompt engineering.

            PIPELINE STAGES:
            • **Pre-run:** Analyzes the agent's task description/prompts to extract intended behavior, generate hypothetical states, and initialize heuristics.
            • **Post-run:** Parses execution logs to identify actual agent actions, tool calls, and outcomes.
            • **Reconciliation:** Aligns intended vs actual behavior, updates the Q-table with rewards, and allows user validation and correction.

            GOALS:
            - Optimize agent behavior by refining its core instructions (prompts/task descriptions).
            - Analyze what an agent intended to do (from its prompt), what it actually did (from execution logs), and how effective those actions were (via a multi-faceted reward system).
            - Provide actionable insights and data-driven prompt engineering.
            - Enable interactive, wizard-guided validation and improvement of agent performance.

            SUPPORTED FRAMEWORKS:
            - **Agentic Framework:** CrewAI (only)
            - **LLM Provider:** OpenAI models (only)

            To run the full AdaptiQ optimization pipeline, use the 'run' command with your configuration file. This will execute all three stages: pre-run, post-run, and reconciliation, guiding you through the process with the Wizard interface.

            **This guide will walk you through the essential steps to prepare your agent for optimization.**

            ---
            ### **Step 1: Configure Your `adaptiq_config.yml`**

            This file is the control center for your optimization task. Here’s what each section means:

            - **`llm_config`**: Your LLM settings.
            - `model_name`: The name of the model to use.
            - `api_key`: Your API key for the provider.
            - **Note:** Currently, only **OpenAI** models are supported.

            - **`framework_adapter`**: Defines how to run your agent.
            - `name`: The agent framework you are using.
            - **Note:** Currently, only **CrewAI** is supported.
            - `settings`: Execution commands and the path to your log file (e.g., `log.json`).

            - **`agent_modifiable_config`**: Points to your agent's core files.
            - `prompt_configuration_file_path`: Path to your tasks file (e.g., `tasks.yaml`).
            - `agent_definition_file_path`: Path to your agents definition file (e.g., `agents.yaml`).
            - **Note:** We recommend placing these in a dedicated config folder, such as: `your_agent_project/src/config/`.

            - **`report_config`**: The output path for the final optimization report.

            ---
            ### **Step 2: Instrument Your Agent Code with Decorators**

            For AdaptiQ to analyze your agent, you MUST use our decorators on your Python functions.

            - **`@instrumental_logger`**:
            - **What it does:** Automatically logs the thoughts and outputs from your agent's methods.
            - **Where to use it:** Apply this to functions inside your CrewAI Tasks or Tools to capture their internal workings for analysis.

            - **`@instrumental_run`**:
            - **What it does:** Wraps your main execution function (e.g., where you call `crew.kickoff()`). It times the execution and automatically triggers the AdaptiQ analysis pipeline right after your agent finishes its run.
            - **Why it's crucial:** This is the link that connects your agent's execution to the AdaptiQ analysis process.

            ---
            ### **Step 3: Follow the Wizard Workflow**

            Once your config file is created and your code is instrumented, use these commands:

            1.  **Validate your setup:**
                `wizard check config /path/to/your/adaptiq_config.yml`

            2.  **Run the optimization pipeline:**
                `wizard run pipeline`

            To see general help about AdaptiQ's concepts at any time, use `wizard info`.

            For more details, see the project README or documentation.
            """
            return help_text

        @function_tool
        async def adaptiq_check_config(config_path: str) -> str:
            """
            Validate the AdaptiQ config file for required keys, file paths, alert_mode settings,
            and ensure all placeholder/example values have been replaced by the user.
            Supports both .yml and .yaml extensions and handles relative paths.

            Args:
                config_path (str): Path to the AdaptiQ YAML config file (.yml or .yaml).

            Returns:
                str: Validation result message (success or details of missing/invalid items).
            """
            if not config_path:
                return "❌ Config path is required. Please provide the path to your YAML config file."

            # Handle both .yml and .yaml extensions
            if not config_path.endswith((".yml", ".yaml")):
                return "❌ Config file must have .yml or .yaml extension."

            # Convert to absolute path to handle relative paths
            abs_config_path = os.path.abspath(config_path)

            if not os.path.exists(abs_config_path):
                return f"❌ Config file not found: {config_path}\nPlease check the path and try again."

            try:
                with open(abs_config_path, "r", encoding="utf-8") as file:
                    config_data = yaml.safe_load(file)

                # Get the current working directory for resolving relative paths
                current_dir = os.getcwd()

                # Check required top-level keys
                required_keys = [
                    "project_name",
                    "email",
                    "llm_config",
                    "framework_adapter",
                    "agent_modifiable_config",
                    "report_config",
                    "alert_mode",
                ]
                missing_keys = [key for key in required_keys if key not in config_data]
                if missing_keys:
                    return f"❌ Missing required configuration keys: {', '.join(missing_keys)}"

                # Check required nested keys
                llm_required = ["model_name", "api_key", "provider"]
                llm_missing = [
                    key
                    for key in llm_required
                    if key not in config_data.get("llm_config", {})
                ]
                if llm_missing:
                    return (
                        f"❌ Missing required llm_config keys: {', '.join(llm_missing)}"
                    )

                framework_required = ["name", "settings"]
                framework_missing = [
                    key
                    for key in framework_required
                    if key not in config_data.get("framework_adapter", {})
                ]
                if framework_missing:
                    return f"❌ Missing required framework_adapter keys: {', '.join(framework_missing)}"

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
                    return f"❌ Missing required agent_modifiable_config keys: {', '.join(agent_config_missing)}"

                report_required = ["output_path"]
                report_missing = [
                    key
                    for key in report_required
                    if key not in config_data.get("report_config", {})
                ]
                if report_missing:
                    return f"❌ Missing required report_config keys: {', '.join(report_missing)}"

                # --- Alert Mode Checks ---
                alert_mode = config_data.get("alert_mode")
                if not alert_mode:
                    return "❌ Missing required section: alert_mode"

                # Check on_demand
                on_demand = alert_mode.get("on_demand")
                if not on_demand or "enabled" not in on_demand:
                    return "❌ Missing 'on_demand' or its 'enabled' key in alert_mode"
                if not isinstance(on_demand["enabled"], bool):
                    return "❌ 'on_demand.enabled' must be true or false"
                if on_demand["enabled"]:
                    if (
                        "runs" not in on_demand
                        or not isinstance(on_demand["runs"], int)
                        or on_demand["runs"] <= 0
                    ):
                        return "❌ 'on_demand.runs' must be a positive integer when 'on_demand.enabled' is true"

                # Check per_run
                per_run = alert_mode.get("per_run")
                if not per_run or "enabled" not in per_run:
                    return "❌ Missing 'per_run' or its 'enabled' key in alert_mode"
                if not isinstance(per_run["enabled"], bool):
                    return "❌ 'per_run.enabled' must be true or false"

                # Helper function to resolve relative paths
                def resolve_path(path_value):
                    """Resolve relative paths relative to the current working directory."""
                    if os.path.isabs(path_value):
                        return path_value
                    return os.path.join(current_dir, path_value)

                # Check if file paths exist (with relative path support)
                # NOTE: Log file path check has been removed as requested
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
                    return "❌ Required file paths not found:\n" + "\n".join(
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
                    return f"❌ Placeholder value '{found_placeholder}' found in your config. Please update all example/template values with your actual project information."

                return "✅ Config validation successful. All required keys, alert_mode, paths, and user-specific values are present and valid."

            except yaml.YAMLError as e:
                return f"❌ Invalid YAML format in config file: {str(e)}"

            except Exception as e:
                return f"❌ Error reading config file: {str(e)}"

        @function_tool
        async def run_adaptiq_command(config_path: str) -> str:
            """
            Execute the complete AdaptiQ pipeline (pre_run) in background with logging.
            Supports both .yml and .yaml extensions and handles relative paths.
            Automatically cleans up existing log files before running.

            Args:
                config_path (str): Path to the AdaptiQ YAML config file (.yml or .yaml).

            Returns:
                str: Information about the launched command and log file location.
            """
            # Clean up completed processes first
            self.running_processes = [p for p in self.running_processes if not p.done()]

            # Check if there are still running processes
            if self.running_processes:
                return "⚠️ An AdaptiQ process is still running. Please wait for it to complete before starting a new one."

            if not config_path:
                return "❌ Config path is required. Please provide the path to your YAML config file."

            # Handle both .yml and .yaml extensions
            if not config_path.endswith((".yml", ".yaml")):
                return "❌ Config file must have .yml or .yaml extension."

            # Convert to absolute path to handle relative paths
            abs_config_path = os.path.abspath(config_path)

            if not os.path.exists(abs_config_path):
                return f"❌ Config file not found: {config_path}\nPlease check the path and try again."

            try:
                # Extract email from config
                with open(abs_config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                email = config_data.get("email", None)

                log_filename = f"adaptiq_run.log"
                results_folder_name = f"results"

                # Get absolute path for log file (in current working directory)
                abs_log_path = os.path.abspath(log_filename)

                # Clean up existing log file if it exists
                if os.path.exists(abs_log_path):
                    try:
                        os.remove(abs_log_path)
                    except Exception as e:
                        return f"❌ Error removing existing log file {abs_log_path}: {str(e)}"

                # Use absolute path for the command to ensure it works regardless of working directory
                cmd_args = [
                    "adaptiq",
                    "default-run",
                    "--config",
                    abs_config_path,
                    "--output_path",
                    results_folder_name,
                    "--log",
                    log_filename,
                ]

                # Initialize log file with UTF-8 encoding
                with open(log_filename, "w", encoding="utf-8") as log_file:
                    log_file.write(
                        f"=== AdaptiQ run started at {asyncio.get_event_loop().time()} ===\n"
                    )
                    log_file.write(f"Command: {' '.join(cmd_args)}\n")
                    log_file.write(
                        f"Config file: {config_path} (resolved to: {abs_config_path})\n"
                    )
                    log_file.write("=" * 50 + "\n\n")

                # Launch process in background with UTF-8 environment (fire and forget)
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                # Create the subprocess without awaiting it - this makes it truly fire-and-forget
                process_task = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        *cmd_args,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        cwd=os.getcwd(),
                        env=env,
                    )
                )

                # Store the process task for tracking
                self.running_processes.append(process_task)

                # Display command with original path for user clarity
                cmd_display = f"adaptiq default-run --config {config_path} --output_path {results_folder_name} --log {log_filename}"
                return (
                    f"✅ Command '{cmd_display}' launched successfully in background!\n"
                    f"📧 The results will be available soon in a report that will be {'sent to your email: ' + email if email else 'saved locally'}\n"
                    f"📝 Logs are being written to: {log_filename}\n"
                    f"📂 Results of the default-run are stored under folder named {results_folder_name}\n"
                    f"⏳ You can execute additional runs when execution of your agent is finished.\n"
                    f"🔧 Make sure please to check the setup of your agents and the decorators used in your code.\n"
                )

            except yaml.YAMLError as e:
                return f"❌ Invalid YAML format in config file: {str(e)}"

            except FileNotFoundError:
                return "❌ AdaptiQ CLI not found. Please ensure AdaptiQ is installed and in your PATH."

            except Exception as e:
                return f"❌ Error launching command: {str(e)}"

        @function_tool
        async def run_adaptiq_command_detached(config_path: str) -> str:
            """
            Execute the complete AdaptiQ pipeline in a fully detached process for non-interactive mode.
            This process will continue running even after the Python script exits.
            Supports both .yml and .yaml extensions and handles relative paths.
            Automatically cleans up existing log files before running.

            Args:
                config_path (str): Path to the AdaptiQ YAML config file (.yml or .yaml).

            Returns:
                str: Information about the launched command and log file location.
            """
            import subprocess
            import sys

            if not config_path:
                return "❌ Config path is required. Please provide the path to your YAML config file."

            # Handle both .yml and .yaml extensions
            if not config_path.endswith((".yml", ".yaml")):
                return "❌ Config file must have .yml or .yaml extension."

            # Convert to absolute path to handle relative paths
            abs_config_path = os.path.abspath(config_path)

            if not os.path.exists(abs_config_path):
                return f"❌ Config file not found: {config_path}\nPlease check the path and try again."

            try:
                # Extract email from config
                with open(abs_config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                email = config_data.get("email", None)

                log_filename = f"adaptiq_run.log"
                results_folder_name = f"results"

                # Get absolute path for log file (in current working directory)
                abs_log_path = os.path.abspath(log_filename)

                # Clean up existing log file if it exists
                if os.path.exists(abs_log_path):
                    try:
                        os.remove(abs_log_path)
                    except Exception as e:
                        return f"❌ Error removing existing log file {abs_log_path}: {str(e)}"

                # Use absolute path for the command to ensure it works regardless of working directory
                cmd_args = [
                    "adaptiq",
                    "default-run",
                    "--config",
                    abs_config_path,
                    "--output_path",
                    results_folder_name,
                    "--log",
                    log_filename,
                ]

                # Initialize log file with UTF-8 encoding
                with open(log_filename, "w", encoding="utf-8") as log_file:
                    log_file.write(f"=== AdaptiQ run started (detached mode) ===\n")
                    log_file.write(f"Command: {' '.join(cmd_args)}\n")
                    log_file.write(
                        f"Config file: {config_path} (resolved to: {abs_config_path})\n"
                    )
                    log_file.write("=" * 50 + "\n\n")

                # Create a fully detached process that survives script termination
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                # Platform-specific detached process creation
                if sys.platform.startswith("win"):
                    # Windows: Use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
                    subprocess.Popen(
                        cmd_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        cwd=os.getcwd(),
                        env=env,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS,
                    )
                else:
                    # Unix/Linux/macOS: Use start_new_session for true detachment
                    subprocess.Popen(
                        cmd_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        cwd=os.getcwd(),
                        env=env,
                        start_new_session=True,  # This creates a new session, detaching from parent
                    )

                # Display command with original path for user clarity
                cmd_display = f"adaptiq default-run --config {config_path} --output_path {results_folder_name} --log {log_filename}"
                return (
                    f"✅ Command '{cmd_display}' launched successfully as detached process!\n"
                    f"📧 The results will be available soon in a report that will be {'sent to your email: ' + email if email else 'saved locally'}\n"
                    f"📝 Process logs are being written to: {log_filename}\n"
                    f"📂 Results of the default-run will be stored under folder named {results_folder_name}\n"
                    f"🔄 The process will continue running even after this script terminates.\n"
                    f"🔧 Make sure to check the setup of your agents and the decorators used in your code.\n"
                )

            except yaml.YAMLError as e:
                return f"❌ Invalid YAML format in config file: {str(e)}"

            except FileNotFoundError:
                return "❌ AdaptiQ CLI not found. Please ensure AdaptiQ is installed and in your PATH."

            except Exception as e:
                return f"❌ Error launching detached command: {str(e)}"

        # Initialize the agent with tools
        self.agent = Agent(
            name="AdaptiQ Wizard Assistant",
            model=self.model,
            instructions=self.system_prompt,
            tools=[
                adaptiq_init,
                adaptiq_check_config,
                get_adaptiq_help,
                run_adaptiq_command,
                run_adaptiq_command_detached,
            ],
        )

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Process a user message and stream the assistant's response.

        Args:
            user_message: The user's input message

        Yields:
            str: Chunks of the assistant's response
        """
        try:
            # Use the agent to process the message with streaming
            self.agent.instructions = self.system_prompt.format(
                memory_context=self.memory
            )
            result = Runner.run_streamed(self.agent, user_message)
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    yield event.data.delta
        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"

    async def chat(self, user_message: str) -> str:
        """
        Chat with the agent and return the complete response

        Args:
            user_input: The user's message

        Returns:
            Complete response from the agent
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        try:
            self.agent.instructions = self.system_prompt.format(
                memory_context=self.memory
            )
            response = await Runner.run(self.agent, input=user_message)
            return response.final_output
        except Exception as e:
            return f"Error: {str(e)}"

    async def start_interactive_session(self):
        """
        Start an interactive chat session with the wizard using streaming responses.
        """
        # Store conversation history for memory
        conversation_history = []

        display_logo_animated()
        print("🧙‍♂️ Welcome to AdaptiQ Wizard Assistant!")
        print(
            "🌟 For better and accurate results, it is recommended to set the LLM provier to 'OPENAI'!"
        )
        print("❌➡️ Type 'exit', 'quit', or 'bye' to end the session.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("🧙‍♂️ Goodbye! Happy optimizing with AdaptiQ!")
                    break

                if not user_input:
                    continue

                # Start the thinking animation
                start_thinking_animation()

                try:
                    # Stream response from the wizard
                    first_chunk = True
                    assistant_response = ""
                    async for chunk in self.chat_stream(user_input):
                        if chunk:  # Only print non-empty chunks
                            # Stop spinner before first chunk and show wizard prefix
                            if first_chunk:
                                stop_thinking_animation()
                                print("🧙‍♂️ AdaptiQ Wizard: ", end="", flush=True)
                                first_chunk = False

                            print(chunk, end="", flush=True)
                            assistant_response += chunk

                    # If no chunks were received, still stop the spinner
                    if first_chunk:
                        stop_thinking_animation()
                        assistant_response = "I didn't receive a response."
                        print(f"🧙‍♂️ AdaptiQ Wizard: {assistant_response}")

                    # Store the conversation turn
                    conversation_history.append(
                        {"user": user_input, "assistant": assistant_response}
                    )

                    # Keep only the last 10 conversations
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]

                    # Update memory with last 10 conversations
                    memory_parts = []
                    for conv in conversation_history:
                        memory_parts.append(f"User: {conv['user']}")
                        memory_parts.append(f"Assistant: {conv['assistant']}")

                    self.memory = "\n".join(
                        ["Previous conversation context:", *memory_parts, "---"]
                    )

                    # Clean up completed processes
                    self.running_processes = [
                        p for p in self.running_processes if not p.done()
                    ]

                except Exception as e:
                    # Make sure to stop spinner if there's an error
                    stop_thinking_animation()
                    print(f"🧙‍♂️ Sorry, I encountered an error: {str(e)}")

                print("\n")  # Add newline after complete response

            except KeyboardInterrupt:
                # Stop spinner on interrupt
                stop_thinking_animation()
                print("\n🧙‍♂️ Goodbye! Happy optimizing with AdaptiQ!")
                break
            except Exception as e:
                # Stop spinner on any error
                stop_thinking_animation()
                print(f"🧙‍♂️ Sorry, I encountered an error: {str(e)}\n")

    async def start_non_interactive_session(
        self, prompt: str, output_format: str = "text"
    ) -> dict:
        """
        Process a single prompt in non-interactive mode and return the response.

        Args:
            prompt: The user's input prompt
            output_format: Output format - 'text', 'json', or 'stream-json'

        Returns:
            dict: Response with status, message, and optional metadata
        """
        try:
            # Get response using the non-streaming chat method
            response = await self.chat(prompt)

            # Update memory with this interaction
            if not hasattr(self, "conversation_history"):
                self.conversation_history = []

            self.conversation_history.append({"user": prompt, "assistant": response})

            # Keep only the last 10 conversations
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Update memory
            memory_parts = []
            for conv in self.conversation_history:
                memory_parts.append(f"User: {conv['user']}")
                memory_parts.append(f"Assistant: {conv['assistant']}")

            self.memory = "\n".join(
                ["Previous conversation context:", *memory_parts, "---"]
            )

            # Clean up completed processes
            self.running_processes = [p for p in self.running_processes if not p.done()]

            # Format response based on requested output format
            if output_format == "json" or output_format == "stream-json":
                return {
                    "status": "success",
                    "response": response,
                    "timestamp": asyncio.get_event_loop().time(),
                    "memory_context_length": len(self.conversation_history),
                }
            else:
                return {"status": "success", "response": response}

        except Exception as e:
            error_response = {
                "status": "error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }
            return error_response


def adaptiq_run_wizard(llm_provider: str, api_key: str):
    """
    Start an interactive session with the AdaptiqWizardAssistant.

    Args:
        llm_provider (str): the Provider to use for the wizard assistant
        api_key (str): The API key for the wizard assistant
    """

    async def main():
        # Initialize the wizard with the provided API key
        wizard = AdaptiqWizardAssistant(llm_provider=llm_provider, api_key=api_key)

        # Start interactive session
        await wizard.start_interactive_session()

    # Run the async main function
    asyncio.run(main())


def adaptiq_run_wizard_headless(
    llm_provider: str, api_key: str, prompt: str, output_format: str = "text"
):
    """
    Run AdaptiqWizardAssistant in non-interactive headless mode for production/automation.

    Args:
        llm_provider (str): The Provider to use for the wizard assistant
        api_key (str): The API key for the wizard assistant
        prompt (str): The prompt/question to process
        output_format (str): Output format - 'text', 'json', or 'stream-json'
    """

    async def main():
        try:
            # Initialize the wizard
            wizard = AdaptiqWizardAssistant(llm_provider=llm_provider, api_key=api_key)

            # Process the prompt
            result = await wizard.start_non_interactive_session(prompt, output_format)

            # Output based on format
            if output_format == "json" or output_format == "stream-json":
                import json

                print(json.dumps(result, indent=2))
            else:
                if result["status"] == "success":
                    print(result["response"])
                else:
                    print(f"Error: {result['error']}")
                    exit(1)

        except (ValueError, RuntimeError, KeyError, TypeError, ImportError) as e:
            if output_format == "json" or output_format == "stream-json":
                import json

                error_result = {"status": "error", "error": str(e)}
                print(json.dumps(error_result, indent=2))
            else:
                print(f"Error: {str(e)}")
            exit(1)

    # Run the async main function
    asyncio.run(main())
