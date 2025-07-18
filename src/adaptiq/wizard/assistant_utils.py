import os


def create_agent_repo_template(project_name=None, base_path=".") -> str:
    """
    Creates a repository template structure for an agent example.

    Args:
        project_name (str): Name of the project (replaces 'agent_example')
        base_path (str): The base directory where the template will be created

    Returns:
        str: Success message or error message
    """

    # Validate project name
    if not project_name:
        return "❌ Error: Project name not provided. Please specify a project name."

    # Clean project name (remove spaces, special characters, etc.)
    project_name = project_name.replace(" ", "_").replace("-", "_")
    project_name = "".join(c for c in project_name if c.isalnum() or c == "_")

    # Check if project directory already exists
    project_path = os.path.join(base_path, "src", project_name)
    if os.path.exists(project_path):
        return f"❌ Error: Folder template already exists at '{project_path}'"

    # Define the directory structure
    directories = [
        f"src/{project_name}",
        f"src/{project_name}/config",
        f"src/{project_name}/tools",
    ]

    # Define the files with their content
    files_content = {
        f"src/{project_name}/config/adaptiq_config.yml": f"""# =============================================================================
# ADAPTIQ PROJECT CONFIGURATION
# =============================================================================
# This file contains all the configuration settings for your AdaptiQ project.
# Please update the placeholder values with your actual project information.
# Reports folder: Contains summary content for both the default run and optimized run, including the final optimized prompt and a general overview.
# Results folder: Stores summary results for both runs, featuring the simulated optimized prompt and Q-table for each mode.
#   - adaptiq_analysis_pre_run_report.txt: First simulated prompt suggestion analysis
#   - adaptiq_q_table.json: Q-table insights with states and actions data
#   - adaptiq_results.json: Full results containing all run information
#   - adaptiq_simulated_scenarios.json: Suggested scenarios for testing and validation
#   - parsed_logs.json: Parsed logs with attributed rewards for learning
#   - raw_logs.json: Raw log data before processing
#   - results.json: Complete information for both default run and optimized run
#   - validated_logs.json: Logs verified with LLM when Q-values are uncertain
#   - validation_summary.json: Summary explaining the validation process and results
# Report_data folder: Holds updated JSON run results for local storage.

# --- Project Information ---
project_name: "{project_name}"        # Name of your project
email: ""          # Developer's email for reporting (By entering your email, you agree to receive the report and allow us to process your data for this purpose. If you do not agree, leave it as an empty string: "")

# --- LLM Configuration ---
# Currently only OpenAI is supported
llm_config:
  provider: "openai"                     # LLM provider (only 'openai' supported)
  model_name: "gpt-4.1-mini"              # OpenAI model to use (Currently, only the gpt-4.1-mini model is supported. More models will be available in future versions.)
  api_key: "your_openai_api_key"         # Your OpenAI API key

# --- Framework Adapter Configuration ---
# Currently only CrewAI is supported
framework_adapter:
  name: "crewai"                         # Framework name (only 'crewai' supported)
  settings:
    execution_mode: "prod"                # Execution mode (dev or prod)
    log_source:
      type: "file_path"                  # Type of log source (for now it is supported only path mode)
      path: "log.json"       # Default path to AdaptiQ agent's logs

# --- Agent Configuration Files ---
# Paths to your agent's configuration files
agent_modifiable_config:
  prompt_configuration_file_path: "./config/tasks.yaml"    # Path to tasks configuration
  agent_definition_file_path: "./config/agents.yaml"       # Path to agents configuration
  agent_name: "your_agent_name"                                       # Name of your agent
  agent_tools:                                 # List of tools your agent uses (Remember to provide the description of the tool for better accuracy)
    - name: "ExampleTool"
      description: "Description of the tool what does actually do"  
    
# --- Report Output Configuration ---
# Where AdaptiQ will save optimization reports
report_config:
  output_path: "reports/{project_name}.md"    # Path for the optimization report (Been set by default)

# --- Alert Mode Configuration ---
# Configure when AdaptiQ should generate reports
alert_mode:
  # On-demand mode: Generate a report after a specified number of runs (useful when the crew is in loop mode or wrapped within an endpoint).
  on_demand:
    enabled: false                        # Enable on-demand reporting
    runs: 5                              # Number of runs before generating report
  
  # Per-run mode: Generate report after each run
  per_run:
    enabled: true                       # Enable per-run reporting

# =============================================================================
# NEXT STEPS:
# 1. Update all placeholder values with your actual project information
# 2. Validate your configuration: wizard check config adaptiq_config.yml
# 3. Run your first optimization: wizard start
# 4. Instrument your code with @instrumental_logger and @instrumental_run decorators
# 5. Run your full optimization: after executing your agent adaptiq will runs automatically the process
# =============================================================================
""",
        f"src/{project_name}/config/agents.yaml": """agent_template:
  role: >
    Describe the role of the agent here.
  goal: >
    Describe the main goal or mission of the agent.
  backstory: >
    Provide a brief background story that helps the agent perform its task with more context or personality.
""",
        f"src/{project_name}/config/tasks.yaml": """generic_task_template:
  description: >
    Describe your task here.
  expected_output: >
    Describe the expected output of the task here.
  agent: "{{ agent_type }}"
""",
        f"src/{project_name}/tools/__init__.py": "",
        f"src/{project_name}/__init__.py": "",
        f"src/{project_name}/tools/custom_tool.py": """from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

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
""",
        f"src/{project_name}/crew.py": '''import sys
import os

# ✅ Add the current directory to the Python path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ Import CrewAI components
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# ✅ Import tools (replace or extend as needed)
from tools.custom_tool import GenericRetrievalTool, GenericActionTool  # 🔁 Replace with your actual tool class names
from crewai_tools import FileReadTool  # Example of a built-in generic tool

# ✅ AdaptiQ-specific decorators for instrumentation
# - Logs agent thoughts, tool usage, reasoning
# - Logs task execution lifecycle and state
from adaptiq import instrumental_agent_logger, instrumental_task_logger

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ✅ Tool initialization (customize these for your use case)
tool_1 = FileReadTool(file_path="knowledge/example_context.txt")  # 📝 Read background info
tool_2 = GenericRetrievalTool()
tool_3 = GenericActionTool()

@CrewBase
class GenericCrew():
    """🧠 Generic AI Crew
    A flexible blueprint for running AI agents on modular tasks.
    Replace tools, agents, and tasks as needed.
    """
    
    # 🔧 YAML configuration paths for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @instrumental_agent_logger  # ✅ Log reasoning, tool use, etc.
    @agent
    def generic_agent(self) -> Agent:
        """Create a generic agent with assigned tools."""
        return Agent(
            config=self.agents_config['generic_agent'],  # 🔁 Replace key as needed
            verbose=True,
            tools=[tool_1, tool_2, tool_3],
        )
    
    @instrumental_task_logger  # ✅ Log task execution status
    @task
    def generic_task(self) -> Task:
        """Define a generic task to be executed."""
        return Task(
            config=self.tasks_config['generic_task'],  # 🔁 Replace key as needed
        )
    
    @crew
    def crew(self) -> Crew:
        """👥 Assembles the agent-task pipeline as a Crew instance."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # 🔁 Change to Process.parallel if needed
            verbose=True
        )
''',
        f"src/{project_name}/main.py": '''import sys
import warnings
import os

# ✅ Add the current directory to the system path
# This allows local imports like `from crew import MyCrew`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ Import your crew (generic name recommended for reusability)
from crew import GenericCrew  # 🔁 Replace `GenericCrew` with your specific crew class

# ✅ Load environment variables from `.env`
from dotenv import load_dotenv
load_dotenv()

# ✅ Suppress known irrelevant warnings (optional)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ✅ Import AdaptiQ instrumentation decorators
# - `instrumental_crew_logger`: Logs execution metrics for agents, tools, and tasks
# - `instrumental_run`: Triggers AdaptiQ run processing, useful for evaluation dashboards
from adaptiq import instrumental_crew_logger, instrumental_run

@instrumental_crew_logger(log_to_console=True)  # ✅ Logs crew-level metrics and agent/task events
def run():
    """
    Main function to run the Crew execution process.
    """
    try:
        # 🧠 Instantiate and run the configured Crew
        crew_instance = GenericCrew().crew()
        result = crew_instance.kickoff()
        
        # ✅ Attach crew instance to result so AdaptiQ can log all details
        result._crew_instance = crew_instance
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

@instrumental_run(
    config_path="./config/adaptiq_config.yml",            # ✅ Path of adaptiq config yml file
)
def main():
    """
    Entry point for the crew run process.
    Also supports post-run logic (e.g., saving outputs, triggering evaluations).
    """
    run()
    # 🔁 Insert any post-execution logic here (e.g., save report, update database, etc.)

# ✅ Standard Python entry point check
if __name__ == "__main__":
    main()
''',
    }

    try:
        # Create directories
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")

        # Create files with content
        for file_path, content in files_content.items():
            full_file_path = os.path.join(base_path, file_path)

            # Create the file with content
            if not os.path.exists(full_file_path):
                with open(full_file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Created file: {full_file_path}")
            else:
                print(f"File already exists: {full_file_path}")

        return f"✅ Repository template created successfully!\n📁 Structure created under: {os.path.join(base_path, 'src', project_name)}"

    except Exception as e:
        return f"❌ Error creating template: {str(e)}"
