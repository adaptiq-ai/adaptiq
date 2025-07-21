import sys
import os

# ✅ Add the current directory to the Python path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ Import CrewAI components
from crewai import  LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from settings import settings

# ✅ AdaptiQ-specific decorators for instrumentation
# - Logs agent thoughts, tool usage, reasoning
# - Logs task execution lifecycle and state
from adaptiq import instrumental_agent_logger, instrumental_task_logger

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

llm = LLM(model="openai/gpt-4.1", api_key=settings.OPENAI_API_KEY)
from tools.describe_image_tool import DescribeImageTool
describe_tool = DescribeImageTool()

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

    def prompt_engineer(self) -> Agent:
        return Agent(
			config=self.agents_config['prompt_engineer'],
            llm=llm,
            tools=[describe_tool],
			verbose=True
		)
    
    @instrumental_task_logger  # ✅ Log task execution status
    @task
    
    def prompt_task(self) -> Task:
        return Task(
			config=self.tasks_config['prompt_task'],
			tools=[describe_tool]
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
