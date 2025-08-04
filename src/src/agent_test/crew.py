# ✅ Import CrewAI components
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# ✅ Import tools (replace or extend as needed)
from crewai_tools import FileReadTool  # Example of a built-in generic tool

from adaptiq.agents.crew_ai import create_crew_instrumental
crew_instrumental = create_crew_instrumental()

# ✅ Tool initialization (customize these for your use case)
tool_1 = FileReadTool(file_path="knowledge/example_context.txt")  # 📝 Read background info

@CrewBase
class GenericCrew():
    """🧠 Generic AI Crew
    A flexible blueprint for running AI agents on modular tasks.
    Replace tools, agents, and tasks as needed.
    """
    
    # 🔧 YAML configuration paths for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @crew_instrumental.agent_logger  
    @agent
    def generic_agent(self) -> Agent:
        """Create a generic agent with assigned tools."""
        return Agent(
            config=self.agents_config['generic_agent'],  # 🔁 Replace key as needed
            verbose=True,
            tools=[tool_1],
        )
    
    @crew_instrumental.task_logger  # ✅ Log task execution status
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