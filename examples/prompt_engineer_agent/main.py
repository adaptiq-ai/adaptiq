import sys
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


@instrumental_crew_logger(
    log_to_console=True
)  # ✅ Logs crew-level metrics and agent/task events
def run():
    """
    Main function to run the Crew execution process.
    """
    try:
        inputs = {"image_path": "./images/demo.jpg"}

        # 🧠 Instantiate and run the configured Crew
        crew_instance = GenericCrew().crew()
        result = crew_instance.kickoff(inputs=inputs)

        # ✅ Attach crew instance to result so AdaptiQ can log all details
        result._crew_instance = crew_instance
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


@instrumental_run(
    config_path="./config/adaptiq_config.yml",
    enabled=False,
    feedback="Add more dramatic rim lighting on the dragon's wings, warmer castle window glows, and atmospheric mist rising from the water to enhance the mystical mood.",
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
