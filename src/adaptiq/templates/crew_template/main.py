import warnings

# ✅ Import your crew (generic name recommended for reusability)
from crew import GenericCrew  # 🔁 Replace `GenericCrew` with your specific crew class

# ✅ Suppress known irrelevant warnings (optional)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from adaptiq.agents.crew_ai import create_crew_instrumental

crew_instrumental = create_crew_instrumental()


@crew_instrumental.crew_logger(
    log_to_console=True
)  # ✅ Logs crew-level metrics and agent/task events
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


@crew_instrumental.run(
    config_path="./config/adaptiq_config.yml",  # ✅ Path of adaptiq config yml file
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
