import argparse
import logging
import sys
import logging

from adaptiq.core.reporting import Aggregator, get_logger
from adaptiq.core.pipelines import PreRunPipeline, PostRunPipeline
from adaptiq.agents.crew_ai import CrewConfig, CrewPromptParser, CrewLogParser

get_logger()

def execute_pre_run_only(args):
    """Execute only the pre_run pipeline."""

    config_path = args.config_path
    template = args.template
    base_config = None
    base_prompt_parser = None
    
    if template == "crew-ai":
        base_config = CrewConfig(config_path=config_path, preload=True)
        base_prompt_parser = CrewPromptParser(config_data=config_path)

    try:
        logging.info("STEP : Executing pre_run pipeline...")
        
        # Execute pre_run pipeline
        pipeline = PreRunPipeline(base_config = base_config , base_prompt_parser =base_prompt_parser, output_path="./results")
        simulation_results = pipeline.execute_pre_run_pipeline()

        logging.info("[SUCCESS] Pre-run pipeline completed successfully")

        return True

    except Exception as e:
        logging.error(f"Error during pre-run pipeline execution: {str(e)}")
        logging.error("Pre-run pipeline execution stopped due to error.")
        return False

def execute_post_run_only(
    config_path: str, output_path: str, feedback: str, template: str, logs_path: str, run_number=None, agent_metrics=None, should_send_report=True
):
    """Execute post_run only."""
    run_prefix = "[RUN %s] " % run_number if run_number is not None else ""

    base_config = None
    base_log_parser = None
    if template == "crew-ai":
        base_config = CrewConfig(config_path=config_path, preload=True)
        base_log_parser = CrewLogParser(logs_path=logs_path, output_path=output_path)
        base_prompt_parser = CrewPromptParser(config_data=config_path)
    
    # Initialize the aggregator
    logging.info("agent_metrics: %s", agent_metrics)
    aggregator = Aggregator(config_data=base_config.get_config(), original_prompt=base_config.get_prompt())
    aggregator._default_run_mode = False

    try:
        # Step 1: Execute post_run pipeline
        logging.info("%sSTEP 1: Executing post_run pipeline...", run_prefix)

        post_run = PostRunPipeline(
            base_config=base_config,
            base_log_parser=base_log_parser,
            output_dir=output_path,
            feedback=feedback
        )
        post_run_results = post_run.execute_post_run_pipeline()

        logging.info("%s[SUCCESS] Post-run pipeline completed successfully", run_prefix)

        validation_summary_path = post_run_results.get(
            "validation_results", {}
        ).get("outputs", {}).get("validation_summary_path")
        reconciliation_results = post_run_results.get(
            "reconciliation_results", {}
        )

        # Use aggregator to handle all metrics processing and reporting
        success = aggregator.aggregate_results(
            agent_metrics=agent_metrics,
            validation_summary_path=validation_summary_path,
            reconciliation_results=reconciliation_results,
            should_send_report=should_send_report,
        )

        find_and_clear_log_files()
        return success

    except Exception as e:
        logging.error("%sError during pipeline execution: %s", run_prefix, str(e))
        logging.error("%sPipeline execution stopped due to error.", run_prefix)

        # Use aggregator to handle error reporting
        success = aggregator.aggregate_results(
            agent_metrics=agent_metrics,
            validation_summary_path=None,
            reconciliation_results=None,
            should_send_report=should_send_report,
        )
        
        return False

def handle_init_command(args):
    """Handles the logic for the 'init' command - initializes a new Adaptiq project."""
    logging

    logging.info("Executing the 'init' command...")
    logging.info(f"Project name: {args.name}")
    logging.info(f"Template: {args.template}")
    logging.info(f"Path: {args.path}")

    # Initialize the project with the specified template
    try:

        if args.template == "crew-ai":
            crew_config = CrewConfig()
            is_created, msg = crew_config.create_project_template(base_path=args.path, project_name=args.name)

            logging.info(msg)
            return is_created

        return False

    except Exception as e:
        logging.error(f"Error initializing project: {str(e)}")
        return False

def handle_validate_command(args):
    """Handles the logic for the 'validate' command - validates project configuration and template structure."""

    logging.info("Executing the 'validate' command...")
    logging.info(f"Configuration file: {args.config_path}")
    logging.info(f"Template type: {args.template}")

    # Validate the project configuration
    try:
        


        if args.template == "crew-ai":
            crew_config = CrewConfig(config_path=args.config_path, preload=True)
            is_valid, msg = crew_config.validate_config()

            if is_valid:
                logging.info("Project configuration and template structure are valid.")
                return True
            else:
                logging.error(f"Validation failed: {msg}")
                return False

        return False

    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        return False

def main():
    """Main entry point for the adaptiq CLI."""
    parser = argparse.ArgumentParser(
        prog="adaptiq",  # Program name shown in help
        description="Adaptiq CLI: Run and manage prompt optimization tasks.",
    )


    # Create subparsers for different commands (like 'run')
    subparsers = parser.add_subparsers(
        dest="command",  # Attribute name to store which subcommand was used
        help="Available commands",
        required=True,  # Make choosing a command mandatory
    )

     # --- Define the 'init' command ---
    parser_init = subparsers.add_parser(
        "init", help="Initialize a new Adaptiq project with configuration templates."
    )

    # Add arguments specific to the 'init' command
    parser_init.add_argument(
        "--name",
        type=str,
        metavar="PROJECT_NAME",
        required=True,
        help="Name of the project to initialize.",
    )

    parser_init.add_argument(
        "--template",
        type=str,
        metavar="TEMPLATE",
        default="crew-ai",
        help="Template to use for initialization (default: crew-ai).",
        
    )

    parser_init.add_argument(
        "--path",
        type=str,
        metavar="PROJECT_PATH",
        default=".",
        help="Path to the current directory.",
    )

    # Set the function to call when 'init' is chosen
    parser_init.set_defaults(func=handle_init_command)

    
    # --- Define the 'validate' command ---
    parser_validate = subparsers.add_parser(
        "validate", help="Validate project configuration and template structure."
    )

    # Add arguments specific to the 'validate' command
    parser_validate.add_argument(
        "--config_path",
        type=str,
        metavar="CONFIG_PATH",
        required=True,
        help="Path to the configuration file to validate.",
    )

    parser_validate.add_argument(
        "--template",
        type=str,
        metavar="TEMPLATE",
        default="crew-ai",
        help="Template to use for initialization (default: crew-ai)",
    )

    # Set the function to call when 'validate' is chosen
    parser_validate.set_defaults(func=handle_validate_command)

    # If no arguments are given (just 'adaptiq'), argparse automatically shows help
    # because subparsers are 'required'.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Execute the function associated with the chosen subcommand
    args.func(args)

if __name__ == "__main__":
    # This allows running the script directly (python src/adaptiq/cli.py run --log ...)
    # although the primary way will be via the installed 'adaptiq' command.
    main()