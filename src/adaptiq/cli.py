import argparse
import json
import logging
import sys  # To get command line arguments
import logging

from adaptiq.core.reporting import Aggregator, AdaptiqLogger, get_logger
from adaptiq.core.pipelines import PreRunPipeline, PostRunPipeline
from adaptiq.agents.crew_ai import CrewConfig, CrewPromptParser, CrewLogParser

get_logger()


def find_and_clear_log_files(
    search_directory=".", log_filename="log.txt", json_filename="log.json"
):
    """
    Search for log files in the directory and clear their content.

    Args:
        search_directory: Directory to search for log files (default: current directory)
        log_filename: Name of the text log file to search for
        json_filename: Name of the JSON log file to search for
    """

    # Search for files in the directory
    # for root, _, files in os.walk(search_directory):
    #     for file in files:
    #         file_path = os.path.join(root, file)

    #         # Check if it's a text log file
    #         if file == log_filename:
    #             print(f"Found text log file: {file_path}")
    #             with open(file_path, "w", encoding="utf-8") as f:
    #                 f.write("")
    #             print(f"Cleared content from: {file_path}")

    #         # Check if it's a JSON log file
    #         elif file == json_filename:
    #             print(f"Found JSON log file: {file_path}")
    #             with open(file_path, "w", encoding="utf-8") as f:
    #                 json.dump([], f, ensure_ascii=False, indent=2)
    #             print(f"Cleared content from: {file_path}")

    pass



def execute_pre_run_only(args):
    """Execute only the pre_run pipeline."""

    config_path = args.config_path
    template = args.template
    base_config = None
    base_prompt_parser = None
    
    if template == "crew-ai":
        base_config = CrewConfig(config_path=config_path, preload=True)
        base_prompt_parser = CrewPromptParser(config_path=config_path)

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


def execute_post_run_and_reconciliation(
    config_path:str, output_path:str, feedback:str, template: str, logs_path:str, run_number=None, agent_metrics=None, should_send_report=True
):
    """Execute post_run and reconciliation pipelines only."""
    run_prefix = f"[RUN {run_number}] " if run_number is not None else ""

    base_config = None
    base_log_parser = None
    if template == "crew-ai":
        base_config = CrewConfig(config_path=config_path, preload=True)
        base_log_parser = CrewLogParser(logs_path= logs_path, output_path=output_path)
    # Initialize the aggregator
    logging.info(f"agent_metrics: {agent_metrics}")
    aggregator = Aggregator(config_data=base_config.get_config(), original_prompt=base_config.get_prompt())
    aggregator._default_run_mode = False
    tracer = AdaptiqLogger.setup()

    # Process ALL crew metrics entries, not just the last one
    total_execution_time = 0
    total_peak_memory = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_successful_requests = 0

    if agent_metrics:
        logging.info(
            f"{run_prefix}Processing {len(agent_metrics)} crew metrics entries..."
        )

        for i, metrics in enumerate(agent_metrics):
            execution_time_seconds = metrics.get("execution_time_seconds", 0)
            peak_memory_mb = metrics.get("peak_memory_mb", 0)
            prompt_tokens = metrics.get("prompt_tokens", 0)
            completion_tokens = metrics.get("completion_tokens", 0)
            successful_requests = metrics.get("successful_requests", 0)

            # Accumulate totals
            total_execution_time += execution_time_seconds
            total_peak_memory = max(
                total_peak_memory, peak_memory_mb
            )  # Use max for peak memory
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_successful_requests += successful_requests

            logging.info(
                f"{run_prefix}Metrics {i + 1}/{len(agent_metrics)}: "
                f"time={execution_time_seconds:.2f}s, "
                f"tokens={prompt_tokens + completion_tokens}, "
                f"requests={successful_requests}"
            )

    try:
        # Step 1: Execute post_run pipeline
        logging.info(f"{run_prefix}STEP 1: Executing post_run pipeline...")

        post_run = PostRunPipeline(
            base_config=base_config,
            base_log_parser=base_log_parser,
            output_dir=output_path,
            feedback=feedback
        )
        post_run_results = post_run.execute_post_run_pipeline()

        logging.info(f"{run_prefix}[SUCCESS] Post-run pipeline completed successfully")



        logging.info(
            f"{run_prefix}Agent token stats: input={total_prompt_tokens}, output={total_completion_tokens}, calls={total_successful_requests}"
        )
        logging.info(
            f"{run_prefix}Total execution time: {total_execution_time:.2f} seconds, Peak memory usage: {total_peak_memory:.2f} MB"
        )

        validation_summary_path = post_run_results.get(
            "validation_results", {}
        ).get("outputs").get("validation_summary_path")
        reconciliation_results= post_run_results.get(
            "reconciliation_results", {}
        )
        # Process each crew metrics entry and add to aggregator
        if agent_metrics:
            for i, metrics in enumerate(agent_metrics):
                execution_time_seconds = metrics.get("execution_time_seconds", 0)
                peak_memory_mb = metrics.get("peak_memory_mb", 0)
                prompt_tokens = metrics.get("prompt_tokens", 0)
                completion_tokens = metrics.get("completion_tokens", 0)
                successful_requests = metrics.get("successful_requests", 0)
                execution_count = metrics.get("execution_count", i + 1)
                

                # Update aggregator with each run's data
                aggregator.increment_run_count()

                # Update token statistics for this specific run
                aggregator.update_avg_run_tokens(
                    pre_input=0,
                    pre_output=0,
                    post_input=prompt_tokens,
                    post_output=completion_tokens,
                    recon_input=0,
                    recon_output=0,
                    default_run_mode=False,
                )

                # Update run time for this specific run
                aggregator.update_avg_run_time(execution_time_seconds)

                # Update error count (0 for successful run)
                aggregator.update_error_count(0)

                # Calculate average reward from simulated scenarios
                avg_reward = aggregator.calculate_avg_reward(
                    validation_summary_path=validation_summary_path, reward_type="execution"
                ) 
                new_prompt = reconciliation_results.get("summary", {}).get("new_prompt", "")
                
                # Add each run summary to the aggregator
                aggregator.add_run_summary(
                    run_name=f"{run_prefix}Execution-{execution_count}",
                    reward=avg_reward,
                    api_calls=successful_requests,
                    suggested_prompt=new_prompt,
                    status="completed",
                    issues=[],
                    error=None,
                    memory_usage=peak_memory_mb,
                    run_time_seconds=execution_time_seconds,
                    execution_logs=tracer.get_logs(),
                )

                logging.info(
                    f"{run_prefix}Added run summary for execution {execution_count}"
                )
        else:
            # Handle case where no crew metrics provided - add single summary
            logging.info(
                f"{run_prefix}No crew metrics provided, adding single run summary..."
            )

            aggregator.increment_run_count()
            aggregator.update_avg_run_tokens(
                pre_input=0,
                pre_output=0,
                post_input=0,
                post_output=0,
                recon_input=0,
                recon_output=0,
                default_run_mode=False,
            )
            aggregator.update_avg_run_time(0)
            aggregator.update_error_count(0)

            avg_reward = aggregator.calculate_avg_reward(
                validation_summary_path=validation_summary_path, reward_type="execution"
            )

            new_prompt = reconciliation_results.get("summary", {}).get("new_prompt", "")

            aggregator.add_run_summary(
                run_name=f"{run_prefix}Single-Run",
                reward=avg_reward,
                api_calls=0,
                suggested_prompt=new_prompt,
                status="completed",
                issues=[],
                error=None,
                memory_usage=0,
                run_time_seconds=0,
                execution_logs=tracer.get_logs(),
            )

        # Only send results if should_send_report is True
        if should_send_report:
            logging.info(
                f"{run_prefix}Building and sending comprehensive project results..."
            )

            # Build project result JSON (now contains ALL runs)
            project_result = aggregator.build_project_result()

            # Merge old with new result then saving the new report
            merged_result = aggregator.merge_json_reports(new_json_data=project_result)
            aggregator.save_json_report(merged_result)

            # Send results to endpoint
            if aggregator.email != "":
                success = aggregator.send_run_results(merged_result)

                if success:
                    logging.info(
                        f"{run_prefix}Successfully sent comprehensive run results to reporting endpoint"
                    )
                else:
                    logging.warning(
                        f"{run_prefix}Failed to send run results to reporting endpoint"
                    )
            else:
                logging.info(f"{run_prefix} are successfully saved locally")
        else:
            logging.info(
                f"{run_prefix}Run summaries added to aggregator - report will be sent when all runs complete"
            )

        find_and_clear_log_files()

        return True

    except Exception as e:
        logging.error(f"{run_prefix}Error during pipeline execution: {str(e)}")
        logging.error(f"{run_prefix}Pipeline execution stopped due to error.")

        # Handle errors for all runs if crew_metrics provided
        if agent_metrics:
            for i, metrics in enumerate(agent_metrics):
                execution_count = metrics.get("execution_count", i + 1)

                # Update aggregator with error information for each run
                aggregator.increment_run_count()
                aggregator.update_error_count(1)
                aggregator.update_avg_run_time(0)

                # Add failed run summary for each execution
                aggregator.add_run_summary(
                    run_name=f"{run_prefix}Execution-{execution_count}",
                    reward=0.0,
                    api_calls=0,
                    suggested_prompt="",
                    status="failed",
                    issues=["Pipeline execution failed"],
                    error=str(e),
                    memory_usage=0,
                    run_time_seconds=0,
                    execution_logs=tracer.get_logs(),
                )
        else:
            # Handle single failed run
            aggregator.increment_run_count()
            aggregator.update_error_count(1)
            aggregator.update_avg_run_time(0)

            aggregator.add_run_summary(
                run_name=f"{run_prefix}Single-Run",
                reward=0.0,
                api_calls=0,
                suggested_prompt="",
                status="failed",
                issues=["Pipeline execution failed"],
                error=str(e),
                memory_usage=0,
                run_time_seconds=0,
                execution_logs=tracer.get_logs(),
            )

        # Only send results if should_send_report is True (even for failed runs)
        if should_send_report:
            logging.info(
                f"{run_prefix}Building and sending project results for failed run..."
            )
            project_result = aggregator.build_project_result()
            aggregator.send_run_results(project_result)
        else:
            logging.info(
                f"{run_prefix}Failed run summary added to aggregator - report will be sent when all runs complete"
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

def handle_default_run_command(args):
    """Handles the logic for the 'default-run' command - executes only pre_run pipeline."""

    logging.info("Executing the 'default-run' command...")

    # if args.output_path:
    #     logging.info(f"Results will be saved to: {args.output_path}")

    # if args.log:
    #     logging.info(f"Logging to file: {args.log}")

    logging.info("DEFAULT-RUN MODE: Executing pre-run pipeline only")

    # Execute pre-run pipeline (runs only once)
    success = execute_pre_run_only(args)

    if success:
        logging.info("=" * 60)
        logging.info("[SUCCESS] ADAPTIQ DEFAULT-RUN COMPLETED SUCCESSFULLY")
        logging.info("Pre-run pipeline executed successfully!")
        logging.info("=" * 60)
    else:
        logging.error("DEFAULT-RUN FAILED!")
        sys.exit(1)

def handle_run_command(args):
    """Handles the logic for the 'run' command - executes post_run and reconciliation sequentially."""

    logging.info("Executing the 'run' command...")
    logging.info(f"Configuration file: {args.config_path}")

    if args.output_path:
        logging.info(f"Results will be saved to: {args.output_path}")

    if args.log:
        logging.info(f"Logging to file: {args.log}")

    if args.template:
        logging.info(f"Using {args.template} configs")

    

    # Handle crew metrics if provided
    agent_metrics_list = []
    if args.agent_metrics:
        logging.info("Crew metrics provided:")
        try:
            agent_metrics_list = json.loads(args.agent_metrics)
            print(f"[CREW_METRICS] {agent_metrics_list}")

            # Extract current execution count from crew metrics
            if agent_metrics_list and len(agent_metrics_list) > 0:
                current_execution_count = agent_metrics_list[-1].get(
                    "execution_count", 0
                )
                logging.info(
                    f"Current execution count from crew metrics: {current_execution_count}"
                )

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format for crew metrics: {e}")
            print(f"[CREW_METRICS_RAW] {args.agent_metrics}")

    # Get send_report flag (defaults to True if not provided)
    should_send_report = True
    if args.send_report:
        should_send_report = args.send_report
        logging.info(f"Send report flag: {should_send_report}")

    # Execute single run (post_run and reconciliation only)
    logging.info("=" * 60)
    logging.info("[START] STARTING ADAPTIQ RUN")
    logging.info("=" * 60)

    success = execute_post_run_and_reconciliation(
        config_path= args.config_path,
        output_path=args.output_path,
        feedback=args.feedback,
        template =args.template,
        logs_path = args.log,
        run_number=None,
        agent_metrics=agent_metrics_list,
        should_send_report=should_send_report,
    )

    if success:
        logging.info("[SUCCESS] ADAPTIQ RUN COMPLETED SUCCESSFULLY")
        logging.info("All pipelines (post_run -> reconciliation) executed successfully!")
        if should_send_report:
            logging.info("[REPORT] Report sent successfully!")
        else:
            logging.info("[REPORT] Report sending skipped (send_report=False)")
    else:
        logging.error("ADAPTIQ RUN FAILED!")
        logging.error("Execution failed during pipeline execution.")
        sys.exit(1)

    logging.info("=" * 60)


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

    # --- Define the 'default-run' command ---
    parser_default_run = subparsers.add_parser(
        "default-run", help="Run only the pre_run pipeline (executes once)."
    )
    # Add arguments specific to the 'default-run' command
    parser_default_run.add_argument(
        "--config_path",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the configuration file for the pre_run pipeline.",
    )
    parser_default_run.add_argument(
        "--template",
        type=str,
        metavar="TEMPLATE",
        default="crew-ai",
        help="Template to use for initialization (default: crew-ai)",
    )
    
    # parser_default_run.add_argument(
    #     "--output_path",
    #     type=str,
    #     metavar="PATH",
    #     help="Path to save the results of the pre_run pipeline.",
    # )
    # parser_default_run.add_argument(
    #     "--log",
    #     type=str,  # Expecting a string (path)
    #     metavar="PATH",  # Placeholder name shown in help message
    #     help="Optional path to a file for logging output.",
    # )
    # Set the function to call when 'default-run' is chosen
    parser_default_run.set_defaults(func=handle_default_run_command)

    # --- Define the 'run' command ---
    parser_run = subparsers.add_parser(
        "run",
        help="Run the post_run and reconciliation pipelines (single execution with optional report sending).",
    )
    # Add arguments specific to the 'run' command
    parser_run.add_argument(
        "--config_path",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the configuration file for post_run and reconciliation pipelines.",
    )

    parser_run.add_argument(
        "--template",
        type=str,
        metavar="TEMPLATE",
        default="crew-ai",
        help="Template to use for initialization (default: crew-ai)",
    )
    parser_run.add_argument(
        "--output_path",
        type=str,
        metavar="PATH",
        help="Path to save the results of post_run and reconciliation pipelines.",
    )
    parser_run.add_argument(
        "--log",
        type=str,  # Expecting a string (path)
        metavar="PATH",  # Placeholder name shown in help message
        help="Optional path to a file for logging output.",
    )
    parser_run.add_argument(
        "--agent_metrics",
        type=str,
        metavar="JSON_STRING",
        help="Optional agent metrics data in JSON format to be processed during the run.",
    )
    parser_run.add_argument(
        "--send_report",
        type=lambda x: x.lower() in ["true", "1", "yes", "on"],
        metavar="BOOL",
        default=True,
        help="Whether to send a report after execution. Accepts: true/false, 1/0, yes/no, on/off (default: true).",
    )
    parser_run.add_argument(
        "--feedback",
        type=str,
        metavar="STRING",
        help="Optional human feedback about the agent's performance to be used for prompt optimization.",
    )
    # Set the function to call when 'run' is chosen
    parser_run.set_defaults(func=handle_run_command)

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