import argparse
import sys  # To get command line arguments
import json
import time
import tracemalloc
import logging
import os

from adaptiq import (
    adaptiq_reconciliation_pipeline,
    adaptiq_pre_run_pipeline,
    adaptiq_post_run_pipeline,
    adaptiq_run_wizard,
    adaptiq_run_wizard_headless,
    get_token_stats,
    AdaptiqAggregator,
    AdaptiqTraceLogger,
)


def setup_logging(log_path=None):
    """
    Sets up logging configuration with both console and file handlers.

    Args:
        log_path (str, optional): Path to log file. If None, only console logging.
    """
    # Create logger
    logger = logging.getLogger("AdaptiQ CLI")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_path is provided)
    if log_path:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if it's a text log file
            if file == log_filename:
                print(f"Found text log file: {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"Cleared content from: {file_path}")

            # Check if it's a JSON log file
            elif file == json_filename:
                print(f"Found JSON log file: {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                print(f"Cleared content from: {file_path}")


def execute_pre_run_only(args, logger):
    """Execute only the pre_run pipeline (runs once) and report time and memory usage."""

    # Initialize the aggregator
    aggregator = AdaptiqAggregator(config_path=args.config)
    tracer = AdaptiqTraceLogger.setup()

    try:
        logger.info("STEP : Executing pre_run pipeline...")

        # Start tracking time and memory
        start_time = time.time()
        tracemalloc.start()

        # Execute pre_run pipeline
        simulation_results, pre_run_prompt = adaptiq_pre_run_pipeline(
            config_path=args.config, output_path=args.output_path
        )

        # Stop tracking
        end_time = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        token_stats = get_token_stats(mode="pre_run")

        input_tokens = token_stats.get("total_input_tokens", 0)
        output_tokens = token_stats.get("total_output_tokens", 0)
        total_calls = token_stats.get("total_calls", 0)

        execution_time = end_time - start_time
        peak_memory = peak / 1024 / 1024  # Convert to MB

        logger.info("[SUCCESS] Pre-run pipeline completed successfully")
        logger.info(
            f"tokens input: {input_tokens}, output: {output_tokens}, total calls: {total_calls}"
        )
        logger.info(
            f"Execution time: {execution_time:.2f} seconds, Peak memory usage: {peak_memory:.2f} MB"
        )

        simulated_scenarios = simulation_results.get("simulated_scenarios", [])

        # Update aggregator with run data
        aggregator.increment_run_count()

        # Update token statistics (assuming pre_run only, so post and recon are 0)
        aggregator.update_avg_run_tokens(
            pre_input=input_tokens,
            pre_output=output_tokens,
            post_input=0,
            post_output=0,
            recon_input=0,
            recon_output=0,
        )

        # Update run time
        aggregator.update_avg_run_time(execution_time)

        # Update error count (0 for successful run)
        aggregator.update_error_count(0)

        # Calculate average reward from simulated scenarios
        avg_reward = aggregator.calculate_avg_reward(
            simulated_scenarios=simulated_scenarios, reward_type="simulation"
        )

        # Store last run data for performance score calculation
        aggregator._last_reward = avg_reward
        aggregator._last_run_time = execution_time
        aggregator._last_original_prompt = (
            ""  # You might want to extract this from config
        )
        aggregator._last_suggested_prompt = pre_run_prompt

        # Build and add run summary
        aggregator.add_run_summary(
            run_name="default_run",
            reward=avg_reward,
            api_calls=total_calls,
            suggested_prompt=pre_run_prompt,
            status="completed",
            issues=[],
            error=None,
            memory_usage=peak_memory,
            run_time_seconds=execution_time,
            execution_logs=tracer.get_logs(),
        )

        # Build project result JSON
        project_result = aggregator.build_project_result()

        # Save default_run report
        aggregator.save_json_report(data=project_result)

        if aggregator.email != "":
            # Send results to endpoint
            success = aggregator.send_run_results(project_result)

            if success:
                logger.info("Successfully sent run results to reporting endpoint")
            else:
                logger.warning("Failed to send run results to reporting endpoint")
        else:
            logger.info("Default run results are saved locally")

        return True

    except Exception as e:
        logger.error(f"Error during pre-run pipeline execution: {str(e)}")
        logger.error("Pre-run pipeline execution stopped due to error.")

        # Update aggregator with error information
        aggregator.increment_run_count()
        aggregator.update_error_count(1)

        # Get execution time even for failed runs
        try:
            execution_time = end_time - start_time if "end_time" in locals() else 0
            peak_memory = peak / 1024 / 1024 if "peak" in locals() else 0
        except Exception:
            execution_time = 0
            peak_memory = 0

        aggregator.update_avg_run_time(execution_time)

        # Add failed run summary
        aggregator.add_run_summary(
            run_name="default_run",
            reward=0.0,
            api_calls=0,
            suggested_prompt="",
            status="failed",
            issues=["Pipeline execution failed"],
            error=str(e),
            memory_usage=peak_memory,
            run_time_seconds=execution_time,
            execution_logs=tracer.get_logs(),
        )

        # Build and send project result even for failed runs
        project_result = aggregator.build_project_result()
        aggregator.send_run_results(project_result)

        return False


def execute_post_run_and_reconciliation(
    args, logger, run_number=None, crew_metrics=None, should_send_report=True
):
    """Execute post_run and reconciliation pipelines only."""
    run_prefix = f"[RUN {run_number}] " if run_number is not None else ""

    # Initialize the aggregator
    aggregator = AdaptiqAggregator(config_path=args.config)
    aggregator._default_run_mode = False
    tracer = AdaptiqTraceLogger.setup()

    # Process ALL crew metrics entries, not just the last one
    total_execution_time = 0
    total_peak_memory = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_successful_requests = 0

    if crew_metrics:
        logger.info(
            f"{run_prefix}Processing {len(crew_metrics)} crew metrics entries..."
        )

        for i, metrics in enumerate(crew_metrics):
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

            logger.info(
                f"{run_prefix}Metrics {i+1}/{len(crew_metrics)}: "
                f"time={execution_time_seconds:.2f}s, "
                f"tokens={prompt_tokens + completion_tokens}, "
                f"requests={successful_requests}"
            )

    try:
        # Step 1: Execute post_run pipeline
        logger.info(f"{run_prefix}STEP 1: Executing post_run pipeline...")

        post_run_results = adaptiq_post_run_pipeline(
            config_path=args.config, output_path=args.output_path
        )

        logger.info(f"{run_prefix}[SUCCESS] Post-run pipeline completed successfully")

        # Step 2: Execute reconciliation pipeline
        logger.info(f"{run_prefix}STEP 2: Executing reconciliation pipeline...")

        reconciliation_results = adaptiq_reconciliation_pipeline(
            config_path=args.config,
            output_path=args.output_path,
            feedback=args.feedback,
        )

        logger.info(
            f"{run_prefix}[SUCCESS] Reconciliation pipeline completed successfully"
        )
        logger.info(
            f"{run_prefix}Agent token stats: input={total_prompt_tokens}, output={total_completion_tokens}, calls={total_successful_requests}"
        )
        logger.info(
            f"{run_prefix}Total execution time: {total_execution_time:.2f} seconds, Peak memory usage: {total_peak_memory:.2f} MB"
        )

        validated_logs = post_run_results.get("outputs", {}).get(
            "validation_summary_path", []
        )

        # Process each crew metrics entry and add to aggregator
        if crew_metrics:
            for i, metrics in enumerate(crew_metrics):
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
                    validation_summary_path=validated_logs, reward_type="execution"
                )

                new_prompt = reconciliation_results.get("summary", {}).get(
                    "new_prompt", ""
                )

                # Store last run data for performance score calculation
                aggregator._last_reward = avg_reward
                aggregator._last_run_time = execution_time_seconds
                aggregator._last_original_prompt = (
                    ""  # You might want to extract this from config
                )
                aggregator._last_suggested_prompt = new_prompt

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

                logger.info(
                    f"{run_prefix}Added run summary for execution {execution_count}"
                )
        else:
            # Handle case where no crew metrics provided - add single summary
            logger.info(
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
                validation_summary_path=validated_logs, reward_type="execution"
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
            logger.info(
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
                    logger.info(
                        f"{run_prefix}Successfully sent comprehensive run results to reporting endpoint"
                    )
                else:
                    logger.warning(
                        f"{run_prefix}Failed to send run results to reporting endpoint"
                    )
            else:
                logger.info(f"{run_prefix} are successfully saved locally")
        else:
            logger.info(
                f"{run_prefix}Run summaries added to aggregator - report will be sent when all runs complete"
            )

        find_and_clear_log_files()

        return True

    except Exception as e:
        logger.error(f"{run_prefix}Error during pipeline execution: {str(e)}")
        logger.error(f"{run_prefix}Pipeline execution stopped due to error.")

        # Handle errors for all runs if crew_metrics provided
        if crew_metrics:
            for i, metrics in enumerate(crew_metrics):
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
            logger.info(
                f"{run_prefix}Building and sending project results for failed run..."
            )
            project_result = aggregator.build_project_result()
            aggregator.send_run_results(project_result)
        else:
            logger.info(
                f"{run_prefix}Failed run summary added to aggregator - report will be sent when all runs complete"
            )

        return False


def handle_default_run_command(args):
    """Handles the logic for the 'default-run' command - executes only pre_run pipeline."""
    logger = setup_logging(args.log)

    logger.info("Executing the 'default-run' command...")
    logger.info(f"Configuration file: {args.config}")

    if args.output_path:
        logger.info(f"Results will be saved to: {args.output_path}")

    if args.log:
        logger.info(f"Logging to file: {args.log}")

    logger.info("DEFAULT-RUN MODE: Executing pre-run pipeline only")

    # Execute pre-run pipeline (runs only once)
    success = execute_pre_run_only(args, logger)

    if success:
        logger.info("=" * 60)
        logger.info("[SUCCESS] ADAPTIQ DEFAULT-RUN COMPLETED SUCCESSFULLY")
        logger.info("Pre-run pipeline executed successfully!")
        logger.info("=" * 60)
    else:
        logger.error("DEFAULT-RUN FAILED!")
        sys.exit(1)


def handle_run_command(args):
    """Handles the logic for the 'run' command - executes post_run and reconciliation sequentially."""
    logger = setup_logging(args.log)

    logger.info("Executing the 'run' command...")
    logger.info(f"Configuration file: {args.config}")

    if args.output_path:
        logger.info(f"Results will be saved to: {args.output_path}")

    if args.log:
        logger.info(f"Logging to file: {args.log}")

    # Handle crew metrics if provided
    crew_metrics_list = []
    if hasattr(args, "crew_metrics") and args.crew_metrics:
        logger.info("Crew metrics provided:")
        try:
            crew_metrics_list = json.loads(args.crew_metrics)
            print(f"[CREW_METRICS] {crew_metrics_list}")

            # Extract current execution count from crew metrics
            if crew_metrics_list and len(crew_metrics_list) > 0:
                current_execution_count = crew_metrics_list[-1].get(
                    "execution_count", 0
                )
                logger.info(
                    f"Current execution count from crew metrics: {current_execution_count}"
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format for crew metrics: {e}")
            print(f"[CREW_METRICS_RAW] {args.crew_metrics}")

    # Get send_report flag (defaults to True if not provided)
    should_send_report = True
    if hasattr(args, "send_report"):
        should_send_report = args.send_report
        logger.info(f"Send report flag: {should_send_report}")

    # Execute single run (post_run and reconciliation only)
    logger.info("=" * 60)
    logger.info("[START] STARTING ADAPTIQ RUN")
    logger.info("=" * 60)

    success = execute_post_run_and_reconciliation(
        args,
        logger,
        run_number=None,
        crew_metrics=crew_metrics_list,
        should_send_report=should_send_report,
    )

    if success:
        logger.info("[SUCCESS] ADAPTIQ RUN COMPLETED SUCCESSFULLY")
        logger.info("All pipelines (post_run -> reconciliation) executed successfully!")
        if should_send_report:
            logger.info("[REPORT] Report sent successfully!")
        else:
            logger.info("[REPORT] Report sending skipped (send_report=False)")
    else:
        logger.error("ADAPTIQ RUN FAILED!")
        logger.error("Execution failed during pipeline execution.")
        sys.exit(1)

    logger.info("=" * 60)


def handle_wizard_command(args):
    """Handles the logic for the 'wizard' subcommand."""
    logger = setup_logging(args.log if hasattr(args, "log") else None)

    logger.info(f"Starting Adaptiq Wizard with LLM provider: {args.llm_provider}")

    try:
        # Call the wizard function with the provided LLM provider and API key
        adaptiq_run_wizard(llm_provider=args.llm_provider, api_key=args.api_key)

        logger.info("Wizard session completed successfully")
    except Exception as e:
        logger.error(f"Error executing wizard: {str(e)}")
        sys.exit(1)


def handle_wizard_headless_command(args):
    """Handles the logic for the 'wizard-headless' subcommand."""
    logger = setup_logging(args.log if hasattr(args, "log") else None)

    logger.info(
        f"Starting Adaptiq Wizard in headless mode with LLM provider: {args.llm_provider}"
    )

    try:
        # Call the headless wizard function with the provided arguments
        adaptiq_run_wizard_headless(
            llm_provider=args.llm_provider,
            api_key=args.api_key,
            prompt=args.prompt,
            output_format=args.output_format,
        )

        logger.info("Wizard headless session completed successfully")
    except Exception as e:
        logger.error(f"Error executing wizard in headless mode: {str(e)}")
        sys.exit(1)


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

    # --- Define the 'default-run' command ---
    parser_default_run = subparsers.add_parser(
        "default-run", help="Run only the pre_run pipeline (executes once)."
    )
    # Add arguments specific to the 'default-run' command
    parser_default_run.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the configuration file for the pre_run pipeline.",
    )
    parser_default_run.add_argument(
        "--output_path",
        type=str,
        metavar="PATH",
        help="Path to save the results of the pre_run pipeline.",
    )
    parser_default_run.add_argument(
        "--log",
        type=str,  # Expecting a string (path)
        metavar="PATH",  # Placeholder name shown in help message
        help="Optional path to a file for logging output.",
    )
    # Set the function to call when 'default-run' is chosen
    parser_default_run.set_defaults(func=handle_default_run_command)

    # --- Define the 'run' command ---
    parser_run = subparsers.add_parser(
        "run",
        help="Run the post_run and reconciliation pipelines (single execution with optional report sending).",
    )
    # Add arguments specific to the 'run' command
    parser_run.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the configuration file for post_run and reconciliation pipelines.",
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
        "--crew_metrics",
        type=str,
        metavar="JSON_STRING",
        help="Optional crew metrics data in JSON format to be processed during the run.",
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

    # --- Define the 'wizard' command ---
    parser_wizard = subparsers.add_parser(
        "wizard", help="Start the interactive Adaptiq wizard assistant."
    )
    # Add arguments specific to the 'wizard' command
    parser_wizard.add_argument(
        "--llm_provider",
        type=str,
        metavar="PROVIDER",
        required=True,
        help="LLM provider to use for the wizard assistant (e.g., 'openai', 'groq').",
    )
    parser_wizard.add_argument(
        "--api_key",
        type=str,
        metavar="API_KEY",
        required=True,
        help="API key for the wizard assistant.",
    )
    parser_wizard.add_argument(
        "--log",
        type=str,
        metavar="PATH",
        help="Optional path to a file for logging output.",
    )
    # Set the function to call when 'wizard' is chosen
    parser_wizard.set_defaults(func=handle_wizard_command)

    # --- Define the 'wizard-headless' command ---
    parser_wizard_headless = subparsers.add_parser(
        "wizard-headless",
        help="Run the Adaptiq wizard assistant in non-interactive headless mode.",
    )
    # Add arguments specific to the 'wizard-headless' command
    parser_wizard_headless.add_argument(
        "--llm_provider",
        type=str,
        metavar="PROVIDER",
        required=True,
        help="LLM provider to use for the wizard assistant (e.g., 'openai', 'groq').",
    )
    parser_wizard_headless.add_argument(
        "--api_key",
        type=str,
        metavar="API_KEY",
        required=True,
        help="API key for the wizard assistant.",
    )
    parser_wizard_headless.add_argument(
        "--prompt",
        type=str,
        metavar="PROMPT",
        required=True,
        help="The prompt/question to process in headless mode.",
    )
    parser_wizard_headless.add_argument(
        "--output_format",
        type=str,
        metavar="FORMAT",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format: 'text', 'json', or 'stream-json' (default: text).",
    )
    parser_wizard_headless.add_argument(
        "--log",
        type=str,
        metavar="PATH",
        help="Optional path to a file for logging output.",
    )
    # Set the function to call when 'wizard-headless' is chosen
    parser_wizard_headless.set_defaults(func=handle_wizard_headless_command)

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
