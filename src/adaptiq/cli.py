import argparse
import logging
import sys
import logging

from adaptiq.core.reporting import get_logger
from adaptiq.agents.crew_ai import CrewConfig

get_logger()


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