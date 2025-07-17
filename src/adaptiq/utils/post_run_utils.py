import os
import json
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple

from langchain_openai import ChatOpenAI

# Set up logger
logger = logging.getLogger(__name__)


class AdaptiqAgentTracer:
    """
    AdaptiqAgentTracer is a utility class for managing and accessing agent execution traces.
    It reads configuration from a YAML file to determine how to access the agent's execution logs.
    The class supports both development and production modes, allowing for flexible log access based on the execution context.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the AdaptiqAgentRunner with the path to the configuration file.
        
        Args:
            config_path (str): Path to the configuration file in YAML format.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file cannot be parsed.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the configuration file from the given path.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            Dict[str, Any]: The loaded configuration.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed as YAML.
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")
    
    def get_agent_trace(self) -> str:
        """
        Access agent trace based on execution mode configuration.
        
        Returns:
            str: The execution trace as text.
        """
        framework_settings = self.config.get("framework_adapter", {}).get("settings", {})
        execution_mode = framework_settings.get("mode_execution", "dev")  # Default to 'dev'

        log_source_config = framework_settings.get("log_source", {})
        log_source_type = log_source_config.get("type", "stdout_capture")  # Default to stdout
        log_file_path = log_source_config.get("path")

        trace_output = ""

        # Log the execution mode
        if execution_mode == "prod":
            logger.info("Running in PROD mode - accessing log file directly")
        else:
            logger.info("Running in DEV mode - accessing log file directly")
        
        # Read the log file regardless of mode
        if log_source_type == "file_path":
            if not log_file_path:
                logger.error("Log source type is 'file_path' but no path is specified in config.")
                return ""
            else:
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        trace_output = f.read()
                    logger.info(f"Successfully read trace from log file: {log_file_path}")
                except FileNotFoundError:
                    logger.error(f"Log file not found: {log_file_path}")
                    return ""
                except Exception as e:
                    logger.error(f"Error reading log file {log_file_path}: {str(e)}")
                    return ""
        else:
            logger.warning(f"Log source type '{log_source_type}' is not 'file_path'. Cannot access logs without execution.")
            return ""
            
        return trace_output
    
    def get_agent_config(self) -> Dict[str, Any]:
        """
        Get the current agent configuration.
        
        Returns:
            Dict[str, Any]: The agent configuration.
        """
        return self.config
        
    def update_log_source(self, log_type: str, log_path: Optional[str] = None) -> None:
        """
        Update the log source configuration.
        
        Args:
            log_type (str): Type of log source, either 'file_path' or 'stdout_capture'.
            log_path (Optional[str]): Path to the log file (required if log_type is 'file_path').
        """
        if log_type not in ["file_path", "stdout_capture"]:
            logger.warning(f"Unsupported log source type: {log_type}. Using 'stdout_capture' instead.")
            log_type = "stdout_capture"
        
        if log_type == "file_path" and not log_path:
            logger.warning("Log source type set to 'file_path' but no path provided. Using 'stdout_capture' instead.")
            log_type = "stdout_capture"
        
        if "framework_adapter" not in self.config:
            self.config["framework_adapter"] = {}
        
        if "settings" not in self.config["framework_adapter"]:
            self.config["framework_adapter"]["settings"] = {}
        
        log_source_config = {"type": log_type}
        if log_type == "file_path" and log_path:
            log_source_config["path"] = log_path
        
        self.config["framework_adapter"]["settings"]["log_source"] = log_source_config
        logger.info(f"Updated log source configuration to: {log_source_config}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            output_path (Optional[str]): Path to save the configuration to. 
                                        If None, the original config path will be used.
        """
        save_path = output_path if output_path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")


class AdaptiqPostRunValidator:
    """
    A class to validate and refine parsed agent logs using an LLM.
    
    This validator checks parsed logs for accuracy, focuses on validating reward values
    and adjusts them if necessary based on the context of agent actions and outcomes.
    """
    
    def __init__(self, raw_logs: List[Dict[str, Any]], parsed_logs: List[Dict[str, Any]], model_name: str, api_key: str, provider: str):
        """
        Initialize the validator with raw and parsed logs.
        
        Args:
            raw_logs: The original agent logs with timestamps, actions, thoughts, etc.
            parsed_logs: The processed logs with state and reward information
            model_name: The name of the LLM model to use
            api_key: OpenAI API key
        """
        self.raw_logs = raw_logs
        self.parsed_logs = parsed_logs
        self.model = model_name
        self.api_key = api_key
        self.provider = provider

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Only 'openai' is currently supported.")
        
    def _create_validation_prompt(self, raw_log_entry: Dict[str, Any], parsed_log_entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM to validate a specific log entry, focusing on reward values.
        
        Args:
            raw_log_entry: Original log entry
            parsed_log_entry: Parsed log entry with state and reward
            
        Returns:
            A list of messages for the LLM to validate the parsing
        """
        system_message = """You are an expert AI agent log validator and Q-learning specialist. Your primary task is to verify that the 
            reward value assigned in the parsed log entry is appropriate given the raw log entry.
            
            Rewards should be between -1.0 and 1.0, with:
            - Positive rewards (0.0 to 1.0) for successful actions, helpful thoughts, and good outcomes
            - Negative rewards (-1.0 to 0.0) for failed actions, irrelevant thoughts, or poor outcomes
            
            IMPORTANT REWARD GUIDELINES FOR Q-LEARNING:
            - For successful actions that directly advance the task, assign rewards in the 0.5 to 0.9 range
            - For partially successful actions or minor progress, assign rewards in the 0.3 to 0.5 range
            - For neutral or minimal progress actions, assign rewards in the 0.1 to 0.3 range
            - For minor errors or slightly unhelpful actions, assign rewards in the -0.3 to 0 range
            - For significant errors or counterproductive actions, assign rewards in the -0.7 to -0.3 range
            - For critical failures that severely impact task completion, assign rewards in the -1.0 to -0.7 range
            
            Additional considerations:
            - Successful information retrieval that directly contributes to the task should get at least 0.5 reward
            - Small positive rewards (< 0.3) are too weak for effective Q-learning when an action is successful
            - Actions that result in errors or failures should have appropriately negative rewards
            - The final action that completes the primary task should receive a high reward (0.8-1.0)
            
            Please analyze the raw log entry and its corresponding parsed version, then provide your assessment 
            in the following JSON format:
            
            ```json
            {
                "reward_assessment": {
                    "original": 0.0,
                    "is_appropriate": true/false,
                    "adjusted": 0.0,
                    "reason": "reason for adjustment or confirmation"
                },
                "corrected_entry": {} // The corrected parsed log entry with adjusted reward if needed
            }
            ```
            
            If the reward is appropriate, set "is_appropriate" to true and keep the original reward value as the "adjusted" value.
            But be strict about following the reward guidelines - small positive rewards for successful actions are usually inappropriate in Q-learning.
            """
            
        human_message = f"""
            Raw Log Entry:
            ```json
            {json.dumps(raw_log_entry, indent=2)}
            ```
            
            Parsed Log Entry:
            ```json
            {json.dumps(parsed_log_entry, indent=2)}
            ```
            
            Please validate the reward value in this parsing and provide your assessment:
            """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message}
        ]
    
    def validate_single_entry(self, raw_log_idx: int, parsed_log_idx: int) -> Dict[str, Any]:
        """
        Validate a single log entry pair, focusing on reward validation.
        
        Args:
            raw_log_idx: Index of the raw log entry
            parsed_log_idx: Index of the parsed log entry
            
        Returns:
            Validation results with any reward corrections
        """
        if raw_log_idx >= len(self.raw_logs) or parsed_log_idx >= len(self.parsed_logs):
            raise IndexError("Log index out of range")
        
        raw_entry = self.raw_logs[raw_log_idx]
        parsed_entry = self.parsed_logs[parsed_log_idx]
        
        messages = self._create_validation_prompt(raw_entry, parsed_entry)
        response = self.llm.invoke(messages)
        
        try:
            parsed_response = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            # Try to extract JSON from the response if it's not already in JSON format
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content if hasattr(response, 'content') else response, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    # If extraction fails, return a basic structure to avoid breaking the pipeline
                    parsed_response = {
                        "reward_assessment": {
                            "original": parsed_entry.get("reward", 0.0),
                            "is_appropriate": True,
                            "adjusted": parsed_entry.get("reward", 0.0),
                            "reason": "Failed to parse LLM response, keeping original reward"
                        },
                        "corrected_entry": parsed_entry
                    }
            else:
                # If no JSON found, return a basic structure
                parsed_response = {
                    "reward_assessment": {
                        "original": parsed_entry.get("reward", 0.0),
                        "is_appropriate": True,
                        "adjusted": parsed_entry.get("reward", 0.0),
                        "reason": "Failed to parse LLM response, keeping original reward"
                    },
                    "corrected_entry": parsed_entry
                }
                
        return parsed_response
    
    def validate_all_entries(self) -> List[Dict[str, Any]]:
        """
        Validate all log entries and return validation results.
        Only entries with reward_exec in range (-0.25, 0.25) are validated.
        
        Returns:
            List of validation results for all entries
        """
        results = []
        min_length = min(len(self.raw_logs), len(self.parsed_logs))
        
        for i in range(min_length):
            parsed_entry = self.parsed_logs[i]
            reward = parsed_entry.get("reward_exec", parsed_entry.get("reward", 0.0))  # fallback to "reward" if "reward_exec" missing
            if -0.25 < reward < 0.25:
                validation = self.validate_single_entry(i, i)
            else:
                # Skip validation, return as appropriate without changes
                validation = {
                    "reward_assessment": {
                        "original": reward,
                        "is_appropriate": True,
                        "adjusted": reward,
                        "reason": "Skipped validation due to reward outside (-0.25, 0.25) range"
                    },
                    "corrected_entry": self.parsed_logs[i]
                }
            results.append(validation)
    
        return results
    
    def get_corrected_logs(self) -> List[Dict[str, Any]]:
        """
        Get the corrected version of all parsed logs after validation.
        
        Returns:
            List of corrected log entries
        """
        validations = self.validate_all_entries()
        corrected_logs = []
        
        for validation in validations:
            # Make sure to update the reward value in the corrected entry
            corrected_entry = validation["corrected_entry"]
            if validation["reward_assessment"]["is_appropriate"] is False:
                corrected_entry["reward"] = validation["reward_assessment"]["adjusted"]
            
            corrected_logs.append(corrected_entry)
            
        return corrected_logs
    
    def summarize_validations(self, validations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a summary of reward validation results.
        
        Args:
            validations: List of validation results (if None, will run validate_all_entries)
            
        Returns:
            Summary statistics about the reward validations
        """
        if validations is None:
            validations = self.validate_all_entries()
            
        total = len(validations)
        appropriate_rewards = sum(1 for v in validations if v["reward_assessment"]["is_appropriate"])
        reward_adjustments = sum(1 for v in validations 
                               if not v["reward_assessment"]["is_appropriate"])
        
        # Calculate average magnitude of adjustments
        adjustments = []
        for v in validations:
            if not v["reward_assessment"]["is_appropriate"]:
                original = v["reward_assessment"]["original"]
                adjusted = v["reward_assessment"]["adjusted"]
                adjustments.append(abs(adjusted - original))
        
        avg_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0
        
        return {
            "total_entries": total,
            "entries_with_appropriate_rewards": appropriate_rewards,
            "entries_with_reward_adjustments": reward_adjustments,
            "appropriate_reward_rate": appropriate_rewards / total if total > 0 else 0,
            "reward_adjustment_rate": reward_adjustments / total if total > 0 else 0,
            "average_adjustment_magnitude": avg_adjustment
        }
    
    def run_validation_pipeline(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the complete validation pipeline and return both the corrected logs and validation results.
        
        Returns:
            Tuple containing:
            - List of corrected logs with validated rewards
            - Dictionary with validation results and summary
        """
        validations = self.validate_all_entries()
        corrected_logs = self.get_corrected_logs()
        summary = self.summarize_validations(validations)
        
        # Create the validation results output
        validation_results = {
            "validations": validations,
            "summary": summary
        }
        
        return corrected_logs, validation_results
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the validation process and return the corrected logs.
        This is a simplified method for use when only the corrected logs are needed.
        
        Returns:
            List of corrected logs with validated rewards
        """
        corrected_logs = self.get_corrected_logs()
        return corrected_logs

    def extract_corrected_entries(self, validation_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract only the corrected entries from validation results.
        
        This method takes validation results (either provided or by running the validation pipeline)
        and extracts just the corrected log entries, removing all the assessment metadata.
        
        Args:
            validation_results: Dictionary containing validations and summary (if None, will run validation pipeline)
            
        Returns:
            List of corrected log entries without validation metadata
        """
        if validation_results is None:
            # Run the validation pipeline to get results
            _, validation_results = self.run_validation_pipeline()
            
        # Extract just the corrected entries from the validations
        corrected_entries = []
        
        if 'validations' in validation_results:
            for validation in validation_results['validations']:
                if 'corrected_entry' in validation:
                    corrected_entries.append(validation['corrected_entry'])
        
        return corrected_entries