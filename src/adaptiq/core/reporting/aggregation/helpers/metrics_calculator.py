import json
import logging
import tiktoken
from typing import Dict, List, Optional


class MetricsCalculator:
    """
    Handles all metrics calculations including token counting, cost computations,
    performance scoring, and statistical aggregations.
    """
    # TODO: Unify the config to generic data model (from base config)
    def __init__(self, config: Dict, pricings: Dict):
        """
        Initialize the metrics calculator.

        Args:
            config (Dict): Configuration dictionary containing LLM and pricing info
            pricings (Dict): Pricing information for different models and providers
        """
        self.logger = logging.getLogger("ADAPTIQ-Aggregator-MetricsCalculator")
        self.config = config
        self.pricings = pricings
        
        # Initialize tracking variables
        self._run_count = 0
        self._reward_sum = 0.0
        self._total_run_time = 0.0
        self._total_errors = 0
        
        # Token tracking
        self.avg_input_tokens = 0.0
        self.avg_output_tokens = 0.0
        self.avg_input = 0.0
        self.avg_output = 0.0
        self.overall_avg = 0.0
        
        # Pricing cache
        self.input_price = 0.0
        self.output_price = 0.0
        
        # Token tracking per run type
        self.run_tokens = {
            "pre_tokens": {"input": 0.0, "output": 0.0},
            "post_tokens": {"input": 0.0, "output": 0.0},
            "recon_tokens": {"input": 0.0, "output": 0.0},
        }
        
        # Runtime state
        self._default_run_mode = True
        self._last_reward = 0.0
        self._last_run_time = 0.0
        self._last_original_prompt = ""
        self._last_suggested_prompt = ""

    def set_run_count(self, count: int):
        """Set the current run count."""
        self._run_count = count

    def set_last_run_data(self, reward: float, run_time: float = 0.0, 
                         original_prompt: str = "", suggested_prompt: str = ""):
        """Store data from the last run for performance calculations."""
        self._last_reward = reward
        self._last_run_time = run_time
        self._last_original_prompt = original_prompt
        self._last_suggested_prompt = suggested_prompt

    def calculate_avg_reward(
        self,
        validation_summary_path: str = None,
        simulated_scenarios: List = None,
        reward_type: str = "execution",
    ) -> float:
        """
        Calculate and update the running average reward across all runs.

        Args:
            validation_summary_path (str, optional): Path to the validation_summary.json file (for execution rewards).
            simulated_scenarios (list, optional): List of simulated scenario dicts (for simulation rewards).
            reward_type (str): "execution" or "simulation" to select which reward to calculate.

        Returns:
            float: The running average reward value, or 0.0 if none found.
        """
        try:
            if reward_type == "simulation" and simulated_scenarios is not None:
                # Calculate average of reward_sim from simulated scenarios
                rewards = [
                    scenario.get("reward_sim")
                    for scenario in simulated_scenarios
                    if "reward_sim" in scenario
                ]
                if not rewards or self._run_count == 0:
                    return 0.0
                avg_this_run = sum(rewards) / len(rewards)
                self._reward_sum += avg_this_run
                return self._reward_sum / self._run_count

            elif reward_type == "execution" and validation_summary_path is not None:
                with open(validation_summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rewards = [
                    entry["corrected_entry"]["reward_exec"]
                    for entry in data.get("validations", [])
                    if "corrected_entry" in entry
                    and "reward_exec" in entry["corrected_entry"]
                ]
                if not rewards or self._run_count == 0:
                    return 0.0
                avg_this_run = sum(rewards) / len(rewards)
                self._reward_sum += avg_this_run
                return self._reward_sum / self._run_count

            else:
                self.logger.error(
                    "Invalid arguments for calculate_avg_reward: must provide either validation_summary_path or simulated_scenarios."
                )
                return (
                    self._reward_sum / self._run_count if self._run_count > 0 else 0.0
                )

        except (OSError, json.JSONDecodeError, TypeError) as e:
            self.logger.error("Failed to calculate average reward: %s", e)
            return self._reward_sum / self._run_count if self._run_count > 0 else 0.0

    def update_avg_run_tokens(
        self,
        pre_input: int,
        pre_output: int,
        post_input: int,
        post_output: int,
        recon_input: int,
        recon_output: int,
        default_run_mode: bool = True,
    ):
        """
        Update the running sum for input/output tokens for each token type.

        Args:
            pre_input (int): Input tokens for pre_tokens.
            pre_output (int): Output tokens for pre_tokens.
            post_input (int): Input tokens for post_tokens.
            post_output (int): Output tokens for post_tokens.
            recon_input (int): Input tokens for recon_tokens.
            recon_output (int): Output tokens for recon_tokens.
            default_run_mode (bool): Whether this is a default mode (True) or not (False).
        """
        if self._run_count == 0:
            return  # Avoid division by zero

        # Update cumulative sums
        self.run_tokens["pre_tokens"]["input"] += pre_input
        self.run_tokens["pre_tokens"]["output"] += pre_output
        self.run_tokens["post_tokens"]["input"] += post_input
        self.run_tokens["post_tokens"]["output"] += post_output
        self.run_tokens["recon_tokens"]["input"] += recon_input
        self.run_tokens["recon_tokens"]["output"] += recon_output

        # Calculate this run's average input and output tokens
        if default_run_mode:
            avg_input_this_run = pre_input
            avg_output_this_run = pre_output
        else:
            avg_input_this_run = (
                pre_input + post_input + recon_input
            ) / self._run_count
            avg_output_this_run = (
                pre_output + post_output + recon_output
            ) / self._run_count

        # Add to running sums for averages
        self.avg_input_tokens += avg_input_this_run
        self.avg_output_tokens += avg_output_this_run

        # Store run mode
        self._default_run_mode = default_run_mode

    def get_avg_run_tokens(self) -> tuple:
        """
        Get the overall average tokens: for each token type, average input/output, then average all three.
        Also return the average input tokens and average output tokens per run.

        Returns:
            tuple: (overall_avg, avg_input_tokens, avg_output_tokens)
        """
        if self._run_count == 0:
            return 0.0, 0.0, 0.0

        avg_pre = (
            self.run_tokens["pre_tokens"]["input"]
            + self.run_tokens["pre_tokens"]["output"]
        ) / self._run_count
        avg_post = (
            self.run_tokens["post_tokens"]["input"]
            + self.run_tokens["post_tokens"]["output"]
        ) / self._run_count
        avg_recon = (
            self.run_tokens["recon_tokens"]["input"]
            + self.run_tokens["recon_tokens"]["output"]
        ) / self._run_count

        if self._default_run_mode:
            self.overall_avg = avg_pre
        else:
            self.overall_avg = (avg_pre + avg_post + avg_recon) / self._run_count

        self.avg_input = self.avg_input_tokens / self._run_count
        self.avg_output = self.avg_output_tokens / self._run_count

        return self.overall_avg, self.avg_input, self.avg_output

    def calculate_avg_cost(self) -> float:
        """
        Calculate the average cost based on avg_input and avg_output tokens,
        using the pricing info from self.pricings and model/provider from self.config.

        Returns:
            float: The average cost for the current averages.
        """
        provider = self.config.get("llm_config", {}).get("provider")
        model = self.config.get("llm_config", {}).get("model_name")
        if not provider or not model:
            self.logger.error("Provider or model not found in config.")
            return 0.0

        pricing = self.pricings.get(provider, {}).get(model)
        if not pricing:
            self.logger.error(
                "Pricing not found for provider '%s' and model '%s'.", provider, model
            )
            return 0.0

        input_price = pricing.get("input", 0.0)
        output_price = pricing.get("output", 0.0)

        avg_cost = ((self.avg_input / 1000) * input_price) + (
            (self.avg_output / 1000) * output_price
        )
        return avg_cost

    def calculate_current_run_cost(
        self, total_input_tokens: int, total_output_tokens: int
    ) -> float:
        """
        Calculate the cost for the current run based on total input and output tokens.

        Args:
            total_input_tokens (int): Total input tokens for the run.
            total_output_tokens (int): Total output tokens for the run.

        Returns:
            float: The cost for the current run.
        """
        provider = self.config.get("llm_config", {}).get("provider")
        model = self.config.get("llm_config", {}).get("model_name")
        if not provider or not model:
            self.logger.error("Provider or model not found in config.")
            return 0.0

        pricing = self.pricings.get(provider, {}).get(model)
        if not pricing:
            self.logger.error(
                "Pricing not found for provider '%s' and model '%s'.", provider, model
            )
            return 0.0

        input_price = pricing.get("input", 0.0)
        self.input_price = input_price
        output_price = pricing.get("output", 0.0)
        self.output_price = output_price

        cost = ((total_input_tokens / 1000) * input_price) + (
            (total_output_tokens / 1000) * output_price
        )
        return cost

    def update_avg_run_time(self, run_time_seconds: float):
        """
        Update the running average of execution time per run.

        Args:
            run_time_seconds (float): The execution time for this run in seconds.
        """
        self._total_run_time += run_time_seconds

    def get_avg_run_time(self) -> float:
        """
        Get the average execution time per run in seconds.

        Returns:
            float: The average run time in seconds.
        """
        if self._run_count == 0:
            return 0.0
        return self._total_run_time / self._run_count

    def update_error_count(self, errors_this_run: int):
        """
        Update the running sum of errors across runs.
        
        Args:
            errors_this_run (int): Number of errors in this run.
        """
        self._total_errors += errors_this_run

    def get_avg_errors(self) -> float:
        """
        Get the average number of errors per run.
        
        Returns:
            float: Average errors per run.
        """
        if self._run_count == 0:
            return 0.0
        return self._total_errors / self._run_count

    def calculate_performance_score(self) -> float:
        """
        Calculate the performance score for the current run using internal state.

        Returns:
            float: The calculated performance score.
        """
        reward = self._last_reward
        exec_time = self._last_run_time
        avg_errors = self.get_avg_errors()
        error_rate = (
            round((avg_errors / self._run_count) * 100, 1) if self._run_count else 0.0
        )

        # Calculate detail_added from last run's prompts if available
        original_prompt = self._last_original_prompt
        suggested_prompt = self._last_suggested_prompt
        orig_len = len(original_prompt) if original_prompt else 0
        sugg_len = len(suggested_prompt) if suggested_prompt else 0
        detail_added = ((sugg_len - orig_len) / orig_len) * 100 if orig_len > 0 else 0.0

        # Normalize metrics (example: reward out of 1, detail_added out of 100, error_rate out of 100, exec_time out of 60s)
        reward_norm = reward  # assuming reward is already 0-1 or 0-100
        detail_norm = min(max(detail_added / 100, 0), 1)
        error_norm = 1 - min(max(error_rate / 100, 0), 1)
        exec_time_norm = 1 - min(exec_time / 60, 1)  # 1 is best, 0 is worst if >60s

        # Weighted sum (adjust weights as needed)
        performance_score = round(
            0.5 * reward_norm
            + 0.2 * detail_norm
            + 0.2 * exec_time_norm
            + 0.1 * error_norm,
            3,
        )
        return performance_score

    def estimate_prompt_tokens(
        self, original_prompt: str, suggested_prompt: str, model_name: str = "gpt-4"
    ) -> tuple:
        """
        Estimate token counts for original and suggested prompts.

        Args:
            original_prompt (str): The original prompt text
            suggested_prompt (str): The suggested/optimized prompt text
            model_name (str): The model name for token encoding (default: "gpt-4")

        Returns:
            tuple: (original_tokens, suggested_tokens)
        """
        try:
            # Get the appropriate encoding for the model
            encoding = tiktoken.encoding_for_model(model_name)

            # Encode and count tokens for original prompt
            original_tokens = 0
            if original_prompt:
                original_encoded = encoding.encode(original_prompt)
                original_tokens = len(original_encoded)

            # Encode and count tokens for suggested prompt
            suggested_tokens = 0
            if suggested_prompt:
                suggested_encoded = encoding.encode(suggested_prompt)
                suggested_tokens = len(suggested_encoded)

            return original_tokens, suggested_tokens

        except Exception as e:
            self.logger.error(f"Error estimating tokens: {e}")
            # Fallback to rough estimation (4 chars per token)
            original_tokens = len(original_prompt) // 4 if original_prompt else 0
            suggested_tokens = len(suggested_prompt) // 4 if suggested_prompt else 0

            return original_tokens, suggested_tokens

    def get_reward_sum(self) -> float:
        """Get the current reward sum."""
        return self._reward_sum

    def get_total_run_time(self) -> float:
        """Get the total run time across all runs."""
        return self._total_run_time

    def get_total_errors(self) -> int:
        """Get the total error count across all runs."""
        return self._total_errors

    def reset_tracking(self):
        """Reset all tracking variables."""
        self._reward_sum = 0.0
        self._total_run_time = 0.0
        self._total_errors = 0
        self.avg_input_tokens = 0.0
        self.avg_output_tokens = 0.0
        self.avg_input = 0.0
        self.avg_output = 0.0
        self.overall_avg = 0.0
        
        # Reset token tracking
        self.run_tokens = {
            "pre_tokens": {"input": 0.0, "output": 0.0},
            "post_tokens": {"input": 0.0, "output": 0.0},
            "recon_tokens": {"input": 0.0, "output": 0.0},
        }