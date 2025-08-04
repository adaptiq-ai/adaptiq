import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from adaptiq.core.abstract.integrations.base_config import BaseConfig
from adaptiq.core.abstract.integrations.base_prompt_parser import BasePromptParser
from adaptiq.core.q_table.q_table_manager import QTableManager
from adaptiq.core.pipelines.pre_run.tools.hypothetical_state_generator import HypotheticalStateGenerator 
from adaptiq.core.pipelines.pre_run.tools.prompt_consulting import PromptConsulting
from adaptiq.core.pipelines.pre_run.tools.scenario_simulator import ScenarioSimulator
from adaptiq.core.pipelines.pre_run.tools.prompt_estimator import PromptEstimator


class PreRunPipeline:
    """
    AdaptiqPreRunOrchestrator coordinates the execution of ADAPTIQ's pre-run module components:
    1. Prompt Parsing - Analyzes agent's task & tools to infer sequence of steps
    2. Hypothetical Representation - Generates hypothetical state-action pairs
    3. Q-table Initialization - Initializes Q-values based on heuristic rules
    4. Prompt Analysis - Analyzes prompt for best practices & improvement opportunities

    This orchestration prepares the agent for execution with optimized configuration.
    """

    def __init__(self, 
    base_config: BaseConfig, 
    base_prompt_parser: BasePromptParser,
    output_path: str,
    ):
        """
        Initialize the PreRunOrchestrator with configuration.

        Args:
            base_config: An instance of BaseConfig (or its subclasses like CrewConfig, OpenAIConfig, etc.)
            base_prompt_parser: An instance of BasePromptParser for prompt parsing functionality
            output_path: Path where output files will be saved
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-PreRun")

        # Store configuration and paths
        self.base_config = base_config
        self.config = base_config.config
        self.config_path = base_config.config_path
        self.output_path = output_path
        self.configuration = self.base_config.get_config()

        # Ensure output directory exists
        self.q_table_path = os.path.join(self._ensure_output_directory(), "adaptiq_q_table.json")

        # Loading the old prompt of agent
        self.old_prompt = self.base_config.get_prompt()

        # Store the prompt parser
        self.prompt_parser = base_prompt_parser

        # Extract key configuration
        self.agent_config = self.configuration.get("agent_modifiable_config", {})

        # Load environment variables for API access
        load_dotenv()
        self.api_key = self.configuration.get("llm_config", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model_name = self.configuration.get("llm_config", {}).get("model_name")
        self.provider = self.configuration.get("llm_config", {}).get("providedr", "openai")

        # Get the list of tools available to the agent
        tools_config = self.configuration.get("agent_modifiable_config", {}).get(
            "agent_tools", []
        )
        self.agent_tools = (
            [{tool["name"]: tool["description"]} for tool in tools_config]
            if tools_config
            else []
        )

        if not self.api_key:
            raise ValueError("API key not provided in config or environment variables")
        if not self.model_name:
            raise ValueError("Model name not provided in configuration")

        # Initialize component instances
        self.state_generator = None
        self.offline_learner = None
        self.prompt_consultant = None
        self.scenario_simulator = None
        self.prompt_estimator = None
        self.offline_learner = QTableManager(file_path=self.q_table_path)

        # Results storage
        self.parsed_steps = []
        self.hypothetical_states = []
        self.prompt_analysis = {}
        self.simulated_scenarios = []


    def _ensure_output_directory(self) -> str:
        """
        Ensure the output directory exists, create it if it doesn't.

        Returns:
            str: The path to the output directory
        """
        if not self.output_path:
            # Use default path in package directory
            self.output_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "results"
            )

        # Create the directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.logger.info("Created output directory: %s", self.output_path)

        return self.output_path

    def run_prompt_parsing(self) -> List[Dict[str, Any]]:
        """
        Execute the prompt parsing step to analyze the agent's task and tools.

        Returns:
            List of dictionaries with parsed steps
        """
        self.logger.info("Starting Prompt Parsing...")

        try:
            # Parse the prompt
            self.parsed_steps = self.prompt_parser.parse_prompt()

            self.logger.info(
                "Prompt Parsing complete. Identified %d steps.",
                len(self.parsed_steps)
            )
            return self.parsed_steps

        except Exception as e:
            self.logger.error("Prompt Parsing failed: %s", str(e))
            raise

    def run_hypothetical_representation(self) -> List[Dict]:
        """
        Generate hypothetical state-action pairs based on parsed steps.

        Returns:
            List of state-action pairs
        """
        self.logger.info("Starting Hypothetical State Generation...")

        if not self.parsed_steps:
            self.logger.warning(
                "No parsed steps available. Running prompt parsing first."
            )
            self.run_prompt_parsing()

        try:
            # Initialize the hypothetical state generator
            self.state_generator = HypotheticalStateGenerator(
                prompt_parsed_plan=self.parsed_steps,
                model_name=self.model_name,
                api_key=self.api_key,
                provider=self.provider,
            )

            # Generate state-action pairs
            raw_states = self.state_generator.generate_hypothetical_state_action_pairs()
            self.hypothetical_states = self.state_generator.clean_representation(
                raw_states
            )

            self.logger.info(
                "Hypothetical State Generation complete. Generated %d state-action pairs.",
                len(self.hypothetical_states)
            )
            return self.hypothetical_states

        except Exception as e:
            self.logger.error("Hypothetical State Generation failed: %s", str(e))
            raise

    def run_simulation(self) -> List[Dict]:
        """
        Run scenario simulation based on the generated hypothetical states.
        Generates multiple plausible scenarios for each state-action pair.

        Returns:
            List of simulated scenarios
        """
        self.logger.info("Starting Scenario Simulation...")

        # Ensure hypothetical states are available
        if not self.hypothetical_states:
            self.logger.warning(
                "No hypothetical states available. Running hypothetical representation first."
            )
            self.run_hypothetical_representation()

        try:
            # Create a filename for the simulation results
            output_dir = self._ensure_output_directory()
            simulation_output_path = os.path.join(
                output_dir, "adaptiq_simulated_scenarios.json"
            )

            # Initialize the scenario simulator
            self.scenario_simulator = ScenarioSimulator(
                hypothetical_states=self.hypothetical_states,
                model_name=self.model_name,
                api_key=self.api_key,
                provider=self.provider,
                output_path=simulation_output_path,
            )

            # Generate simulated scenarios
            self.simulated_scenarios = (
                self.scenario_simulator.generate_simulated_scenarios()
            )

            self.logger.info(
                "Scenario Simulation complete. Generated %d scenarios.",
                len(self.simulated_scenarios)
            )
            return self.simulated_scenarios

        except Exception as e:
            self.logger.error("Scenario Simulation failed: %s", str(e))
            raise

    def run_qtable_initialization(self, alpha: float = 0.8, gamma: float = 0.8) -> Dict:
        """
        Q-table initialization using the simulated scenarios.
        Ensures all seen states have Q-values for available actions.

        Args:
            alpha: Learning rate for Q-value updates
            gamma: Discount factor for future rewards

        Returns:
            The initialized Q-table
        """
        self.logger.info("Running Q-table initialization from simulated scenarios...")

        # Validate availability of scenarios
        if not hasattr(self, "simulated_scenarios") or not self.simulated_scenarios:
            raise ValueError("No simulated scenarios available for initialization.")

        # Initialize offline learner if not present
        if not self.offline_learner:
            self.offline_learner = QTableManager(file_path=self.q_table_path)
        else:
            self.offline_learner.alpha = alpha
            self.offline_learner.gamma = gamma

        # Collect all possible actions from scenarios
        all_actions = set()
        for scenario in self.simulated_scenarios:
            action = scenario.get("simulated_action", scenario.get("intended_action"))
            if action:
                all_actions.add(action)

        # Process each scenario
        for scenario in self.simulated_scenarios:
            try:
                state = scenario.get("original_state")
                action = scenario.get(
                    "simulated_action", scenario.get("intended_action")
                )
                reward = scenario.get("reward_sim", 0.0)
                next_state = str(
                    scenario.get("next_state")
                )  # use string representation directly

                def ensure_tuple(s):
                    if isinstance(s, str) and s.startswith("(") and s.endswith(")"):
                        try:
                            import ast
                            return ast.literal_eval(
                                s
                            )  # safer than eval for parsing tuples
                        except (ValueError, SyntaxError):
                            return s
                    return s

                state = ensure_tuple(state)
                next_state = ensure_tuple(next_state)

                if not state or not action:
                    self.logger.warning("Skipping incomplete scenario: %s", scenario)
                    continue

                self.offline_learner.seen_states.add(state)
                self.offline_learner.seen_states.add(next_state)

                # Gather possible actions from scenarios with matching next_state
                actions_prime = list(
                    {
                        s.get("simulated_action", s.get("intended_action"))
                        for s in self.simulated_scenarios
                        if str(s.get("original_state")) == next_state
                    }
                )

                # Update Q-table using update_policy if next state and actions_prime exist
                if next_state and actions_prime:
                    self.offline_learner.update_policy(
                        s=state,
                        a=action,
                        R=reward,
                        s_prime=next_state,
                        actions_prime=actions_prime,
                    )
                else:
                    # Direct assignment if no next state info
                    self.offline_learner.Q_table[(state, action)] = reward

            except (KeyError, TypeError, ValueError) as e:
                self.logger.error("Failed to process scenario: %s", e)
                continue

        # Add default Q-values for all seen states and actions
        # This ensures every state in seen_states has entries in the Q-table
        for state in self.offline_learner.seen_states:
            for action in all_actions:
                if (state, action) not in self.offline_learner.Q_table:
                    # Initialize with a default value of 0.0
                    self.offline_learner.Q_table[(state, action)] = 0.0

        self.logger.info(
            "Q-table initialized with %d entries.",
            len(self.offline_learner.Q_table)
        )

        save_success = self.offline_learner.save_q_table(prefix_version="pre_run")

        if not save_success:
            self.logger.warning("Failed to save Q-table to %s", self.q_table_path)

        return self.offline_learner.Q_table

    def run_prompt_analysis(self) -> Dict:
        """
        Analyze the agent's prompt for best practices and improvement opportunities.

        Returns:
            Dictionary with prompt analysis results
        """
        self.logger.info("Starting Prompt Analysis...")

        try:
            # Load the agent prompt
            agent_prompt = self.configuration.get("agent_modifiable_config", {}).get("prompt_configuration_file_path")

            # Initialize the prompt consultant
            self.prompt_consultant = PromptConsulting(
                agent_prompt=agent_prompt,
                model_name=self.model_name,
                api_key=self.api_key,
                provider=self.provider,
            )

            # Analyze the prompt
            raw_analysis = self.prompt_consultant.analyze_prompt()
            self.prompt_analysis = self.prompt_consultant.get_formatted_analysis(
                raw_analysis
            )

            self.logger.info("Prompt Analysis complete.")
            return self.prompt_analysis

        except Exception as e:
            self.logger.error("Prompt Analysis failed: %s", str(e))
            raise

    def run_prompt_estimation(self) -> str:
        """
        Generate an optimized system prompt for the agent based on pre-run analysis results.

        Returns:
            The generated system prompt as a string
        """
        self.logger.info("Starting Prompt Estimation...")

        try:
            # Initialize the prompt estimator
            self.prompt_estimator = PromptEstimator(
                status=self.get_status_summary(),
                agent_id=self.configuration.get("project_name", "N/A"),
                old_prompt=self.old_prompt,
                parsed_steps=self.parsed_steps,
                hypothetical_states=self.hypothetical_states,
                offline_learner=self.offline_learner.Q_table if self.offline_learner else {},
                prompt_analysis=self.prompt_analysis,
                model_name=self.model_name,
                api_key=self.api_key,
                provider=self.provider,
                agent_tools= self.agent_tools,
                output_path= self.output_path,
            )

            # Generate the optimized prompt
            optimized_prompt = self.prompt_estimator.generate_estimated_prompt()

            self.logger.info("Prompt Estimation complete.")
            return optimized_prompt

        except Exception as e:
            self.logger.error("Prompt Estimation failed: %s", str(e))
            raise

    def execute_pre_run_pipeline(self, save_results: bool = True) -> Dict:
        """
        Execute the complete pre-run pipeline: parsing, hypothetical representation,
        scenario simulation, Q-table initialization, and prompt analysis.

        Args:
            save_results: Whether to save the results to files

        Returns:
            Dictionary with the results of all steps
        """
        self.logger.info("Starting ADAPTIQ Pre-Run Pipeline...")

        # Execute all steps
        parsed_steps = self.run_prompt_parsing()
        hypothetical_states = self.run_hypothetical_representation()
        simulated_scenarios = self.run_simulation()
        q_table = self.run_qtable_initialization()
        prompt_analysis = self.run_prompt_analysis()
        new_prompt = self.run_prompt_estimation()

        # Compile results
        results = {
            "parsed_steps": parsed_steps,
            "hypothetical_states": hypothetical_states,
            "simulated_scenarios": simulated_scenarios,
            "q_table_size": len(q_table),
            "prompt_analysis": prompt_analysis,
            "new_prompt": new_prompt,
        }

        # Save results if requested
        if save_results:
            output_dir = self._ensure_output_directory()
            results_path = os.path.join(output_dir, "adaptiq_results.json")
            try:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                self.logger.info("Results saved to %s", results_path)
            except (OSError, TypeError, json.JSONDecodeError) as e:
                self.logger.error("Failed to save results: %s", str(e))

        self.logger.info("ADAPTIQ Pre-Run Pipeline complete.")
        return results

    def get_status_summary(self) -> Dict:
        """
        Get a summary of the current status of each pre-run component.

        Returns:
            Dictionary with status information
        """
        return {
            "prompt_parsing": {
                "completed": len(self.parsed_steps) > 0,
                "steps_found": len(self.parsed_steps),
            },
            "hypothetical_representation": {
                "completed": len(self.hypothetical_states) > 0,
                "states_generated": len(self.hypothetical_states),
            },
            "scenario_simulation": {
                "completed": hasattr(self, "simulated_scenarios")
                and len(self.simulated_scenarios) > 0,
                "scenarios_generated": (
                    len(self.simulated_scenarios)
                    if hasattr(self, "simulated_scenarios")
                    else 0
                ),
            },
            "qtable_initialization": {
                "completed": self.offline_learner is not None
                and len(self.offline_learner.Q_table) > 0,
                "q_entries": (
                    len(self.offline_learner.Q_table) if self.offline_learner else 0
                ),
            },
            "prompt_analysis": {
                "completed": bool(self.prompt_analysis),
                "weaknesses_found": (
                    len(self.prompt_analysis.get("weaknesses", []))
                    if self.prompt_analysis
                    else 0
                ),
                "suggestions_provided": (
                    len(self.prompt_analysis.get("suggested_modifications", []))
                    if self.prompt_analysis
                    else 0
                ),
            },
        }



def adaptiq_pre_run_pipeline(config_path: str, output_path: str = None) -> Any:
    """Execute full pre-run pipeline workflow."""
    
    return {}
