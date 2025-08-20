import json
import logging
import os
from typing import Dict, List


from adaptiq.core.abstract.integrations import BaseConfig, BasePromptParser
from adaptiq.core.entities import TaskIntent, ScenarioModel, HypotheticalStateRepresentation, StatusSummary, PromptParsingStatus, HypotheticalRepresentationStatus, ScenarioSimulationStatus, QTableInitializationStatus,PromptAnalysisStatus, FormattedAnalysis, QTableState, QTableAction, QTableQValue
from adaptiq.core.q_table import QTableManager
from adaptiq.core.pipelines.pre_run.tools import HypotheticalStateGenerator, PromptConsulting, ScenarioSimulator, PromptEstimator


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
        self.output_path = output_path
        self.config_data = self.base_config.get_config()

        # Ensure output directory exists
        self.q_table_path = os.path.join(output_path, "adaptiq_q_table.json")

        # Loading the old prompt of agent
        self.old_prompt = self.base_config.get_prompt(get_newest=True)

        # Store the prompt parser
        self.prompt_parser = base_prompt_parser

        # Load environment variables for API access
        self.api_key = self.config_data.llm_config.api_key
        self.model_name = self.config_data.llm_config.model_name.value
        self.provider = self.config_data.llm_config.provider.value

        # Get the list of tools available to the agent
        self.agent_tools = self.config_data.agent_modifiable_config.agent_tools

        if not self.api_key:
            raise ValueError("API key not provided in config or environment variables")
        if not self.model_name:
            raise ValueError("Model name not provided in configuration")

        # Initialize component instances
        self.state_generator = None
        self.prompt_consultant = None
        self.scenario_simulator = None
        self.prompt_estimator = None
        self.offline_learner = QTableManager(file_path=self.q_table_path)

        # Results storage
        self.parsed_steps:List[TaskIntent] = []
        self.hypothetical_states: List[HypotheticalStateRepresentation] = []
        self.prompt_analysis: FormattedAnalysis = None
        self.simulated_scenarios: List[ScenarioModel] = []

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

    def run_prompt_parsing(self):
        """
        Execute the prompt parsing step to analyze the agent's task and tools.

        Returns:
            List of dictionaries with parsed steps
        """
        self.logger.info("Starting Prompt Parsing...")

        try:
            # Parse the prompt
            self.parsed_steps = self.prompt_parser.run_parse_prompt()

            self.logger.info(
                "Prompt Parsing complete. Identified %d steps.",
                len(self.parsed_steps)
            )

        except Exception as e:
            self.logger.error("Prompt Parsing failed: %s", str(e))
            raise

    def run_hypothetical_representation(self):
        """
        Generate hypothetical state-action pairs based on parsed steps.

        Returns:
            List of state-action pairs
        """
        self.logger.info("Starting Hypothetical State Generation...")

        try:
            # Initialize the hypothetical state generator
            self.state_generator = HypotheticalStateGenerator(
                prompt_parsed_plan=self.parsed_steps,
                llm=self.base_config.get_llm_instance()
            )

            # Generate state-action pairs
            self.hypothetical_states = self.state_generator.generate_hypothetical_state_action_pairs()

            self.logger.info(
                "Hypothetical State Generation complete. Generated %d state-action pairs.",
                len(self.hypothetical_states)
            )

        except Exception as e:
            self.logger.error("Hypothetical State Generation failed: %s", str(e))
            raise

    def run_simulation(self):
        """
        Run scenario simulation based on the generated hypothetical states.
        Generates multiple plausible scenarios for each state-action pair.

        Returns:
            List of simulated scenarios
        """
        self.logger.info("Starting Scenario Simulation...")

        try:
            # Create a filename for the simulation results
            output_dir = self._ensure_output_directory()
            simulation_output_path = os.path.join(
                output_dir, "adaptiq_simulated_scenarios.json"
            )

            # Initialize the scenario simulator
            self.scenario_simulator = ScenarioSimulator(
                hypothetical_states=self.hypothetical_states,
                llm=self.base_config.get_llm_instance(),
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


        except Exception as e:
            self.logger.error("Scenario Simulation failed: %s", str(e))
            raise

    def _create_qtable_state_from_scenario(self, scenario: ScenarioModel) -> QTableState:
        """
        Create a QTableState from scenario data.
        
        Args:
            scenario: ScenarioModel containing state information
            
        Returns:
            QTableState: Properly structured state object
        """
        # Extract state components from scenario
        # Assuming scenario has attributes we can use to construct the state
        current_subtask = str(scenario.original_state) if scenario.original_state else "unknown"
        
        # For hypothetical states, we may not have complete information
        # so we'll use reasonable defaults
        last_action_taken = getattr(scenario, 'previous_action', 'none')
        last_outcome = getattr(scenario, 'previous_outcome', 'unknown')
        key_context = getattr(scenario, 'context', str(scenario.simulated_action)[:50] if scenario.simulated_action else 'none')
        
        return QTableState(
            current_subtask=current_subtask,
            last_action_taken=last_action_taken,
            last_outcome=last_outcome,
            key_context=key_context
        )

    def _create_next_qtable_state_from_scenario(self, scenario: ScenarioModel) -> QTableState:
        """
        Create a next state QTableState from scenario data.
        
        Args:
            scenario: ScenarioModel containing next state information
            
        Returns:
            QTableState: Properly structured next state object
        """
        current_subtask = str(scenario.next_state) if scenario.next_state else "terminal"
        last_action_taken = str(scenario.simulated_action) if scenario.simulated_action else "none"
        last_outcome = "success" if scenario.reward_sim > 0 else "failure" if scenario.reward_sim < 0 else "neutral"
        key_context = getattr(scenario, 'outcome_context', str(scenario.next_state)[:50] if scenario.next_state else 'terminal')
        
        return QTableState(
            current_subtask=current_subtask,
            last_action_taken=last_action_taken,
            last_outcome=last_outcome,
            key_context=key_context
        )

    def run_qtable_initialization(self, alpha: float = 0.8, gamma: float = 0.8) -> Dict:
        """
        Q-table initialization using the simulated scenarios.
        Ensures all seen states have Q-values for available actions.

        Args:
            alpha: Learning rate for Q-value updates
            gamma: Discount factor for future rewards

        Returns:
            The initialized Q-table as a dictionary
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
            action = scenario.simulated_action 
            if action:
                all_actions.add(action)

        # Convert to QTableAction objects
        all_qtable_actions = [QTableAction(action=action) for action in all_actions]

        # Process each scenario
        for scenario in self.simulated_scenarios:
            try:
                if not scenario.simulated_action:
                    self.logger.warning("Skipping scenario with no action: %s", scenario)
                    continue

                # Create structured state objects
                state = self._create_qtable_state_from_scenario(scenario)
                next_state = self._create_next_qtable_state_from_scenario(scenario)
                action = QTableAction(action=scenario.simulated_action)
                reward = scenario.reward_sim

                # Mark states as seen
                self.offline_learner.seen_states.add(state)
                self.offline_learner.seen_states.add(next_state)

                # Find possible actions from scenarios with matching next_state
                actions_prime = []
                next_state_str = str(next_state.current_subtask)
                
                for other_scenario in self.simulated_scenarios:
                    if (other_scenario.original_state and 
                        str(other_scenario.original_state) == next_state_str and 
                        other_scenario.simulated_action):
                        actions_prime.append(QTableAction(action=other_scenario.simulated_action))

                # If no specific actions found for next state, use all available actions
                if not actions_prime:
                    actions_prime = all_qtable_actions

                # Update Q-table using the structured update_policy method
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
                    if state not in self.offline_learner.Q_table:
                        self.offline_learner.Q_table[state] = {}
                    self.offline_learner.Q_table[state][action] = QTableQValue(q_value=reward)

            except (KeyError, TypeError, ValueError) as e:
                self.logger.error("Failed to process scenario: %s", e)
                continue

        # Add default Q-values for all seen states and actions
        # This ensures every state in seen_states has entries in the Q-table
        for state in self.offline_learner.seen_states:
            if state not in self.offline_learner.Q_table:
                self.offline_learner.Q_table[state] = {}
            
            for action in all_qtable_actions:
                if action not in self.offline_learner.Q_table[state]:
                    # Initialize with a default value of 0.0
                    self.offline_learner.Q_table[state][action] = QTableQValue(q_value=0.0)

        self.logger.info(
            "Q-table initialized with %d states and %d total entries.",
            len(self.offline_learner.Q_table),
            sum(len(actions) for actions in self.offline_learner.Q_table.values())
        )

        save_success = self.offline_learner.save_q_table(prefix_version="pre_run")

        if not save_success:
            self.logger.warning("Failed to save Q-table to %s", self.q_table_path)

        # Return a dictionary representation for backward compatibility
        return self.offline_learner.get_q_table_dict()

    def run_prompt_analysis(self) -> Dict:
        """
        Analyze the agent's prompt for best practices and improvement opportunities.

        Returns:
            Dictionary with prompt analysis results
        """
        self.logger.info("Starting Prompt Analysis...")

        try:
            # Load the agent prompt
            agent_prompt = self.config_data.agent_modifiable_config.prompt_configuration_file_path
            # Initialize the prompt consultant
            self.prompt_consultant = PromptConsulting(
                agent_prompt=agent_prompt,
                llm=self.base_config.get_llm_instance()
            )

            # Analyze the prompt
            self.prompt_analysis = self.prompt_consultant.analyze_prompt()


            self.logger.info("Prompt Analysis complete.")


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
                agent_id=self.config_data.project_name,
                llm=self.base_config.get_llm_instance(),
                old_prompt=self.old_prompt,
                parsed_steps=self.parsed_steps,
                hypothetical_states=self.hypothetical_states,
                offline_learner=self.offline_learner,
                prompt_analysis=self.prompt_analysis,
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
        try:
            self.logger.info("Starting ADAPTIQ Pre-Run Pipeline...")

            # Execute all steps
            self.run_prompt_parsing()
            self.run_hypothetical_representation()
            self.run_simulation()
            self.run_prompt_analysis()
            self.run_qtable_initialization()
            new_prompt = self.run_prompt_estimation()

            # Compile results
            results = {
                "parsed_steps": [step.model_dump() for step in self.parsed_steps],
                "hypothetical_states": [state.model_dump() for state in self.hypothetical_states],
                "simulated_scenarios": [sim.model_dump() for sim in self.simulated_scenarios],
                "q_table_size": len(self.offline_learner.Q_table) if self.offline_learner else 0,
                "prompt_analysis": self.prompt_analysis.model_dump(),
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
        
        except Exception as e:
            self.logger.error("ADAPTIQ Pre-Run Pipeline failed: %s", str(e))
            return {"error": str(e),}
            

    def get_status_summary(self) -> StatusSummary:
        """
        Get a summary of the current status of each pre-run component.

        Returns:
            StatusSummary model with structured status info
        """
        return StatusSummary(
            prompt_parsing=PromptParsingStatus(
                completed=len(self.parsed_steps) > 0,
                steps_found=len(self.parsed_steps),
            ),
            hypothetical_representation=HypotheticalRepresentationStatus(
                completed=len(self.hypothetical_states) > 0,
                states_generated=len(self.hypothetical_states),
            ),
            scenario_simulation=ScenarioSimulationStatus(
                completed=hasattr(self, "simulated_scenarios")
                and self.simulated_scenarios is not None
                and len(self.simulated_scenarios) > 0,
                scenarios_generated=(
                    len(self.simulated_scenarios)
                    if getattr(self, "simulated_scenarios", None)
                    else 0
                ),
            ),
            qtable_initialization=QTableInitializationStatus(
                completed=self.offline_learner is not None
                and len(self.offline_learner.Q_table) > 0,
                q_entries=(
                    sum(len(actions) for actions in self.offline_learner.Q_table.values())
                    if self.offline_learner
                    else 0
                ),
            ),
            prompt_analysis=PromptAnalysisStatus(
                completed=bool(self.prompt_analysis),
                weaknesses_found=(
                    len(self.prompt_analysis.weaknesses)
                    if self.prompt_analysis
                    else 0
                ),
                suggestions_provided=(
                    len(self.prompt_analysis.suggested_modifications)
                    if self.prompt_analysis
                    else 0
                ),
            ),
        )