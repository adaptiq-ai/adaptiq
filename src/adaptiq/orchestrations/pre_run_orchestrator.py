import json
import logging
import os
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adaptiq.instrumental.instrumental import (
    capture_llm_response,
    instrumental_track_tokens,
)

# Import the four components
from adaptiq.parser.prompt_parser import AdaptiqPromptParser
from adaptiq.q_learning.q_learning import AdaptiqOfflineLearner
from adaptiq.utils.pre_run_utils import (
    AdaptiqHypotheticalStateGenerator,
    AdaptiqPromptConsulting,
    AdaptiqScenarioSimulator,
)


class AdaptiqPreRunOrchestrator:
    """
    AdaptiqPreRunOrchestrator coordinates the execution of ADAPTIQ's pre-run module components:
    1. Prompt Parsing - Analyzes agent's task & tools to infer sequence of steps
    2. Hypothetical Representation - Generates hypothetical state-action pairs
    3. Q-table Initialization - Initializes Q-values based on heuristic rules
    4. Prompt Analysis - Analyzes prompt for best practices & improvement opportunities

    This orchestration prepares the agent for execution with optimized configuration.
    """

    def __init__(self, config_path: str, output_path: str):
        """
        Initialize the PreRunOrchestrator with configuration.

        Args:
            config_path: Path to the ADAPTIQ configuration YAML file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-PreRun")

        # Load configuration
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.output_path = output_path

        # Extract key configuration
        self.llm_config = self.config.get("llm_config", {})
        self.agent_config = self.config.get("agent_modifiable_config", {})

        # Load environment variables for API access
        load_dotenv()
        self.api_key = self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model_name = self.llm_config.get("model_name")
        self.provider = self.llm_config.get("provider")

        # Get the list of tools available to the agent
        tools_config = self.config.get("agent_modifiable_config", {}).get(
            "agent_tools", []
        )
        self.agent_tools = (
            [{tool["name"]: tool["description"]} for tool in tools_config]
            if tools_config
            else []
        )

        self.tool_strings = [
            f"{name}: {desc}"
            for tool_dict in self.agent_tools
            for name, desc in tool_dict.items()
        ]
        self.tools_string = "\n".join(self.tool_strings)

        if not self.api_key:
            raise ValueError("API key not provided in config or environment variables")
        if not self.model_name:
            raise ValueError("Model name not provided in configuration")

        # Initialize component instances
        self.prompt_parser = None
        self.state_generator = None
        self.offline_learner = None
        self.prompt_consultant = None

        # Results storage
        self.parsed_steps = []
        self.hypothetical_states = []
        self.prompt_analysis = {}
        self.simulated_scenarios = []

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and parse the ADAPTIQ configuration YAML file

        Args:
            config_path: Path to the configuration file

        Returns:
            dict: The parsed configuration
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

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
            self.logger.info(f"Created output directory: {self.output_path}")

        return self.output_path

    def _load_agent_prompt(self) -> str:
        """
        Load the agent's prompt from the specified file.

        Returns:
            String containing the agent's prompt
        """
        prompt_path = self.agent_config.get("prompt_configuration_file_path")
        if not prompt_path:
            raise ValueError("Agent prompt path not specified in configuration")

        try:
            with open(prompt_path, "r") as file:
                content = file.read()
            return content
        except Exception as e:
            self.logger.error(f"Failed to load agent prompt: {str(e)}")
            raise

    def run_prompt_parsing(self) -> List[Dict[str, Any]]:
        """
        Execute the prompt parsing step to analyze the agent's task and tools.

        Returns:
            List of dictionaries with parsed steps
        """
        self.logger.info("Starting Prompt Parsing...")

        try:
            # Initialize the prompt parser
            self.prompt_parser = AdaptiqPromptParser(self.config_path)

            # Parse the prompt
            self.parsed_steps = self.prompt_parser.parse_prompt()

            self.logger.info(
                f"Prompt Parsing complete. Identified {len(self.parsed_steps)} steps."
            )
            return self.parsed_steps

        except Exception as e:
            self.logger.error(f"Prompt Parsing failed: {str(e)}")
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
            self.state_generator = AdaptiqHypotheticalStateGenerator(
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
                f"Hypothetical State Generation complete. Generated {len(self.hypothetical_states)} state-action pairs."
            )
            return self.hypothetical_states

        except Exception as e:
            self.logger.error(f"Hypothetical State Generation failed: {str(e)}")
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
            self.scenario_simulator = AdaptiqScenarioSimulator(
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
                f"Scenario Simulation complete. Generated {len(self.simulated_scenarios)} scenarios."
            )
            return self.simulated_scenarios

        except Exception as e:
            self.logger.error(f"Scenario Simulation failed: {str(e)}")
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
            self.offline_learner = AdaptiqOfflineLearner(alpha=alpha, gamma=gamma)
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
                            return eval(
                                s
                            )  # safer if you trust input; otherwise parse manually
                        except Exception:
                            return s
                    return s

                state = ensure_tuple(state)
                next_state = ensure_tuple(next_state)

                if not state or not action:
                    self.logger.warning(f"Skipping incomplete scenario: {scenario}")
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

            except Exception as e:
                self.logger.error(f"Failed to process scenario: {e}")
                continue

        # Add default Q-values for all seen states and actions
        # This ensures every state in seen_states has entries in the Q-table
        for state in self.offline_learner.seen_states:
            for action in all_actions:
                if (state, action) not in self.offline_learner.Q_table:
                    # Initialize with a default value of 0.0
                    self.offline_learner.Q_table[(state, action)] = 0.0

        self.logger.info(
            f"Q-table initialized with {len(self.offline_learner.Q_table)} entries."
        )

        output_dir = self._ensure_output_directory()
        q_table_path = os.path.join(output_dir, "adaptiq_q_table.json")
        save_success = self.offline_learner.save_q_table(
            file_path=q_table_path, prefix_version="pre_run"
        )

        if not save_success:
            self.logger.warning(f"Failed to save Q-table to {q_table_path}")

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
            agent_prompt = self._load_agent_prompt()

            # Initialize the prompt consultant
            self.prompt_consultant = AdaptiqPromptConsulting(
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
            self.logger.error(f"Prompt Analysis failed: {str(e)}")
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

        # Compile results
        results = {
            "parsed_steps": parsed_steps,
            "hypothetical_states": hypothetical_states,
            "simulated_scenarios": simulated_scenarios,
            "q_table_size": len(q_table),
            "prompt_analysis": prompt_analysis,
        }

        # Save results if requested
        if save_results:
            output_dir = self._ensure_output_directory()
            results_path = os.path.join(output_dir, "adaptiq_results.json")
            try:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Results saved to {results_path}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {str(e)}")

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

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def generate_estimated_prompt(self) -> str:
        """
        Generate an optimized system prompt for the agent based on the full results of the pre-run pipeline.

        This method:
        - Ensures all pre-run pipeline components have completed (parsing, hypothetical states, simulation, Q-table, analysis).
        - Summarizes key findings from each phase, including parsed steps, hypothetical states, Q-table heuristics, and prompt analysis.
        - Uses an LLM to synthesize a new, improved system prompt that incorporates best practices and recommendations.
        - Saves the generated prompt to a report file in the output directory.

        Returns:
            str: The optimized system prompt generated by the LLM.
        """
        self.logger.info("Generating comprehensive pre-run analysis report...")

        # Ensure we have results from all pipeline components
        status = self.get_status_summary()
        missing_components = []

        for component, data in status.items():
            if not data["completed"]:
                missing_components.append(component)

        if missing_components:
            self.logger.warning(
                "Missing results from: %s. Running complete pipeline...",
                ", ".join(missing_components),
            )
            self.execute_pre_run_pipeline(save_results=False)

        try:
            # Extract key information from results
            agent_id = self.config.get("project_name", "N/A")
            prompt_file_path = self.agent_config.get(
                "prompt_configuration_file_path", "N/A"
            )
            old_prompt = self._load_config(prompt_file_path)
            task_name = list(old_prompt.keys())[0]
            description = old_prompt[task_name]["description"]

            # Process parsed steps
            num_parsed_steps = len(self.parsed_steps)
            first_few_subtasks = []
            for step in self.parsed_steps[:3]:  # Get first three steps
                subtask_name = step.get("subtask_name", "Unnamed task")
                first_few_subtasks.append(subtask_name)

            # Process hypothetical states
            num_hypothetical_states = len(self.hypothetical_states)

            # Process Q-table information
            q_table_size = (
                len(self.offline_learner.Q_table) if self.offline_learner else 0
            )

            # Get heuristics summary
            heuristic_counts = {}
            if self.offline_learner and hasattr(self.offline_learner, "Q_table"):
                for i, hypothetical_step in enumerate(
                    self.hypothetical_states[:20]
                ):  # Sample from first 20 states
                    state_repr = hypothetical_step.get("state")
                    action = hypothetical_step.get("action")

                    # Convert state representation to a tuple if it's in string form
                    state_key = None
                    try:
                        import ast

                        if (
                            isinstance(state_repr, str)
                            and state_repr.strip().startswith("(")
                            and state_repr.strip().endswith(")")
                        ):
                            state_key = ast.literal_eval(state_repr)
                        else:
                            state_key = state_repr
                    except Exception:
                        state_key = state_repr

                    if (state_key, action) in self.offline_learner.Q_table:
                        q_value = self.offline_learner.Q_table[(state_key, action)]

                        # Infer which heuristics might have been applied based on Q-value
                        if q_value < 0:
                            if "undeclared_tool" in action.lower() or not any(
                                tool in action
                                for tool in self.config.get(
                                    "agent_modifiable_config", {}
                                ).get("agent_tools", [])
                            ):
                                heuristic_counts["undeclared_tool_penalty"] = (
                                    heuristic_counts.get("undeclared_tool_penalty", 0)
                                    + 1
                                )
                            if "unknown" in action.lower():
                                heuristic_counts["ambiguous_action_penalty"] = (
                                    heuristic_counts.get("ambiguous_action_penalty", 0)
                                    + 1
                                )
                        elif q_value > 0:
                            if isinstance(state_key, tuple) and len(state_key) >= 2:
                                if (state_key[1] == "None" or not state_key[1]) and any(
                                    x in str(state_key[0])
                                    for x in ["Information", "Initial", "Query"]
                                ):
                                    heuristic_counts["good_first_step_reward"] = (
                                        heuristic_counts.get(
                                            "good_first_step_reward", 0
                                        )
                                        + 1
                                    )

            # Format heuristics summary
            key_heuristics = []
            for heuristic, count in heuristic_counts.items():
                key_heuristics.append(f"{heuristic} (applied {count} times)")

            # Process prompt analysis
            weaknesses = self.prompt_analysis.get("weaknesses", [])
            suggestions = self.prompt_analysis.get("suggested_modifications", [])
            strengths = self.prompt_analysis.get("strengths", [])

            # Create prompt analysis summary text
            prompt_analysis_summary = ""
            if strengths:
                prompt_analysis_summary += "Strengths:\n"
                for i, strength in enumerate(
                    strengths[:3]
                ):  # Limit to first 3 for brevity
                    prompt_analysis_summary += f"- {strength}\n"

            if weaknesses:
                prompt_analysis_summary += "\nWeaknesses:\n"
                for i, weakness in enumerate(
                    weaknesses[:3]
                ):  # Limit to first 3 for brevity
                    prompt_analysis_summary += f"- {weakness}\n"

            if suggestions:
                prompt_analysis_summary += "\nSuggested Modifications:\n"
                for i, suggestion in enumerate(
                    suggestions[:3]
                ):  # Limit to first 3 for brevity
                    prompt_analysis_summary += f"- {suggestion}\n"

            # Create LLM report generation prompt
            template = """
            You are Adaptiq, an AI assistant specializing in optimizing AI agent prompts through reinforcement learning analysis. You have completed a comprehensive 'Pre-Run Analysis Phase' for a developer's agent and must now generate an improved, optimized version of their agent's prompt.

            ## Analysis Summary:

            ### Agent Configuration Context:
            - **Agent ID**: {agent_id}
            - **Current Prompt**: {old_prompt}

            ### Available Agent Tools:
            {agent_tools}

            ### Analysis Results:
            - **Prompt Structure Analysis**: {num_parsed_steps} intended steps identified
            - **Key Sub-Tasks Extracted**: {first_few_subtasks}
            - **Hypothetical State Generation**: {num_hypothetical_states} state-action pairs created
            - **Representative States**: {hypothetical_states_sample}
            - **Q-Table Initialization**: {q_table_size} entries with heuristics: {key_heuristics}
            - **Automated Prompt Analysis**: {prompt_analysis_summary}

            ## Optimization Guidelines:

            Your optimized prompt must:

            1. **Preserve All Placeholders**: Any placeholders (variables in curly braces, template syntax, or dynamic content markers) found in the original prompt are agent inputs that process runtime data. These MUST be preserved exactly as they appear - they represent essential dynamic content that the agent needs to function.

            2. **Structural Optimization**: 
            - Reflect the intended workflow from the parsed steps and sub-tasks
            - Create clear, logical instruction sequences
            - Eliminate redundancy and ambiguity

            3. **Analysis-Driven Improvements**:
            - Incorporate specific recommendations from the prompt analysis phase
            - Address weaknesses identified in the hypothetical state generation
            - Leverage insights from Q-table heuristics

            4. **Tool Integration**:
            - Ensure seamless integration with available tools
            - Provide clear guidance on when and how to use each tool
            - Align tool usage with the hypothetical states and decision patterns

            5. **Clarity and Efficiency**:
            - Use precise, actionable language
            - Remove unnecessary verbosity
            - Structure information hierarchically for better comprehension
            - Focus on decision-making support

            6. **Quality Assurance**:
            - Ensure the prompt supports robust error handling
            - Include guidance for edge cases identified in analysis
            - Maintain consistency with the agent's intended behavior patterns

            ## Output Requirements:

            Return ONLY the optimized prompt. Do not include:
            - Explanations of changes made
            - Reasoning behind modifications  
            - Additional commentary or analysis
            - Formatting beyond the prompt itself
            """
            # Create the prompt
            prompt = ChatPromptTemplate.from_template(template)

            # Initialize the LLM
            if self.provider == "openai":
                chat_model = ChatOpenAI(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.7,  # Balanced between creativity and consistency
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
                )

            # Format the message with our data
            formatted_prompt = prompt.format(
                agent_id=agent_id,
                old_prompt=description,
                agent_tools=self.tools_string,
                num_parsed_steps=num_parsed_steps,
                first_few_subtasks=", ".join(first_few_subtasks),
                num_hypothetical_states=num_hypothetical_states,
                hypothetical_states_sample=self.hypothetical_states,
                q_table_size=q_table_size,
                key_heuristics=(
                    ", ".join(key_heuristics) if key_heuristics else "None identified"
                ),
                prompt_analysis_summary=prompt_analysis_summary,
            )

            # Generate the report
            response = chat_model.invoke(formatted_prompt)
            capture_llm_response(response)
            report = response.content

            self.logger.info("Report generation complete")

            # Save the report if desired
            output_dir = self._ensure_output_directory()
            report_path = os.path.join(
                output_dir, "adaptiq_analysis_pre_run_report.txt"
            )

            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                self.logger.info(f"Report saved to {report_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save report: {str(e)}")

            return report

        except ImportError as e:
            self.logger.error(f"Required package not installed: {str(e)}")
            return f"Failed to generate report: {str(e)}. Please ensure langchain and openai packages are installed."
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return f"Failed to generate report: {str(e)}"


def adaptiq_pre_run_pipeline(config_path: str, output_path: str = None) -> Any:
    """Execute full pre-run pipeline workflow."""
    pre_run_orchestrator = AdaptiqPreRunOrchestrator(
        config_path=config_path, output_path=output_path
    )
    results = pre_run_orchestrator.execute_pre_run_pipeline()
    new_prompt = pre_run_orchestrator.generate_estimated_prompt()

    return results, new_prompt
