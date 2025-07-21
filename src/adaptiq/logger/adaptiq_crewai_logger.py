import datetime
import json
import os
import re

from crewai.agents.parser import AgentAction, AgentFinish, OutputParserException


class AdaptiqLogger:
    """
    AdaptiqLogger provides structured logging for CrewAI agent runs.

    This logger writes both human-readable text logs and structured JSON logs for each agent action,
    final result, or error during execution. It supports:
      - Logging agent thoughts, actions, tool usage, and results.
      - Logging final answers and output from the agent.
      - Logging parsing errors and exceptions.
      - Logging high-level task information and summaries.
      - Maintaining a JSON log file as an array of structured entries for downstream analysis.

    Designed for use with CrewAI agents and compatible with AdaptiQ's reporting and analytics tools.
    """

    def __init__(self, log_file="log.txt", json_file="log.json"):
        """
        Initialize an AdaptiqLogger.

        Ensures that the structured JSON log file exists and is initialized
        as an empty JSON array so that subsequent logging calls can append
        structured entries safely.

        Args:
            log_file (str, optional): Path to the human‑readable plaintext log
                file. Written in append mode. Defaults to "log.txt".
            json_file (str, optional): Path to the structured JSON log file that
                stores a list of log entry objects. Defaults to "log.json".
        """
        self.log_file = log_file
        self.json_file = json_file

        # Initialize JSON log file with array if not existing
        if not os.path.exists(self.json_file):
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _append_to_json(self, log_data):
        """
        Append a structured log entry to the JSON log file.

        Reads the current JSON array from disk (recovering gracefully from
        JSON decode errors by starting a new list), appends `log_data`,
        and rewrites the file. Entries that contain only a timestamp (i.e.,
        no additional diagnostic fields) are ignored to avoid noise.

        Args:
            log_data (dict): Structured log payload to append. Must contain at
                least one key in addition to "timestamp" to be written.
        """
        # Skip writing logs with only timestamp (invalid)
        if len(log_data.keys()) <= 1:
            return

        with open(self.json_file, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

            data.append(log_data)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

    def log_thoughts(self, formatted_answer):
        """
        Log an agent step, final result, or parsing error from a CrewAI run.

        Interprets the incoming object type and emits a rich, human-readable
        log block plus a structured JSON record:

        * AgentAction  – logs thought, action text, tool name, tool input, result.
        * AgentFinish  – logs final thought and output text.
        * OutputParserException – logs the parser error message.

        Text logs are appended to `self.log_file`; structured entries are appended
        to `self.json_file`.

        Args:
            formatted_answer (AgentAction | AgentFinish | OutputParserException | Any):
                Object produced during agent execution. Only the above three types
                receive specialized formatting; other types result in a minimal entry.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        divider = "=" * 80
        log_entry = "- Agent In Progress -"
        json_log = {"timestamp": timestamp}

        if isinstance(formatted_answer, AgentAction):
            thought = re.sub(r"\n+", "\n", formatted_answer.thought.strip())
            formatted_json = json.dumps(
                formatted_answer.tool_input,
                indent=2,
                ensure_ascii=False,
            )

            # Extract Action and Observation from full text
            action_match = re.search(r"Action: (.+)", formatted_answer.text)
            observation_match = re.search(
                r"Observation: (.+)", formatted_answer.text, re.DOTALL
            )

            action = action_match.group(1).strip() if action_match else ""
            observation = (
                observation_match.group(1).strip() if observation_match else ""
            )

            log_entry = f"""
{divider}
[ LOG TIME: {timestamp} ]
[ ACTION STEP ]
  🧠 Thought:     {thought}
  📝 Action:      {action}
  🔧 Using Tool:  {formatted_answer.tool}
  📥 Tool Input:
{formatted_json}
  📤 Tool Output:
{formatted_answer.result}
{divider}
"""

            json_log.update(
                {
                    "type": "AgentAction",
                    "thought": formatted_answer.thought,
                    "text": f"Action: {action}\nObservation: {observation}",
                    "tool": formatted_answer.tool,
                    "tool_input": formatted_answer.tool_input,
                    "result": formatted_answer.result,
                }
            )

        elif isinstance(formatted_answer, AgentFinish):
            # Extract "Final Answer:" part from text
            final_answer_match = re.search(
                r"Final Answer:\s*(.+)", formatted_answer.text, re.DOTALL
            )
            final_answer = (
                final_answer_match.group(1).strip()
                if final_answer_match
                else formatted_answer.output.strip()
            )

            log_entry = f"""
{divider}
[ LOG TIME: {timestamp} ]
[ FINAL RESULT ]
  🧠 Thought:     {formatted_answer.thought}
  ✅ Output:
{final_answer}
{divider}
"""

            json_log.update(
                {
                    "type": "AgentFinish",
                    "thought": formatted_answer.thought,
                    "text": final_answer,
                    "output": formatted_answer.output,
                }
            )

        elif isinstance(formatted_answer, OutputParserException):
            log_entry = f"""
{divider}
[ LOG TIME: {timestamp} ]
[ PARSING ERROR ]
  ❌ Error:
{formatted_answer.error}
{divider}
"""
            json_log.update(
                {"type": "OutputParserException", "error": formatted_answer.error}
            )

        # Write to text file
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(log_entry)

        # Write to JSON file
        self._append_to_json(json_log)

    def log_task(self, output):
        """
        Log a high-level task record summarizing an agent's work.

        Intended for top-level task checkpoints (e.g., after a CrewAI Task run).
        Captures the agent name, task description, raw details, and a short
        summary into both plaintext and structured JSON logs.

        Args:
            output: An object with the attributes `agent`, `description`, `raw`,
                and `summary` (such as a CrewAI Task output object).
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        divider = "=" * 80

        log_entry = f"""
{divider}
[ LOG TIME: {timestamp} ]
[ TASK LOG ]
  🤖 Agent: {output.agent}
  📝 Task Description: {output.description}
  🔍 Raw Details:
{output.raw}
  📌 Summary:
{output.summary}
{divider}
"""

        json_log = {
            "timestamp": timestamp,
            "type": "TaskLog",
            "agent": output.agent,
            "description": output.description,
            "raw": output.raw,
            "summary": output.summary,
        }

        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(log_entry)

        self._append_to_json(json_log)


logger = AdaptiqLogger()
