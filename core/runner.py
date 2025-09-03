import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from utils.settings import logger
from . import utils
from .dag import DAG


class DAGRunner:
    """Executes a DAG, managing file system state and dependencies."""

    def __init__(self, dag: DAG, working_dir: str = "workspace"):
        self.dag = dag
        self.working_dir = Path(working_dir)
        self.experiment_path: Optional[Path] = None

    def _resolve_input(self, input_value: str) -> Any:
        """Resolves an input string from the agent's input_spec."""
        try:
            input_type, value = input_value.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid input format: '{input_value}'. Expected 'type:value'.")

        if input_type == "agent":
            # e.g., "agent:agent_name/relative/path.json"
            if not self.experiment_path:
                raise RuntimeError("Experiment path not set. Cannot resolve agent dependency.")

            path = self.experiment_path / value
            if not path.exists():
                raise FileNotFoundError(f"Dependency not found: Agent output '{value}' does not exist.")

            if path.suffix == ".jsonl":
                return utils.load_jsonl(path)
            return utils.load_json(path)

        elif input_type == "file":
            # e.g., "file:/absolute/path/to/file.json"
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"External file not found: '{value}'")
            return utils.load_json(path)

        elif input_type == "url":
            # e.g., "url:https://api.example.com/data"
            return value

        elif input_type == "str":
            # e.g., "str:hello"
            return value

        elif input_type == "env":
            # e.g., "str:hello"
            return value

        elif input_type == "json":
            # e.g., "json:{\"key\": \"value\"}"
            return json.loads(value)

        else:
            raise ValueError(f"Unknown input type: '{input_type}'")

    async def run(self, experiment_id: Optional[str] = None, force_rerun: Optional[List[str]] = None):
        """
        Runs the entire DAG.

        Args:
            experiment_id (str, optional): A specific ID for the run. If not provided,
                a new one is generated with a timestamp. If provided, the runner
                will attempt to resume from this experiment's state.
            force_rerun (list, optional): A list of agent names to force re-running,
                even if they have a 'complete.lock' file.
        """
        force_rerun = force_rerun or []

        if experiment_id:
            self.experiment_path = self.working_dir / experiment_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_path = self.working_dir / f"{self.dag.name}-{timestamp}"

        self.experiment_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- Starting DAG run in: {self.experiment_path} ---")

        run_order = self.dag.get_run_order()

        for agent in run_order:
            logger.info(f"[AGENT] {agent.name}")
            agent_dir = self.experiment_path / agent.name
            agent_dir.mkdir(exist_ok=True)
            lock_file = agent_dir / "complete.lock"
            agent.workspace = agent_dir
            # Check for completion or forced rerun
            if agent.name in force_rerun and lock_file.exists():
                logger.warning(f"Forcing rerun for '{agent.name}'. Deleting lock file.")
                lock_file.unlink()

            if lock_file.exists():
                logger.warning(f"Already complete. Skipping.")
                continue

            # Prepare inputs
            try:
                logger.warning("Resolving inputs...")
                resolved_inputs = {
                    key: self._resolve_input(value)
                    for key, value in agent.input_spec.items()
                }
                # Execute agent
                outputs = await agent.run(**resolved_inputs)

                # Save outputs
                if not isinstance(outputs, dict):
                    raise TypeError("Agent's run() method must return a dictionary.")

                logger.warning(f"Saving outputs...")
                for filename, data in outputs.items():
                    output_path = agent_dir / filename
                    if filename.endswith(".jsonl"):
                        utils.save_jsonl(data, output_path)
                    else:  # Default to JSON
                        utils.save_json(data, output_path)

                # Create lock file on success
                lock_file.touch()
                logger.warning(f"  - Agent '{agent.name}' completed successfully.")

            except Exception as e:
                logger.error(f"[ERROR] Agent '{agent.name}' failed: {e}")
                logger.error("--- DAG run interrupted due to error. ---")
                return  # Stop execution

        logger.info("\n--- DAG run finished successfully. ---")
