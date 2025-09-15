import json
from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Set, Optional


class Agent(ABC):
    """Abstract Base Class for an agent in a DAG."""

    config: Any
    workspace: Path
    env: Dict[str, Any] = {}

    def __init__(self, name: str, input_spec: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initializes an Agent.

        Args:
            name (str): A unique name for the agent. Used for directory creation.
            input_spec (dict, optional): A dictionary specifying the agent's inputs.
                Format: 'input_name': 'type:value'
                Types: 'agent', 'file', 'url', 'json'
        """
        if not name or not name.isidentifier():
            raise ValueError("Agent name must be a valid Python identifier.")
        self._name = name
        self.input_spec = input_spec or {}

        self.downstream_tasks: Set['Agent'] = set()
        self.upstream_tasks: Set['Agent'] = set()

        if hasattr(self, "Config"):
            # Get the field names from the nested Config dataclass
            config_fields = {f.name for f in fields(self.Config)}

            # Separate kwargs into those for the model_config and others
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

            # Instantiate the nested Config dataclass with the provided arguments
            self.config = self.Config(**config_kwargs)
        else:
            # If no Config class, just store kwargs in a simple object
            from types import SimpleNamespace
            self.config = SimpleNamespace(**kwargs)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        The main logic of the agent. This method is called by the DAGRunner.

        Args:
            **kwargs: The inputs to the agent, resolved from the input_spec.

        Returns:
            A dictionary where keys are filenames (e.g., 'output.json') and
            values are the data to be saved to those files.
        """
        raise NotImplementedError

    def save_locally(self, rel_path: str, data: Any) -> Path:
        local_p = self.workspace / rel_path
        with open(local_p, "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(data, ensure_ascii=False))
        return local_p.resolve().absolute()

    def save_raw_ouput_locally(self, rel_path: str, data: Any) -> Path:
        local_p = self.workspace / rel_path
        with open(local_p, "w", encoding="utf-8") as f_out:
            f_out.write(data)
        return local_p.resolve().absolute()

    def __rshift__(self, other: 'Agent'):
        self.downstream_tasks.add(other)
        other.upstream_tasks.add(self)
        return other

    def __lshift__(self, other: 'Agent'):
        other.__rshift__(self)
        return other

    def __repr__(self):
        return f"<Agent(name='{self.name}')>"
