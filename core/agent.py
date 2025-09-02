from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Optional


class Agent(ABC):
    """Abstract Base Class for an agent in a DAG."""

    def __init__(self, name: str, input_spec: Optional[Dict[str, str]] = None):
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

    def __rshift__(self, other: 'Agent'):
        self.downstream_tasks.add(other)
        other.upstream_tasks.add(self)
        return other

    def __lshift__(self, other: 'Agent'):
        other.__rshift__(self)
        return other

    def __repr__(self):
        return f"<Agent(name='{self.name}')>"
