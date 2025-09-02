from collections import deque
from typing import List
from .agent import Agent


class DAG:
    """Represents a Directed Acyclic Graph of agents."""

    def __init__(self, name: str, tasks: List[Agent]):
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self._validate()

    def _validate(self):
        """Ensures the DAG is valid (e.g., no cycles)."""
        try:
            self.get_run_order()
        except ValueError as e:
            raise ValueError(f"Invalid DAG structure: {e}")

    def get_run_order(self) -> List[Agent]:
        """
        Performs a topological sort to get the execution order of agents.

        Raises:
            ValueError: If a cycle is detected in the graph.

        Returns:
            A list of Agent instances in the correct execution order.
        """
        in_degree = {name: len(task.upstream_tasks) for name, task in self.tasks.items()}

        queue = deque([self.tasks[name] for name, degree in in_degree.items() if degree == 0])
        sorted_order = []
        while queue:
            task = queue.popleft()
            sorted_order.append(task)

            for downstream_task in task.downstream_tasks:
                in_degree[downstream_task.name] -= 1
                if in_degree[downstream_task.name] == 0:
                    queue.append(downstream_task)

        if len(sorted_order) != len(self.tasks):
            raise ValueError("Cycle detected in the DAG. Execution order cannot be determined.")

        return sorted_order
