from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from numpy import float64
from numpy.typing import NDArray

from decent_bench.cost_functions import CostFunction
from decent_bench.schemes import AgentActivationScheme


class Agent:
    """Agent with unique id, local cost function, and activation scheme."""

    def __init__(self, agent_id: int, cost_function: CostFunction, activation_scheme: AgentActivationScheme):
        self._id = agent_id
        self._cost_function = cost_function
        self._activation_scheme = activation_scheme
        self._x_per_iteration: list[NDArray[float64]] = []
        self._auxiliary_variables: dict[str, NDArray[float64]] = {}
        self._received_messages: dict[Agent, NDArray[float64]] = {}
        self._n_sent_messages = 0
        self._n_received_messages = 0
        self._n_sent_messages_dropped = 0
        self._n_evaluate_calls = 0
        self._n_gradient_calls = 0
        self._n_hessian_calls = 0
        self._n_proximal_calls = 0
        cost_function.evaluate = self._call_counting_evaluate  # type: ignore[method-assign]
        cost_function.gradient = self._call_counting_gradient  # type: ignore[method-assign]
        cost_function.hessian = self._call_counting_hessian  # type: ignore[method-assign]
        cost_function.proximal = self._call_counting_proximal  # type: ignore[method-assign]

    @property
    def id(self) -> int:
        """Unique id for the agent."""
        return self._id

    @property
    def cost_function(self) -> CostFunction:
        """Local cost function."""
        return self._cost_function

    @property
    def x(self) -> NDArray[float64]:
        """
        Local optimization variable x.

        Raises:
            RuntimeError: if x is retrieved before being set or initialized

        """
        if not self._x_per_iteration:
            raise RuntimeError("x must be initialized before being accessed")
        return self._x_per_iteration[-1]

    @x.setter
    def x(self, x: NDArray[float64]) -> None:
        self._x_per_iteration.append(x)

    @property
    def received_messages(self) -> Mapping[Agent, NDArray[float64]]:
        """Messages received by neighbors."""
        return MappingProxyType(self._received_messages)

    @property
    def aux_vars(self) -> dict[str, NDArray[float64]]:
        """Auxiliary optimization variables used by algorithms that require more variables than x."""
        return self._auxiliary_variables

    def initialize(
        self,
        *,
        x: NDArray[float64] | None = None,
        aux_vars: dict[str, NDArray[float64]] | None = None,
        received_msgs: dict[Agent, NDArray[float64]] | None = None,
    ) -> None:
        """
        Initialize local variables and messages before running an algorithm.

        Args:
            x: initial x
            aux_vars: initial auxiliary variables
            received_msgs: initial messages from neighbors

        """
        if x is not None:
            self._x_per_iteration = [x]
        if aux_vars:
            self._auxiliary_variables = aux_vars
        if received_msgs:
            self._received_messages = received_msgs

    def _call_counting_evaluate(self, x: NDArray[float64]) -> float:
        self._n_evaluate_calls += 1
        return self._cost_function.__class__.evaluate(self.cost_function, x)

    def _call_counting_gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        self._n_gradient_calls += 1
        return self._cost_function.__class__.gradient(self.cost_function, x)

    def _call_counting_hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        self._n_hessian_calls += 1
        return self._cost_function.__class__.hessian(self.cost_function, x)

    def _call_counting_proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        self._n_proximal_calls += 1
        return self._cost_function.__class__.proximal(self.cost_function, y, rho)

    def __index__(self) -> int:
        """Enable using agent as index, for example ``W[a1, a2]`` instead of ``W[a1.id, a2.id]``."""
        return self._id


@dataclass(frozen=True, eq=False)
class AgentMetricsView:
    """Immutable view of agent that exposes useful properties for calculating metrics."""

    cost_function: CostFunction
    x_per_iteration: list[NDArray[float64]]
    n_evaluate_calls: int
    n_gradient_calls: int
    n_hessian_calls: int
    n_proximal_calls: int
    n_sent_messages: int
    n_received_messages: int
    n_sent_messages_dropped: int

    @staticmethod
    def from_agent(agent: Agent) -> AgentMetricsView:
        """Create from agent."""
        return AgentMetricsView(
            cost_function=agent.cost_function,
            x_per_iteration=agent._x_per_iteration,  # noqa: SLF001
            n_evaluate_calls=agent._n_evaluate_calls,  # noqa: SLF001
            n_gradient_calls=agent._n_gradient_calls,  # noqa: SLF001
            n_hessian_calls=agent._n_hessian_calls,  # noqa: SLF001
            n_proximal_calls=agent._n_proximal_calls,  # noqa: SLF001
            n_sent_messages=agent._n_sent_messages,  # noqa: SLF001
            n_received_messages=agent._n_received_messages,  # noqa: SLF001
            n_sent_messages_dropped=agent._n_sent_messages_dropped,  # noqa: SLF001
        )
