from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from decent_bench.costs import Cost
from decent_bench.schemes import AgentActivationScheme
from decent_bench.utils.parameter import X
from decent_bench.utils.types import SupportedXTypes


class Agent[Xtype: SupportedXTypes]:
    """Agent with unique id, local cost function, and activation scheme."""

    def __init__(self, agent_id: int, cost: Cost[Xtype], activation: AgentActivationScheme):
        self._id = agent_id
        self._cost = cost
        self._activation = activation
        self._x_history: list[X[Xtype]] = []
        self._auxiliary_variables: dict[str, X[Xtype]] = {}
        self._received_messages: dict[Agent[Xtype], X[Xtype]] = {}
        self._n_sent_messages = 0
        self._n_received_messages = 0
        self._n_sent_messages_dropped = 0
        self._n_function_calls = 0
        self._n_gradient_calls = 0
        self._n_hessian_calls = 0
        self._n_proximal_calls = 0
        cost.function = self._call_counting_function  # type: ignore[method-assign]
        cost.gradient = self._call_counting_gradient  # type: ignore[method-assign]
        cost.hessian = self._call_counting_hessian  # type: ignore[method-assign]
        cost.proximal = self._call_counting_proximal  # type: ignore[method-assign]

    @property
    def id(self) -> int:
        """Unique id for the agent."""
        return self._id

    @property
    def cost(self) -> Cost[Xtype]:
        """
        Local cost function.

        Alias: :class:`f`, :class:`loss`
        """
        return self._cost

    # Aliases for cost
    f = cost
    loss = cost

    @property
    def x(self) -> X[Xtype]:
        """
        Local optimization variable x.

        Raises:
            RuntimeError: if x is retrieved before being set or initialized

        """
        if not self._x_history:
            raise RuntimeError("x must be initialized before being accessed")
        return self._x_history[-1]

    @x.setter
    def x(self, x: X[Xtype]) -> None:
        self._x_history.append(x)

    @property
    def messages(self) -> Mapping[Agent[Xtype], X[Xtype]]:
        """Messages received by neighbors."""
        return MappingProxyType(self._received_messages)

    @property
    def aux_vars(self) -> dict[str, X[Xtype]]:
        """Auxiliary optimization variables used by algorithms that require more variables than x."""
        return self._auxiliary_variables

    def initialize(
        self,
        *,
        x: X[Xtype] | None = None,
        aux_vars: dict[str, X[Xtype]] | None = None,
        received_msgs: dict[Agent[Xtype], X[Xtype]] | None = None,
    ) -> None:
        """
        Initialize local variables and messages before running an algorithm.

        Args:
            x: initial x
            aux_vars: initial auxiliary variables
            received_msgs: initial messages from neighbors

        """
        if x is not None:
            self._x_history = [x]
        if aux_vars:
            self._auxiliary_variables = aux_vars
        if received_msgs:
            self._received_messages = received_msgs

    def _call_counting_function(self, x: X[Xtype]) -> float:
        self._n_function_calls += 1
        return self._cost.__class__.function(self.cost, x)

    def _call_counting_gradient(self, x: X[Xtype]) -> X[Xtype]:
        self._n_gradient_calls += 1
        return self._cost.__class__.gradient(self.cost, x)

    def _call_counting_hessian(self, x: X[Xtype]) -> X[Xtype]:
        self._n_hessian_calls += 1
        return self._cost.__class__.hessian(self.cost, x)

    def _call_counting_proximal(self, y: X[Xtype], rho: float) -> X[Xtype]:
        self._n_proximal_calls += 1
        return self._cost.__class__.proximal(self.cost, y, rho)

    def __index__(self) -> int:
        """Enable using agent as index, for example ``W[a1, a2]`` instead of ``W[a1.id, a2.id]``."""
        return self._id


@dataclass(frozen=True, eq=False)
class AgentMetricsView:
    """Immutable view of agent that exposes useful properties for calculating metrics."""

    cost: Cost
    x_history: list[X]
    n_function_calls: int
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
            cost=agent.cost,
            x_history=agent._x_history,  # noqa: SLF001
            n_function_calls=agent._n_function_calls,  # noqa: SLF001
            n_gradient_calls=agent._n_gradient_calls,  # noqa: SLF001
            n_hessian_calls=agent._n_hessian_calls,  # noqa: SLF001
            n_proximal_calls=agent._n_proximal_calls,  # noqa: SLF001
            n_sent_messages=agent._n_sent_messages,  # noqa: SLF001
            n_received_messages=agent._n_received_messages,  # noqa: SLF001
            n_sent_messages_dropped=agent._n_sent_messages_dropped,  # noqa: SLF001
        )
