from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.costs import Cost
from decent_bench.schemes import AgentActivationScheme
from decent_bench.utils.array import Array


class Agent:
    """Agent with unique id, local cost function, activation scheme and state snapshot period."""

    def __init__(self, agent_id: int, cost: Cost, activation: AgentActivationScheme, state_snapshot_period: int):
        if state_snapshot_period <= 0:
            raise ValueError("state_snapshot_period must be a positive integer")

        self._id = agent_id
        self._cost = cost
        self._activation = activation
        self._state_snapshot_period = state_snapshot_period
        self._current_x: Array | None = None
        self._x_history: dict[int, Array] = {}
        self._auxiliary_variables: dict[str, Array] = {}
        self._received_messages: dict[Agent, Array] = {}
        self._n_x_updates = 0
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
    def cost(self) -> Cost:
        """
        Local cost function.

        Alias: :class:`f`, :class:`loss`
        """
        return self._cost

    # Aliases for cost
    f = cost
    loss = cost

    @property
    def x(self) -> Array:
        """
        Local optimization variable x.

        Raises:
            RuntimeError: if x is retrieved before being set or initialized

        """
        if self._current_x is None:
            raise RuntimeError("x must be initialized before being accessed")
        return self._current_x

    @x.setter
    def x(self, x: Array) -> None:
        self._n_x_updates += 1
        self._current_x = x
        if self._n_x_updates % self._state_snapshot_period == 0:
            self._x_history[self._n_x_updates] = iop.copy(x)

    @property
    def messages(self) -> Mapping[Agent, Array]:
        """Messages received by neighbors."""
        return MappingProxyType(self._received_messages)

    @property
    def aux_vars(self) -> dict[str, Array]:
        """Auxiliary optimization variables used by algorithms that require more variables than x."""
        return self._auxiliary_variables

    def initialize(
        self,
        *,
        x: Array | None = None,
        aux_vars: dict[str, Array] | None = None,
        received_msgs: dict[Agent, Array] | None = None,
    ) -> None:
        """
        Initialize local variables and messages before running an algorithm.

        Args:
            x: initial x
            aux_vars: initial auxiliary variables
            received_msgs: initial messages from neighbors

        Raises:
            ValueError: if initialized x has incorrect shape

        """
        if x is not None:
            if iop.shape(x) != self.cost.shape:
                raise ValueError(f"Initialized x has shape {iop.shape(x)}, expected {self.cost.shape}")
            self._x_history = {0: iop.copy(x)}
            self._current_x = iop.copy(x)
            self._n_x_updates = 0
        if aux_vars:
            self._auxiliary_variables = {k: iop.copy(v) for k, v in aux_vars.items()}
        if received_msgs:
            self._received_messages = {k: iop.copy(v) for k, v in received_msgs.items()}

    def _call_counting_function(self, x: Array, *args: Any, **kwargs: Any) -> float:  # noqa: ANN401
        self._n_function_calls += 1
        return self._cost.__class__.function(self.cost, x, *args, **kwargs)

    def _call_counting_gradient(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        self._n_gradient_calls += 1
        return self._cost.__class__.gradient(self.cost, x, *args, **kwargs)

    def _call_counting_hessian(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        self._n_hessian_calls += 1
        return self._cost.__class__.hessian(self.cost, x, *args, **kwargs)

    def _call_counting_proximal(self, x: Array, rho: float, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        self._n_proximal_calls += 1
        return self._cost.__class__.proximal(self.cost, x, rho, *args, **kwargs)

    def __index__(self) -> int:
        """Enable using agent as index, for example ``W[a1, a2]`` instead of ``W[a1.id, a2.id]``."""
        return self._id


@dataclass(frozen=True, eq=False)
class AgentMetricsView:
    """Immutable view of agent that exposes useful properties for calculating metrics."""

    cost: Cost
    x_history: dict[int, Array]
    n_x_updates: int
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
        # Append the last x if not already recorded
        if agent._current_x is not None and agent._n_x_updates not in agent._x_history:  # noqa: SLF001
            agent._x_history[agent._n_x_updates] = iop.copy(agent._current_x)  # noqa: SLF001

        return AgentMetricsView(
            cost=agent.cost,
            x_history=agent._x_history,  # noqa: SLF001
            n_x_updates=agent._n_x_updates,  # noqa: SLF001
            n_function_calls=agent._n_function_calls,  # noqa: SLF001
            n_gradient_calls=agent._n_gradient_calls,  # noqa: SLF001
            n_hessian_calls=agent._n_hessian_calls,  # noqa: SLF001
            n_proximal_calls=agent._n_proximal_calls,  # noqa: SLF001
            n_sent_messages=agent._n_sent_messages,  # noqa: SLF001
            n_received_messages=agent._n_received_messages,  # noqa: SLF001
            n_sent_messages_dropped=agent._n_sent_messages_dropped,  # noqa: SLF001
        )
