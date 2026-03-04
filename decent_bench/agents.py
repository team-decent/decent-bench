from __future__ import annotations

import bisect
import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.costs import Cost, EmpiricalRiskCost
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
        self._x_history: AgentHistory = AgentHistory()
        self._auxiliary_variables: dict[str, Array] = {}
        self._received_messages: dict[Agent, Array] = {}
        self._n_x_updates = 0
        self._n_sent_messages = 0
        self._n_received_messages = 0
        self._n_sent_messages_dropped = 0
        self._n_function_calls: float = 0
        self._n_gradient_calls: float = 0
        self._n_hessian_calls: float = 0
        self._n_proximal_calls: float = 0
        self._no_count_depth: int = 0  # Nesting counter; counting disabled when > 0
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

    @property
    def state_snapshot_period(self) -> int:
        """Number of iterations between snapshots of the agent's state."""
        return self._state_snapshot_period

    def snapshot(self, iteration: int, force: bool = False) -> None:
        """
        Snapshot the agent's state.

        This saves the current optimization variable x every :attr:`state_snapshot_period` iterations.

        Args:
            iteration: Algorithm iteration
            force: If true, skip :attr:`state_snapshot_period` and forcefully snapshot the agent state.
                Useful when saving the agents final state.

        """
        if (force or iteration % self.state_snapshot_period == 0) and self._current_x is not None:
            self._x_history[iteration] = iop.copy(self._current_x)

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
            self._x_history = AgentHistory()
            self._x_history[0] = iop.copy(x)
            self._current_x = iop.copy(x)
            self._n_x_updates = 0
        if aux_vars:
            self._auxiliary_variables = {k: iop.copy(v) for k, v in aux_vars.items()}
        if received_msgs:
            self._received_messages = {k: iop.copy(v) for k, v in received_msgs.items()}

    def _call_counting_function(self, x: Array, *args: Any, **kwargs: Any) -> float:  # noqa: ANN401
        # Call the function first so "batch_used" is populated for EmpiricalRiskCost before counting function calls
        res = self._cost.__class__.function(self.cost, x, *args, **kwargs)
        if self._no_count_depth > 0:
            return res
        if isinstance(self._cost, EmpiricalRiskCost):
            self._n_function_calls += len(self._cost.batch_used) / self._cost.n_samples
        else:
            self._n_function_calls += 1
        return res

    def _call_counting_gradient(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        res = self._cost.__class__.gradient(self.cost, x, *args, **kwargs)
        if self._no_count_depth > 0:
            return res
        if isinstance(self._cost, EmpiricalRiskCost):
            self._n_gradient_calls += len(self._cost.batch_used) / self._cost.n_samples
        else:
            self._n_gradient_calls += 1
        return res

    def _call_counting_hessian(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        res = self._cost.__class__.hessian(self.cost, x, *args, **kwargs)
        if self._no_count_depth > 0:
            return res
        if isinstance(self._cost, EmpiricalRiskCost):
            self._n_hessian_calls += len(self._cost.batch_used) / self._cost.n_samples
        else:
            self._n_hessian_calls += 1
        return res

    def _call_counting_proximal(self, x: Array, rho: float, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        res = self._cost.__class__.proximal(self.cost, x, rho, *args, **kwargs)
        if self._no_count_depth > 0:
            return res
        if isinstance(self._cost, EmpiricalRiskCost):
            self._n_proximal_calls += len(self._cost.batch_used) / self._cost.n_samples
        else:
            self._n_proximal_calls += 1
        return res

    def __index__(self) -> int:
        """Enable using agent as index, for example ``W[a1, a2]`` instead of ``W[a1.id, a2.id]``."""
        return self._id

    @staticmethod
    @contextlib.contextmanager
    def no_count(agents: Sequence[Agent]) -> Iterator[None]:
        """
        Context manager that disables call counting for a sequence of agents.

        Use this when computing metrics or other operations that should not
        be counted as algorithm function/gradient calls.

        Args:
            agents: sequence of agents to disable call counting for

        Example::

            with Agent.no_count(agents):
                value = metric.compute(problem, agents, iteration)

        """
        for agent in agents:
            agent._no_count_depth += 1  # noqa: SLF001
        try:
            yield
        finally:
            for agent in agents:
                agent._no_count_depth -= 1  # noqa: SLF001


class AgentHistory:
    """
    Ordered history of an agent's optimization variable x, indexed by algorithm iteration.

    Snapshots are stored sparsely — only iterations at which :meth:`Agent.snapshot` was called
    are recorded. Lookups for iterations between snapshots fall back to the nearest preceding
    snapshot, matching the behaviour of :meth:`Agent.snapshot` with a ``state_snapshot_period > 1``.

    Internally, snapshots are kept in a dict for O(1) exact lookup and a parallel sorted list
    for O(log n) predecessor search via :mod:`bisect`.
    """

    def __init__(self) -> None:
        self._x_history: dict[int, Array] = {}
        self._sorted_keys: list[int] = []

    def max(self) -> int:
        """
        Return the latest iteration for which a snapshot exists.

        Raises:
            ValueError: if no snapshots have been recorded yet.

        """
        if len(self._sorted_keys) < 1:
            raise ValueError("No history available")
        return self._sorted_keys[-1]

    def min(self) -> int:
        """
        Return the earliest iteration for which a snapshot exists.

        Raises:
            ValueError: if no snapshots have been recorded yet.

        """
        if len(self._sorted_keys) < 1:
            raise ValueError("No history available")
        return self._sorted_keys[0]

    def items(self) -> Iterator[tuple[int, Array]]:
        """Yield ``(iteration, x)`` pairs for every snapshot, in ascending iteration order."""
        return ((iteration, self._x_history[iteration]) for iteration in self._sorted_keys)

    def values(self) -> Iterator[Array]:
        """Yield the x snapshot for every recorded iteration, in ascending iteration order."""
        return (self._x_history[iteration] for iteration in self._sorted_keys)

    def keys(self) -> list[int]:
        """Return a sorted list of all iterations for which a snapshot has been recorded."""
        return self._sorted_keys.copy()

    def set_x(self, iteration: int, x: Array) -> None:
        """
        Record ``x`` at ``iteration``, replacing any existing snapshot at that iteration.

        Also available as ``history[iteration] = x``.
        """
        if iteration not in self._x_history:
            bisect.insort(self._sorted_keys, iteration)
        self._x_history[iteration] = x

    def get_x(self, iteration: int) -> Array:
        """
        Return x at ``iteration``, falling back to the nearest preceding snapshot if needed.

        Snapshots are not necessarily recorded at every iteration (controlled by
        :attr:`Agent.state_snapshot_period`). When the exact iteration is not found, the
        closest snapshot with an iteration number ``<= iteration`` is returned instead.
        For example, if snapshots exist at iterations 0, 10, 20 and iteration 23 is requested,
        the snapshot from iteration 20 is returned.

        Also available as ``value = history[iteration]``.

        Args:
            iteration: The algorithm iteration to retrieve x for.

        Raises:
            ValueError: if ``iteration`` is before the first recorded snapshot.

        """
        if iteration not in self._x_history:
            # Binary search for the closest previous snapshot
            idx = bisect.bisect_right(self._sorted_keys, iteration) - 1
            if idx < 0:
                raise ValueError(f"No snapshot available for iteration {iteration}")
            iteration = self._sorted_keys[idx]
        return self._x_history[iteration]

    def __setitem__(self, iteration: int, x: Array) -> None:
        """Record ``x`` at ``iteration``, replacing any existing snapshot at that iteration."""
        self.set_x(iteration, x)

    def __getitem__(self, iteration: int) -> Array:
        """
        Return x at ``iteration``.

        Falls back to the nearest preceding snapshot when no exact match exists.
        See :meth:`get_x` for full semantics.
        """
        return self.get_x(iteration)

    def __contains__(self, iteration: int) -> bool:
        """Return ``True`` if an exact snapshot was recorded at ``iteration``."""
        return iteration in self._x_history

    def __iter__(self) -> Iterator[int]:
        """Iterate over recorded iteration numbers in ascending order."""
        return iter(self._sorted_keys)

    def __len__(self) -> int:
        """Return the number of snapshots recorded."""
        return len(self._x_history)


@dataclass(frozen=True, eq=False)
class AgentMetricsView:
    """Immutable view of agent that exposes useful properties for calculating metrics."""

    cost: Cost
    x_history: AgentHistory
    n_x_updates: int
    n_function_calls: float
    n_gradient_calls: float
    n_hessian_calls: float
    n_proximal_calls: float
    n_sent_messages: int
    n_received_messages: int
    n_sent_messages_dropped: int

    @staticmethod
    def from_agent(agent: Agent) -> AgentMetricsView:
        """Create from agent."""
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
