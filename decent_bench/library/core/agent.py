from __future__ import annotations

from types import MappingProxyType

from numpy import float64
from numpy.typing import NDArray

from decent_bench.library.core.benchmark_problem.schemes.agent_activation_schemes import AgentActivationScheme
from decent_bench.library.core.cost_functions import CostFunction


class Agent:
    """Agent with unique id, local cost function, and activation scheme."""

    def __init__(self, agent_id: int, cost_function: CostFunction, activation_scheme: AgentActivationScheme):
        self._id = agent_id
        self._historical_x: list[NDArray[float64]] = []
        self._auxiliary_variables: dict[str, NDArray[float64]] = {}
        self._received_messages: dict[Agent, NDArray[float64]] = {}
        self._n_messages_sent = 0
        self._n_messages_received = 0
        self._activation_scheme = activation_scheme
        self._cost_function_proxy = _CallCountingCostFunctionProxy(cost_function)

    @property
    def id(self) -> int:
        """Unique id for the agent."""
        return self._id

    @property
    def cost_function(self) -> CostFunction:
        """Local cost function."""
        return self._cost_function_proxy

    @property
    def x(self) -> NDArray[float64]:
        """
        Local optimization variable x.

        Raises:
            RuntimeError: if x is retrieved before being set or initialized

        """
        if not self._historical_x:
            raise RuntimeError("x must be initialized before being accessed")
        return self._historical_x[-1]

    @x.setter
    def x(self, x: NDArray[float64]) -> None:
        self._historical_x.append(x)

    @property
    def received_messages(self) -> MappingProxyType[Agent, NDArray[float64]]:
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
            self._historical_x = [x]
        if aux_vars:
            self._auxiliary_variables = aux_vars
        if received_msgs:
            self._received_messages = received_msgs

    def __index__(self) -> int:
        """Enable using agent as index, for example ``W[a1, a2]`` instead of ``W[a1.id, a2.id]``."""
        return self._id


class AgentMetricView:
    """View of agent that exposes useful properties for calculating metrics."""

    def __init__(self, agent: Agent):
        cost_function_proxy = agent._cost_function_proxy  # noqa: SLF001
        self.historical_x = agent._historical_x  # noqa: SLF001
        self.n_messages_sent = agent._n_messages_sent  # noqa: SLF001
        self.n_messages_received = agent._n_messages_received  # noqa: SLF001
        self.n_evaluate_calls = cost_function_proxy.evaluate_calls
        self.n_gradient_calls = cost_function_proxy.gradient_calls
        self.n_hessian_calls = cost_function_proxy.hessian_calls
        self.n_proximal_calls = cost_function_proxy.proximal_calls
        self.cost_function = cost_function_proxy.inner_cost_function


class _CallCountingCostFunctionProxy(CostFunction):
    def __init__(self, inner_cost_function: CostFunction):
        self.inner_cost_function = inner_cost_function
        self.evaluate_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0
        self.proximal_calls = 0

    @property
    def m_smooth(self) -> float:
        return self.inner_cost_function.m_smooth

    @property
    def m_cvx(self) -> float:
        return self.inner_cost_function.m_cvx

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self.inner_cost_function.domain_shape

    def evaluate(self, x: NDArray[float64]) -> float:
        self.evaluate_calls += 1
        return self.inner_cost_function.evaluate(x)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        self.gradient_calls += 1
        return self.inner_cost_function.gradient(x)

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        self.hessian_calls += 1
        return self.inner_cost_function.hessian(x)

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        self.proximal_calls += 1
        return self.inner_cost_function.proximal(y, rho)

    def __add__(self, other: CostFunction) -> CostFunction:
        return self.inner_cost_function + other
