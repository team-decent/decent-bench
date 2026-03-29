import copy

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent, AgentHistory
from decent_bench.costs import L2RegularizerCost, LinearRegressionCost, QuadraticCost
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    TF_GPU_AVAILABLE = False

try:
    import jax

    JAX_AVAILABLE = True
    JAX_GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except (ImportError, ModuleNotFoundError):
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False
except RuntimeError:
    # JAX raises RuntimeError if no GPU is available when querying devices
    JAX_GPU_AVAILABLE = False


@pytest.mark.parametrize(
    "framework,device",
    [
        pytest.param(
            SupportedFrameworks.NUMPY,
            SupportedDevices.CPU,
        ),
        pytest.param(
            SupportedFrameworks.PYTORCH,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            SupportedFrameworks.PYTORCH,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            SupportedFrameworks.TENSORFLOW,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            SupportedFrameworks.TENSORFLOW,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            SupportedFrameworks.JAX,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            SupportedFrameworks.JAX,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
def test_in_place_operations_history(framework: SupportedFrameworks, device: SupportedDevices):
    """Test that in-place operations on agent.x properly update the history."""
    agent = Agent(
        0,
        LinearRegressionCost([(np.array([1.0, 1.0, 1.0]), np.array([1.0]))]),
        None,
        state_snapshot_period=1,
    )

    initial = iop.zeros(framework=framework, device=device, shape=(3,))
    agent.initialize(x=initial)

    def assert_state(expected_x, expected_history):
        """Helper to verify agent state and history."""
        np.testing.assert_array_almost_equal(
            iop.to_numpy(agent.x),
            expected_x,
            decimal=5,
            err_msg=f"Expected x: {expected_x}, but got: {iop.to_numpy(agent.x)}",
        )
        assert len(agent._x_history) == len(expected_history), (
            f"Expected history length: {len(expected_history)}, but got: {len(agent._x_history)}"
        )
        for i, expected in enumerate(expected_history):
            np.testing.assert_array_almost_equal(
                iop.to_numpy(agent._x_history[i]),
                expected,
                decimal=5,
                err_msg=f"At history index {i}, expected: {expected}, but got: {iop.to_numpy(agent._x_history)}",
            )

    # Initial state
    assert_state(
        np.array([0.0, 0.0, 0.0]),
        [
            np.array([0.0, 0.0, 0.0]),
        ],
    )

    # Test += operator
    agent.x += 1.0
    agent._snapshot(1)
    assert_state(
        np.array([1.0, 1.0, 1.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ],
    )

    # Test *= operator
    agent.x *= 2.0
    agent._snapshot(2)
    assert_state(
        np.array([2.0, 2.0, 2.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
        ],
    )

    # Test **= operator
    agent.x **= 2.0
    agent._snapshot(3)
    assert_state(
        np.array([4.0, 4.0, 4.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([4.0, 4.0, 4.0]),
        ],
    )

    # Test /= operator
    agent.x /= 2.0
    agent._snapshot(4)
    assert_state(
        np.array([2.0, 2.0, 2.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([4.0, 4.0, 4.0]),
            np.array([2.0, 2.0, 2.0]),
        ],
    )

    # Test -= operator
    agent.x -= 1.0
    agent._snapshot(5)
    assert_state(
        np.array([1.0, 1.0, 1.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([4.0, 4.0, 4.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
        ],
    )


@pytest.mark.parametrize(
    "framework,device",
    [
        pytest.param(
            SupportedFrameworks.NUMPY,
            SupportedDevices.CPU,
        ),
        pytest.param(
            SupportedFrameworks.PYTORCH,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            SupportedFrameworks.PYTORCH,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            SupportedFrameworks.TENSORFLOW,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            SupportedFrameworks.TENSORFLOW,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            SupportedFrameworks.JAX,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            SupportedFrameworks.JAX,
            SupportedDevices.GPU,
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize("state_snapshot_period", [1, 5, 10])
def test_agent_state_snapshot_period(
    framework: SupportedFrameworks, device: SupportedDevices, state_snapshot_period: int
):
    """Test that agent history is recorded according to the specified history period."""
    agent = Agent(
        0,
        LinearRegressionCost([(np.array([1.0, 1.0, 1.0]), np.array([1.0]))]),
        None,
        state_snapshot_period=state_snapshot_period,
    )

    initial = iop.zeros(shape=(3,), framework=framework, device=device)
    agent.initialize(x=initial)

    def assert_state(expected_x, expected_history):
        """Helper to verify agent state and history."""
        np.testing.assert_array_almost_equal(
            iop.to_numpy(agent.x),
            expected_x,
            decimal=5,
            err_msg=f"Expected x: {expected_x}, but got: {iop.to_numpy(agent.x)}",
        )
        assert len(agent._x_history) == len(expected_history), (
            f"Expected history length: {len(expected_history)}, but got: {len(agent._x_history)}"
        )
        steps = sorted(agent._x_history.keys())
        assert steps == list(range(0, state_snapshot_period * (len(expected_history)), state_snapshot_period)), (
            f"Expected history steps: {list(range(0, state_snapshot_period * (len(expected_history)), state_snapshot_period))}, "
            f"but got: {steps}"
        )
        for i, expected in zip(steps, expected_history, strict=True):
            np.testing.assert_array_almost_equal(
                iop.to_numpy(agent._x_history[i]),
                expected,
                decimal=5,
                err_msg=f"At history index {i}, expected: {expected}, but got: {iop.to_numpy(agent._x_history)}",
            )

    expected_history_length = 5  # Excluding the initial state, so +1 later
    n_updates = expected_history_length * state_snapshot_period
    for k in range(n_updates):
        agent.x += 1.0
        agent._snapshot(k + 1)

    assert_state(
        np.array([n_updates, n_updates, n_updates]),
        [
            np.array([0.0, 0.0, 0.0]),
        ]
        + [
            np.array([i * state_snapshot_period, i * state_snapshot_period, i * state_snapshot_period])
            for i in range(1, expected_history_length + 1)
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quadratic_agent() -> Agent:
    """Return an agent with a simple 2-D quadratic cost f(x) = x^T x."""
    A = np.eye(2) * 2.0
    b = np.zeros(2)
    return Agent(0, QuadraticCost(A, b, 0.0), None, state_snapshot_period=1)


def _make_empirical_agent(batch_size="all") -> Agent:
    """Return an agent backed by a 4-sample LinearRegressionCost."""
    dataset = [(np.array([float(i)]), np.array([float(i)])) for i in range(1, 5)]
    return Agent(0, LinearRegressionCost(dataset, batch_size=batch_size), None, state_snapshot_period=1)


# ---------------------------------------------------------------------------
# Basic call counting
# ---------------------------------------------------------------------------


class TestCallCounting:
    def test_function_call_increments_counter(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        assert agent._n_function_calls == 0
        agent.cost.function(agent.x)
        assert agent._n_function_calls == 1
        agent.cost.function(agent.x)
        assert agent._n_function_calls == 2

    def test_gradient_call_increments_counter(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        agent.cost.gradient(agent.x)
        assert agent._n_gradient_calls == 1
        agent.cost.gradient(agent.x)
        assert agent._n_gradient_calls == 2

    def test_hessian_call_increments_counter(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        agent.cost.hessian(agent.x)
        assert agent._n_hessian_calls == 1

    def test_proximal_call_increments_counter(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        agent.cost.proximal(agent.x, 1.0)
        assert agent._n_proximal_calls == 1

    def test_independent_counters(self):
        """Each counter tracks only its own call type."""
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        agent.cost.function(agent.x)
        agent.cost.gradient(agent.x)
        agent.cost.gradient(agent.x)
        assert agent._n_function_calls == 1
        assert agent._n_gradient_calls == 2
        assert agent._n_hessian_calls == 0
        assert agent._n_proximal_calls == 0

    @pytest.mark.parametrize(
        "make_wrapper",
        [
            lambda cost: 2.0 * cost,
            lambda cost: cost + L2RegularizerCost(shape=cost.shape),
        ],
    )
    def test_shared_wrapper_calls_increment_original_agent_counter(self, make_wrapper):
        """
        Shared-reference wrappers reuse the patched base cost object.

        Calling the wrapper therefore increments the same agent counter as calling the original cost directly.
        """
        base_cost = QuadraticCost(np.eye(2) * 2.0, np.zeros(2), 0.0)
        wrapper = make_wrapper(base_cost)
        agent = Agent(0, base_cost, None, state_snapshot_period=1)
        agent.initialize(x=np.zeros(2))

        agent.cost.function(agent.x)
        wrapper.function(agent.x)

        assert agent._n_function_calls == 2

    @pytest.mark.parametrize(
        "make_wrapper",
        [
            lambda cost: 2.0 * cost,
            lambda cost: cost + L2RegularizerCost(shape=cost.shape),
        ],
    )
    def test_deepcopied_wrapper_avoids_shared_counter_effects(self, make_wrapper):
        """Deep-copying a wrapper breaks the shared reference to the patched base cost."""
        base_cost = QuadraticCost(np.eye(2) * 2.0, np.zeros(2), 0.0)
        wrapper = copy.deepcopy(make_wrapper(base_cost))
        agent = Agent(0, base_cost, None, state_snapshot_period=1)
        agent.initialize(x=np.zeros(2))

        wrapper.function(agent.x)

        assert agent._n_function_calls == 0

    @pytest.mark.parametrize(
        "make_wrapper",
        [
            lambda cost: 2.0 * cost,
            lambda cost: cost + L2RegularizerCost(shape=cost.shape),
        ],
    )
    def test_agents_sharing_wrapped_cost_object_share_counter_side_effects(self, make_wrapper):
        """
        Reusing the same underlying cost across agents propagates counting side effects.

        This is current shared-reference behavior: calling the wrapped agent also increments the counter on the agent
        that owns the reused base cost object.
        """
        base_cost = QuadraticCost(np.eye(2) * 2.0, np.zeros(2), 0.0)
        wrapped_cost = make_wrapper(base_cost)
        base_agent = Agent(0, base_cost, None, state_snapshot_period=1)
        wrapped_agent = Agent(1, wrapped_cost, None, state_snapshot_period=1)
        base_agent.initialize(x=np.zeros(2))
        wrapped_agent.initialize(x=np.zeros(2))

        wrapped_agent.cost.function(wrapped_agent.x)

        assert wrapped_agent._n_function_calls == 1
        assert base_agent._n_function_calls == 1


# ---------------------------------------------------------------------------
# no_count context manager
# ---------------------------------------------------------------------------


class TestNoCount:
    def test_suppresses_function_counting(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        with Agent.no_count([agent]):
            agent.cost.function(agent.x)
            agent.cost.function(agent.x)
        assert agent._n_function_calls == 0

    def test_suppresses_all_call_types(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        with Agent.no_count([agent]):
            agent.cost.function(agent.x)
            agent.cost.gradient(agent.x)
            agent.cost.hessian(agent.x)
            agent.cost.proximal(agent.x, 1.0)
        assert agent._n_function_calls == 0
        assert agent._n_gradient_calls == 0
        assert agent._n_hessian_calls == 0
        assert agent._n_proximal_calls == 0

    def test_counting_resumes_after_context_exit(self):
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        agent.cost.function(agent.x)  # counted
        with Agent.no_count([agent]):
            agent.cost.function(agent.x)  # not counted
        agent.cost.function(agent.x)  # counted again
        assert agent._n_function_calls == 2

    def test_counting_resumes_after_exception(self):
        """Counter should re-enable even when an exception is raised inside the block."""
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        with pytest.raises(RuntimeError):
            with Agent.no_count([agent]):
                raise RuntimeError("boom")
        agent.cost.function(agent.x)
        assert agent._n_function_calls == 1

    def test_nested_no_count_blocks(self):
        """Counting should stay suppressed until the outermost block exits."""
        agent = _make_quadratic_agent()
        agent.initialize(x=np.zeros(2))
        with Agent.no_count([agent]):
            agent.cost.function(agent.x)  # suppressed (depth=1)
            with Agent.no_count([agent]):
                agent.cost.function(agent.x)  # suppressed (depth=2)
            # depth back to 1 — still suppressed
            agent.cost.function(agent.x)
        # depth back to 0 — counting resumes
        agent.cost.function(agent.x)
        assert agent._n_function_calls == 1

    def test_multiple_agents(self):
        """no_count should suppress counting on all supplied agents."""
        agents = [_make_quadratic_agent(), _make_quadratic_agent()]
        for a in agents:
            a.initialize(x=np.zeros(2))
        with Agent.no_count(agents):
            for a in agents:
                a.cost.function(a.x)
        for a in agents:
            assert a._n_function_calls == 0

    def test_only_listed_agents_are_suppressed(self):
        """Agents not passed to no_count should still be counted."""
        a1 = _make_quadratic_agent()
        a2 = _make_quadratic_agent()
        a1.initialize(x=np.zeros(2))
        a2.initialize(x=np.zeros(2))
        with Agent.no_count([a1]):
            a1.cost.function(a1.x)  # suppressed
            a2.cost.function(a2.x)  # NOT suppressed
        assert a1._n_function_calls == 0
        assert a2._n_function_calls == 1


# ---------------------------------------------------------------------------
# EmpiricalRiskCost fractional counting
# ---------------------------------------------------------------------------


class TestEmpiricalRiskCallCounting:
    def test_full_batch_counts_as_one(self):
        """With batch_size='all' (4 samples), each call should add 4/4 = 1.0."""
        agent = _make_empirical_agent(batch_size="all")
        agent.initialize(x=np.zeros(1))
        agent.cost.function(agent.x)
        assert agent._n_function_calls == pytest.approx(agent.cost.n_samples)

    def test_mini_batch_counts_as_fraction(self):
        """With batch_size=2 out of 4 samples, each call should add 2/4 = 0.5."""
        agent = _make_empirical_agent(batch_size=2)
        agent.initialize(x=np.zeros(1))
        agent.cost.function(agent.x)
        assert agent._n_function_calls == pytest.approx(0.5 * agent.cost.n_samples)

    def test_no_count_suppresses_empirical_risk_counting(self):
        agent = _make_empirical_agent(batch_size="all")
        agent.initialize(x=np.zeros(1))
        with Agent.no_count([agent]):
            agent.cost.function(agent.x)
            agent.cost.gradient(agent.x)
        assert agent._n_function_calls == 0
        assert agent._n_gradient_calls == 0


# ---------------------------------------------------------------------------
# AgentHistory
# ---------------------------------------------------------------------------


class TestAgentHistory:
    # -- construction & length -----------------------------------------------

    def test_empty_on_construction(self):
        h = AgentHistory()
        assert len(h) == 0

    def test_len_tracks_insertions(self):
        h = AgentHistory()
        h[0] = np.array([1.0])
        h[5] = np.array([2.0])
        assert len(h) == 2

    # -- set_x / __setitem__ -------------------------------------------------

    def test_setitem_stores_value(self):
        h = AgentHistory()
        h[3] = np.array([7.0])
        np.testing.assert_array_equal(h[3], [7.0])

    def test_overwrite_existing_iteration(self):
        """Writing to the same iteration replaces the value but does not grow the history."""
        h = AgentHistory()
        h[0] = np.array([1.0])
        h[0] = np.array([9.0])
        assert len(h) == 1
        np.testing.assert_array_equal(h[0], [9.0])

    def test_keys_remain_sorted_after_out_of_order_inserts(self):
        h = AgentHistory()
        for iteration in [10, 0, 5, 20, 3]:
            h[iteration] = np.array([float(iteration)])
        assert h.keys() == sorted([10, 0, 5, 20, 3])

    # -- get_x / __getitem__ -------------------------------------------------

    def test_exact_lookup(self):
        h = AgentHistory()
        h[0] = np.array([0.0])
        h[10] = np.array([10.0])
        np.testing.assert_array_equal(h[10], [10.0])

    def test_fallback_to_nearest_preceding_snapshot(self):
        """Requesting an iteration between snapshots should return the preceding one."""
        h = AgentHistory()
        h[0] = np.array([0.0])
        h[10] = np.array([10.0])
        h[20] = np.array([20.0])
        # Iteration 13 lies between 10 and 20 → should return snapshot at 10
        np.testing.assert_array_equal(h[13], [10.0])

    def test_raises_when_before_first_snapshot(self):
        h = AgentHistory()
        h[5] = np.array([5.0])
        with pytest.raises(ValueError):
            _ = h[3]

    def test_raises_on_empty_history(self):
        h = AgentHistory()
        with pytest.raises(ValueError):
            _ = h[0]

    # -- contains ------------------------------------------------------------

    def test_contains_exact_iteration(self):
        h = AgentHistory()
        h[7] = np.array([1.0])
        assert 7 in h

    def test_not_contains_missing_iteration(self):
        h = AgentHistory()
        h[7] = np.array([1.0])
        assert 3 not in h
        assert 8 not in h

    # -- min / max -----------------------------------------------------------

    def test_min_max(self):
        h = AgentHistory()
        for it in [5, 1, 9, 3]:
            h[it] = np.array([float(it)])
        assert h.min() == 1
        assert h.max() == 9

    def test_min_raises_on_empty(self):
        with pytest.raises(ValueError):
            AgentHistory().min()

    def test_max_raises_on_empty(self):
        with pytest.raises(ValueError):
            AgentHistory().max()

    # -- keys / values / items / iter ----------------------------------------

    def test_keys_returns_sorted_copy(self):
        h = AgentHistory()
        for it in [4, 2, 0]:
            h[it] = np.array([float(it)])
        assert h.keys() == [0, 2, 4]

    def test_keys_returns_independent_copy(self):
        """Mutating the returned list must not affect internal state."""
        h = AgentHistory()
        h[0] = np.array([0.0])
        keys = h.keys()
        keys.append(99)
        assert 99 not in h

    def test_values_in_ascending_order(self):
        h = AgentHistory()
        for it in [20, 0, 10]:
            h[it] = np.array([float(it)])
        result = [v[0] for v in h.values()]
        assert result == [0.0, 10.0, 20.0]

    def test_items_yields_sorted_pairs(self):
        h = AgentHistory()
        for it in [3, 1, 2]:
            h[it] = np.array([float(it)])
        pairs = list(h.items())
        assert [k for k, _ in pairs] == [1, 2, 3]
        assert [v[0] for _, v in pairs] == [1.0, 2.0, 3.0]

    def test_iter_yields_sorted_iterations(self):
        h = AgentHistory()
        for it in [5, 1, 3]:
            h[it] = np.array([float(it)])
        assert list(iter(h)) == [1, 3, 5]
