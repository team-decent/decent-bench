import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.costs import LinearRegressionCost
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
            SupportedFrameworks.TORCH,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            SupportedFrameworks.TORCH,
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
    agent = Agent(0, LinearRegressionCost(np.array([[1.0, 1.0, 1.0]]), np.array([1.0])), None, history_period=1)  # type: ignore  # noqa: PGH003

    initial = iop.zeros((3,), framework=framework, device=device)
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
    assert_state(
        np.array([1.0, 1.0, 1.0]),
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ],
    )

    # Test *= operator
    agent.x *= 2.0
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
            SupportedFrameworks.TORCH,
            SupportedDevices.CPU,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            SupportedFrameworks.TORCH,
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
@pytest.mark.parametrize("history_period", [1, 5, 10])
def test_agent_state_snapshot_period(framework: SupportedFrameworks, device: SupportedDevices, history_period: int):
    """Test that agent history is recorded according to the specified history period."""
    agent = Agent(
        0,
        LinearRegressionCost(np.array([[1.0, 1.0, 1.0]]), np.array([1.0])),
        None,
        history_period=history_period,
    )

    initial = iop.zeros((3,), framework=framework, device=device)
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
        assert steps == list(range(0, history_period * (len(expected_history)), history_period)), (
            f"Expected history steps: {list(range(0, history_period * (len(expected_history)), history_period))}, "
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
    n_updates = expected_history_length * history_period
    for _ in range(n_updates):
        agent.x += 1.0

    assert_state(
        np.array([n_updates, n_updates, n_updates]),
        [
            np.array([0.0, 0.0, 0.0]),
        ]
        + [
            np.array([i * history_period, i * history_period, i * history_period])
            for i in range(1, expected_history_length + 1)
        ],
    )
