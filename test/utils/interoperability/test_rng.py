import random

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from .test_core import assert_same_type, assert_shapes_equal, create_array

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
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    JAX_GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except (ImportError, ModuleNotFoundError):
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False
except RuntimeError:
    # JAX raises RuntimeError if no GPU is available when querying devices
    JAX_GPU_AVAILABLE = False


@pytest.fixture(autouse=True)
def restore_rng_state_after_test():
    """Keep RNG-global side effects from leaking between tests."""
    previous_state = iop.get_rng_state()
    yield
    iop.set_rng_state(previous_state)


def _assert_same_random_values(first, second) -> None:
    np.testing.assert_allclose(iop.to_numpy(first), iop.to_numpy(second), rtol=0, atol=0)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_rand_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test rand_like function for all frameworks and devices."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    rand_arr = iop.uniform_like(arr)

    # Compute expected shape using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)

    assert_shapes_equal(rand_arr, np_arr, framework)
    assert_same_type(rand_arr, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_randn_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test randn_like function for all frameworks and devices."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    rand_arr = iop.normal_like(arr)

    # Compute expected shape using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)

    assert_shapes_equal(rand_arr, np_arr, framework)
    assert_same_type(rand_arr, framework)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(2, 3), (5,)],
)
def test_randn_frameworks(framework: str, device: str, shape: tuple[int, ...]) -> None:
    """Test randn function for all frameworks and devices."""
    randn_arr = iop.normal(framework=SupportedFrameworks(framework), shape=shape, device=SupportedDevices(device))
    expected = np.random.default_rng().normal(size=shape)

    assert_shapes_equal(randn_arr, expected, framework)
    assert_same_type(randn_arr, framework)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(2, 3), (5,)],
)
def test_rand_frameworks(framework: str, device: str, shape: tuple[int, ...]) -> None:
    """Test rand function for all frameworks and devices."""
    rand_arr = iop.uniform(framework=SupportedFrameworks(framework), shape=shape, device=SupportedDevices(device))
    expected = np.random.default_rng().random(size=shape)

    assert_shapes_equal(rand_arr, expected, framework)
    assert_same_type(rand_arr, framework)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
def test_set_seed_makes_rand_and_randn_reproducible(framework: str, device: str) -> None:
    """Setting the same seed twice should reproduce the same random samples."""
    framework_enum = SupportedFrameworks(framework)
    device_enum = SupportedDevices(device)
    shape = (3, 2)
    seed = 1234

    iop.set_seed(seed=seed, frameworks=[framework_enum])
    assert iop.get_seed() == seed
    rand_first = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    randn_first = iop.normal(framework=framework_enum, shape=shape, device=device_enum)

    iop.set_seed(seed=seed, frameworks=[framework_enum])
    assert iop.get_seed() == seed
    rand_second = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    randn_second = iop.normal(framework=framework_enum, shape=shape, device=device_enum)

    _assert_same_random_values(rand_first, rand_second)
    _assert_same_random_values(randn_first, randn_second)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize("set_seed", [True, False])
def test_set_rng_state_restores_sequence(framework: str, device: str, set_seed: bool) -> None:
    """Restoring RNG state should continue the exact same random sequence."""
    framework_enum = SupportedFrameworks(framework)
    device_enum = SupportedDevices(device)
    shape = (4,)

    if set_seed:
        iop.set_seed(seed=2026, frameworks=[framework_enum])
    _ = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    state_after_first_draw = iop.get_rng_state(frameworks=[framework_enum])

    second_draw = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    iop.set_rng_state(state_after_first_draw)
    second_draw_after_restore = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)

    _assert_same_random_values(second_draw, second_draw_after_restore)


def test_set_seed_python() -> None:
    """Test that set_seed sets the seed for the Python random module."""
    seed = 2027
    iop.set_seed(seed=seed, frameworks=[])
    assert iop.get_seed() == seed
    first = random.random()
    second = random.random()

    iop.set_seed(seed=seed, frameworks=[])
    assert iop.get_seed() == seed
    first_after_restore = random.random()
    second_after_restore = random.random()

    _assert_same_random_values(first, first_after_restore)
    _assert_same_random_values(second, second_after_restore)


@pytest.mark.parametrize("set_seed", [True, False])
def test_set_rng_state_python(set_seed: bool) -> None:
    """Test that set_rng_state restores the state for the Python random module."""
    if set_seed:
        iop.set_seed(seed=2028, frameworks=[])
    _ = random.random()
    state_after_first_draw = iop.get_rng_state(frameworks=[])

    second_draw = random.random()
    iop.set_rng_state(state_after_first_draw)
    second_draw_after_restore = random.random()

    _assert_same_random_values(second_draw, second_draw_after_restore)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
def test_set_rng_state_with_different_seeds(framework: str, device: str) -> None:
    """Setting the same seed twice should reproduce the same random samples."""
    framework_enum = SupportedFrameworks(framework)
    device_enum = SupportedDevices(device)
    shape = (3, 2)
    seed1 = 1234
    seed2 = 5678

    iop.set_seed(seed=seed1, frameworks=[framework_enum])
    assert iop.get_seed() == seed1
    _ = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    _ = iop.normal(framework=framework_enum, shape=shape, device=device_enum)

    state = iop.get_rng_state(frameworks=[framework_enum])
    rand_first = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    randn_first = iop.normal(framework=framework_enum, shape=shape, device=device_enum)

    iop.set_seed(seed=seed2, frameworks=[framework_enum])
    assert iop.get_seed() == seed2
    _ = iop.uniform(
        framework=framework_enum, shape=shape, device=device_enum
    )  # Advance the RNG state with the new seed
    _ = iop.normal(framework=framework_enum, shape=shape, device=device_enum)  # Advance the RNG state with the new seed

    iop.set_rng_state(state)
    rand_second = iop.uniform(framework=framework_enum, shape=shape, device=device_enum)
    randn_second = iop.normal(framework=framework_enum, shape=shape, device=device_enum)

    _assert_same_random_values(rand_first, rand_second)
    _assert_same_random_values(randn_first, randn_second)


def test_set_rng_state_with_different_seeds_python() -> None:
    """Test that set_rng_state restores the state even if the current seed is different."""
    seed1 = 2029
    seed2 = 2030
    iop.set_seed(seed=seed1, frameworks=[])
    _ = random.random()
    state_after_first_draw = iop.get_rng_state(frameworks=[])

    iop.set_seed(seed=seed2, frameworks=[])
    _ = random.random()  # Advance the RNG state with the new seed

    iop.set_rng_state(state_after_first_draw)
    first_after_restore = random.random()

    iop.set_seed(seed=seed1, frameworks=[])
    _ = random.random()  # Advance the RNG state to match the state after first draw
    first_after_restore_again = random.random()

    _assert_same_random_values(first_after_restore, first_after_restore_again)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize(("size", "replace"), [(4, True), (3, False)])
def test_choice_frameworks(framework: str, device: str, size: int, replace: bool) -> None:
    """choice should return same-framework values with expected shape and replacement semantics."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    arr = create_array(data, framework, device)

    sampled = iop.choice(arr, size=size, replace=replace)

    assert_same_type(sampled, framework)
    sampled_np = iop.to_numpy(sampled)
    assert sampled_np.shape == (size,)
    assert set(sampled_np.tolist()).issubset(set(data))
    if not replace:
        assert np.unique(sampled_np).shape[0] == size


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
def test_set_seed_makes_choice_reproducible(framework: str, device: str) -> None:
    """Setting the same seed twice should reproduce identical choice samples."""
    framework_enum = SupportedFrameworks(framework)
    seed = 2031
    data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    arr = create_array(data, framework, device)

    iop.set_seed(seed=seed, frameworks=[framework_enum])
    first = iop.choice(arr, size=4, replace=True)

    iop.set_seed(seed=seed, frameworks=[framework_enum])
    second = iop.choice(arr, size=4, replace=True)

    _assert_same_random_values(first, second)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
@pytest.mark.parametrize("set_seed", [True, False])
def test_set_rng_state_restores_choice_sequence(framework: str, device: str, set_seed: bool) -> None:
    """Restoring RNG state should continue the same choice sequence."""
    framework_enum = SupportedFrameworks(framework)
    data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    arr = create_array(data, framework, device)

    if set_seed:
        iop.set_seed(seed=2032, frameworks=[framework_enum])

    _ = iop.choice(arr, size=3, replace=True)
    state_after_first_draw = iop.get_rng_state(frameworks=[framework_enum])

    second_draw = iop.choice(arr, size=3, replace=True)
    iop.set_rng_state(state_after_first_draw)
    second_draw_after_restore = iop.choice(arr, size=3, replace=True)

    _assert_same_random_values(second_draw, second_draw_after_restore)
