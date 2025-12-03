import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as np_assert_almost_equal

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array

try:
    import torch
    from torch.testing import assert_close as torch_assert_close

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

# ============================================================================
# Helpers
# ============================================================================


def create_array(data: list, framework: str, device: str = "cpu"):
    """
    Factory function to create arrays in different frameworks and devices.

    Args:
        data: Python list/nested list to convert
        framework: One of 'numpy', 'torch', 'tensorflow', 'jax'
        device: 'cpu' or 'gpu'

    Returns:
        Array in the specified framework and device

    """  # noqa: D401, DOC501
    if data is None:
        raise ValueError("Data cannot be None")

    if framework == "numpy":
        return Array(np.array(data, dtype=np.float32))
    if framework == "torch":
        array1 = torch.tensor(data, dtype=torch.float32)
        if device == "gpu" and TORCH_CUDA_AVAILABLE:
            array1 = array1.to("cuda")
        return Array(array1)
    if framework == "tensorflow":
        device_str = "/GPU:0" if device == "gpu" and TF_GPU_AVAILABLE else "/CPU:0"
        with tf.device(device_str):
            array2: tf.Tensor = tf.constant(data, dtype=tf.float32)  # type: ignore
            return Array(array2)
    elif framework == "jax":
        array3 = jnp.array(data, dtype=jnp.float32)
        if device == "gpu" and JAX_GPU_AVAILABLE:
            gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
            if gpu_devices:
                array3 = jax.device_put(array3, device=gpu_devices[0])
        return Array(array3)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# ============================================================================
# Tests for Array class
# ============================================================================


def assert_arrays_equal(result, expected, framework: str):
    """Framework-agnostic assertion for array equality."""
    result_np = iop.to_numpy(result)
    expected_np = iop.to_numpy(expected)

    if framework == "torch" and isinstance(result, torch.Tensor):
        # For torch, use torch_assert_close if result is still a tensor
        expected_torch = torch.tensor(expected_np).to(result.dtype)
        if result.is_cuda:
            expected_torch = expected_torch.to("cuda")
        torch_assert_close(result, expected_torch)
    else:
        np_assert_almost_equal(result_np, expected_np)


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
class TestArrayOperators:
    """Test suite for Array class operators."""

    def test_add(self, framework: str, device: str) -> None:
        """Test addition operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        c = a + b
        expected = a_np + b_np
        assert_arrays_equal(c, expected, framework)

        d = c + a
        expected = expected + a_np
        assert_arrays_equal(d, expected, framework)

        c = a + 2
        expected = a_np + 2
        assert_arrays_equal(c, expected, framework)

        c = 2 + a
        expected = 2 + a_np
        assert_arrays_equal(c, expected, framework)

    def test_sub(self, framework: str, device: str) -> None:
        """Test subtraction operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        c = a - b
        expected = a_np - b_np
        assert_arrays_equal(c, expected, framework)

        d = c - a
        expected = expected - a_np
        assert_arrays_equal(d, expected, framework)

        c = a - 1
        expected = a_np - 1
        assert_arrays_equal(c, expected, framework)

        c = 5 - a
        expected = 5 - a_np
        assert_arrays_equal(c, expected, framework)

    def test_mul(self, framework: str, device: str) -> None:
        """Test multiplication operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        c = a * b
        expected = a_np * b_np
        assert_arrays_equal(c, expected, framework)

        d = c * a
        expected = expected * a_np
        assert_arrays_equal(d, expected, framework)

        c = a * 2
        expected = a_np * 2
        assert_arrays_equal(c, expected, framework)

        e = c * 2
        expected = expected * 2
        assert_arrays_equal(e, expected, framework)

        c = 2 * a
        expected = 2 * a_np
        assert_arrays_equal(c, expected, framework)

    def test_truediv(self, framework: str, device: str) -> None:
        """Test true division operator."""
        a = create_array([[10, 20], [30, 40]], framework, device)
        b = create_array([[2, 5], [10, 8]], framework, device)
        a_np = create_array([[10, 20], [30, 40]], "numpy")
        b_np = create_array([[2, 5], [10, 8]], "numpy")
        c = a / b
        expected = a_np / b_np
        assert_arrays_equal(c, expected, framework)

        d = c / a
        expected = expected / a_np
        assert_arrays_equal(d, expected, framework)

        c = a / 10
        expected = a_np / 10
        assert_arrays_equal(c, expected, framework)

        c = 100 / a
        expected = 100 / a_np
        assert_arrays_equal(c, expected, framework)

    def test_matmul(self, framework: str, device: str) -> None:
        """Test matrix multiplication operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        c = a @ b
        expected = a_np @ b_np
        assert_arrays_equal(c, expected, framework)

        d = c @ a
        expected = expected @ a_np
        assert_arrays_equal(d, expected, framework)

    def test_pow(self, framework: str, device: str) -> None:
        """Test power operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        c = a**2
        expected = a_np**2
        assert_arrays_equal(c, expected, framework)

        d = c**2
        expected = expected**2
        assert_arrays_equal(d, expected, framework)

    def test_iadd(self, framework: str, device: str) -> None:
        """Test in-place addition."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)

        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        a += b
        expected = a_np + b_np

        assert_arrays_equal(a, expected, framework)

        a += 2
        expected = expected + 2
        assert_arrays_equal(a, expected, framework)

        a = create_array([[1, 2], [3, 4]], framework, device)
        a += 2
        expected = a_np + 2
        assert_arrays_equal(a, expected, framework)

    def test_isub(self, framework: str, device: str) -> None:
        """Test in-place subtraction."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        a -= b
        expected = a_np - b_np
        assert_arrays_equal(a, expected, framework)

        a -= 2
        expected = expected - 2
        assert_arrays_equal(a, expected, framework)

        a = create_array([[1, 2], [3, 4]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        a -= 1
        expected = a_np - 1
        assert_arrays_equal(a, expected, framework)

    def test_imul(self, framework: str, device: str) -> None:
        """Test in-place multiplication."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")
        a *= b
        expected = a_np * b_np
        assert_arrays_equal(a, expected, framework)

        a *= 2
        expected = expected * 2
        assert_arrays_equal(a, expected, framework)

        a = create_array([[1, 2], [3, 4]], framework, device)
        a *= 2
        expected = a_np * 2
        assert_arrays_equal(a, expected, framework)

    def test_itruediv(self, framework: str, device: str) -> None:
        """Test in-place true division."""
        a = create_array([[10, 20], [30, 40]], framework, device)
        b = create_array([[2, 5], [10, 8]], framework, device)
        a_np = create_array([[10, 20], [30, 40]], "numpy")
        b_np = create_array([[2, 5], [10, 8]], "numpy")
        a /= b
        expected = a_np / b_np
        assert_arrays_equal(a, expected, framework)

        a /= b
        expected = expected / b_np
        assert_arrays_equal(a, expected, framework)

        a = create_array([[10, 20], [30, 40]], framework, device)
        a /= 10
        expected = a_np / 10
        assert_arrays_equal(a, expected, framework)

    def test_ipow(self, framework: str, device: str) -> None:
        """Test in-place power operator."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        a **= 2
        expected = a_np**2
        assert_arrays_equal(a, expected, framework)

        a **= 2
        expected = expected**2
        assert_arrays_equal(a, expected, framework)

    def test_neg(self, framework: str, device: str) -> None:
        """Test negation operator."""
        a = create_array([[1, -2], [-3, 4]], framework, device)
        a_np = create_array([[1, -2], [-3, 4]], "numpy")
        c = -a
        expected = -a_np
        assert_arrays_equal(c, expected, framework)

        d = -c
        expected = -expected
        assert_arrays_equal(d, expected, framework)

    def test_abs(self, framework: str, device: str) -> None:
        """Test absolute value."""
        a = create_array([[1, -2], [-3, 4]], framework, device)
        a_np = create_array([[1, -2], [-3, 4]], "numpy")
        c = abs(a)
        expected = abs(a_np)
        assert_arrays_equal(c, expected, framework)

        d = abs(c)
        expected = abs(expected)
        assert_arrays_equal(d, expected, framework)

    def test_getitem(self, framework: str, device: str) -> None:
        """Test __getitem__ method."""
        a = create_array([[1, 2, 3], [4, 5, 6]], framework, device)
        a_np = create_array([[1, 2, 3], [4, 5, 6]], "numpy")
        item = a[0, 1]
        expected = a_np[0, 1]
        assert_arrays_equal(item, expected, framework)

        slice_ = a[0, :]
        expected = a_np[0, :]
        assert_arrays_equal(slice_, expected, framework)

    def test_setitem(self, framework: str, device: str) -> None:
        """Test __setitem__ method."""

        if framework in ["jax", "tensorflow"]:
            pytest.skip("Setitem not supported for JAX and TensorFlow due to immutability.")

        a = create_array([[1, 2], [3, 4]], framework, device)
        a[0, 0] = 99.0
        expected = create_array([[1, 2], [3, 4]], "numpy")
        expected[0, 0] = 99.0
        assert_arrays_equal(a, expected, framework)

        b = create_array([10, 20], framework, device)
        b_np = create_array([10, 20], "numpy")
        a[1, :] = b
        expected[1, :] = b_np
        assert_arrays_equal(a, expected, framework)

    def test_len(self, framework: str, device: str) -> None:
        """Test __len__ method."""
        a = create_array([[1, 2, 3], [4, 5, 6]], framework, device)
        assert len(a) == 2

        a_scalar = create_array(5.0, framework, device)
        with pytest.raises(TypeError):
            len(a_scalar)

    def test_iter(self, framework: str, device: str) -> None:
        """Test __iter__ method."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        it = iter(a)
        row1 = next(it)
        assert_arrays_equal(Array(row1), np.array([1, 2]), framework)
        row2 = next(it)
        assert_arrays_equal(Array(row2), np.array([3, 4]), framework)
        with pytest.raises(StopIteration):
            next(it)

        a_scalar = create_array(5, framework, device)
        with pytest.raises(TypeError):
            iter(a_scalar)

    def test_float(self, framework: str, device: str) -> None:
        """Test __float__ method."""
        a = create_array(42.0, framework, device)
        f = float(a)
        assert isinstance(f, float)
        assert f == 42.0

        f = float(a)
        assert isinstance(f, float)
        assert f == 42.0

        a_array = create_array([42.0], framework, device)
        f = float(a_array)
        assert isinstance(f, float)
        assert f == 42.0

        a_array_2d = create_array([[42.0]], framework, device)
        f = float(a_array_2d)
        assert isinstance(f, float)
        assert f == 42.0

        a_non_scalar = create_array([1.0, 2.0], framework, device)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            float(a_non_scalar)

    def test_combinations(self, framework: str, device: str) -> None:
        """Test combinations of operations."""
        a = create_array([[1, 2], [3, 4]], framework, device)
        b = create_array([[5, 6], [7, 8]], framework, device)
        a_np = create_array([[1, 2], [3, 4]], "numpy")
        b_np = create_array([[5, 6], [7, 8]], "numpy")

        c = (a + b) * 2 - 3 / (a - 0.5)
        expected = (a_np + b_np) * 2 - 3 / (a_np - 0.5)
        assert_arrays_equal(c, expected, framework)
