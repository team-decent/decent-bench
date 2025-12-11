from __future__ import annotations

from typing import cast

import numpy as np

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedArrayTypes

from ._helpers import _return_array
from ._imports_types import _jnp_types, _np_types, _tf_types, _torch_types, jnp, tf, torch


def add(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Element-wise addition of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise addition in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types):
        return _return_array(value1 + value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return _return_array(torch.add(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return _return_array(tf.add(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return _return_array(jnp.add(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def iadd[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
    """
    Element-wise in-place addition of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise in-place addition in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        value1 += value2
        return cast("T", _return_array(value1))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 += value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 += value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, _jnp_types):
        value1 += value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def sub(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Element-wise subtraction of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise subtraction in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types):
        return _return_array(value1 - value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return _return_array(torch.sub(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return _return_array(tf.subtract(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return _return_array(jnp.subtract(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def isub[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
    """
    Element-wise in-place subtraction of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise in-place subtraction in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        value1 -= value2
        return cast("T", _return_array(value1))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 -= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 -= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, _jnp_types):
        value1 -= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def mul(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Element-wise multiplication of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types):
        return _return_array(value1 * value2)  # pyright: ignore[reportOperatorIssue]
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return _return_array(torch.mul(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return _return_array(tf.multiply(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return _return_array(jnp.multiply(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def imul[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
    """
    Element-wise in-place multiplication of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise in-place multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        value1 *= value2
        return cast("T", _return_array(value1))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 *= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 *= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, _jnp_types):
        value1 *= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def div(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Element-wise division of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise division in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types):
        return _return_array(value1 / value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return _return_array(torch.div(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return _return_array(tf.divide(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return _return_array(jnp.divide(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def idiv[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
    """
    Element-wise in-place division of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of element-wise in-place division in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        value1 /= value2
        return cast("T", _return_array(value1))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 /= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 /= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, _jnp_types):
        value1 /= value2  # pyright: ignore[reportOperatorIssue]
        return cast("T", _return_array(value1))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def matmul(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Matrix multiplication of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of matrix multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        return _return_array(value1 @ value2)
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return _return_array(value1 @ value2)  # pyright: ignore[reportOperatorIssue]
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, tf.Tensor):
        return _return_array(value1 @ value2)  # pyright: ignore[reportOperatorIssue]
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, jnp.ndarray | jnp.generic):
        return _return_array(value1 @ value2)  # pyright: ignore[reportOperatorIssue]

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def dot(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
    """
    Dot product of two arrays.

    Args:
        array1 (Array | SupportedArrayTypes): First input array.
        array2 (Array | SupportedArrayTypes): Second input array.

    Returns:
        Array: Result of the dot product in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray | np.generic):
        return _return_array(value1.dot(value2))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return _return_array(value1.dot(value2))  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, tf.Tensor):
        return _return_array(value1.dot(value2))  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    if jnp and isinstance(value1, jnp.ndarray | jnp.generic) and isinstance(value2, jnp.ndarray | jnp.generic):
        return _return_array(value1.dot(value2))  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def power(array: Array | SupportedArrayTypes, p: float) -> Array:
    """
    Raise array to p power.

    Args:
        array (Array | SupportedArrayTypes): The tensor.
        p (float): The power.

    Returns:
        Array: The result of the operation.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.power(value, p))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.pow(value, p))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.pow(value, p))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.power(value, p))

    raise TypeError(f"Unsupported type: {type(value)}")


def ipow[T: Array](array: T, p: float) -> T:
    """
    Element-wise in-place power of an array.

    Args:
        array (Array | SupportedArrayTypes): Input array.
        p (float): The power.

    Returns:
        Array: Result of element-wise in-place power in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        value **= p
        return cast("T", _return_array(value))
    if torch and isinstance(value, torch.Tensor):
        value **= p
        return cast("T", _return_array(value))
    if tf and isinstance(value, tf.Tensor):
        value **= p
        return cast("T", _return_array(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        value **= p
        return cast("T", _return_array(value))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def negative(array: Array | SupportedArrayTypes) -> Array:
    """
    Negate array.

    Args:
        array (Array | SupportedArrayTypes): The tensor.

    Returns:
        Array: The negated tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.negative(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.neg(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.negative(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.negative(value))

    raise TypeError(f"Unsupported type: {type(value)}")


def absolute(array: Array | SupportedArrayTypes) -> Array:
    """
    Return the absolute value of a tensor.

    Args:
        array (Array | SupportedArrayTypes): The tensor.

    Returns:
        Array: The absolute value tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.abs(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.abs(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.abs(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.abs(value))

    raise TypeError(f"Unsupported type: {type(value)}")


def sqrt(array: Array | SupportedArrayTypes) -> Array:
    """
    Return the square root of a tensor.

    Args:
        array (Array | SupportedArrayTypes): The tensor.

    Returns:
        Array: The square root tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.sqrt(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.sqrt(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.sqrt(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.sqrt(value))

    raise TypeError(f"Unsupported type: {type(value)}")
