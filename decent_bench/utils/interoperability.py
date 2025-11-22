from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import (
    Any,
    TypeVar,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

torch = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch as _torch

    torch = _torch

tf = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import tensorflow as _tf

    tf = _tf

jnp = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax.numpy as _jnp

    jnp = _jnp

jax_random = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from jax import random as _jax_random

    jax_random = _jax_random


_T = TypeVar("_T", bound=ArrayLike)
"""
TypeVar for ArrayLike types such as NumPy arrays, PyTorch tensors or
nested containers (lists/tuples).

This TypeVar is used throughout the Interoperability class to ensure that
operations preserve the input framework type.
"""


class Interoperability:
    """Utility class for operating on arrays from different deep learning frameworks."""

    @staticmethod
    def to_numpy(array: ArrayLike) -> NDArray[Any]:
        """
        Convert input array to a NumPy array.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            NDArray: Converted NumPy array.

        """
        if isinstance(array, np.ndarray):
            return array
        if torch and isinstance(array, torch.Tensor):
            return cast("np.ndarray", array.cpu().numpy())
        if tf and isinstance(array, tf.Tensor):
            return cast("np.ndarray", array.numpy())
        if (jnp and isinstance(array, jnp.ndarray)) or isinstance(array, (list, tuple)):
            return np.array(array)
        return np.array(array)

    @staticmethod
    def sum(
        array: _T,
        dims: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> _T:
        """
        Sum elements of an array.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dims (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
                Same semantics as NumPy/TensorFlow/JAX.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Summed value in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.sum(array, axis=dims, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.sum(array, dim=dims, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.reduce_sum(array, axis=dims, keepdims=keepdims))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.sum(array, axis=dims, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            summed = np.sum(np_array, axis=dims, keepdims=keepdims).tolist()
            return cast("_T", type(array)(summed if isinstance(summed, list) else [summed]))
        return cast(
            "_T",
            np.sum(
                Interoperability.to_numpy(array),
                axis=dims,
                keepdims=keepdims,
            ),
        )

    @staticmethod
    def mean(
        array: _T,
        dims: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> _T:
        """
        Compute mean of array elements.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dims (int | tuple[int, ...] | None): Dimension or dimensions along which to compute mean.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Mean value in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.mean(array, axis=dims, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.mean(array, dim=dims, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.reduce_mean(array, axis=dims, keepdims=keepdims))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.mean(array, axis=dims, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            meaned = np.mean(np_array, axis=dims, keepdims=keepdims).tolist()
            return cast("_T", type(array)(meaned if isinstance(meaned, list) else [meaned]))
        return cast(
            "_T",
            np.mean(
                Interoperability.to_numpy(array),
                axis=dims,
                keepdims=keepdims,
            ),
        )

    @staticmethod
    def min(
        array: _T,
        dims: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> _T:
        """
        Compute minimum of array elements.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dims (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Minimum value in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.min(array, axis=dims, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.amin(array, dim=dims, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.reduce_min(array, axis=dims, keepdims=keepdims))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.min(array, axis=dims, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            mined = np.min(np_array, axis=dims, keepdims=keepdims).tolist()
            return cast("_T", type(array)(mined if isinstance(mined, list) else [mined]))
        return cast("_T", np.min(Interoperability.to_numpy(array), axis=dims, keepdims=keepdims))

    @staticmethod
    def max(
        array: _T,
        dims: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> _T:
        """
        Compute maximum of array elements.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dims (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Maximum value in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.max(array, axis=dims, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.amax(array, dim=dims, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.reduce_max(array, axis=dims, keepdims=keepdims))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.max(array, axis=dims, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            maxed = np.max(np_array, axis=dims, keepdims=keepdims).tolist()
            return cast("_T", type(array)(maxed if isinstance(maxed, list) else [maxed]))
        return cast("_T", np.max(Interoperability.to_numpy(array), axis=dims, keepdims=keepdims))

    @staticmethod
    def argmax(array: _T, dim: int | None = None, keepdims: bool = False) -> _T:
        """
        Compute index of maximum value along an axis.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Indices of maximum values in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.argmax(array, axis=dim, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.argmax(array, dim=dim, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            ret = None
            if dim is None:
                # TensorFlow's argmax does not support dim=None directly
                dims = array.ndim if array.ndim is not None else 0
                array = tf.reshape(array, [-1])
                ret = (
                    cast("_T", tf.math.argmax(array, axis=0))
                    if not keepdims
                    else cast("_T", tf.reshape(tf.math.argmax(array, axis=0), [1] * dims))
                )
            else:
                ret = (
                    cast("_T", tf.math.argmax(array, axis=dim))
                    if not keepdims
                    else cast("_T", tf.expand_dims(tf.math.argmax(array, axis=dim), axis=dim))
                )
            return ret
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.argmax(array, axis=dim, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            argmaxed = np.argmax(np_array, axis=dim, keepdims=keepdims).tolist()
            return cast(
                "_T",
                type(array)(argmaxed if isinstance(argmaxed, list) else [argmaxed]),
            )
        return cast(
            "_T",
            np.argmax(Interoperability.to_numpy(array), axis=dim, keepdims=keepdims),
        )

    @staticmethod
    def argmin(array: _T, dim: int | None = None, keepdims: bool = False) -> _T:
        """
        Compute index of minimum value along an axis.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            ArrayLike: Indices of minimum values in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.argmin(array, axis=dim, keepdims=keepdims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.argmin(array, dim=dim, keepdim=keepdims))
        if tf and isinstance(array, tf.Tensor):
            ret = None
            if dim is None:
                # TensorFlow's argmin does not support dim=None directly
                dims = array.ndim if array.ndim is not None else 0
                array = tf.reshape(array, [-1])
                ret = (
                    cast("_T", tf.math.argmin(array, axis=0))
                    if not keepdims
                    else cast("_T", tf.reshape(tf.math.argmin(array, axis=0), [1] * dims))
                )
            else:
                ret = (
                    cast("_T", tf.math.argmin(array, axis=dim))
                    if not keepdims
                    else cast("_T", tf.expand_dims(tf.math.argmin(array, axis=dim), axis=dim))
                )
            return ret
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.argmin(array, axis=dim, keepdims=keepdims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            argmined = np.argmin(np_array, axis=dim, keepdims=keepdims).tolist()
            return cast(
                "_T",
                type(array)(argmined if isinstance(argmined, list) else [argmined]),
            )
        return cast(
            "_T",
            np.argmin(Interoperability.to_numpy(array), axis=dim, keepdims=keepdims),
        )

    @staticmethod
    def copy(array: _T) -> _T:
        """
        Create a copy of the input array.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            ArrayLike: A copy of the input array in the same framework type.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.copy(array))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.clone(array))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.identity(array))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.array(array, copy=True))
        return deepcopy(array)

    @staticmethod
    def stack(arrays: Sequence[_T], dim: int = 0) -> _T:
        """
        Stack a sequence of arrays along a new axis.

        Args:
            arrays (Sequence[ArrayLike]): Sequence of input arrays (NumPy, torch, tf, jax)
                or nested containers (list, tuple).
            dim (int): Axis along which to stack the arrays.

        Returns:
            ArrayLike: Stacked array in the same framework type as the inputs.

        """
        if isinstance(arrays[0], np.ndarray):
            return cast("_T", np.stack(arrays, axis=dim))
        if torch and isinstance(arrays[0], torch.Tensor):
            return cast("_T", torch.stack(arrays, dim=dim))
        if tf and isinstance(arrays[0], tf.Tensor):
            return cast("_T", tf.stack(arrays, axis=dim))
        if jnp and isinstance(arrays[0], jnp.ndarray):
            return cast("_T", jnp.stack(arrays, axis=dim))
        if isinstance(arrays[0], (list, tuple)):
            np_arrays = [Interoperability.to_numpy(arr) for arr in arrays]
            stacked = np.stack(np_arrays, axis=dim)
            return cast("_T", type(arrays[0])(stacked.tolist()))
        np_arrays = [Interoperability.to_numpy(arr) for arr in arrays]
        return cast("_T", np.stack(np_arrays, axis=dim))

    @staticmethod
    def reshape(array: _T, shape: tuple[int, ...]) -> _T:
        """
        Reshape an array to the specified shape.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            shape (tuple[int, ...]): Desired shape for the output array.

        Returns:
            ArrayLike: Reshaped array in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.reshape(array, shape))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.reshape(array, shape))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.reshape(array, shape))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.reshape(array, shape))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            reshaped = np.reshape(np_array, shape)
            return cast("_T", type(array)(reshaped.tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.reshape(np_array, shape))

    @staticmethod
    def zeros_like(array: _T) -> _T:
        """
        Create an array of zeros with the same shape as the input.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            ArrayLike: Array of zeros in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return np.zeros_like(array)
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.zeros_like(array))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.zeros_like(array))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.zeros_like(array))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            return cast("_T", type(array)(np.zeros_like(np_array).tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.zeros_like(np_array))

    @staticmethod
    def ones_like(array: _T) -> _T:
        """
        Create an array of ones with the same shape as the input.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            ArrayLike: Array of ones in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return np.ones_like(array)
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.ones_like(array))
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.ones_like(array))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.ones_like(array))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            return cast("_T", type(array)(np.ones_like(np_array).tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.ones_like(np_array))

    @staticmethod
    def rand_like(array: _T) -> _T:
        """
        Create an array of random values with the same shape as the input.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            ArrayLike: Array of random values in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.random.default_rng().random(array.shape))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.rand_like(array))
        if tf and isinstance(array, tf.Tensor):
            shape = tf.shape(array)
            return cast("_T", tf.random.uniform(shape))
        if jnp and jax_random and isinstance(array, jnp.ndarray):
            return cast(
                "_T",
                jax_random.uniform(jax_random.key(random.randint(0, 2**32 - 1)), shape=array.shape),
            )
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            random_array = np.random.default_rng().random(np_array.shape)
            return cast("_T", type(array)(random_array.tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.random.default_rng().random(np_array.shape))

    @staticmethod
    def eye_like(array: _T) -> _T:
        """
        Create an identity matrix with the same shape as the input.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

        Returns:
            ArrayLike: Identity matrix in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.eye(*array.shape[-2:], dtype=array.dtype, device=array.device))
        if torch and isinstance(array, torch.Tensor):
            return cast(
                "_T",
                torch.eye(*array.shape[-2:], dtype=array.dtype, device=array.device),
            )
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.eye(*array.shape[-2:], dtype=array.dtype))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.eye(*array.shape[-2:], dtype=array.dtype, device=array.device))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            eye_array = np.eye(*np_array.shape[-2:])
            return cast("_T", type(array)(eye_array.tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.eye(*np_array.shape[-2:]))

    @staticmethod
    def transpose(array: _T, dims: tuple[int, ...] | None = None) -> _T:
        """
        Transpose an array.

        Args:
            array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).
            dims (tuple[int, ...] | None): Desired dims order. If None, reverses the dimensions.

        Returns:
            ArrayLike: Transposed array in the same framework type as the input.

        """
        if isinstance(array, np.ndarray):
            return cast("_T", np.transpose(array, axes=dims))
        if torch and isinstance(array, torch.Tensor):
            return cast("_T", torch.permute(array, dims=dims)) if dims else cast("_T", array.T)
        if tf and isinstance(array, tf.Tensor):
            return cast("_T", tf.transpose(array, perm=dims))
        if jnp and isinstance(array, jnp.ndarray):
            return cast("_T", jnp.transpose(array, axes=dims))
        if isinstance(array, (list, tuple)):
            np_array = Interoperability.to_numpy(array)
            transposed = np.transpose(np_array, axes=dims)
            return cast("_T", type(array)(transposed.tolist()))
        np_array = Interoperability.to_numpy(array)
        return cast("_T", np.transpose(np_array, axes=dims))
