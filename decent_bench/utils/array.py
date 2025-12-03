from __future__ import annotations

from collections.abc import Iterator
from typing import Self

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import ArrayKey, SupportedArrayTypes


class Array:  # noqa: PLR0904
    """
    A wrapper class for :data:`decent_bench.utils.types.SupportedArrayTypes` objects to enable operator overloading.

    This class allows for seamless interoperability between different tensor frameworks
    by overloading standard arithmetic operators.It supports operations such as addition, subtraction, multiplication,
    division, matrix multiplication, and in-place operations.
    """

    def __init__(
        self,
        value: SupportedArrayTypes,
    ):
        """
        Initialize the X object.

        Can be initialized either by providing a tensor-like `value`,
        framework and its device or using one of the methods in
        :mod:`decent_bench.utils.interoperability`.

        Args:
            value (SupportedArrayTypes): The tensor-like object to wrap.

        """
        self.value: SupportedArrayTypes = value

    def __add__(self, other: Array | SupportedArrayTypes) -> Array:
        """
        Add another X object or a scalar to this one.

        Args:
            other: The object to add.

        Returns:
            The result of the addition.

        """
        return iop.add(self, other)

    __radd__ = __add__

    def __sub__(self, other: Array | SupportedArrayTypes) -> Array:
        """
        Subtract another X object or a scalar from this one.

        Args:
            other: The object to subtract.

        Returns:
            The result of the subtraction.

        """
        return iop.sub(self, other)

    def __rsub__(self, other: SupportedArrayTypes) -> Array:
        """
        Handle right-side subtraction.

        Args:
            other: The scalar to subtract from.

        Returns:
            The result of the subtraction.

        """
        return iop.sub(other, self)

    def __mul__(self, other: Array | SupportedArrayTypes) -> Array:
        """
        Multiply this object by another X object or a scalar.

        Args:
            other: The object to multiply by.

        Returns:
            The result of the multiplication.

        """
        return iop.mul(self, other)

    def __truediv__(self, other: Array | SupportedArrayTypes) -> Array:
        """
        Divide this object by another X object or a scalar.

        Args:
            other: The object to divide by.

        Returns:
            The result of the division.

        """
        return iop.div(self, other)

    def __matmul__(self, other: Array | SupportedArrayTypes) -> Array:
        """
        Perform matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        return iop.matmul(self, other)

    def __rmatmul__(self, other: SupportedArrayTypes) -> Array:
        """
        Perform right-side matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        return iop.matmul(other, self)

    def __rmul__(self, other: SupportedArrayTypes) -> Array:
        """
        Handle right-side multiplication.

        Args:
            other: The scalar to multiply by.

        Returns:
            The result of the multiplication.

        """
        return iop.mul(other, self)

    def __rtruediv__(self, other: SupportedArrayTypes) -> Array:
        """
        Handle right-side division.

        Args:
            other: The scalar to be divided by the object.

        Returns:
            The result of the division.

        """
        return iop.div(other, self)

    def __pow__(self, other: float) -> Array:
        """
        Raise the wrapped tensor to a power.

        Args:
            other: The power.

        Returns:
            The result of the operation.

        """
        return iop.power(self, other)

    def __iadd__(self, other: Array | SupportedArrayTypes) -> Self:
        """
        Perform in-place addition.

        Args:
            other: The object to add.

        Returns:
            The modified object.

        """
        return iop.iadd(self, other)

    def __isub__(self, other: Array | SupportedArrayTypes) -> Self:
        """
        Perform in-place subtraction.

        Args:
            other: The object to subtract.

        Returns:
            The modified object.

        """
        return iop.isub(self, other)

    def __imul__(self, other: Array | SupportedArrayTypes) -> Self:
        """
        Perform in-place multiplication.

        Args:
            other: The object to multiply by.

        Returns:
            The modified object.

        """
        return iop.imul(self, other)

    def __itruediv__(self, other: Array | SupportedArrayTypes) -> Self:
        """
        Perform in-place division.

        Args:
            other: The object to divide by.

        Returns:
            The modified object.

        """
        return iop.idiv(self, other)

    def __ipow__(self, other: float) -> Self:
        """
        Perform in-place power operation.

        Args:
            other: The power.

        Returns:
            The modified object.

        """
        return iop.ipow(self, other)

    def __neg__(self) -> Array:
        """
        Negate the wrapped tensor.

        Returns:
            The negated tensor.

        """
        return iop.negative(self)

    def __abs__(self) -> Array:
        """
        Return the absolute value of the wrapped tensor.

        Returns:
            The absolute value.

        """
        return iop.absolute(self)

    def __getitem__(self, key: ArrayKey) -> Array:
        """
        Get an item or slice from the wrapped tensor.

        Args:
            key (ArrayKey): The key or slice.

        Returns:
            The resulting item or slice.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        return iop.get_item(self, key)

    def __setitem__(self, key: ArrayKey, value: Array | SupportedArrayTypes) -> None:
        """
        Set an item or slice in the wrapped tensor.

        Be aware that this operation may not be supported by all underlying frameworks.
        JAX and TensorFlow, for example, use immutable arrays by default.

        Args:
            key (ArrayKey): The key or slice.
            value: The value to set.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        iop.set_item(self, key, value)

    def __hash__(self) -> int:
        """Return the hash of the wrapped tensor."""
        return hash(self.value)

    def __repr__(self) -> str:
        """Return the official string representation of the object."""
        return f"X({self.value!r})"

    def __str__(self) -> str:
        """Return the user-friendly string representation of the object."""
        return str(self.value)

    def __len__(self) -> int:
        """
        Return the length of the first dimension.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not have length.")
        return len(self.value)

    def __iter__(self) -> Iterator[SupportedArrayTypes]:
        """
        Return an iterator over the first dimension, yielding array elements.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values are not iterable.")
        return iter(self.value)

    def __array__(self) -> SupportedArrayTypes:  # noqa: PLW3201
        """
        Return the underlying array-like object.

        Returns:
            The wrapped tensor-like object.

        """
        return self.value

    def __float__(self) -> float:
        """
        Return the wrapped tensor as a float.

        Returns:
            The float representation of the wrapped tensor.

        """
        return iop.astype(self, float)
