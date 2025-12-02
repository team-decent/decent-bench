"""Custom parameter class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Self, SupportsIndex

from . import interoperability as iop
from .types import SupportedXTypes


class Array:  # noqa: PLR0904
    """A wrapper class for :type:`SupportedXTypes` objects to enable operator overloading."""

    def __init__(
        self,
        value: SupportedXTypes,
    ):
        """
        Initialize the X object.

        Can be initialized either by providing a tensor-like `value`,
        framework and its device or using one of the methods in
        :mod:`decent_bench.utils.interoperability`.

        Args:
            value (SupportedXTypes): The tensor-like object to wrap.

        """
        self.value: SupportedXTypes = value

    def __add__(self, other: Array | SupportedXTypes) -> Array:
        """
        Add another X object or a scalar to this one.

        Args:
            other: The object to add.

        Returns:
            The result of the addition.

        """
        if not TYPE_CHECKING:
            return self.value + (other.value if isinstance(other, Array) else other)

        if isinstance(other, Array):
            return iop.add(self, other)
        return iop.add(self, Array(other))

    def __sub__(self, other: Array | SupportedXTypes) -> Array:
        """
        Subtract another X object or a scalar from this one.

        Args:
            other: The object to subtract.

        Returns:
            The result of the subtraction.

        """
        if not TYPE_CHECKING:
            return self.value - (other.value if isinstance(other, Array) else other)

        if isinstance(other, Array):
            return iop.sub(self, other)
        return iop.sub(self, Array(other))

    def __mul__(self, other: Array | SupportedXTypes) -> Array:
        """
        Multiply this object by another X object or a scalar.

        Args:
            other: The object to multiply by.

        Returns:
            The result of the multiplication.

        """
        if not TYPE_CHECKING:
            return self.value * (other.value if isinstance(other, Array) else other)

        if isinstance(other, Array):
            return iop.mul(self, other)
        return iop.mul(self, Array(other))

    def __truediv__(self, other: Array | SupportedXTypes) -> Array:
        """
        Divide this object by another X object or a scalar.

        Args:
            other: The object to divide by.

        Returns:
            The result of the division.

        """
        if not TYPE_CHECKING:
            return self.value / (other.value if isinstance(other, Array) else other)

        if isinstance(other, Array):
            return iop.div(self, other)
        return iop.div(self, Array(other))

    def __matmul__(self, other: Array | SupportedXTypes) -> Array:
        """
        Perform matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        if not TYPE_CHECKING:
            return self.value @ (other.value if isinstance(other, Array) else other)

        if isinstance(other, Array):
            return iop.matmul(self, other)
        return iop.matmul(self, Array(other))

    def __rmatmul__(self, other: SupportedXTypes) -> Array:
        """
        Perform right-side matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        if not TYPE_CHECKING:
            return other @ self.value

        return iop.matmul(Array(other), self)

    __radd__ = __add__
    __rsub__ = __sub__

    def __rmul__(self, other: SupportedXTypes) -> Array:
        """
        Handle right-side multiplication.

        Args:
            other: The scalar to multiply by.

        Returns:
            The result of the multiplication.

        """
        if not TYPE_CHECKING:
            return other * self.value

        return iop.mul(Array(other), self)

    def __rtruediv__(self, other: SupportedXTypes) -> Array:
        """
        Handle right-side division.

        Args:
            other: The scalar to be divided by the object.

        Returns:
            The result of the division.

        """
        if not TYPE_CHECKING:
            return other / self.value

        return iop.div(Array(other), self)

    def __pow__(self, other: float) -> Array:
        """
        Raise the wrapped tensor to a power.

        Args:
            other: The power.

        Returns:
            The result of the operation.

        """
        if not TYPE_CHECKING:
            return self.value**other

        return iop.power(self, other)

    def __iadd__(self, other: Array | SupportedXTypes) -> Self:
        """
        Perform in-place addition.

        Args:
            other: The object to add.

        Returns:
            The modified object.

        """
        if not TYPE_CHECKING:
            self.value += other.value if isinstance(other, Array) else other
            return self.value

        if isinstance(other, Array):
            return iop.iadd(self, other)
        return iop.iadd(self, Array(other))

    def __isub__(self, other: Array | SupportedXTypes) -> Self:
        """
        Perform in-place subtraction.

        Args:
            other: The object to subtract.

        Returns:
            The modified object.

        """
        if not TYPE_CHECKING:
            self.value -= other.value if isinstance(other, Array) else other
            return self.value

        if isinstance(other, Array):
            return iop.isub(self, other)
        return iop.isub(self, Array(other))

    def __imul__(self, other: Array | SupportedXTypes) -> Self:
        """
        Perform in-place multiplication.

        Args:
            other: The object to multiply by.

        Returns:
            The modified object.

        """
        if not TYPE_CHECKING:
            self.value *= other.value if isinstance(other, Array) else other
            return self.value

        if isinstance(other, Array):
            return iop.imul(self, other)
        return iop.imul(self, Array(other))

    def __itruediv__(self, other: Array | SupportedXTypes) -> Self:
        """
        Perform in-place division.

        Args:
            other: The object to divide by.

        Returns:
            The modified object.

        """
        if not TYPE_CHECKING:
            self.value /= other.value if isinstance(other, Array) else other
            return self.value

        if isinstance(other, Array):
            return iop.idiv(self, other)
        return iop.idiv(self, Array(other))

    def __ipow__(self, other: float) -> Self:
        """
        Perform in-place power operation.

        Args:
            other: The power.

        Returns:
            The modified object.

        """
        if not TYPE_CHECKING:
            self.value **= other
            return self.value

        return iop.ipow(self, other)

    def __neg__(self) -> Array:
        """
        Negate the wrapped tensor.

        Returns:
            The negated tensor.

        """
        if not TYPE_CHECKING:
            return -self.value

        return iop.negative(self)

    def __abs__(self) -> Array:
        """
        Return the absolute value of the wrapped tensor.

        Returns:
            The absolute value.

        """
        if not TYPE_CHECKING:
            return abs(self.value)

        return iop.absolute(self)

    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Array:
        """
        Get an item or slice from the wrapped tensor.

        Args:
            key: The key or slice.

        Returns:
            The resulting item or slice.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        if not TYPE_CHECKING:
            return self.value[key]

        return Array(self.value[key])  # type: ignore[index]

    def __setitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...], value: Array | SupportedXTypes) -> None:
        """
        Set an item or slice in the wrapped tensor.

        Args:
            key: The key or slice.
            value: The value to set.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        if isinstance(value, Array):
            iop.set_item(self, key, value)
            return
        iop.set_item(self, key, Array(value))

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

    def __iter__(self) -> Iterator[SupportedXTypes]:
        """
        Return an iterator over the first dimension, yielding TensorLike elements.

        Raises:
            TypeError: If the wrapped value is a scalar.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values are not iterable.")
        return iter(self.value)

    def __array__(self) -> SupportedXTypes:  # noqa: PLW3201
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
