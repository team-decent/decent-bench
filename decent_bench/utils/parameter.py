"""Custom parameter class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self, SupportsIndex

from numpy.typing import NDArray

from . import interoperability as iop
from .types import SupportedDevices, SupportedFrameworks, SupportedXTypes


class X:  # noqa: PLR0904
    """A wrapper class for :type:`SupportedXTypes` objects to enable operator overloading."""

    def __init__(
        self,
        value: SupportedXTypes | None = None,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        shape: tuple[int, ...] | None = None,
        device: SupportedDevices = SupportedDevices.CPU,
    ):
        """
        Initialize the X object.

        Can be initialized either by providing a tensor-like `value`,
        or by providing a `framework`, `shape`, and optionally `dtype`
        to create a zero tensor.

        Args:
            value (T | None): The tensor-like object to wrap. Defaults to None.
            framework (SupportedFrameworks): The framework to use for zero tensor creation.
              Defaults to None.
            shape (tuple[int, ...] | None): The shape for zero tensor creation. Defaults to None.
            device (SupportedDevices): The device for the tensor. Defaults to "cpu".

        Raises:
            ValueError: If initialization parameters are incorrect.

        """
        if value is not None and shape is not None:
            raise ValueError("Either 'value' or'shape' must be provided, not a mix.")

        if value is not None:
            self.value = value
        elif shape is not None:
            self.value = iop.zeros(framework=framework, shape=shape, device=device).value
        else:
            raise ValueError("Either 'value' or both 'framework' and 'shape' must be provided.")

        self.framework: SupportedFrameworks = framework
        self.device: SupportedDevices = device

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the wrapped tensor."""
        return iop.shape(self)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the wrapped tensor."""
        return len(self.shape)

    @property
    def T(self) -> X:  # noqa: N802
        """Return the transpose of the wrapped tensor."""
        return iop.transpose(self)

    def dot(self, other: X | SupportedXTypes) -> X:
        """
        Compute the dot product with another X object.

        Args:
            other: The other X object.

        Returns:
            The result of the dot product.

        """
        if isinstance(other, X):
            return iop.dot(self, other)
        return iop.dot(self, X(other))

    def copy(self) -> X:
        """
        Create a copy of the X object.

        Returns:
            A new X object that is a copy of the current one.

        """
        return iop.copy(self)

    def to_numpy(self) -> NDArray[Any]:
        """Convert the wrapped tensor to a NumPy array."""
        return iop.to_numpy(self.value)

    def __add__(self, other: X | SupportedXTypes) -> X:
        """
        Add another X object or a scalar to this one.

        Args:
            other: The object to add.

        Returns:
            The result of the addition.

        """
        if isinstance(other, X):
            return iop.add(self, other)
        return iop.add(self, X(other))

    def __sub__(self, other: X | SupportedXTypes) -> X:
        """
        Subtract another X object or a scalar from this one.

        Args:
            other: The object to subtract.

        Returns:
            The result of the subtraction.

        """
        if isinstance(other, X):
            return iop.sub(self, other)
        return iop.sub(self, X(other))

    def __mul__(self, other: X | SupportedXTypes) -> X:
        """
        Multiply this object by another X object or a scalar.

        Args:
            other: The object to multiply by.

        Returns:
            The result of the multiplication.

        """
        if isinstance(other, X):
            return iop.mul(self, other)
        return iop.mul(self, X(other))

    def __truediv__(self, other: X | SupportedXTypes) -> X:
        """
        Divide this object by another X object or a scalar.

        Args:
            other: The object to divide by.

        Returns:
            The result of the division.

        """
        if isinstance(other, X):
            return iop.div(self, other)
        return iop.div(self, X(other))

    def __matmul__(self, other: X | SupportedXTypes) -> X:
        """
        Perform matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        if isinstance(other, X):
            return iop.matmul(self, other)
        return iop.matmul(self, X(other))

    __radd__ = __add__
    __rsub__ = __sub__

    def __rmul__(self, other: SupportedXTypes) -> X:
        """
        Handle right-side multiplication.

        Args:
            other: The scalar to multiply by.

        Returns:
            The result of the multiplication.

        """
        if isinstance(other, X):
            return iop.mul(other, self)
        return iop.mul(X(other), self)

    def __rtruediv__(self, other: SupportedXTypes) -> X:
        """
        Handle right-side division.

        Args:
            other: The scalar to be divided by the object.

        Returns:
            The result of the division.

        """
        if isinstance(other, X):
            return iop.div(other, self)
        return iop.div(X(other), self)

    def __pow__(self, other: float) -> X:
        """
        Raise the wrapped tensor to a power.

        Args:
            other: The power.

        Returns:
            The result of the operation.

        """
        return iop.power(self, other)

    def __iadd__(self, other: X | SupportedXTypes) -> Self:
        """
        Perform in-place addition.

        Args:
            other: The object to add.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            return iop.iadd(self, other)
        return iop.iadd(self, X(other))

    def __isub__(self, other: X | SupportedXTypes) -> Self:
        """
        Perform in-place subtraction.

        Args:
            other: The object to subtract.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            return iop.isub(self, other)
        return iop.isub(self, X(other))

    def __imul__(self, other: X | SupportedXTypes) -> Self:
        """
        Perform in-place multiplication.

        Args:
            other: The object to multiply by.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            return iop.imul(self, other)
        return iop.imul(self, X(other))

    def __itruediv__(self, other: X | SupportedXTypes) -> Self:
        """
        Perform in-place division.

        Args:
            other: The object to divide by.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            return iop.idiv(self, other)
        return iop.idiv(self, X(other))

    def __ipow__(self, other: float) -> Self:
        """
        Perform in-place power operation.

        Args:
            other: The power.

        Returns:
            The modified object.

        """
        return iop.ipow(self, other)

    def __neg__(self) -> X:
        """
        Negate the wrapped tensor.

        Returns:
            The negated tensor.

        """
        return iop.negative(self)

    def __abs__(self) -> X:
        """
        Return the absolute value of the wrapped tensor.

        Returns:
            The absolute value.

        """
        return iop.absolute(self)

    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> X:
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

        return X(self.value[key])  # type: ignore[index]

    def __setitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...], value: X | SupportedXTypes) -> None:
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

        if isinstance(value, X):
            iop.set_item(self, key, value)  # type: ignore[arg-type]
            return
        iop.set_item(self, key, X(value))  # type: ignore[arg-type]

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
