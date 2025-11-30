"""Custom parameter class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

from numpy.typing import NDArray

from decent_bench.agents import Agent

from . import interoperability as iop
from .types import SupportedDevices, SupportedFrameworks, SupportedXTypes


class X:  # noqa: PLR0904
    """A wrapper class for TensorLike objects to enable operator overloading."""

    def __init__(
        self,
        value: SupportedXTypes | None = None,
        framework: SupportedFrameworks = "numpy",
        shape: tuple[int, ...] | None = None,
        device: SupportedDevices = "cpu",
    ):
        """
        Initialize the X object.

        Can be initialized either by providing a tensor-like `value`,
        or by providing a `framework`, `shape`, and optionally `dtype`
        to create a zero tensor.

        Args:
            value (SupportedXTypes | None): The tensor-like object to wrap. Defaults to None.
            framework (SupportedFrameworks): The framework to use for zero tensor creation.
              Defaults to None.
            shape (tuple[int, ...] | None): The shape for zero tensor creation. Defaults to None.
            device (SupportedDevices): The device for the tensor. Defaults to "cpu".

        Raises:
            ValueError: If initialization parameters are incorrect.

        """
        if value and shape:
            raise ValueError("Either 'value' or'shape' must be provided, not a mix.")

        if value is not None:
            self.value = value
        elif shape:
            self.value = iop.zeros(framework=framework, shape=shape, device=device).value
        else:
            raise ValueError("Either 'value' or both 'framework' and 'shape' must be provided.")

        self.framework: SupportedFrameworks = framework
        self.device: SupportedDevices = device

    def __add__(self, other: X | float) -> X:
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

    def __sub__(self, other: X | float) -> X:
        """
        Subtract another X object or a scalar from this one.

        Args:
            other: The object to subtract.

        Returns:
            The result of the subtraction.

        """
        if isinstance(other, X):
            return X(iop.sub(self.value, other.value), framework=self.framework, device=self.device)
        return X(iop.sub(self.value, other), framework=self.framework, device=self.device)

    def __mul__(self, other: X | SupportedXTypes) -> X:
        """
        Multiply this object by another X object or a scalar.

        Args:
            other: The object to multiply by.

        Returns:
            The result of the multiplication.

        """
        if isinstance(other, X):
            return X(
                iop.mul(self.value, other.value),
            )
        return X(iop.mul(self.value, other))

    def __truediv__(self, other: X | SupportedXTypes) -> X:
        """
        Divide this object by another X object or a scalar.

        Args:
            other: The object to divide by.

        Returns:
            The result of the division.

        """
        if isinstance(other, X):
            return X(iop.div(self.value, other.value))
        return X(iop.div(self.value, other))

    def __matmul__(self, other: X) -> X:
        """
        Perform matrix multiplication with another X object.

        Args:
            other: The object to multiply with.

        Returns:
            The result of the matrix multiplication.

        """
        return X(iop.matmul(self.value, other.value))

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__

    def __rtruediv__(self, other: float) -> X:
        """
        Handle right-side division with a scalar.

        Args:
            other: The scalar to be divided by the object.

        Returns:
            The result of the division.

        """
        return X(other / self.value)

    def __pow__(self, other: float) -> X:
        """
        Raise the wrapped tensor to a power.

        Args:
            other: The power.

        Returns:
            The result of the operation.

        """
        return X(iop.power(self.value, other))

    def __neg__(self) -> X:
        """
        Negate the wrapped tensor.

        Returns:
            The negated tensor.

        """
        return X(iop.negative(self.value))

    def __abs__(self) -> X:
        """
        Return the absolute value of the wrapped tensor.

        Returns:
            The absolute value.

        """
        return X(iop.absolute(self.value))

    def __hash__(self) -> int:
        """Return the hash of the wrapped tensor."""
        return hash(self.value)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the wrapped tensor."""
        return iop.shape(self.value)

    def to_numpy(self) -> NDArray:
        """Convert the wrapped tensor to a NumPy array."""
        return iop.to_numpy(self.value)

    def __iadd__(self, other: X | float) -> Self:
        """
        Perform in-place addition.

        Args:
            other: The object to add.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            return iop.iadd(self.value, other.value)
        else:
            return iop.iadd(self.value, other)

    def __isub__(self, other: SupportedXTypes) -> Self:
        """
        Perform in-place subtraction.

        Args:
            other: The object to subtract.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            self.value = iop.isub(self.value, other.value)
        else:
            self.value = iop.isub(self.value, other)
        return self

    def __imul__(self, other: SupportedXTypes) -> Self:
        """
        Perform in-place multiplication.

        Args:
            other: The object to multiply by.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            self.value = iop.imul(self.value, other.value)
        else:
            self.value = iop.imul(self.value, other)
        return self

    def __itruediv__(self, other: SupportedXTypes) -> Self:
        """
        Perform in-place division.

        Args:
            other: The object to divide by.

        Returns:
            The modified object.

        """
        if isinstance(other, X):
            self.value = iop.idiv(self.value, other.value)
        else:
            self.value = iop.idiv(self.value, other)
        return self

    def __ipow__(self, other: float) -> Self:
        """
        Perform in-place power operation.

        Args:
            other: The power.

        Returns:
            The modified object.

        """
        self.value = iop.ipow(self.value, other)
        return self

    def __getitem__(self, key: tuple[int | Agent, ...] | int | Agent) -> X:
        """
        Get an item or slice from the wrapped tensor.

        Args:
            key: The key or slice.

        Returns:
            The resulting item or slice.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        return X(self.value[key])

    def __setitem__(self, key: tuple[int | Agent, ...] | int | Agent, value: X | SupportedXTypes) -> None:
        """
        Set an item or slice in the wrapped tensor.

        Args:
            key: The key or slice.
            value: The value to set.

        """
        if isinstance(self.value, (float, int, complex)):
            raise TypeError("Scalar values do not support indexing.")

        iop.setitem(self.value, key, value)

    def __repr__(self) -> str:
        """Return the official string representation of the object."""
        return f"X({self.value!r})"

    def __str__(self) -> str:
        """Return the user-friendly string representation of the object."""
        return str(self.value)

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return len(self.value)

    def __iter__(self) -> Iterator[TensorLike]:
        """Return an iterator over the first dimension, yielding TensorLike elements."""
        return iter(self.value)
