from abc import ABC, abstractmethod

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class CompressionScheme(ABC):
    """Schema defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:
        """Apply compression and return a new, compressed message."""


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102, PLR6301
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, significant_digits: int):
        self.significant_digits = significant_digits

    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102
        return np.vectorize(lambda x: float(f"%.{self.significant_digits - 1}e" % x))(msg)  # type: ignore[no-any-return]
