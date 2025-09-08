from abc import ABC, abstractmethod

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class CompressionScheme(ABC):
    """Scheme defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:
        """Apply compression and return a new, compressed message."""


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, n_significant_digits: int):
        self.n_significant_digits = n_significant_digits

    def compress(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102
        res: NDArray[float64] = np.vectorize(lambda x: float(f"%.{self.n_significant_digits - 1}e" % x))(msg)
        return res
