from __future__ import annotations

from abc import ABC, abstractmethod

from decent_array import Array
from decent_array import interoperability as iop


# later remove framework and device when iop refactored
class NoiseScheme(ABC):
    """Scheme defining the noise impacting messages."""

    @abstractmethod
    def make_noise(self, shape: tuple[int, ...]) -> Array | None:
        """Generate noise array of given shape (None if no noise)."""


class NoNoise(NoiseScheme):
    """Scheme representing transmission without noise."""

    def make_noise(self, _: tuple[int, ...]) -> Array | None:
        return None


class GaussianNoise(NoiseScheme):
    """
    Scheme generating normal noise.

    The scheme generates independent noise sampled from a normal distribution with mean ``mean`` and standard deviation
    ``std`` to each message entry.

    Args:
        mean: mean of the normal noise.
        std: standard deviation of the normal noise.

    Raises:
        ValueError: if ``std`` is negative.

    """

    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError("Standard deviation (std) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.std = std

    def make_noise(self, shape: tuple[int, ...]) -> Array:
        return iop.normal(shape=shape, mean=self.mean, std=self.std)
