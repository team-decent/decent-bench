from __future__ import annotations

from abc import ABC, abstractmethod

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


# later remove framework and device when iop refactored
class NoiseScheme(ABC):
    """Scheme defining the noise impacting messages."""

    @abstractmethod
    def make_noise(
        self, shape: tuple[int, ...], framework: SupportedFrameworks, device: SupportedDevices
    ) -> Array | None:
        """Generate noise array of given shape (None if no noise)."""


class NoNoise(NoiseScheme):
    """Scheme representing transmission without noise."""

    def make_noise(
        self,
        _: tuple[int, ...],
        _framework: SupportedFrameworks,
        _device: SupportedDevices,
    ) -> Array | None:
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

    def make_noise(self, shape: tuple[int, ...], framework: SupportedFrameworks, device: SupportedDevices) -> Array:
        return iop.normal(framework=framework, device=device, shape=shape, mean=self.mean, std=self.std)
