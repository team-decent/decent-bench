from abc import ABC, abstractmethod

from numpy import float64
from numpy.random import MT19937, Generator
from numpy.typing import NDArray


class NoiseScheme(ABC):
    """Scheme defining how noise impacts messages."""

    @abstractmethod
    def make_noise(self, msg: NDArray[float64]) -> NDArray[float64]:
        """Apply noise scheme without mutating the *msg* passed in."""


class NoNoise(NoiseScheme):
    """Scheme that leaves messages untouched."""

    def make_noise(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102
        return msg


class GaussianNoise(NoiseScheme):
    """Scheme that applies Gaussian noise - that is, noise following a normal distribution."""

    def __init__(self, mean: float, sd: float):
        if sd < 0:
            raise ValueError("Standard deviation (sd) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.sd = sd
        self.generator = Generator(MT19937())

    def make_noise(self, msg: NDArray[float64]) -> NDArray[float64]:  # noqa: D102
        return msg + self.generator.normal(self.mean, self.sd, msg.shape)
