import random
from abc import ABC, abstractmethod

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.parameter import X
from decent_bench.utils.types import SupportedXTypes


class AgentActivationScheme(ABC):
    """Scheme defining how agents go active/inactive over the course of the algorithm execution."""

    @abstractmethod
    def is_active(self, iteration: int) -> bool:
        """
        Whether or not the agent is active.

        Args:
            iteration: current iteration of algorithm execution

        """


class AlwaysActive(AgentActivationScheme):
    """Scheme that makes the agent always active."""

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        return True


class UniformActivationRate(AgentActivationScheme):
    """Scheme where the agent's probability of being active is uniformly distributed."""

    def __init__(self, activation_probability: float):
        self.activation_probability = activation_probability

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        return random.random() < self.activation_probability


class CompressionScheme(ABC):
    """Scheme defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress[T: SupportedXTypes](self, msg: X[T]) -> X[T]:
        """Apply compression and return a new, compressed message."""


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress[T: SupportedXTypes](self, msg: X[T]) -> X[T]:  # noqa: D102
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, n_significant_digits: int):
        self.n_significant_digits = n_significant_digits

    def compress[T: SupportedXTypes](self, msg: X[T]) -> X[T]:  # noqa: D102
        res = np.vectorize(lambda x: float(f"%.{self.n_significant_digits - 1}e" % x))(iop.to_numpy(msg))
        return iop.numpy_to_X(res, msg.framework, msg.device)


class DropScheme(ABC):
    """Scheme defining how message drops occur over the network."""

    @abstractmethod
    def should_drop(self) -> bool:
        """Whether or not to drop."""


class NoDrops(DropScheme):
    """Scheme that never drops messages."""

    def should_drop(self) -> bool:  # noqa: D102
        return False


class UniformDropRate(DropScheme):
    """Scheme that drops messages with uniform probability."""

    def __init__(self, drop_rate: float):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("Drop rate must be in [0, 1]")
        self.drop_rate = drop_rate

    def should_drop(self) -> bool:  # noqa: D102
        return random.random() < self.drop_rate


class NoiseScheme(ABC):
    """Scheme defining how noise impacts messages."""

    @abstractmethod
    def make_noise[T: SupportedXTypes](self, msg: X[T]) -> X[T]:
        """Apply noise scheme without mutating the *msg* passed in."""


class NoNoise(NoiseScheme):
    """Scheme that leaves messages untouched."""

    def make_noise[T: SupportedXTypes](self, msg: X[T]) -> X[T]:  # noqa: D102
        return msg


class GaussianNoise(NoiseScheme):
    """Scheme that applies Gaussian noise - that is, noise following a normal distribution."""

    def __init__(self, mean: float, sd: float):
        if sd < 0:
            raise ValueError("Standard deviation (sd) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.sd = sd

    def make_noise[T: SupportedXTypes](self, msg: X[T]) -> X[T]:  # noqa: D102
        return msg + iop.randn_like(msg, mean=self.mean, std=self.sd)
