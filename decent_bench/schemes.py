import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    from decent_bench.agents import Agent


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


class ClientSelectionScheme(ABC):
    """Scheme defining how to select a subset of available clients."""

    @abstractmethod
    def select(self, clients: Sequence["Agent"], iteration: int) -> list["Agent"]:
        """
        Select a subset of available clients.

        Args:
            clients: available clients
            iteration: current iteration of algorithm execution

        """


class UniformClientSelection(ClientSelectionScheme):
    """Uniformly sample clients without replacement."""

    def __init__(
        self,
        *,
        clients_per_round: int | None = None,
        client_fraction: float | None = None,
        seed: int | None = None,
    ) -> None:
        if clients_per_round is None and client_fraction is None:
            raise ValueError("Provide clients_per_round or client_fraction")
        if clients_per_round is not None and client_fraction is not None:
            raise ValueError("Provide only one of clients_per_round or client_fraction")
        if clients_per_round is not None and clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive")
        if client_fraction is not None and not (0 < client_fraction <= 1):
            raise ValueError("client_fraction must be in (0, 1]")
        self.clients_per_round = clients_per_round
        self.client_fraction = client_fraction
        self._rng = random.Random(seed)

    def select(self, clients: Sequence["Agent"], iteration: int) -> list["Agent"]:  # noqa: D102, ARG002
        if not clients:
            return []
        if self.clients_per_round is not None:
            k = min(self.clients_per_round, len(clients))
        else:
            k = max(1, int(self.client_fraction * len(clients)))  # type: ignore[operator]
            k = min(k, len(clients))
        if k >= len(clients):
            return list(clients)
        return self._rng.sample(list(clients), k)


class CompressionScheme(ABC):
    """Scheme defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress(self, msg: Array) -> Array:
        """Apply compression and return a new, compressed message."""


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: Array) -> Array:  # noqa: D102
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, n_significant_digits: int):
        self.n_significant_digits = n_significant_digits

    def compress(self, msg: Array) -> Array:  # noqa: D102
        res = np.vectorize(lambda x: float(f"%.{self.n_significant_digits - 1}e" % x))(iop.to_numpy(msg))
        return iop.to_array_like(res, msg)


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
    def make_noise(self, msg: Array) -> Array:
        """Apply noise scheme without mutating the *msg* passed in."""


class NoNoise(NoiseScheme):
    """Scheme that leaves messages untouched."""

    def make_noise(self, msg: Array) -> Array:  # noqa: D102
        return msg


class GaussianNoise(NoiseScheme):
    """Scheme that applies Gaussian noise - that is, noise following a normal distribution."""

    def __init__(self, mean: float, sd: float):
        if sd < 0:
            raise ValueError("Standard deviation (sd) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.sd = sd

    def make_noise(self, msg: Array) -> Array:  # noqa: D102
        return msg + iop.randn_like(msg, mean=self.mean, std=self.sd)
