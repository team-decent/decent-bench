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


class MarkovChainActivation(AgentActivationScheme):
    """
    Scheme modeling activation with a 2-state Markov chain.

    The scheme models activation with a 2-state (active and inactive) Markov chain. The agent transitions
    between the two states with the given probabilities.

    Args:
        inactive_to_active: transition probability from inactive to active
        active_to_inactive: transition probability from active to inactive

    Raises:
        ValueError: if `inactive_to_active` or `active_to_inactive` are not in :math:`[0, 1]`

    """

    def __init__(self, inactive_to_active: float = 0.5, active_to_inactive: float = 0.5):
        if (inactive_to_active < 0 or inactive_to_active > 1) or (active_to_inactive < 0 or active_to_inactive > 1):
            raise ValueError("Transition probabilities must be in [0, 1]")
        self.inactive_to_active = inactive_to_active
        self.active_to_inactive = active_to_inactive
        self._states = np.array([0, 1])  # inactive = 0, active = 1
        self._P = np.array([
            [1 - inactive_to_active, inactive_to_active],
            [active_to_inactive, 1 - active_to_inactive],
        ])  # transition matrix
        self._current_state = iop.rng_numpy().choice(self._states, p=[0, 1])

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        self._current_state = iop.rng_numpy().choice(
            self._states,
            p=self._P[self._current_state],
        )  # evolve the Markov chain

        return bool(self._current_state)


class PoissonActivation(AgentActivationScheme):
    """
    Scheme modeling activation at random intervals determined by a Poisson distribution.

    The agent activates at random intervals of length sampled from a Poisson distribution of given mean.

    Args:
        mean_interval: mean interval of inactivity

    Raises:
        ValueError: if `mean_interval` is negative

    """

    def __init__(self, mean_interval: float = 1.0):
        if mean_interval < 0:
            raise ValueError("`mean_interval` must be non-negative")
        self.mean_interval = mean_interval
        self._countdown = int(iop.rng_numpy().poisson(self.mean_interval))

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        if self._countdown == 0:
            self._countdown = int(iop.rng_numpy().poisson(self.mean_interval))
            return True
        self._countdown -= 1
        return False


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
        return random.sample(list(clients), k)


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
        res = np.vectorize(lambda x: float(format(x, f".{self.n_significant_digits - 1}e")))(iop.to_numpy(msg))
        return iop.to_array_like(res, msg)


class TopK(CompressionScheme):
    """
    Top-k compression which transmits only the k elements with largest magnitude.

    Raises:
        ValueError: if ``k <= 0``

    Note:
        If the message has fewer than ``k`` elements, no compression is applied.

    """

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(f"`k` must be a positive integer, got {k}")
        self.k = k

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_np = np.copy(iop.to_numpy(msg))
        k = min(self.k, msg_np.size)
        idx = np.argpartition(np.abs(msg_np), -k, axis=None)[-k:]
        mask = np.isin(np.arange(msg_np.size), idx).reshape(msg_np.shape)
        msg_np[~mask] = 0
        return iop.to_array_like(msg_np, msg)


class RandK(CompressionScheme):
    """
    Rand-k compression which transmits only k randomly selected elements.

    Raises:
        ValueError: if ``k <= 0``

    Note:
        If the message has fewer than ``k`` elements, no compression is applied.

    """

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("`k` must be a positive integer")
        self.k = k

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_np = np.copy(iop.to_numpy(msg))
        k = min(self.k, msg_np.size)
        idx = iop.rng_numpy().choice(msg_np.size, size=k, replace=False)
        mask = np.isin(np.arange(msg_np.size), idx).reshape(msg_np.shape)
        msg_np[~mask] = 0
        return iop.to_array_like(msg_np, msg)


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


class GilbertElliott(DropScheme):
    """
    Drop scheme based on the Gilbert-Elliott model :footcite:p:`Scheme_GilbertElliott`.

    The Gilbert-Elliott model is characterized by a Markov chain with two states (good and bad), which
    can stay the same or transition into each other. In the bad state message drops occur with probability
    `drop_rate`, while in the good state no message drops occur.

    Args:
        drop_rate: message drop rate while in the bad state
        bad_to_good: transition probability from bad to good state
        good_to_bad: transition probability from good to bad state

    Raises:
        ValueError: if `drop_rate`, `bad_to_good` or `good_to_bad` are not in :math:`[0, 1]`

    .. footbibliography::

    """

    def __init__(self, drop_rate: float, bad_to_good: float = 0.5, good_to_bad: float = 0.5):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("Drop rate must be in [0, 1]")
        if (bad_to_good < 0 or bad_to_good > 1) or (good_to_bad < 0 or good_to_bad > 1):
            raise ValueError("Transition probabilities `bad_to_good` and `good_to_bad` must be in [0, 1]")
        self.drop_rate = drop_rate
        self.bad_to_good = bad_to_good
        self.good_to_bad = good_to_bad
        self._states = np.array([0, 1])  # good = 0, bad = 1
        self._P = np.array([[1 - good_to_bad, good_to_bad], [bad_to_good, 1 - bad_to_good]])  # transition matrix
        self._current_state = iop.rng_numpy().choice(self._states)  # initialize uniformly at random

    def should_drop(self) -> bool:  # noqa: D102
        self._current_state = iop.rng_numpy().choice(
            self._states, p=self._P[self._current_state]
        )  # evolve the Markov chain

        return iop.rng_numpy().random() < self.drop_rate if self._current_state else False


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

    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError("Standard deviation (std) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.std = std

    def make_noise(self, msg: Array) -> Array:  # noqa: D102
        return msg + iop.normal_like(msg, mean=self.mean, std=self.std)
