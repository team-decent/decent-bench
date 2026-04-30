from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.agent_utils import infer_client_data_size
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    from decent_bench.agents import Agent


class AgentActivationScheme(ABC):
    """
    Scheme defining how agents go active/inactive over the course of the algorithm execution.

    Activation schemes are attached to agents by networks and are queried during algorithm execution.
    """

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
    """
    Scheme where the agent is active with fixed probability.

    Each call samples an independent Bernoulli event with probability ``activation_probability``.

    Args:
        activation_probability: probability that the agent is active at a queried iteration.

    Raises:
        ValueError: if ``activation_probability`` is not in :math:`[0, 1]`.

    """

    def __init__(self, activation_probability: float):
        if activation_probability < 0 or activation_probability > 1:
            raise ValueError("activation_probability must be in [0, 1]")
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
    """
    Scheme defining how to select a subset of available clients.

    Federated algorithms call :meth:`select` once per round with the currently active clients. Implementations may be
    stateful, but they should not mutate the client sequence they receive.
    """

    @staticmethod
    def _validate_selection_size(
        clients_per_round: int | None,
        client_fraction: float | None,
    ) -> None:
        if clients_per_round is None and client_fraction is None:
            raise ValueError("Provide clients_per_round or client_fraction")
        if clients_per_round is not None and client_fraction is not None:
            raise ValueError("Provide only one of clients_per_round or client_fraction")
        if clients_per_round is not None and clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive")
        if client_fraction is not None and not (0 < client_fraction <= 1):
            raise ValueError("client_fraction must be in (0, 1]")

    @staticmethod
    def _num_selected_clients(
        clients: Sequence[Agent],
        clients_per_round: int | None,
        client_fraction: float | None,
    ) -> int:
        if clients_per_round is not None:
            return min(clients_per_round, len(clients))
        k = max(1, int(client_fraction * len(clients)))  # type: ignore[operator]
        return min(k, len(clients))

    @abstractmethod
    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:
        """
        Select a subset of available clients.

        Args:
            clients: available clients
            iteration: current iteration of algorithm execution

        """


class UniformClientSelection(ClientSelectionScheme):
    """
    Uniform client selection.

    The scheme samples clients uniformly without replacement. It selects either a fixed number of clients or a fraction
    of the clients passed to :meth:`select`.

    Args:
        clients_per_round: number of clients to sample per round.
        client_fraction: fraction of provided clients to sample per round.

    Raises:
        ValueError: if the selection size is invalid.

    """

    def __init__(
        self,
        *,
        clients_per_round: int | None = None,
        client_fraction: float | None = None,
    ) -> None:
        self._validate_selection_size(clients_per_round, client_fraction)
        self.clients_per_round = clients_per_round
        self.client_fraction = client_fraction

    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:  # noqa: D102, ARG002
        if not clients:
            return []
        k = self._num_selected_clients(clients, self.clients_per_round, self.client_fraction)
        if k >= len(clients):
            return list(clients)
        return random.sample(list(clients), k)


class DataSizeClientSelection(ClientSelectionScheme):
    """
    Data-size weighted client selection :footcite:p:`Scheme_FedSampling`.

    The scheme samples clients without replacement with probability proportional to each client's local data size.

    Args:
        clients_per_round: number of clients to sample per round.
        client_fraction: fraction of provided clients to sample per round.
        data_size_key: key used to read explicit data sizes from ``Agent.data``.

    Raises:
        ValueError: if the selection size is invalid or any client's data size cannot be inferred.

    .. footbibliography::

    """

    def __init__(
        self,
        *,
        clients_per_round: int | None = None,
        client_fraction: float | None = None,
        data_size_key: str = "n_samples",
    ) -> None:
        self._validate_selection_size(clients_per_round, client_fraction)
        self.clients_per_round = clients_per_round
        self.client_fraction = client_fraction
        self.data_size_key = data_size_key

    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:  # noqa: D102, ARG002
        if not clients:
            return []
        k = self._num_selected_clients(clients, self.clients_per_round, self.client_fraction)
        clients_list = list(clients)
        if k >= len(clients_list):
            return clients_list

        data_sizes = np.array(
            [infer_client_data_size(client, self.data_size_key) for client in clients_list],
            dtype=np.float64,
        )
        probabilities = data_sizes / data_sizes.sum()
        selected_indices = iop.rng_numpy().choice(len(clients_list), size=k, replace=False, p=probabilities)
        return [clients_list[int(index)] for index in selected_indices]


class ParticipationFairClientSelection(ClientSelectionScheme):
    """
    Participation-fair client selection :footcite:p:`Scheme_FairFedCS`.

    The scheme prioritizes clients with fewer past selections, breaking ties at random.

    Args:
        clients_per_round: number of clients to sample per round.
        client_fraction: fraction of provided clients to sample per round.

    Raises:
        ValueError: if the selection size is invalid.

    .. footbibliography::

    """

    def __init__(
        self,
        *,
        clients_per_round: int | None = None,
        client_fraction: float | None = None,
    ) -> None:
        self._validate_selection_size(clients_per_round, client_fraction)
        self.clients_per_round = clients_per_round
        self.client_fraction = client_fraction
        self._selection_counts: dict[Agent, int] = {}

    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:  # noqa: D102, ARG002
        if not clients:
            return []
        k = self._num_selected_clients(clients, self.clients_per_round, self.client_fraction)
        clients_list = list(clients)
        if k >= len(clients_list):
            selected_clients = clients_list
        else:
            tie_breakers = iop.rng_numpy().permutation(len(clients_list))
            ranked_indices = sorted(
                range(len(clients_list)),
                key=lambda index: (self._selection_counts.get(clients_list[index], 0), int(tie_breakers[index])),
            )
            selected_clients = [clients_list[index] for index in ranked_indices[:k]]

        for client in selected_clients:
            self._selection_counts[client] = self._selection_counts.get(client, 0) + 1
        return selected_clients


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
    """
    Scheme that rounds each element in a message to *significant_digits*.

    The scheme rounds finite, non-zero message entries to ``n_significant_digits`` significant digits and leaves zeros,
    infinities, and NaNs unchanged.

    Args:
        n_significant_digits: number of significant digits to retain.

    Raises:
        ValueError: if ``n_significant_digits`` is not positive.

    """

    def __init__(self, n_significant_digits: int):
        if n_significant_digits <= 0:
            raise ValueError("`n_significant_digits` must be a positive integer")
        self.n_significant_digits = n_significant_digits

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_np = iop.to_numpy(msg, dtype=np.float64)

        # Round finite non-zero entries to the requested number of significant digits.
        mask = np.isfinite(msg_np) & (msg_np != 0)
        if np.any(mask):
            magnitudes = np.floor(np.log10(np.abs(msg_np[mask])))
            scale = np.power(10.0, self.n_significant_digits - 1 - magnitudes)
            msg_np[mask] = np.round(msg_np[mask] * scale) / scale

        return iop.to_array_like(msg_np, msg)


class TopK(CompressionScheme):
    """
    Top-k compression which transmits only a subset of elements with largest magnitude.

    The parameter ``k`` can be either:

    - an ``int``: transmit exactly ``k`` elements, or
    - a ``float`` in :math:`(0, 1]`: transmit a fraction ``k`` of elements.

    Message size is preserved by transmitting zeros in place of non-transmitted elements.

    Raises:
        ValueError: if ``k`` is a float and not in :math:`(0, 1]`
        ValueError: if ``k`` is an int and less than 1

    Note:
        If ``k * n_elements < 1``, at least one element is still transmitted.

    """

    def __init__(self, k: float):
        if isinstance(k, int):
            if k < 1:
                raise ValueError(f"If `k` is an integer, it must be at least 1, got {k}")
        elif k <= 0 or k > 1:
            raise ValueError(f"If `k` is a float, it must be in (0, 1], got {k}")
        self.k = k
        self.is_integer_k = isinstance(self.k, int)

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_np = iop.to_numpy(msg)
        n_elements = msg_np.size
        k_count = min(int(self.k), n_elements) if self.is_integer_k else max(1, int(np.ceil(self.k * n_elements)))

        flat_msg = msg_np.reshape(-1)
        idx = np.argpartition(np.abs(flat_msg), -k_count)[-k_count:]
        compressed_flat = np.zeros_like(flat_msg)
        compressed_flat[idx] = flat_msg[idx]

        return iop.to_array_like(compressed_flat.reshape(msg_np.shape), msg)


class RandK(CompressionScheme):
    """
    Rand-k compression which transmits only a random subset of elements.

    The parameter ``k`` can be either:

    - an ``int``: transmit exactly ``k`` elements chosen uniformly at random (without replacement), or
    - a ``float`` in :math:`(0, 1]`: transmit a fraction ``k`` of elements.

    Message size is preserved by transmitting zeros in place of non-transmitted elements.

    Raises:
        ValueError: if ``k`` is a float and not in :math:`(0, 1]`
        ValueError: if ``k`` is an int and less than 1

    Note:
        If ``k * n_elements < 1``, at least one element is still transmitted.

    """

    def __init__(self, k: float):
        if isinstance(k, int):
            if k < 1:
                raise ValueError(f"`k` must be at least 1 if an integer, got {k}")
        elif k <= 0 or k > 1:
            raise ValueError(f"`k` must be in (0, 1], got {k}")
        self.k = k
        self.is_integer_k = isinstance(self.k, int)

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_np = iop.to_numpy(msg)
        n_elements = msg_np.size
        k_count = min(int(self.k), n_elements) if self.is_integer_k else max(1, int(np.ceil(self.k * n_elements)))

        flat_msg = msg_np.reshape(-1)
        idx = iop.rng_numpy().choice(n_elements, size=k_count, replace=False)
        compressed_flat = np.zeros_like(flat_msg)
        compressed_flat[idx] = flat_msg[idx]

        return iop.to_array_like(compressed_flat.reshape(msg_np.shape), msg)


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
    """
    Scheme that drops messages with uniform probability.

    Each call samples an independent Bernoulli event with probability ``drop_rate``.

    Args:
        drop_rate: probability that a message is dropped.

    Raises:
        ValueError: if ``drop_rate`` is not in :math:`[0, 1]`.

    """

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
    """
    Scheme that applies additive Gaussian noise.

    The scheme adds independent noise sampled from a normal distribution with mean ``mean`` and standard deviation
    ``std`` to each message entry.

    Args:
        mean: mean of the Gaussian noise.
        std: standard deviation of the Gaussian noise.

    Raises:
        ValueError: if ``std`` is negative.

    """

    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError("Standard deviation (std) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.std = std

    def make_noise(self, msg: Array) -> Array:  # noqa: D102
        return msg + iop.normal_like(msg, mean=self.mean, std=self.std)
