from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs import EmpiricalRiskCost
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
        self._P = np.array(
            [
                [1 - inactive_to_active, inactive_to_active],
                [active_to_inactive, 1 - active_to_inactive],
            ]
        )  # transition matrix
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


class CyclicActivation(AgentActivationScheme):
    """
    Scheme where an agent cycles through active and inactive intervals.

    The agent is active for ``active_for`` iterations and inactive for ``inactive_for`` iterations in each cycle.
    If ``inactive_for`` is not provided, it defaults to ``active_for``. ``offset`` shifts the phase of the cycle,
    allowing agents to follow the same cycle with staggered active windows.

    Args:
        active_for: number of active iterations in each cycle.
        inactive_for: number of inactive iterations in each cycle. If ``None``, it defaults to ``active_for``.
        offset: phase offset applied to the cycle.

    Raises:
        ValueError: if ``active_for``, ``inactive_for``, or ``offset`` is negative, both intervals are zero, or
            ``iteration`` is negative.

    """

    def __init__(self, active_for: int, inactive_for: int | None = None, offset: int = 0):
        inactive_for = active_for if inactive_for is None else inactive_for
        if active_for < 0 or inactive_for < 0:
            raise ValueError("active_for and inactive_for must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if active_for == 0 and inactive_for == 0:
            raise ValueError("At least one of active_for or inactive_for must be positive")
        self.active_for = active_for
        self.inactive_for = inactive_for
        self.offset = offset

    def is_active(self, iteration: int) -> bool:  # noqa: D102
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        period = self.active_for + self.inactive_for
        phase = (iteration + self.offset) % period
        return phase < self.active_for


class ClientSelectionScheme(ABC):
    """
    Scheme defining how to select a subset of available clients.

    Federated algorithms call :meth:`select` once per round with the currently active clients. Implementations
    should return a subset without modifying the input sequence.
    """

    @staticmethod
    def _validate_selection_size(
        num_selected_clients: int | None,
        fraction_selected_clients: float | None,
    ) -> None:
        """
        Validate that exactly one selection-size parameter is provided.

        Raises:
            ValueError: if neither or both size parameters are provided, or if the provided value is outside the
                accepted range.

        """
        if num_selected_clients is None and fraction_selected_clients is None:
            raise ValueError("Provide num_selected_clients or fraction_selected_clients")
        if num_selected_clients is not None and fraction_selected_clients is not None:
            raise ValueError("Provide only one of num_selected_clients or fraction_selected_clients")
        if num_selected_clients is not None and num_selected_clients <= 0:
            raise ValueError("num_selected_clients must be positive")
        if fraction_selected_clients is not None and not (0 < fraction_selected_clients <= 1):
            raise ValueError("fraction_selected_clients must be in (0, 1]")

    @staticmethod
    def _resolve_num_selected_clients(
        clients: Sequence[Agent],
        num_selected_clients: int | None,
        fraction_selected_clients: float | None,
    ) -> int:
        """
        Resolve the number of selected clients for a given input client pool.

        If ``num_selected_clients`` is provided, it is capped at ``len(clients)``. If
        ``fraction_selected_clients`` is provided, at least one client is selected from a non-empty input.
        """
        if num_selected_clients is not None:
            return min(num_selected_clients, len(clients))
        k = max(1, int(fraction_selected_clients * len(clients)))  # type: ignore[operator]
        return min(k, len(clients))

    @staticmethod
    def _client_loss(client: Agent) -> float:
        """
        Evaluate a client's current local loss for selection.

        Empirical-risk costs are evaluated on all local samples to avoid consuming a stochastic mini-batch during
        client selection.
        """
        if isinstance(client.cost, EmpiricalRiskCost):
            return client.cost.function(client.x, indices="all")
        return client.cost.function(client.x)

    @abstractmethod
    def select(
        self,
        clients: Sequence[Agent],
        iteration: int,
    ) -> list[Agent]:
        """
        Select a subset of available clients.

        Args:
            clients: available clients
            iteration: current iteration of algorithm execution

        """


class UniformSelection(ClientSelectionScheme):
    """
    Uniform client selection.

    The scheme samples clients uniformly without replacement. It selects either a fixed number of clients or a fraction
    of the clients passed to :meth:`select`.

    Args:
        num_selected_clients: number of provided clients to sample.
        fraction_selected_clients: fraction of provided clients to sample.

    Raises:
        ValueError: if the selection size is invalid.

    """

    def __init__(
        self,
        *,
        num_selected_clients: int | None = None,
        fraction_selected_clients: float | None = None,
    ) -> None:
        self._validate_selection_size(num_selected_clients, fraction_selected_clients)
        self.num_selected_clients = num_selected_clients
        self.fraction_selected_clients = fraction_selected_clients

    def select(  # noqa: D102
        self,
        clients: Sequence[Agent],
        iteration: int,  # noqa: ARG002
    ) -> list[Agent]:
        if not clients:
            return []
        k = self._resolve_num_selected_clients(clients, self.num_selected_clients, self.fraction_selected_clients)
        if k == len(clients):
            return list(clients)
        return random.sample(list(clients), k)


class DataSizeSelection(ClientSelectionScheme):
    r"""
    Data-size weighted client selection :footcite:p:`Scheme_FedSampling`.

    The scheme samples clients without replacement with probability proportional to each client's local data size.
    The sampling probability for client :math:`i` is

    .. math::

        p_i = \frac{n_i}{\sum_{j \in \mathcal{C}} n_j},

    where :math:`n_i` is the client's inferred local data size and :math:`\mathcal{C}` is the client pool passed to
    :meth:`select`.

    Args:
        num_selected_clients: number of provided clients to sample.
        fraction_selected_clients: fraction of provided clients to sample.

    Raises:
        ValueError: if the selection size is invalid or any client's data size cannot be inferred.

    .. footbibliography::

    """

    def __init__(
        self,
        *,
        num_selected_clients: int | None = None,
        fraction_selected_clients: float | None = None,
    ) -> None:
        self._validate_selection_size(num_selected_clients, fraction_selected_clients)
        self.num_selected_clients = num_selected_clients
        self.fraction_selected_clients = fraction_selected_clients

    def select(  # noqa: D102
        self,
        clients: Sequence[Agent],
        iteration: int,  # noqa: ARG002
    ) -> list[Agent]:
        if not clients:
            return []
        k = self._resolve_num_selected_clients(clients, self.num_selected_clients, self.fraction_selected_clients)
        if k == len(clients):
            return list(clients)

        clients_list = list(clients)
        data_sizes = np.array(
            [infer_client_data_size(client) for client in clients_list],
            dtype=np.float64,
        )
        probabilities = data_sizes / data_sizes.sum()
        selected_indices = iop.rng_numpy().choice(len(clients_list), size=k, replace=False, p=probabilities)
        return [clients_list[int(index)] for index in selected_indices]


class FairSelection(ClientSelectionScheme):
    r"""
    Fair client selection inspired by fairness-aware client selection :footcite:p:`Scheme_FairFedCS`.

    The scheme is a simplified count-based fairness rule that prioritizes clients with fewer past selections. It acts
    as a participation-balancing exploration rule: clients selected fewer times are prioritized so that the algorithm
    keeps exploring under-represented clients instead of repeatedly selecting the same ones.
    At round :math:`t`, let :math:`c_i(t)` be the number of previous rounds in which client :math:`i` was selected.
    For the client pool :math:`\mathcal{C}_t` passed to :meth:`select`, the selected set is

    .. math::

        S_t \in \operatorname{arg\,min}_{S \subseteq \mathcal{C}_t,\ |S| = m}
        \sum_{i \in S} c_i(t),

    where :math:`m` is the resolved number of selected clients. Clients with the same count keep the order in which
    they were provided to :meth:`select`. After selecting :math:`S_t`, the counts are updated as

    .. math::

        c_i(t+1) = c_i(t) + \mathbf{1}\{i \in S_t\}.

    Args:
        num_selected_clients: number of provided clients to sample.
        fraction_selected_clients: fraction of provided clients to sample.

    Raises:
        ValueError: if the selection size is invalid.

    .. footbibliography::

    """

    def __init__(
        self,
        *,
        num_selected_clients: int | None = None,
        fraction_selected_clients: float | None = None,
    ) -> None:
        self._validate_selection_size(num_selected_clients, fraction_selected_clients)
        self.num_selected_clients = num_selected_clients
        self.fraction_selected_clients = fraction_selected_clients
        self._selection_counts: dict[Agent, int] = {}

    def select(  # noqa: D102
        self,
        clients: Sequence[Agent],
        iteration: int,  # noqa: ARG002
    ) -> list[Agent]:
        if not clients:
            return []
        k = self._resolve_num_selected_clients(clients, self.num_selected_clients, self.fraction_selected_clients)
        if k == len(clients):
            selected_clients = list(clients)
        else:
            clients_list = list(clients)
            selected_clients = sorted(clients_list, key=lambda client: self._selection_counts.get(client, 0))[:k]

        for client in selected_clients:
            self._selection_counts[client] = self._selection_counts.get(client, 0) + 1
        return selected_clients


class HighLossSelection(ClientSelectionScheme):
    r"""
    High-loss client selection inspired by Power-of-Choice :footcite:p:`Scheme_PowerOfChoice`.

    The scheme evaluates each client's local loss at its current local state ``x`` and selects the clients with
    highest loss, breaking ties at random. Unlike the Power-of-Choice strategy, this scheme does not trigger extra
    communication to evaluate losses at the current server model.

    At round :math:`t`, for the client pool :math:`\mathcal{C}_t` passed to :meth:`select`, the selected set is

    .. math::

        S_t \in \operatorname{arg\,max}_{S \subseteq \mathcal{C}_t,\ |S| = m}
        \sum_{i \in S} F_i(x_i),

    where :math:`m` is the resolved number of selected clients, :math:`F_i` is client :math:`i`'s local cost, and
    :math:`x_i` is its current local state.

    Args:
        num_selected_clients: number of provided clients to sample.
        fraction_selected_clients: fraction of provided clients to sample.

    Raises:
        ValueError: if the selection size is invalid.
        RuntimeError: if any evaluated client's ``x`` has not been initialized.

    .. footbibliography::

    """

    def __init__(
        self,
        *,
        num_selected_clients: int | None = None,
        fraction_selected_clients: float | None = None,
    ) -> None:
        self._validate_selection_size(num_selected_clients, fraction_selected_clients)
        self.num_selected_clients = num_selected_clients
        self.fraction_selected_clients = fraction_selected_clients

    def select(  # noqa: D102
        self,
        clients: Sequence[Agent],
        iteration: int,  # noqa: ARG002
    ) -> list[Agent]:
        if not clients:
            return []

        n_selected_clients = self._resolve_num_selected_clients(
            clients, self.num_selected_clients, self.fraction_selected_clients
        )
        if n_selected_clients == len(clients):
            return list(clients)

        clients_list = list(clients)
        losses = [self._client_loss(client) for client in clients_list]
        tie_breakers = iop.rng_numpy().permutation(len(clients_list))
        ranked_indices = sorted(
            range(len(clients_list)),
            key=lambda index: (-losses[index], int(tie_breakers[index])),
        )
        return [clients_list[index] for index in ranked_indices[:n_selected_clients]]


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


class StochasticQuantization(CompressionScheme):
    r"""
    Stochastic quantization used in QSGD :footcite:p:`Scheme_QSGD`.

    The scheme quantizes each coordinate using ``n_levels`` stochastic levels scaled by the message norm. This keeps the
    compressed message unbiased in expectation while preserving the original message shape. Given a message
    :math:`x` and :math:`s=\texttt{n\_levels}`, the quantizer computes

    .. math::

        a_i = \frac{s |x_i|}{\lVert x \rVert_2}, \qquad
        \ell_i = \lfloor a_i \rfloor, \qquad
        p_i = a_i - \ell_i.

    The quantization level is sampled as

    .. math::

        \xi_i =
        \begin{cases}
            \ell_i + 1, & \text{with probability } p_i, \\
            \ell_i, & \text{with probability } 1 - p_i,
        \end{cases}

    and the compressed coordinate is

    .. math::

        Q_s(x_i) = \lVert x \rVert_2 \operatorname{sign}(x_i) \frac{\xi_i}{s}.

    Args:
        n_levels: number of stochastic quantization levels. Larger values give a finer quantization grid and usually
            lower quantization error. Smaller values give coarser quantization and stronger compression noise.

    Raises:
        ValueError: if ``n_levels`` is not positive.

    Warning:
        This scheme computes the :math:`\ell_2` norm of each message. This can be computationally expensive for large
        messages or when messages live on accelerator devices.

    .. footbibliography::

    """

    def __init__(self, n_levels: int):
        if n_levels <= 0:
            raise ValueError("`n_levels` must be a positive integer")
        self.n_levels = n_levels

    def compress(self, msg: Array) -> Array:  # noqa: D102
        msg_norm = float(iop.norm(msg))
        if msg_norm == 0:
            return iop.zeros_like(msg)

        msg_np = iop.to_numpy(msg, dtype=np.float64)
        magnitudes = np.abs(msg_np)
        signs = np.sign(msg_np)
        scaled_magnitudes = self.n_levels * magnitudes / msg_norm
        lower_levels = np.floor(scaled_magnitudes)
        probabilities = scaled_magnitudes - lower_levels
        quantized_levels = lower_levels + (iop.rng_numpy().random(size=magnitudes.shape) < probabilities)
        compressed_msg = msg_norm * signs * quantized_levels / self.n_levels
        return iop.to_array_like(compressed_msg, msg)


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
