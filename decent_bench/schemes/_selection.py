from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from decent_array import interoperability as iop

from decent_bench.agents._utils import infer_client_data_size
from decent_bench.costs import EmpiricalRiskCost

if TYPE_CHECKING:
    from decent_bench.agents import Agent


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

    def select(
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

    def select(
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
        selected_indices = iop.get_numpy_rng().choice(len(clients_list), size=k, replace=False, p=probabilities)
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

    def select(
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

    def select(
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
        tie_breakers = iop.get_numpy_rng().permutation(len(clients_list))
        ranked_indices = sorted(
            range(len(clients_list)),
            key=lambda index: (-losses[index], int(tie_breakers[index])),
        )
        return [clients_list[index] for index in ranked_indices[:n_selected_clients]]
