from dataclasses import dataclass
from typing import TYPE_CHECKING

from decent_bench.utils._tags import tags

from ._fed_opt import FedOpt

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


@tags("federated")
@dataclass(eq=False)
class FedAdagrad(FedOpt):
    r"""
    FedAdagrad uses local SGD on clients and an Adagrad-style adaptive server update :footcite:p:`Alg_FedOpt`.

    Each selected client starts from the broadcast global model :math:`\mathbf{x}_t` and performs
    ``num_local_epochs`` local SGD steps with client step size ``step_size``.

    .. math::
        \mathbf{x}_{i, t}^{(k+1)} = \mathbf{x}_{i, t}^{(k)} - \eta_l
        \nabla f_i(\mathbf{x}_{i, t}^{(k)}).

    The final client model defines the uploaded delta

    .. math::
        \delta_i^t = \mathbf{x}_{i, t}^{(K)} - \mathbf{x}_t.

    The server aggregates client model deltas uniformly over the participating clients:

    .. math::
        \Delta_t = \frac{1}{|S_t|} \sum_{i \in S_t} \delta_i^t.

    FedAdagrad then updates its moment buffers and global model as

    .. math::
        \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \Delta_t

    .. math::
        \mathbf{v}_t = \mathbf{v}_{t-1} + \Delta_t^2

    .. math::
        \mathbf{x}_{t+1} = \mathbf{x}_t + \eta
        \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \tau}.

    Here :math:`\eta_l` is the client learning rate (``step_size``), :math:`K` is the number of local SGD steps
    (``num_local_epochs``), :math:`\eta` is the server learning rate (``server_step_size``), :math:`\beta_1` is the
    first-moment coefficient, :math:`\tau` is the numerical stability term, and :math:`S_t` is the set of clients
    whose uploads are actually received in round :math:`t`. Aggregation is always uniform across the received clients.
    Costs that preserve the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use mini-batch local updates;
    generic costs use their usual full-gradient updates.

    .. footbibliography::
    """

    name: str = "FedAdagrad"

    def _update_second_moment(self, second_moment: "Array", average_delta: "Array") -> "Array":
        return second_moment + (average_delta * average_delta)
