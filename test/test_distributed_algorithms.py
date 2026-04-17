import pytest

from decent_bench.algorithms.decentralized import (
    ADMM,
    ATC,
    ATG,
    DGD,
    DLM,
    ED,
    EXTRA,
    GT_SAGA,
    GT_SARAH,
    GT_VR,
    KGT,
    LED,
    LT_ADMM,
    LT_ADMM_VR,
    NIDS,
    ATCTracking,
    AugDGM,
    DiNNO,
    ProxSkip,
    SimpleGT,
    WangElia,
)
from decent_bench.algorithms.federated import FedAvg, FedProx


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (FedAvg, {"iterations": 10, "step_size": 0.1}),
        (FedProx, {"iterations": 10, "step_size": 0.1}),
        (DGD, {"iterations": 10, "step_size": 0.1}),
        (ATC, {"iterations": 10, "step_size": 0.1}),
        (SimpleGT, {"iterations": 10, "step_size": 0.1}),
        (ED, {"iterations": 10, "step_size": 0.1}),
        (AugDGM, {"iterations": 10, "step_size": 0.1}),
        (WangElia, {"iterations": 10, "step_size": 0.1}),
        (EXTRA, {"iterations": 10, "step_size": 0.1}),
        (ATCTracking, {"iterations": 10, "step_size": 0.1}),
        (NIDS, {"iterations": 10, "step_size": 0.1}),
        (ADMM, {"iterations": 10, "rho": 1.0, "alpha": 0.5}),
        (ATG, {"iterations": 10, "rho": 1.0, "alpha": 0.5}),
        (DLM, {"iterations": 10, "step_size": 0.1, "penalty": 1.0}),
        (DiNNO, {"iterations": 10, "step_size": 0.1, "local_steps": 5}),
        (GT_VR, {"iterations": 10, "step_size": 0.1, "snapshot_prob": 0.5}),
        (GT_SAGA, {"iterations": 10, "step_size": 0.1}),
        (GT_SARAH, {"iterations": 10, "step_size": 0.1, "local_steps": 5}),
        (KGT, {"iterations": 10, "step_size": 0.1, "local_steps": 5}),
        (LED, {"iterations": 10, "step_size": 0.1, "local_steps": 5}),
        (LT_ADMM, {"iterations": 10, "step_size": 0.1, "local_steps": 5}),
        (LT_ADMM_VR, {"iterations": 10, "step_size": 0.1, "local_steps": 5, "v2": False}),
        (LT_ADMM_VR, {"iterations": 10, "step_size": 0.1, "local_steps": 5, "v2": True}),
        (ProxSkip, {"iterations": 10, "step_size": 0.1, "comm_probability": 0.5}),
    ],
)
def test_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)

    assert algorithm.iterations == 10
    assert isinstance(algorithm.name, str)
