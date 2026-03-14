import pytest

from decent_bench.distributed_algorithms import (
    ADMM,
    ATC,
    ATCTracking,
    ATG,
    DGD,
    DLM,
    ED,
    EXTRA,
    FedAvg,
    NIDS,
    AugDGM,
    SimpleGT,
    WangElia,
)


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (FedAvg, {"iterations": 10, "step_size": 0.1}),
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
    ],
)
def test_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)

    assert algorithm.iterations == 10
    assert isinstance(algorithm.name, str)
