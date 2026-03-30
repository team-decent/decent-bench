import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs import LinearRegressionCost


def _make_dataset(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array([float(i)]), np.array([0.0])) for i in range(n_samples)]


def test_batch_sampling_no_reselection_while_enough_unseen_samples_remain() -> None:
    iop.set_seed(7)
    cost = LinearRegressionCost(dataset=_make_dataset(10), batch_size=3)

    seen: set[int] = set()
    for _ in range(3):
        batch = cost._sample_batch_indices("batch")  # noqa: SLF001
        assert len(batch) == 3
        assert len(set(batch)) == 3
        assert set(batch).isdisjoint(seen)
        seen.update(batch)

    # After four draws of size 3 over 10 samples, all samples should have been seen at least once.
    batch_4 = cost._sample_batch_indices("batch")  # noqa: SLF001
    assert len(batch_4) == 3
    assert len(set(batch_4)) == 3
    seen.update(batch_4)
    assert seen == set(range(10))


def test_batch_sampling_with_large_batch_uses_full_dataset() -> None:
    cost = LinearRegressionCost(dataset=_make_dataset(5), batch_size="all")
    assert cost._sample_batch_indices("batch") == [0, 1, 2, 3, 4]  # noqa: SLF001
