from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decent_bench.metrics import Metric


def _flatten_plot_metrics(plot_metrics: list["Metric"] | list[list["Metric"]]) -> list["Metric"]:
    if any(isinstance(metric, list) for metric in plot_metrics):
        return [metric for group in plot_metrics for metric in group]  # type: ignore[union-attr]

    return plot_metrics  # type: ignore[return-value]


def _find_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return sorted(duplicates)
