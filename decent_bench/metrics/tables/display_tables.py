from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import tabulate as tb

from decent_bench.costs import EmpiricalRiskCost
from decent_bench.metrics.metric_library import FunctionCalls, GradientCalls, HessianCalls, ProximalCalls
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import MetricResult

SCALE_METRICS = (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls)


def display_tables(
    metrics_result: "MetricResult",
    table_fmt: Literal["grid", "latex"] = "grid",
    scale_compute: float = 1.0,
    table_path: Path | None = None,
) -> None:
    """Display table of metrics as mean ± std across trials."""
    if metrics_result.table_metrics is None or metrics_result.table_results is None:
        return

    if (
        any(isinstance(metric, SCALE_METRICS) for metric in metrics_result.table_metrics)
        and metrics_result.network_views
    ):
        network_view = next(iter(metrics_result.network_views.values()))[0]
        metric_views = network_view.agents()
        scale_metrics_in_use = [
            metric.description for metric in metrics_result.table_metrics if isinstance(metric, SCALE_METRICS)
        ]
        if any(isinstance(a.cost, EmpiricalRiskCost) for a in metric_views):
            LOGGER.info(
                f"Empirical-risk cost functions are in use. Compute counters increment by the number of samples "
                f"processed in each method call, which can lead to large raw counts. Applying scaling factor of "
                f"'scale_compute={scale_compute}' to {scale_metrics_in_use} metrics for display."
            )

    table_results = metrics_result.table_results
    if table_results.empty:
        LOGGER.warning("No table rows available to display")
        return

    algs = list(table_results.index.get_level_values("algorithm").unique())

    row_header_1 = ["Metric", "Statistic\nacross agents", "mean ± std\nacross trials"] + [""] * (len(algs) - 1)
    row_header_2 = ["", ""] + [str(alg) for alg in algs]
    rows: list[list[str] | str] = [
        row_header_1,
        row_header_2,
        tb.SEPARATING_LINE,
    ]

    metric_lookup = {metric.description: metric for metric in metrics_result.table_metrics}

    for metric in metrics_result.table_metrics:
        metric_name = metric.description
        if metric_name not in table_results.index.get_level_values("metric"):
            continue

        metric_frame = table_results.xs(metric_name, level="metric", drop_level=False)
        statistic_names = list(metric_frame.index.get_level_values("statistic").unique())
        first_statistic_row = True
        for statistic_name in statistic_names:
            row = [metric.description if first_statistic_row else "", statistic_name]
            for alg in algs:
                key = (metric_name, statistic_name, alg)
                if key not in table_results.index:
                    row.append("-")
                    continue

                mean = float(table_results.loc[key, "mean"])
                std = float(table_results.loc[key, "std"])

                if isinstance(metric, SCALE_METRICS):
                    mean, std = mean * scale_compute, std * scale_compute

                row.append(_format_mean_std(mean, std, metric_lookup[metric_name].fmt))
            rows.append(row)
            first_statistic_row = False

    grid_table = tb.tabulate(rows, tablefmt="grid")
    latex_table = tb.tabulate(rows, tablefmt="latex")
    LOGGER.info("\n" + latex_table if table_fmt == "latex" else "\n" + grid_table)

    if table_path:
        table_path.mkdir(parents=True, exist_ok=True)
        latex_path = table_path / "table.tex"
        grid_path = table_path / "table.txt"
        latex_path.write_text(latex_table, encoding="utf-8")
        grid_path.write_text(grid_table, encoding="utf-8")
        LOGGER.info(f"Saved LaTeX table to {latex_path}")
        LOGGER.info(f"Saved grid table to {grid_path}")


def _format_mean_std(mean: float, std: float, fmt: str) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    return f"{mean:{fmt}} \u00b1 {std:{fmt}}"


def _is_valid_float_format_spec(fmt: str) -> bool:
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
