import copy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from decent_bench.metrics.metric_library import FunctionCalls, GradientCalls, HessianCalls, ProximalCalls
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import MetricResult

SCALE_METRICS = (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls)


def display_tables(
    metrics_result: "MetricResult",
    table_fmt: Literal["text", "latex"] = "text",
    scale_compute: float = 1.0,
    table_path: Path | None = None,
) -> None:
    """Display and optionally save table metrics."""
    if metrics_result.raw_table_results is None or metrics_result.table_results is None:
        return

    # avoid modifying table_results
    table_results = copy.deepcopy(metrics_result.table_results)

    # detect if there are +/-inf values
    has_infinite = bool(np.any(np.isinf(table_results[["mean", "std"]])))

    # scale mean and std (if needed) and format
    for metric in metrics_result.raw_table_results:

        mask = table_results["metric"] == metric.description
        scaling = scale_compute if isinstance(metric, SCALE_METRICS) else 1
        fmt = metric.fmt

        table_results.loc[mask, ["mean", "std"]] = _scale_and_format(
            table_results.loc[mask, ["mean", "std"]],
            fmt,
            scaling
        )

    # join mean and std into single string
    # the result is a DataFrame with columns (metric, algorithm, statistic, value={mean:fmt} +/- {std:fmt})
    table_results["value"] = table_results["mean"] + " \u00b1 " + table_results["std"]
    table_results = table_results.drop(columns=["mean", "std"])

    # reorganize the DataFrame so that it has a MultiIndex (metric, statistic)
    # and algorithms as the columns; each value in a column is mean +/- std (formatted)
    table_results = table_results.pivot_table(
        index=["metric", "statistic"],
        columns="algorithm",
        values="value",
        aggfunc="first",
    )

    # info to user
    if any(isinstance(metric, SCALE_METRICS) for metric in metrics_result.raw_table_results):
        LOGGER.info("Compute counters (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls) can yield very large "
                    "numbers. Set ``scale_compute < 1`` to scale their values for display.")
    if has_infinite:
        LOGGER.info("Infinite values likely indicate divergence. Inspect plots to confirm.")

    # prepare for printing and storing
    text_table = table_results.to_string()
    latex_table = table_results.to_latex(column_format="ll" + "c"*len(table_results.columns))  # index left-align, algorithm center-align  # noqa: E501

    LOGGER.info("\n" + latex_table if table_fmt == "latex" else "\n" + text_table)

    if table_path:
        table_path.mkdir(parents=True, exist_ok=True)
        latex_path = table_path / "table.tex"
        grid_path = table_path / "table.txt"
        latex_path.write_text(latex_table, encoding="utf-8")
        grid_path.write_text(text_table, encoding="utf-8")
        LOGGER.info(f"Saved LaTeX table to {latex_path}")
        LOGGER.info(f"Saved text table to {grid_path}")


def _scale_and_format(frame: pd.DataFrame, fmt: str, scaling: float = 1) -> pd.DataFrame:
    # scale
    if scaling != 1:
        frame = frame * scaling
    # format
    fmt_map = f"{{:{_get_format(fmt)}}}".format
    return frame.map(fmt_map)


def _get_format(fmt: str) -> str:
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    return fmt
