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
    """Display table of metrics as mean ± std across trials."""
    if metrics_result.table_metrics is None or not metrics_result.table_results:
        return

    formatted_frames = []
    has_infinite = False

    for metric, frame in metrics_result.table_results.items():
        # detect if there are +/-inf values
        has_infinite = has_infinite or bool(np.any(np.isinf(frame[["mean", "std"]])))

        # scale mean and std if needed
        if isinstance(metric, SCALE_METRICS):
            new_frame = frame.assign(
                mean=frame["mean"] * scale_compute,
                std=frame["std"] * scale_compute,
            )

        # format mean and std, add a metric column
        # result is a DataFrame with columns (metric, algorithm, statistic, value={mean:fmt} +/- {std:fmt})
        fmt = f"{{:{_get_format(metric.fmt)}}}".format
        new_frame = new_frame.assign(
            metric=metric.description,
            value=fmt(new_frame["mean"]) + " \u00b1 " + fmt(new_frame["std"]),
        )
        formatted_frames.append(new_frame[["metric", "algorithm", "statistic", "value"]])  # reorder columns

    # stack into a big frame with columns (metric, algorithm, statistic, value={mean:fmt} +/- {std:fmt})
    table_frame = pd.concat(formatted_frames, ignore_index=True)

    # reorganize the DataFrame so that it has a MultiIndex (metric, statistic)
    # and algorithms as the columns; each value in a column is mean +/- std (formatted)
    table_frame = table_frame.pivot_table(
        index=["metric", "statistic"],
        columns="algorithm",
        values="value",
        aggfunc="first",
    )

    # info to user
    if any(isinstance(metric, SCALE_METRICS) for metric in metrics_result.table_metrics):
        LOGGER.info("Compute counters (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls) can yield very large "
                    "numbers. Set ``scale_compute < 1`` to scale their values for display.")
    if has_infinite:
        LOGGER.info("Infinite values likely indicate divergence. Inspect plots to confirm.")

    # prepare for printing and storing
    text_table = table_frame.to_string()
    latex_table = table_frame.to_latex(column_format="ll" + "c"*len(table_frame.columns))  # index left-align, algorithm center-align  # noqa: E501

    LOGGER.info("\n" + latex_table if table_fmt == "latex" else "\n" + text_table)

    if table_path:
        table_path.mkdir(parents=True, exist_ok=True)
        latex_path = table_path / "table.tex"
        grid_path = table_path / "table.txt"
        latex_path.write_text(latex_table, encoding="utf-8")
        grid_path.write_text(text_table, encoding="utf-8")
        LOGGER.info(f"Saved LaTeX table to {latex_path}")
        LOGGER.info(f"Saved text table to {grid_path}")


def _get_format(fmt: str) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    return fmt


def _is_valid_float_format_spec(fmt: str) -> bool:
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
