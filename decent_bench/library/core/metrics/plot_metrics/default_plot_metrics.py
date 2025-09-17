from decent_bench.library.core.metrics.plot_metrics.plot_metrics_constructs import PlotMetric
from decent_bench.library.core.metrics.plot_metrics.plot_metrics_data_extractors import (
    global_cost_error_per_iteration,
    global_gradient_optimality_per_iteration,
)

DEFAULT_PLOT_METRICS = [
    PlotMetric(
        x_label="iteration",
        y_label="global cost error",
        x_log=False,
        y_log=True,
        get_data_from_trial=global_cost_error_per_iteration,
    ),
    PlotMetric(
        x_label="iteration",
        y_label="global gradient optimality",
        x_log=False,
        y_log=True,
        get_data_from_trial=global_gradient_optimality_per_iteration,
    ),
]
"""
- Global cost error (y-axis) per iteration (x-axis). Semi-log plot.
  Details: :func:`~decent_bench.library.core.metrics.plot_metrics.plot_metrics_data_extractors.\
global_cost_error_per_iteration`.
- Global gradient optimality (y-axis) per iteration (x-axis). Semi-log plot.
  Details: :func:`~decent_bench.library.core.metrics.plot_metrics.plot_metrics_data_extractors.\
global_gradient_optimality_per_iteration`.

:meta hide-value:
"""
