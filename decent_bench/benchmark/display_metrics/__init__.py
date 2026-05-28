from ...benchmark.compute_metrics.compute_plot_metrics import compute_plot_metrics
from ...benchmark.compute_metrics.compute_plots import MAX_Y_PLOT_VALUE, compute_plots
from ...benchmark.display_metrics.display_plots import (
    _add_legend_and_save,
    _create_separate_legend_figure,
    _get_marker_style_color,
    _get_separate_legend_path,
    _select_legend_mode,
    display_plots,
)

__all__ = [
    "MAX_Y_PLOT_VALUE",
    "_add_legend_and_save",
    "_create_separate_legend_figure",
    "_get_marker_style_color",
    "_get_separate_legend_path",
    "_select_legend_mode",
    "compute_plot_metrics",
    "compute_plots",
    "display_plots",
]


from ...benchmark.compute_metrics.compute_tables import compute_tables, validate_statistics_across_agents
from ...benchmark.display_metrics.display_tables import display_tables
from ..compute_metrics.compute_metrics_at_iter import compute_table_metrics

__all__ = [
    "compute_table_metrics",
    "compute_tables",
    "display_tables",
    "validate_statistics_across_agents",
]
