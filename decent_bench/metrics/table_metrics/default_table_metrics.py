from decent_bench.library.core.metrics.table_metrics import table_metrics_data_extractors as data_extractors
from decent_bench.library.core.metrics.table_metrics.table_metrics_constructs import (
    Avg,
    Max,
    Min,
    Single,
    Sum,
    TableMetric,
)

DEFAULT_TABLE_METRICS = [
    TableMetric("global cost error (< 1e-9 = exact convergence)", [Single], data_extractors.global_cost_error),
    TableMetric("global gradient optimality", [Single], data_extractors.global_gradient_optimality),
    TableMetric("x error", [Min, Avg, Max], data_extractors.x_error),
    TableMetric("asymptotic convergence order", [Avg], data_extractors.asymptotic_convergence_order),
    TableMetric("asymptotic convergence rate", [Avg], data_extractors.asymptotic_convergence_rate),
    TableMetric("iterative convergence order", [Avg], data_extractors.iterative_convergence_order),
    TableMetric("iterative convergence rate", [Avg], data_extractors.iterative_convergence_rate),
    TableMetric("nr x updates", [Avg, Sum], data_extractors.n_x_updates),
    TableMetric("nr evaluate calls", [Avg, Sum], data_extractors.n_evaluate_calls),
    TableMetric("nr gradient calls", [Avg, Sum], data_extractors.n_gradient_calls),
    TableMetric("nr hessian calls", [Avg, Sum], data_extractors.n_hessian_calls),
    TableMetric("nr proximal calls", [Avg, Sum], data_extractors.n_proximal_calls),
    TableMetric("nr sent messages", [Avg, Sum], data_extractors.n_sent_messages),
    TableMetric("nr received messages", [Avg, Sum], data_extractors.n_received_messages),
    TableMetric("nr sent messages dropped", [Avg, Sum], data_extractors.n_sent_messages_dropped),
]
"""
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.global_cost_error` (single)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.global_gradient_optimality` \
(single)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.x_error` (min, avg, max)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.asymptotic_convergence_order` \
(avg)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.asymptotic_convergence_rate` \
(avg)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.iterative_convergence_order` \
(avg)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.iterative_convergence_rate` \
(avg)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_x_updates` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_evaluate_calls` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_gradient_calls` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_hessian_calls` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_proximal_calls` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_sent_messages` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_received_messages` (avg, sum)
- :func:`~decent_bench.library.core.metrics.table_metrics.table_metrics_data_extractors.n_sent_messages_dropped` \
(avg, sum)


:meta hide-value:
"""
