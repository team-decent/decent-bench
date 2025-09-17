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
    TableMetric("global cost error", [Single], data_extractors.global_cost_error),
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
- Global cost error (single) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.global_cost_error`.
- Global gradient optimality (single) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.global_gradient_optimality`.
- x error (min, avg, max) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.x_error`.
- Asymptotic convergence order (avg) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.asymptotic_convergence_order`.
- Asymptotic convergence rate (avg) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.asymptotic_convergence_rate`.
- Iterative convergence order (avg) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.iterative_convergence_order`.
- Iterative convergence rate (avg) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.iterative_convergence_rate`.
- Nr of x updates (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_x_updates`.
- Nr of evaluate calls (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_evaluate_calls`.
- Nr of gradient calls (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_gradient_calls`.
- Nr of hessian calls (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_hessian_calls`.
- Nr of proximal calls (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_proximal_calls`.
- Nr of sent messages (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_sent_messages`.
- Nr of received messages (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_received_messages`.
- Nr of sent messages dropped (avg, sum) defined at :func:`~decent_bench.library.core.metrics.table_metrics.\
table_metrics_data_extractors.n_sent_messages_dropped`.

:meta hide-value:
"""
