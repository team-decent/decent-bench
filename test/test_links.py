import requests

from decent_bench.library.core.metrics.table_metrics.tabulate import DOC_LINK as TABLE_METRICS_DOC_LINK
from decent_bench.library.core.metrics.plot_metrics.plot import DOC_LINK as PLOT_METRICS_DOC_LINK


def test_table_metrics_doc_link_works():
    assert requests.head(TABLE_METRICS_DOC_LINK, allow_redirects=True, timeout=10).status_code == 200


def test_plot_metrics_doc_link_works():
    assert requests.head(PLOT_METRICS_DOC_LINK, allow_redirects=True, timeout=10).status_code == 200
