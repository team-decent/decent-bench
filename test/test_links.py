from urllib.request import Request, urlopen
from decent_bench.library.core.metrics.table_metrics.tabulate import DOC_LINK as TABLE_METRICS_DOC_LINK
from decent_bench.library.core.metrics.plot_metrics.plot import DOC_LINK as PLOT_METRICS_DOC_LINK


def open_url(url: str):
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    assert urlopen(url=req, timeout=10).status == 200


def test_table_metrics_doc_link_works():
    open_url(TABLE_METRICS_DOC_LINK)


def test_plot_metrics_doc_link_works():
    open_url(PLOT_METRICS_DOC_LINK)
