from urllib.request import Request, urlopen
from decent_bench.metrics.table_metrics import TABLE_METRICS_DOC_LINK
from decent_bench.metrics.plot_metrics import PLOT_METRICS_DOC_LINK


def open_url(url: str):
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    assert urlopen(url=req, timeout=10).status == 200


def test_table_metrics_doc_link_works():
    open_url(TABLE_METRICS_DOC_LINK)


def test_plot_metrics_doc_link_works():
    open_url(PLOT_METRICS_DOC_LINK)
