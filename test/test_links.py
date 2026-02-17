from urllib.request import Request, urlopen

DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/"


def open_url(url: str):
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    assert urlopen(url=req, timeout=10).status == 200


def test_metrics_doc_link_works():
    open_url(DOC_LINK)
