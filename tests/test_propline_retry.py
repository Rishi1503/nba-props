"""PropLine requests retry transient failures (a read timeout failed the Oct 27 6 PM run)."""

import pytest
import requests

from nbaprops.pipeline.odds import PropLineProvider


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self.headers = {}
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class _Flaky:
    def __init__(self, failures):
        self.failures = list(failures)
        self.headers = {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.failures:
            f = self.failures.pop(0)
            if f == "timeout":
                raise requests.ReadTimeout("Read timed out. (read timeout=20.0)")
            return _Resp(f, {})
        return _Resp(200, {"data": []})


@pytest.mark.parametrize("failures", [["timeout"], [503], ["timeout", 429]])
def test_transient_failures_are_retried(failures):
    waits = []
    http = _Flaky(failures)
    p = PropLineProvider("k", session=http, sleep=waits.append)
    assert p._get("/x") == {"data": []}
    assert http.calls == len(failures) + 1
    assert waits == [2.0, 5.0][: len(failures)]


def test_a_persistent_timeout_is_reported_after_three_tries():
    waits = []
    http = _Flaky(["timeout"] * 3)
    with pytest.raises(requests.ReadTimeout):
        PropLineProvider("k", session=http, sleep=waits.append)._get("/x")
    assert http.calls == 3 and waits == [2.0, 5.0]


def test_a_real_error_is_not_retried():
    http = _Flaky([404])
    with pytest.raises(requests.HTTPError):
        PropLineProvider("k", session=http, sleep=lambda s: None)._get("/x")
    assert http.calls == 1
