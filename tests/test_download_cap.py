from pathlib import Path

from scripts.download_cap import download_subjects


def test_download_subjects_reports_only_failed_files(tmp_path: Path):
    calls = []

    def fake_downloader(url: str, destination: Path) -> bool:
        calls.append((url, destination.name))
        return destination.name != "n1.txt"

    failures = download_subjects(["n1"], [".edf", ".txt"], tmp_path, fake_downloader)

    assert failures == ["n1.txt"]
    assert calls == [
        ("https://physionet.org/files/capslpdb/1.0.0/n1.edf", "n1.edf"),
        ("https://physionet.org/files/capslpdb/1.0.0/n1.txt", "n1.txt"),
    ]


# --- wznawianie i ponowienia (poprawka po zerwanych pobraniach 180 MB) -----
import io
import urllib.error

import pytest

from scripts.download_cap import download_file

PAYLOAD = bytes(range(256)) * 40  # 10 240 B


class _Resp:
    """Udaje odpowiedz urlopen; opcjonalnie zrywa polaczenie po `fail_after` bajtach."""

    def __init__(self, body: bytes, status: int, headers: dict, fail_after: int | None = None):
        self._buf, self.status, self.headers, self._fail_after, self._sent = io.BytesIO(body), status, headers, fail_after, 0

    def read(self, n):
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise ConnectionResetError(104, "Connection reset by peer")
        chunk = self._buf.read(min(n, 1000))
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Server:
    """Obsluguje Range; pierwsze `drops` polaczen zrywa w polowie."""

    def __init__(self, drops=0, honor_range=True, dns_failures=0):
        self.drops, self.honor_range, self.dns_failures, self.ranges = drops, honor_range, dns_failures, []

    def __call__(self, request, timeout):
        if self.dns_failures:
            self.dns_failures -= 1
            raise urllib.error.URLError("[Errno -3] Temporary failure in name resolution")
        rng = request.headers.get("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and self.honor_range else 0
        body = PAYLOAD[start:]
        if start:
            status, headers = 206, {"Content-Range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}", "Content-Length": str(len(body))}
        else:
            status, headers = 200, {"Content-Length": str(len(body))}
        fail_after = len(body) // 2 if self.drops else None
        self.drops = max(0, self.drops - 1)
        return _Resp(body, status, headers, fail_after)


def _download(tmp_path, server, retries=5):
    dest = tmp_path / "rbd1.edf"
    ok = download_file("https://x/rbd1.edf", dest, retries=retries, opener=server, sleep=lambda s: None)
    return ok, dest


def test_resumes_from_partial_file_after_connection_reset(tmp_path):
    server = _Server(drops=2)
    ok, dest = _download(tmp_path, server)
    assert ok and dest.read_bytes() == PAYLOAD
    assert server.ranges[0] is None and server.ranges[1].startswith("bytes=")  # wznowienie, nie od zera
    assert not (tmp_path / "rbd1.edf.part").exists()


def test_partial_file_survives_final_failure_and_next_run_completes(tmp_path):
    ok, dest = _download(tmp_path, _Server(drops=10), retries=2)
    part = tmp_path / "rbd1.edf.part"
    assert not ok and not dest.exists() and part.stat().st_size > 0  # nic nie skasowane
    ok, dest = _download(tmp_path, _Server())  # "ponowne uruchomienie skryptu"
    assert ok and dest.read_bytes() == PAYLOAD


def test_retries_after_dns_failure(tmp_path):
    ok, dest = _download(tmp_path, _Server(dns_failures=2))
    assert ok and dest.read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts_cleanly(tmp_path):
    (tmp_path / "rbd1.edf.part").write_bytes(PAYLOAD[:3000])
    ok, dest = _download(tmp_path, _Server(honor_range=False))
    assert ok and dest.read_bytes() == PAYLOAD  # bez zdublowanych bajtow


def test_truncated_body_is_not_accepted_as_complete(tmp_path):
    def short_server(request, timeout):
        return _Resp(PAYLOAD[:100], 200, {"Content-Length": str(len(PAYLOAD))})

    ok, dest = _download(tmp_path, short_server, retries=1)
    assert not ok and not dest.exists()


def test_404_is_not_retried(tmp_path):
    calls = []

    def missing(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    ok, _ = _download(tmp_path, missing)
    assert not ok and len(calls) == 1


def test_already_complete_part_file_is_finalized_on_416(tmp_path):
    (tmp_path / "rbd1.edf.part").write_bytes(PAYLOAD)

    def complete(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 416, "Range Not Satisfiable", {"Content-Range": f"bytes */{len(PAYLOAD)}"}, None)

    ok, dest = _download(tmp_path, complete)
    assert ok and dest.read_bytes() == PAYLOAD
