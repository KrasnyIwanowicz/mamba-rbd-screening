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
