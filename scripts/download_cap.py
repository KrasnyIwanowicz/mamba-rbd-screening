"""Download selected CAP Sleep Database files with an honest failure summary."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path
import urllib.request

from tqdm import tqdm


BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"
DEFAULT_DEST_DIR = Path("data/raw/capslpdb")
DEFAULT_SUBJECTS = tuple(
    [f"rbd{i}" for i in range(1, 23)] + [f"n{i}" for i in range(1, 17)]
)
DEFAULT_EXTENSIONS = (".edf", ".txt")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path) -> bool:
    """Download a non-empty file atomically; return False instead of hiding errors."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[-] Plik istnieje, pomijam: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "mamba-rbd-screening/1.0"})
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path.name) as progress:
            with urllib.request.urlopen(request) as response, open(temp_path, "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(len(chunk))
        if temp_path.stat().st_size == 0:
            raise OSError("pobrany plik ma 0 bajtow")
        temp_path.replace(dest_path)
        return True
    except Exception as error:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[!] Blad pobierania {url}: {error}")
        return False


def download_subjects(
    subjects: Iterable[str],
    extensions: Iterable[str],
    dest_dir: Path,
    downloader: Callable[[str, Path], bool] = download_file,
) -> list[str]:
    """Download requested subject/file pairs and return the filenames that failed."""
    failures = []
    for subject in subjects:
        for extension in extensions:
            filename = f"{subject}{extension}"
            if not downloader(f"{BASE_URL}{filename}", dest_dir / filename):
                failures.append(filename)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS), help="np. n1 rbd1")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS), help="np. .txt albo .edf .txt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DEST_DIR)
    args = parser.parse_args()

    invalid_extensions = [extension for extension in args.extensions if not extension.startswith(".")]
    if invalid_extensions:
        parser.error(f"Rozszerzenia musza zaczynac sie od kropki: {', '.join(invalid_extensions)}")

    print(f"[*] Pobieranie do: {args.data_dir.resolve()}")
    print(f"[*] Podmioty: {', '.join(args.subjects)}; rozszerzenia: {', '.join(args.extensions)}")
    failures = download_subjects(args.subjects, args.extensions, args.data_dir)
    if failures:
        print("\n[!] Nie pobrano: " + ", ".join(failures))
        print("    Rekord bez .txt nie moze wejsc do analizy REM/RSWA.")
        return 1

    print("\n[+] Wszystkie zadane pliki sa obecne i niepuste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
