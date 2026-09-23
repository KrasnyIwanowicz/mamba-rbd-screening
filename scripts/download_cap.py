"""Download selected CAP Sleep Database files with an honest failure summary."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path
import time
import urllib.error
import urllib.request

from tqdm import tqdm


BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"
DEFAULT_DEST_DIR = Path("data/raw/capslpdb")
DEFAULT_SUBJECTS = tuple(
    [f"rbd{i}" for i in range(1, 23)] + [f"n{i}" for i in range(1, 17)]
)
DEFAULT_EXTENSIONS = (".edf", ".txt")


def _expected_size(response, offset: int) -> int | None:
    """Pelny rozmiar pliku z Content-Range (206) albo Content-Length (200)."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        return int(total) if total.isdigit() else None
    length = response.headers.get("Content-Length")
    return offset + int(length) if length and length.isdigit() else None


def _fetch_to_part(url: str, part_path: Path, opener: Callable, timeout_s: float) -> None:
    """Dociaga part_path do konca (HTTP Range od aktualnego rozmiaru) i sprawdza rozmiar."""
    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "mamba-rbd-screening/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = opener(urllib.request.Request(url, headers=headers), timeout=timeout_s)
    except urllib.error.HTTPError as error:
        # 416 = zakres poza plikiem: albo .part jest juz kompletny, albo plik na
        # serwerze sie zmienil. Content-Range "bytes */TOTAL" rozstrzyga.
        if error.code == 416 and offset:
            server_size = (error.headers.get("Content-Range") or "").rsplit("/", 1)[-1]
            if server_size.isdigit() and int(server_size) == offset:
                return
            part_path.unlink()
        raise

    with response:
        if offset and getattr(response, "status", 200) != 206:
            offset = 0  # serwer zignorowal Range -> pobieramy od zera
        total = _expected_size(response, offset)
        mode = "ab" if offset else "wb"
        with tqdm(unit="B", unit_scale=True, desc=part_path.name, initial=offset, total=total) as progress:
            with open(part_path, mode) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(len(chunk))

    size = part_path.stat().st_size
    if size == 0:
        raise OSError("pobrany plik ma 0 bajtow")
    if total is not None and size != total:
        raise OSError(f"niepelny plik: {size} z {total} bajtow")


def download_file(
    url: str,
    dest_path: Path,
    retries: int = 5,
    backoff_s: float = 10.0,
    timeout_s: float = 60.0,
    opener: Callable = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Pobiera plik z wznawianiem i ponowieniami; zwraca False zamiast ukrywac bledy.

    Czesciowe dane leza w <plik>.part i NIE sa kasowane po bledzie -- kolejna
    proba (albo kolejne uruchomienie skryptu) dociaga reszte przez HTTP Range.
    Gotowy plik pojawia sie pod docelowa nazwa dopiero po sprawdzeniu rozmiaru.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[-] Plik istnieje, pomijam: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            _fetch_to_part(url, part_path, opener, timeout_s)
            part_path.replace(dest_path)
            return True
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404, 410):
                print(f"[!] {url}: HTTP {error.code} -- pliku nie ma pod tym adresem, nie ponawiam.")
                return False
            last_error: Exception = error
        except Exception as error:  # zerwane polaczenie, DNS, timeout, niepelny plik
            last_error = error
        if attempt < retries:
            wait = backoff_s * 2 ** (attempt - 1)
            print(f"[!] {dest_path.name}: {last_error} -- proba {attempt}/{retries}, ponawiam za {wait:.0f} s")
            sleep(wait)

    kept = part_path.stat().st_size if part_path.exists() else 0
    print(f"[!] Blad pobierania {url} po {retries} probach: {last_error}")
    if kept:
        print(f"    Zachowano {kept / 1e6:.1f} MB w {part_path.name} -- uruchom skrypt ponownie, aby dokonczyc.")
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
    parser.add_argument("--retries", type=int, default=5, help="proby na plik (z wznawianiem od miejsca przerwania)")
    args = parser.parse_args()

    invalid_extensions = [extension for extension in args.extensions if not extension.startswith(".")]
    if invalid_extensions:
        parser.error(f"Rozszerzenia musza zaczynac sie od kropki: {', '.join(invalid_extensions)}")

    print(f"[*] Pobieranie do: {args.data_dir.resolve()}")
    print(f"[*] Podmioty: {', '.join(args.subjects)}; rozszerzenia: {', '.join(args.extensions)}")
    failures = download_subjects(
        args.subjects, args.extensions, args.data_dir,
        downloader=lambda url, dest: download_file(url, dest, retries=args.retries),
    )
    if failures:
        print("\n[!] Nie pobrano: " + ", ".join(failures))
        print("    Rekord bez .txt nie moze wejsc do analizy REM/RSWA.")
        print("    Uruchom te sama komende ponownie -- pobrane pliki sa pomijane, a .part wznawiane.")
        return 1

    print("\n[+] Wszystkie zadane pliki sa obecne i niepuste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
