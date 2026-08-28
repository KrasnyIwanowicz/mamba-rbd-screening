from pathlib import Path
import urllib.request
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"
DEST_DIR = Path("data/raw/capslpdb")

# Lista pacjentów: RBD (1-22) i Healthy/Normal (1-16)
SUBJECTS = [f"rbd{i}" for i in range(1, 23)] + [f"n{i}" for i in range(1, 17)]
EXTENSIONS = [".edf", ".txt"]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[-] Plik istnieje, pomijam: {dest_path.name}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path.name) as t:
            urllib.request.urlretrieve(url, filename=temp_path, reporthook=t.update_to)
        temp_path.rename(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[!] Błąd pobierania {url}: {e}")


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Rozpoczynam pobieranie CAP Sleep Database do: {DEST_DIR.resolve()}")
    print(f"[*] Liczba podmiotów: {len(SUBJECTS)} (RBD: 22, Healthy: 16)")

    for subject in SUBJECTS:
        for ext in EXTENSIONS:
            filename = f"{subject}{ext}"
            url = f"{BASE_URL}{filename}"
            dest = DEST_DIR / filename
            download_file(url, dest)

    print("\n[+] Wszystkie pliki zostały pobrane pomyślnie!")


if __name__ == "__main__":
    main()