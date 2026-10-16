import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "http://btsd.ethz.ch/shareddata/BelgiumTSC/"
FILES = ["BelgiumTSC_Training.zip", "BelgiumTSC_Testing.zip"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_and_extract(filename: str) -> None:
    zip_path = DATA_DIR / filename
    if not zip_path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, zip_path)
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        download_and_extract(name)
    print("Done.")