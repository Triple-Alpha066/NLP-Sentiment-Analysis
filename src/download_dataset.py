from pathlib import Path
import urllib.request
import tarfile

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DATA_DIR / "aclImdb_v1.tar.gz"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print("Dataset archive already exists.")
        return

    print("Downloading IMDb dataset...")
    urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
    print(f"Downloaded to: {ARCHIVE_PATH}")


def extract_dataset():
    extracted_dir = RAW_DATA_DIR / "aclImdb"

    if extracted_dir.exists():
        print("IMDb dataset is already extracted.")
        return

    print("Extracting IMDb dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DATA_DIR)

    print(f"Extracted to: {extracted_dir}")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    print("IMDb dataset preparation complete.")