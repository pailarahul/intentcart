"""Download the UCI Online Shoppers Purchasing Intention dataset and verify it.

Usage:  python src/download_data.py
Source: https://archive.ics.uci.edu/dataset/468  (licence: CC BY 4.0)
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intentcart import DATA_PATH, DATA_SHA256, sha256  # noqa: E402

URL = ("https://archive.ics.uci.edu/static/public/468/"
       "online+shoppers+purchasing+intention+dataset.zip")


def main() -> None:
    if DATA_PATH.exists() and sha256(DATA_PATH) == DATA_SHA256:
        print(f"Already present and verified: {DATA_PATH}")
        return
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_bytes(z.read("online_shoppers_intention.csv"))
    got = sha256(DATA_PATH)
    if got != DATA_SHA256:
        sys.exit(f"Checksum mismatch: {got}")
    print(f"Saved and verified: {DATA_PATH}")


if __name__ == "__main__":
    main()
