"""Fetch official KSDD2 for local noncommercial research; never redistribute data."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        parser.error(
            "Read https://www.vicos.si/resources/kolektorsdd2/ and explicitly accept CC BY-NC-SA 4.0"
        )
    root = Path("data/ksdd2")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "KolektorSDD2.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            while block := response.read(1 << 20):
                output.write(block)
                total += len(block)
                if total % (64 << 20) == 0:
                    print(f"Downloaded {total // (1 << 20)} MiB", flush=True)
        temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    with zipfile.ZipFile(archive) as stream:
        for name in stream.namelist():
            if not (root / name).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive member")
        stream.extractall(root)
    metadata = {
        "url": URL,
        "sha256": digest.hexdigest(),
        "license": "CC BY-NC-SA 4.0",
        "official_page": "https://www.vicos.si/resources/kolektorsdd2/",
    }
    (root / "download.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
