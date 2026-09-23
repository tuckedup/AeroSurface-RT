"""Download official ONNX Runtime C++ SDK into ignored third_party/."""

import argparse
import hashlib
import json
import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.19.2")
    args = parser.parse_args()
    target = "win-x64" if platform.system() == "Windows" else "linux-x64"
    ext = "zip" if target.startswith("win") else "tgz"
    name = f"onnxruntime-{target}-{args.version}"
    root = Path("third_party")
    root.mkdir(exist_ok=True)
    archive = root / f"{name}.{ext}"
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{args.version}/{name}.{ext}"
    urllib.request.urlretrieve(url, archive)

    # All extraction destinations are confined to third_party.
    def validate(names):
        for member in names:
            if not (root / member).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive member")

    if ext == "zip":
        with zipfile.ZipFile(archive) as stream:
            validate(stream.namelist())
            stream.extractall(root)
    else:
        with tarfile.open(archive) as stream:
            validate(stream.getnames())
            # Official Linux SDK includes library symlinks; Python's data filter
            # permits confined links while rejecting traversal and special files.
            stream.extractall(root, filter="data")
    (root / "download.json").write_text(
        json.dumps(
            {"url": url, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2
        )
    )
    print(root / name)


if __name__ == "__main__":
    main()
