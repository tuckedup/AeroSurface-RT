"""Audit small repository deliverables without staging or committing generated data."""

import json
import re
import subprocess
from pathlib import Path


def main():
    root = Path.cwd()
    run = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    )
    files = [Path(p) for p in run.stdout.splitlines()]
    errors = []
    for path in files:
        if path.suffix in {".pt", ".engine", ".onnx", ".zip", ".npy"}:
            errors.append(f"Generated binary would be tracked: {path}")
        if path.stat().st_size > 2_000_000:
            errors.append(f"Unexpected large deliverable: {path}")
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                if not (path.parent / target.split("#")[0]).exists():
                    errors.append(f"Broken documentation link: {path} -> {target}")
    for directory in ["data/ksdd2", "models/edge", "third_party", ".venv"]:
        probe = subprocess.run(
            ["git", "check-ignore", directory], capture_output=True, text=True, check=False
        )
        if probe.returncode != 0:
            errors.append(f"Generated directory not ignored: {directory}")
    if errors:
        raise RuntimeError("\n".join(errors))
    result = {
        "status": "PASS",
        "deliverable_files": len(files),
        "deliverable_bytes": sum(p.stat().st_size for p in files),
        "checks": [
            "no weights/engines/dataset archives tracked",
            "no files above 2 MB",
            "relative Markdown links resolve",
            "generated directories ignored",
        ],
    }
    (root / "benchmarks/repository_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
