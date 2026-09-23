"""Check actual native C++ outputs against Python ORT and postprocessing."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("artifacts/demo")
result = {}
for name in ["mask", "sandable", "avoid", "defect", "protected"]:
    python = np.array(Image.open(root / f"{name}.png"))
    cpp = np.array(Image.open(root / f"cpp_{name}.pgm"))
    np.testing.assert_array_equal(python, cpp)
    result[name] = "EXACT_MATCH"
Path("benchmarks/cpp_parity.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
