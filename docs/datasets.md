# Data provenance and valid comparisons

## Procedural fallback

`python -m aerosurface.cli data` creates 192 training, 40 validation and 64 test frames from seed
42. Splits use seed offsets 0, 100000 and 200000. Metadata records illumination/color/direction,
material offsets, texture/noise, curved shading, camera-slant proxy, object locations and sizes.
Train and validation are nominal with randomized scenes; test rotates through nine stress slices.
No real photograph or Isaac rendering is involved. Analytic depth and metallic metadata are
explicit proxies, not a physical renderer. This is useful for software integration and domain-gap
demonstration, not proof of aviation performance.

## Isaac Sim

`simulation/isaac_generate.py` is a standalone Isaac Sim **5.1** recipe based on the official
[Replicator examples](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_getting_started.html).
It creates a tessellated curved panel, a flat panel, protected seam/fixture, defect patch, obstacle,
USD PreviewSurface materials and randomized lighting/camera/texture/geometry. BasicWriter requests
RGB, semantic/instance labels and depth from the same render product; synchronous capture and
subframes prevent intentional frame-to-metadata lag. Noise/blur stress can be applied downstream.

```bash
# Use Isaac Sim's bundled Python, from this repository:
/path/to/isaac-sim/python.sh simulation/isaac_generate.py --config configs/isaac.json --frames 12
python -m simulation.prepare_isaac data/isaac_raw data/isaac_prepared
python -m aerosurface.cli train --data data/isaac_prepared --run models/isaac
```

Isaac Sim is not installed here. The recipe was syntax-checked and ID-remapping logic was tested
with fixtures; scene execution, camera framing, writer filenames and material rendering remain
unvalidated. Isaac Sim 6 changes Replicator APIs; do not assume this 5.1 recipe works unchanged.
Inspect actual frame alignment/ID mappings before training. Separate scene families across splits
for a serious experiment; consecutive randomized frames alone are a weak independence assumption.

## KolektorSDD2

The [official ViCoS dataset page](https://www.vicos.si/resources/kolektorsdd2/) supplies images,
pixel labels, original train/test partitions and **CC BY-NC-SA 4.0** terms. Commercial use requires
contacting the authors. Third-party mirrors reported inconsistent licenses; the original source
was used. Cite Božič, Tabernik and Skočaj, *Mixed supervision for surface-defect detection: from
weakly to fully supervised learning*, Computers in Industry (2021).

```bash
python scripts/download_ksdd2.py --accept-noncommercial-license
python -m training.real_domain --epochs 6
```

The archive was actually downloaded and verified locally. SHA256:
`edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`.
Images, derived images, weights adapted on this dataset and the archive remain in ignored local
directories. They are not covered by the repository's MIT code license and are not redistributed.
Two unlabelled `(copy)` files in the archive were excluded only after checking byte-identical
content against their originals. Any other missing mask raises an error.

Positive mask pixels mean defect; all other pixels mean **nondefect**, not sandable. Five-class
synthetic logits are collapsed with logsumexp over IDs 0,1,2,4 versus ID 3, exactly preserving the
defect softmax probability. Binary fine-tuned models must not be deployed as five-class models.

The bounded experiment uses 64 positive + 192 negative training images and a disjoint validation
subset of 16 positive + 48 negative images selected from the official training split using seed
123. All 1,004 official test images remain untouched for evaluation. Selection is best validation
defect IoU, fixed threshold 0.5, six epochs, the same data/order policy and optimizer settings for
real-only and synthetic-then-real. Images are resized to 128×320; small defects can lose pixels.
Splits are image identities, not verified manufacturing-part groups. There is one seed, no
pretrained external backbone and no hyperparameter search; results are a feasibility study.
