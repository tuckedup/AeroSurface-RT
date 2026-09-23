# Executed binary transfer experiment

KolektorSDD2; CC BY-NC-SA 4.0. 256 train / 64 validation / 1,004 official test images, one seed, six training epochs. This compares defect versus nondefect, never five-class sandability accuracy.

| Initialization/training | Defect IoU | Defect Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| synthetic_only | 0.0112 | 0.0221 | 0.8674 | 0.0112 |
| real_only | 0.0000 | 0.0000 | N/A | 0.0000 |
| synthetic_then_real | 0.2729 | 0.4288 | 0.4143 | 0.4443 |

Synthetic pretraining helped this short fine-tuning run, but the scratch baseline collapsed to nondefect. Its zero recall must not be advertised as a competitive real-only baseline. The comparisons share an architecture, training subset, epoch budget and validation selection rule, but synthetic pretraining adds compute. There are no confidence intervals, repeated seeds, full training-set sweeps or pretrained industrial baseline. Accuracy at downsampled resolution is not native-resolution defect accuracy. See `docs/datasets.md` for label mapping and exact split policy.

Raw confusion matrices, histories and train/validation identities are saved in `benchmarks/real_domain.json`. Adapted weights and real imagery remain ignored under `artifacts/`.
