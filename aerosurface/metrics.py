"""Dataset-aggregated confusion matrices; absent-class metrics are null."""

import numpy as np

from aerosurface import CLASSES


def confusion(target: np.ndarray, prediction: np.ndarray, classes: int = 5) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError("Mismatched prediction shape")
    if np.any((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)):
        raise ValueError("Class id out of range")
    return np.bincount((target * classes + prediction).ravel(), minlength=classes**2).reshape(
        classes, classes
    )


def summarize(cm: np.ndarray, names: list[str] = CLASSES) -> dict:
    tp = np.diag(cm).astype(float)
    truth, predicted = cm.sum(1), cm.sum(0)

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)

    iou = ratio(tp, truth + predicted - tp)
    arrays = {
        "iou": iou,
        "dice": ratio(2 * tp, truth + predicted),
        "precision": ratio(tp, predicted),
        "recall": ratio(tp, truth),
    }
    return {
        "miou": float(np.nanmean(iou)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                key: float(value[i]) if np.isfinite(value[i]) else None
                for key, value in arrays.items()
            }
            for i, name in enumerate(names)
        },
    }
