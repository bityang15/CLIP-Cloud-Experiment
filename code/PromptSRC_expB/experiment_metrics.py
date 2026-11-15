import numpy as np


def calibration_bins(confidence, correct, bins):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    indices = np.clip(np.ceil(confidence * bins).astype(np.int64) - 1, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins)
    confidence_sum = np.bincount(indices, weights=confidence, minlength=bins)
    correct_sum = np.bincount(indices, weights=correct, minlength=bins)
    nonempty = np.maximum(counts, 1)
    accuracy = correct_sum / nonempty
    mean_confidence = confidence_sum / nonempty
    ece = np.sum(np.abs(correct_sum - confidence_sum)) / len(confidence)
    return dict(ece=100.0 * float(ece), counts=counts.tolist(),
                bin_accuracy=accuracy.tolist(), bin_confidence=mean_confidence.tolist())


def metrics(logits, labels, bins=10, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert logits.ndim == 2 and len(logits) > 0
    assert labels.shape == (len(logits),)
    assert np.isfinite(logits).all() and np.isfinite(temperature) and temperature > 0
    assert labels.min() >= 0 and labels.max() < logits.shape[1]
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    normalizer = exponential.sum(axis=1)
    confidence = exponential.max(axis=1) / normalizer
    correct = scaled.argmax(axis=1) == labels
    nll = np.mean(np.log(normalizer) - shifted[np.arange(len(labels)), labels])
    calibration = calibration_bins(confidence, correct, bins)
    return dict(accuracy=100.0 * float(correct.mean()), nll=float(nll), n=len(labels),
                classes=logits.shape[1], bins=bins, **calibration)


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert len(labels) > 0
    grid = np.geomspace(0.05, 20.0, 401)
    grid[np.argmin(np.abs(grid - 1.0))] = 1.0
    losses = []
    for temperature in grid:
        scaled = logits / temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        losses.append(np.mean(np.log(np.exp(shifted).sum(axis=1))
                              - shifted[np.arange(len(labels)), labels]))
    losses = np.asarray(losses)
    candidates = np.flatnonzero(losses == losses.min())
    selected = candidates[np.argmin(np.abs(grid[candidates] - 1.0))]
    temperature = float(grid[selected])
    assert np.isfinite(losses).all()
    return dict(temperature=temperature, validation_nll_before=float(losses[grid == 1.0][0]),
                validation_nll_after=float(losses[selected]), validation_n=len(labels),
                grid_min=0.05, grid_max=20.0, grid_points=401,
                boundary_hit=bool(selected in [0, len(grid) - 1]),
                grid=grid.tolist(), validation_nll_grid=losses.tolist())
