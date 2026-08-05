"""Synthetic compact telemetry dataset and feature extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_NAMES = ("normal", "drift", "spike", "dropout")


@dataclass(frozen=True)
class TelemetryDataset:
    """A generated telemetry classification dataset."""

    windows: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...] = CLASS_NAMES


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Extract compact per-sensor features from telemetry windows.

    Input shape is ``(samples, timesteps, sensors)``. Output shape is
    ``(samples, sensors * 8)``.
    """

    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (samples, timesteps, sensors)")

    _, seq_len, _ = windows.shape
    if seq_len < 2:
        raise ValueError("windows must contain at least two timesteps")

    t = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    t = t - np.mean(t)
    denom = float(np.sum(t * t))

    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    mins = np.min(windows, axis=1)
    maxs = np.max(windows, axis=1)
    ranges = maxs - mins
    slopes = np.einsum("nts,t->ns", windows, t, optimize=True) / denom
    energy = np.mean(windows * windows, axis=1)
    endpoints = windows[:, -1, :] - windows[:, 0, :]

    features = np.concatenate(
        [means, stds, mins, maxs, ranges, slopes, energy, endpoints], axis=1
    )
    return features.astype(np.float32, copy=False)


def generate_synthetic_telemetry(
    n_samples: int = 768,
    seq_len: int = 32,
    n_sensors: int = 4,
    seed: int = 7,
) -> TelemetryDataset:
    """Generate a deterministic synthetic telemetry classification dataset."""

    if n_samples < len(CLASS_NAMES):
        raise ValueError(f"n_samples must be at least {len(CLASS_NAMES)}")
    if seq_len < 12:
        raise ValueError("seq_len must be at least 12")
    if n_sensors < 2:
        raise ValueError("n_sensors must be at least 2")

    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples, dtype=np.int64) % len(CLASS_NAMES)
    rng.shuffle(labels)

    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    windows = np.empty((n_samples, seq_len, n_sensors), dtype=np.float32)

    for idx, label in enumerate(labels):
        phase = rng.uniform(0.0, 1.0, size=n_sensors).astype(np.float32)
        frequency = rng.uniform(0.7, 1.4, size=n_sensors).astype(np.float32)
        seasonal = 0.04 * np.sin(
            2.0 * np.pi * (t[:, None] * frequency[None, :] + phase[None, :])
        )
        baseline = rng.normal(0.0, 0.055, size=(seq_len, n_sensors)).astype(np.float32)
        offsets = rng.normal(0.0, 0.035, size=(1, n_sensors)).astype(np.float32)
        sample = seasonal.astype(np.float32) + baseline + offsets

        sensor = int(rng.integers(0, n_sensors))
        paired = (sensor + 1) % n_sensors

        if label == 1:
            ramp = (t - 0.2).clip(min=0.0)
            ramp = ramp / float(np.max(ramp))
            amplitude = float(rng.uniform(0.55, 0.9))
            sample[:, sensor] += amplitude * ramp
            sample[:, paired] += 0.25 * amplitude * ramp
        elif label == 2:
            center = int(rng.integers(seq_len // 4, seq_len - seq_len // 4))
            width = max(1, seq_len // 18)
            spike_t = np.arange(seq_len, dtype=np.float32)
            spike = np.exp(-0.5 * ((spike_t - center) / width) ** 2)
            amplitude = float(rng.uniform(0.85, 1.25))
            sample[:, sensor] += amplitude * spike
        elif label == 3:
            width = max(3, seq_len // 5)
            start = int(rng.integers(1, seq_len - width - 1))
            sample[start : start + width, sensor] -= float(rng.uniform(0.7, 1.05))
            sample[start : start + width, paired] -= float(rng.uniform(0.2, 0.35))

        windows[idx] = sample.astype(np.float32, copy=False)

    return TelemetryDataset(
        windows=windows,
        features=extract_features(windows),
        labels=labels,
        class_names=CLASS_NAMES,
    )
