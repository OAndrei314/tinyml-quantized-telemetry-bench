"""Accuracy, latency, throughput, and size metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    repeats: int
    samples: int
    p50_batch_latency_ms: float
    mean_batch_latency_ms: float
    sample_latency_us: float
    throughput_samples_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "repeats": self.repeats,
            "samples": self.samples,
            "p50_batch_latency_ms": self.p50_batch_latency_ms,
            "mean_batch_latency_ms": self.mean_batch_latency_ms,
            "sample_latency_us": self.sample_latency_us,
            "throughput_samples_s": self.throughput_samples_s,
        }


def accuracy_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if labels.size == 0:
        raise ValueError("accuracy requires at least one sample")
    return float(np.mean(labels == predictions))


def measure_predict_latency(
    model: object,
    features: np.ndarray,
    repeats: int = 50,
    warmups: int = 5,
) -> LatencyStats:
    """Measure CPU prediction latency for a full feature batch."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if len(features) == 0:
        raise ValueError("features must contain at least one sample")

    predict = getattr(model, "predict")
    for _ in range(max(0, warmups)):
        predict(features)

    durations_ns = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        predict(features)
        durations_ns.append(time.perf_counter_ns() - start)

    durations = np.asarray(durations_ns, dtype=np.float64)
    p50_ns = float(np.median(durations))
    mean_ns = float(np.mean(durations))
    samples = int(len(features))

    return LatencyStats(
        repeats=int(repeats),
        samples=samples,
        p50_batch_latency_ms=p50_ns / 1.0e6,
        mean_batch_latency_ms=mean_ns / 1.0e6,
        sample_latency_us=(p50_ns / samples) / 1.0e3,
        throughput_samples_s=samples / (p50_ns / 1.0e9),
    )
