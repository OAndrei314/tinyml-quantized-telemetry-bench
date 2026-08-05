"""FP32 and int8 compact classifiers for telemetry features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import TelemetryDataset


def _standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features.astype(np.float32, copy=False) - mean) / scale).astype(np.float32)


def _symmetric_scale(values: np.ndarray, qmax: int = 127) -> float:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    return max(max_abs / qmax, 1.0e-8)


@dataclass(frozen=True)
class LinearClassifier:
    """A small fp32 linear classifier over standardized telemetry features."""

    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    class_names: tuple[str, ...]

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        x = _standardize(features, self.feature_mean, self.feature_scale)
        return x @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(features), axis=1).astype(np.int64)

    def classifier_nbytes(self) -> int:
        return int(self.weights.nbytes + self.bias.nbytes)

    def nbytes(self) -> int:
        return int(
            self.weights.nbytes
            + self.bias.nbytes
            + self.feature_mean.nbytes
            + self.feature_scale.nbytes
        )


@dataclass(frozen=True)
class QuantizedLinearClassifier:
    """An int8 post-training quantized linear classifier.

    The implementation quantizes standardized inputs and per-class weights to
    int8, accumulates in int32, then dequantizes scores before adding fp32 bias.
    """

    q_weights: np.ndarray
    weight_scales: np.ndarray
    input_scale: float
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    class_names: tuple[str, ...]

    @classmethod
    def from_fp32(
        cls, model: LinearClassifier, calibration_features: np.ndarray
    ) -> "QuantizedLinearClassifier":
        x = _standardize(calibration_features, model.feature_mean, model.feature_scale)
        input_scale = _symmetric_scale(x)
        weight_scales = np.max(np.abs(model.weights), axis=0) / 127.0
        weight_scales = np.maximum(weight_scales, 1.0e-8).astype(np.float32)
        q_weights = np.clip(np.round(model.weights / weight_scales), -127, 127).astype(
            np.int8
        )
        return cls(
            q_weights=q_weights,
            weight_scales=weight_scales,
            input_scale=float(input_scale),
            bias=model.bias.copy(),
            feature_mean=model.feature_mean.copy(),
            feature_scale=model.feature_scale.copy(),
            class_names=model.class_names,
        )

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        x = _standardize(features, self.feature_mean, self.feature_scale)
        q_inputs = np.clip(np.round(x / self.input_scale), -127, 127).astype(np.int8)
        accum = q_inputs.astype(np.int32) @ self.q_weights.astype(np.int32)
        scales = (self.input_scale * self.weight_scales).astype(np.float32)
        return accum.astype(np.float32) * scales[None, :] + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(features), axis=1).astype(np.int64)

    def classifier_nbytes(self) -> int:
        return int(
            self.q_weights.nbytes
            + self.weight_scales.nbytes
            + np.dtype(np.float32).itemsize
            + self.bias.nbytes
        )

    def nbytes(self) -> int:
        return int(
            self.classifier_nbytes()
            + self.feature_mean.nbytes
            + self.feature_scale.nbytes
        )


def train_linear_classifier(dataset: TelemetryDataset, l2: float = 1.0e-2) -> LinearClassifier:
    """Fit a small fp32 linear classifier with closed-form ridge regression."""

    features = dataset.features.astype(np.float32, copy=False)
    labels = dataset.labels.astype(np.int64, copy=False)
    n_classes = len(dataset.class_names)

    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_scale = np.std(features, axis=0).astype(np.float32)
    feature_scale = np.where(feature_scale < 1.0e-6, 1.0, feature_scale).astype(np.float32)

    x = _standardize(features, feature_mean, feature_scale).astype(np.float64)
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.eye(n_classes, dtype=np.float64)[labels]

    reg = np.eye(x_bias.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    params = np.linalg.solve(x_bias.T @ x_bias + reg, x_bias.T @ y)

    return LinearClassifier(
        weights=params[:-1].astype(np.float32),
        bias=params[-1].astype(np.float32),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        class_names=dataset.class_names,
    )
