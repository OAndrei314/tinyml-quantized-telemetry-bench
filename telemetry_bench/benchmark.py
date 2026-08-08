"""End-to-end benchmark runner."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import generate_synthetic_telemetry
from .metrics import accuracy_score, measure_predict_latency
from .models import QuantizedLinearClassifier, train_linear_classifier
from .report import build_report


@dataclass(frozen=True)
class BenchmarkConfig:
    train_samples: int = 768
    test_samples: int = 384
    seq_len: int = 32
    n_sensors: int = 4
    seed: int = 7
    repeats: int = 50
    l2: float = 1.0e-2


def _evaluate_model(name: str, model: object, test_dataset: object, repeats: int) -> dict[str, Any]:
    predictions = model.predict(test_dataset.features)
    latency = measure_predict_latency(model, test_dataset.features, repeats=repeats)
    return {
        "name": name,
        "accuracy": accuracy_score(test_dataset.labels, predictions),
        "latency": latency.as_dict(),
        "total_model_bytes": model.nbytes(),
        "classifier_bytes": model.classifier_nbytes(),
    }


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Generate data, train fp32, quantize to int8, and collect metrics."""

    config = config or BenchmarkConfig()
    train_dataset = generate_synthetic_telemetry(
        n_samples=config.train_samples,
        seq_len=config.seq_len,
        n_sensors=config.n_sensors,
        seed=config.seed,
    )
    test_dataset = generate_synthetic_telemetry(
        n_samples=config.test_samples,
        seq_len=config.seq_len,
        n_sensors=config.n_sensors,
        seed=config.seed + 1,
    )

    fp32_model = train_linear_classifier(train_dataset, l2=config.l2)
    int8_model = QuantizedLinearClassifier.from_fp32(fp32_model, train_dataset.features)

    models = [
        _evaluate_model("fp32", fp32_model, test_dataset, config.repeats),
        _evaluate_model("int8-ptq", int8_model, test_dataset, config.repeats),
    ]
    fp32_bytes = models[0]["total_model_bytes"]
    for row in models:
        row["compression_vs_fp32"] = fp32_bytes / row["total_model_bytes"]

    return {
        "benchmark": "tinyml-quantized-telemetry-bench",
        "config": asdict(config),
        "dataset": {
            "train_samples": config.train_samples,
            "test_samples": config.test_samples,
            "seq_len": config.seq_len,
            "n_sensors": config.n_sensors,
            "feature_count": int(train_dataset.features.shape[1]),
            "class_names": list(train_dataset.class_names),
        },
        "models": models,
    }


def write_metrics(summary: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return path


def run_and_write(config: BenchmarkConfig, out_dir: str | Path) -> dict[str, Path]:
    summary = run_benchmark(config)
    out_dir = Path(out_dir)
    metrics_path = write_metrics(summary, out_dir)
    report_path = out_dir / "report.md"
    report_path.write_text(build_report(summary), encoding="utf-8")
    return {"metrics": metrics_path, "report": report_path}
