"""Markdown report generation for benchmark metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def build_report(summary: Mapping[str, Any]) -> str:
    models = list(summary.get("models", []))
    if not models:
        return "# TinyML Telemetry Benchmark Report\n\nNo model results found.\n"

    dataset = summary.get("dataset", {})
    config = summary.get("config", {})
    class_names = ", ".join(dataset.get("class_names", []))

    lines = [
        "# TinyML Telemetry Benchmark Report",
        "",
        "## Dataset",
        "",
        f"- Train samples: {dataset.get('train_samples')}",
        f"- Test samples: {dataset.get('test_samples')}",
        f"- Window shape: {dataset.get('seq_len')} timesteps x {dataset.get('n_sensors')} sensors",
        f"- Feature count: {dataset.get('feature_count')}",
        f"- Classes: {class_names}",
        f"- Seed: {config.get('seed')}",
        "",
        "## Results",
        "",
        "| model | accuracy | p50_batch_latency_ms | sample_latency_us | throughput_samples_s | total_model_bytes | classifier_bytes | compression_vs_fp32 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in models:
        latency = row["latency"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    _fmt_float(row["accuracy"], 4),
                    _fmt_float(latency["p50_batch_latency_ms"], 3),
                    _fmt_float(latency["sample_latency_us"], 3),
                    _fmt_float(latency["throughput_samples_s"], 1),
                    str(row["total_model_bytes"]),
                    str(row["classifier_bytes"]),
                    f"{_fmt_float(row['compression_vs_fp32'], 2)}x",
                ]
            )
            + " |"
        )

    deployment = summary.get("deployment", {})
    if deployment:
        lines.extend(
            [
                "",
                "## Deployment Readiness",
                "",
                f"- Recommended model: `{deployment.get('recommended_model')}`",
                "",
                "| model | readiness_score | passed | violations |",
                "| --- | ---: | :---: | --- |",
            ]
        )
        for row in deployment.get("models", []):
            violations = ", ".join(row.get("violations", [])) or "none"
            lines.append(
                f"| {row.get('model')} | {_fmt_float(row.get('readiness_score', 0), 3)} | "
                f"{'yes' if row.get('passed') else 'no'} | {violations} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Latency is measured for a full test-feature batch on the current CPU process.",
            "- The int8 path uses quantized activations and weights with int32 accumulation.",
            "- Model bytes include classifier state plus feature standardization parameters.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(summary: Mapping[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(summary), encoding="utf-8")
    return out_path
