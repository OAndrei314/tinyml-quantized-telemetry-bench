# tinyml-quantized-telemetry-bench

Maintained by: codex-daily-routine

A small, dependency-light CPU benchmark for compact telemetry classifiers and int8
post-training quantization. It is meant to be boring in the best way: synthetic
data, inspectable models, no network calls, and markdown output you can commit.

## Research + Money Thesis

**Research question:** what accuracy, latency, memory, and model-size trade-offs appear
when telemetry models are quantized for embedded-style deployment?

**Money question:** efficient inference compounds across fleets. Smaller local telemetry
models can reduce memory footprint, CPU load, power draw, and upstream data movement while
enabling local fault detection near the hardware.

**Engineering evidence:** the benchmark reports fp32 versus int8 accuracy, latency,
throughput, classifier bytes, total model bytes, and compression, with deterministic data
and reproducible markdown output.

## Why this exists

TinyML telemetry classifiers often ship under tight memory and latency budgets. A
full edge deployment stack can obscure the basics, so this repo isolates the core
question:

> How much accuracy, latency, throughput, and model size changes when a compact
> fp32 telemetry classifier is post-training quantized to int8?

The MVP uses a deterministic synthetic dataset with normal, drift, spike, and
dropout classes. A small fp32 linear classifier is trained with closed-form ridge
regression, then wrapped in an int8 implementation that quantizes activations and
weights for integer dot products on CPU.

## How it works

```
synthetic windows -> feature extractor -> fp32 linear model -> int8 PTQ model
                                                     |
                                                     v
                                      metrics.json + report.md
```

- **Dataset**: generated locally with seeded NumPy RNG; no downloads.
- **Features**: compact per-sensor statistics such as mean, standard deviation,
  range, slope, energy, endpoints, and peak magnitude.
- **FP32 model**: closed-form ridge classifier over standardized features.
- **INT8 model**: post-training quantized weights and activations with int32
  accumulation and fp32 bias.
- **Metrics**: accuracy, latency, throughput, total model bytes, classifier bytes,
  and compression against the fp32 baseline.

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m telemetry_bench.cli run --out results/demo

pytest -v
```

The run command writes:

- `results/demo/metrics.json`
- `results/demo/report.md`

You can rebuild a report from a saved metrics file:

```bash
python -m telemetry_bench.cli report --metrics results/demo/metrics.json --out report.md
```

## Example report

| model | accuracy | p50_batch_latency_ms | sample_latency_us | throughput_samples_s | total_model_bytes | compression_vs_fp32 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | 1.00x |
| int8-ptq | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime | generated at runtime |

## Status

MVP: deterministic synthetic data, fp32 classifier, int8 post-training quantized
classifier, benchmark runner, CLI, markdown report, and network-free tests.

## License

MIT - see [LICENSE](LICENSE).
