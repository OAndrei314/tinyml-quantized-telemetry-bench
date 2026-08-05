from telemetry_bench import __version__
from telemetry_bench.benchmark import BenchmarkConfig


def test_version_is_defined():
    assert __version__ == "0.1.0"


def test_benchmark_config_defaults_are_small():
    assert BenchmarkConfig.train_samples <= 1024
    assert BenchmarkConfig.repeats <= 100
