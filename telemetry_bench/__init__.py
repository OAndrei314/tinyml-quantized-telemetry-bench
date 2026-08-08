"""TinyML telemetry quantization benchmark."""

from .benchmark import BenchmarkConfig, DeploymentTarget, run_benchmark, score_deployment_readiness

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "DeploymentTarget",
    "run_benchmark",
    "score_deployment_readiness",
]
