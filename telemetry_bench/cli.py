"""Command-line entry point: `python -m telemetry_bench.cli run|report ...`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import BenchmarkConfig, run_and_write
from .report import build_report, load_metrics, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="telemetry-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the fp32 vs int8 telemetry benchmark")
    run_p.add_argument("--out", required=True, help="output directory for metrics and report")
    run_p.add_argument("--train-samples", type=int, default=BenchmarkConfig.train_samples)
    run_p.add_argument("--test-samples", type=int, default=BenchmarkConfig.test_samples)
    run_p.add_argument("--seq-len", type=int, default=BenchmarkConfig.seq_len)
    run_p.add_argument("--sensors", type=int, default=BenchmarkConfig.n_sensors)
    run_p.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    run_p.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats)
    run_p.add_argument("--l2", type=float, default=BenchmarkConfig.l2)

    report_p = sub.add_parser("report", help="build markdown from a metrics JSON file")
    report_p.add_argument("--metrics", required=True, help="path to metrics.json")
    report_p.add_argument("--out", required=True, help="output markdown file path")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = BenchmarkConfig(
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seq_len=args.seq_len,
            n_sensors=args.sensors,
            seed=args.seed,
            repeats=args.repeats,
            l2=args.l2,
        )
        paths = run_and_write(config, Path(args.out))
        print(f"wrote metrics -> {paths['metrics']}")
        print(f"wrote report  -> {paths['report']}")
        return 0

    if args.command == "report":
        summary = load_metrics(args.metrics)
        write_report(summary, args.out)
        print(build_report(summary))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
