import json

from telemetry_bench.cli import main


def test_cli_run_writes_metrics_and_report(tmp_path):
    out_dir = tmp_path / "run"

    code = main(
        [
            "run",
            "--out",
            str(out_dir),
            "--train-samples",
            "192",
            "--test-samples",
            "96",
            "--repeats",
            "3",
        ]
    )

    assert code == 0
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"
    assert metrics_path.exists()
    assert report_path.exists()

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_names = {row["name"] for row in summary["models"]}
    assert model_names == {"fp32", "int8-ptq"}
    assert summary["dataset"]["feature_count"] == 32
    assert "TinyML Telemetry Benchmark Report" in report_path.read_text(encoding="utf-8")


def test_cli_report_rebuilds_markdown(tmp_path):
    out_dir = tmp_path / "run"
    report_path = tmp_path / "rebuilt.md"

    assert main(["run", "--out", str(out_dir), "--repeats", "2"]) == 0
    assert (
        main(
            [
                "report",
                "--metrics",
                str(out_dir / "metrics.json"),
                "--out",
                str(report_path),
            ]
        )
        == 0
    )

    assert report_path.exists()
    assert "| fp32 |" in report_path.read_text(encoding="utf-8")
