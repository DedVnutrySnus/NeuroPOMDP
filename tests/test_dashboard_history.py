from __future__ import annotations

from neuropomdp.config import BenchmarkConfig
from neuropomdp.dashboard.history import compare_runs, list_runs, load_run, save_run


def test_history_round_trip_and_empty_metrics(tmp_path) -> None:
    record = save_run("Demo", BenchmarkConfig(), None, history_dir=tmp_path)

    history = list_runs(tmp_path)
    assert len(history) == 1
    assert history[0]["run_id"] == record["run_id"]
    assert history[0]["mode"] == "Demo"
    assert history[0]["metrics"] == []
    assert load_run(str(record["run_id"]), tmp_path) == record
    assert compare_runs(history) == []
