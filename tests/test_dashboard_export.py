from __future__ import annotations

import json

from neuropomdp.dashboard.export import (
    metrics_csv_bytes,
    result_json_bytes,
    rows_to_csv_bytes,
    timestep_csv_bytes,
)


def test_empty_exports_do_not_fail() -> None:
    assert rows_to_csv_bytes([]) == b""
    assert metrics_csv_bytes(None) == b""
    assert timestep_csv_bytes(None) == b""

    payload = json.loads(result_json_bytes(None, None))
    assert payload == {
        "model_parameters": {},
        "results": {},
        "run_config": {},
    }
