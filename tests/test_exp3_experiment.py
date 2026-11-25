"""Fast integration test for Experiment 3 (Sec. 3, Theorem 3.2 / Corollary 3.3)."""
from __future__ import annotations

from pathlib import Path

from exp.exp3 import M_STEPS_LIST, RESULTS_DIR, run


def test_exp3_run_end_to_end(monkeypatch):
    """run() completes, satisfies Corollary 3.3 / nondegeneracy, and writes
    both result artifacts to disk, using a small, fast step-count list."""
    import exp.exp3 as exp3_mod

    monkeypatch.setattr(exp3_mod, "M_STEPS_LIST", (4, 8))

    result = exp3_mod.run()
    table_rows = result["table_rows"]

    assert len(table_rows) > 0
    assert any(row["x0_star"] >= row["frictionless_price"] - 1e-6 for row in table_rows)

    midpoint_rows = [row for row in table_rows if row["regime"] == "midpoint"]
    assert len(midpoint_rows) > 0
    assert all(row["nondegenerate"] for row in midpoint_rows)

    json_path = RESULTS_DIR / "exp3_replication_cost.json"
    png_path = RESULTS_DIR / "exp3_replication.png"
    assert json_path.exists()
    assert png_path.exists()
