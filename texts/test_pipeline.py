"""
tests/test_pipeline.py
-----------------------
Unit tests for pipeline/validator.py, pipeline/checkpoint.py,
pipeline/reporter.py, and pipeline/processor.process_single_case().

Run:  python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.validator   import validate, REQUIRED_COLS
from pipeline.checkpoint  import Checkpoint
from pipeline.reporter    import save_outputs, generate_summary_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _good_df(n: int = 20) -> pd.DataFrame:
    """Minimal valid ETL output DataFrame."""
    return pd.DataFrame({
        "Case_ID":       [str(i) for i in range(n)],
        "Year":          ["1958"] * n,
        "Court":         ["Connecticut Superior Court"] * n,
        "Case_Text":     ["plaintiff filed suit for negligence damages"] * n,
        "Verdict":       ["Judgment AFFIRMED"] * n,
        "Legal_Citations": ["12 Conn. 345"] * n,
        "Case_Type":     ["Torts"] * n,
        "Sub_Type":      ["Negligence"] * n,
        "Num_Citations": [3] * n,
        "Source_Folder": ["test_source"] * n,
    })


# ── validator ─────────────────────────────────────────────────────────────────

class TestValidator:

    def test_valid_df_passes(self):
        result = validate(_good_df())
        assert result.ok is True
        assert result.errors == []

    def test_empty_df_fails(self):
        result = validate(pd.DataFrame())
        assert result.ok is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_missing_required_column_fails(self):
        df     = _good_df().drop(columns=["Verdict"])
        result = validate(df)
        assert result.ok is False
        assert any("Verdict" in e for e in result.errors)

    def test_all_required_columns_checked(self):
        for col in REQUIRED_COLS:
            df     = _good_df().drop(columns=[col])
            result = validate(df)
            assert result.ok is False, f"Should fail when {col} is missing"

    def test_empty_case_text_warns(self):
        df              = _good_df()
        df.loc[0, "Case_Text"] = ""
        result          = validate(df)
        assert any("empty Case_Text" in w for w in result.warnings)

    def test_duplicate_case_ids_warns(self):
        df             = _good_df(10)
        df.loc[0, "Case_ID"] = df.loc[1, "Case_ID"]
        result         = validate(df)
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_report_returns_string(self):
        result = validate(_good_df())
        text   = result.report()
        assert isinstance(text, str)
        assert "VALIDATION REPORT" in text

    def test_stats_contains_total_rows(self):
        result = validate(_good_df(15))
        assert result.stats.get("total_rows") == 15

    def test_stats_contains_year_range(self):
        result = validate(_good_df())
        assert "year_range" in result.stats

    def test_low_verdict_coverage_warns(self):
        df              = _good_df(20)
        df["Verdict"]   = "Verdict Unknown"
        result          = validate(df)
        assert any("Verdict" in w for w in result.warnings)


# ── checkpoint ────────────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_new_checkpoint_is_empty(self, tmp_path):
        ck = Checkpoint(str(tmp_path / "ck.json"))
        assert len(ck) == 0

    def test_mark_done_and_is_done(self, tmp_path):
        ck = Checkpoint(str(tmp_path / "ck.json"))
        ck.mark_done("case_001")
        assert ck.is_done("case_001")
        assert not ck.is_done("case_002")

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "ck.json")
        ck1  = Checkpoint(path)
        ck1.mark_done("abc")
        ck2  = Checkpoint(path)
        assert ck2.is_done("abc")

    def test_mark_done_batch(self, tmp_path):
        ck = Checkpoint(str(tmp_path / "ck.json"))
        ck.mark_done_batch(["a", "b", "c"])
        assert ck.is_done("a") and ck.is_done("b") and ck.is_done("c")

    def test_reset_clears_state(self, tmp_path):
        path = str(tmp_path / "ck.json")
        ck   = Checkpoint(path)
        ck.mark_done("x")
        ck.reset()
        assert len(ck) == 0
        assert not os.path.exists(path)

    def test_stats_returns_dict(self, tmp_path):
        ck    = Checkpoint(str(tmp_path / "ck.json"))
        stats = ck.stats()
        assert "completed_cases" in stats
        assert "checkpoint_file" in stats

    def test_len_reflects_count(self, tmp_path):
        ck = Checkpoint(str(tmp_path / "ck.json"))
        ck.mark_done_batch(["1", "2", "3"])
        assert len(ck) == 3

    def test_ids_coerced_to_string(self, tmp_path):
        ck = Checkpoint(str(tmp_path / "ck.json"))
        ck.mark_done(42)           # integer ID
        assert ck.is_done("42")    # string lookup must work


# ── reporter ──────────────────────────────────────────────────────────────────

class TestReporter:

    def test_save_outputs_creates_csv(self, tmp_path):
        df = _good_df(5)
        save_outputs(df, str(tmp_path))
        assert (tmp_path / "legal_cases_complete.csv").exists()

    def test_save_outputs_creates_json(self, tmp_path):
        df = _good_df(5)
        save_outputs(df, str(tmp_path))
        assert (tmp_path / "legal_cases_complete.json").exists()

    def test_csv_is_readable(self, tmp_path):
        df = _good_df(5)
        save_outputs(df, str(tmp_path))
        loaded = pd.read_csv(tmp_path / "legal_cases_complete.csv")
        assert len(loaded) == 5
        assert "Case_ID" in loaded.columns

    def test_json_is_valid(self, tmp_path):
        df = _good_df(5)
        save_outputs(df, str(tmp_path))
        with open(tmp_path / "legal_cases_complete.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_generate_summary_report_creates_file(self, tmp_path, capsys):
        df = _good_df(10)
        generate_summary_report(df, str(tmp_path))
        assert (tmp_path / "etl_summary_report.txt").exists()

    def test_summary_report_contains_key_sections(self, tmp_path, capsys):
        df = _good_df(10)
        generate_summary_report(df, str(tmp_path))
        text = (tmp_path / "etl_summary_report.txt").read_text()
        assert "CASE TYPE" in text
        assert "VERDICT"   in text
        assert "QUALITY"   in text

    def test_summary_report_total_cases(self, tmp_path, capsys):
        df = _good_df(7)
        generate_summary_report(df, str(tmp_path))
        text = (tmp_path / "etl_summary_report.txt").read_text()
        assert "7" in text