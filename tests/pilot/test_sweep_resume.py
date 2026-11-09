"""A long sweep must save each result and retry failed work on restart."""

import csv
import sys

import pytest

from scripts.pilot import sweep


def _row(tag, status):
    row = dict.fromkeys(sweep.FIELDS, "")
    row.update(tag=tag, status=status, wall_sec=1.0)
    return row


def test_resume_retries_failures(tmp_path):
    path = tmp_path / "results.csv"
    sweep._append_row(path, _row("complete", "ok"))
    sweep._append_row(path, _row("retry", "error"))
    assert sweep._load_done(path) == {"complete"}
    sweep._append_row(path, _row("retry", "ok"))
    assert sweep._load_done(path) == {"complete", "retry"}


def test_result_is_saved_before_next_run_and_failures_are_counted(tmp_path):
    path = tmp_path / "results.csv"

    def rows():
        yield _row("first", "ok")
        with path.open() as source:
            assert [r["tag"] for r in csv.DictReader(source)] == ["first"]
        yield _row("second", "error")

    assert sweep._record_rows(path, rows()) == 1
    with path.open() as source:
        assert len(list(csv.DictReader(source))) == 2


def test_failed_sweep_has_nonzero_exit_and_keeps_its_result(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sweep", "--device", "cpu", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(sweep, "enumerate_grid", lambda cfg: [("gru", "off", 1.0, 0)])

    def fail(spec):
        assert spec[-1] == str(tmp_path)
        return _row("failed", "error")

    monkeypatch.setattr(sweep, "run_one", fail)
    with pytest.raises(SystemExit, match="1"):
        sweep.main()
    assert sweep._load_done(tmp_path / "results.csv") == set()
    assert (tmp_path / "results.csv").is_file()
