from geort.testing.repro_log import compare_logs


def test_identical_logs_match_line_by_line(tmp_path):
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    baseline.write_text("line 1\nline 2\n", encoding="utf-8")
    candidate.write_text("line 1\nline 2\n", encoding="utf-8")

    assert compare_logs(baseline, candidate) == {
        "equal": True,
        "first_difference": None,
    }


def test_log_comparator_reports_first_distinct_line(tmp_path):
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    baseline.write_text("line 1\nline 2\n", encoding="utf-8")
    candidate.write_text("line 1\nchanged\n", encoding="utf-8")

    assert compare_logs(baseline, candidate) == {
        "equal": False,
        "first_difference": {
            "line": 2,
            "baseline": "line 2\n",
            "candidate": "changed\n",
        },
    }
