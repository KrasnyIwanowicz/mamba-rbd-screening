import numpy as np

from scripts.diagnose_subject import describe, rms


def test_rms_returns_expected_value():
    assert rms(np.array([3.0, 4.0])) == np.sqrt(12.5)


def test_describe_reports_empty_and_summary_statistics():
    assert describe([]) == "n=0"
    summary = describe([1.0, 3.0])
    assert "n=2" in summary
    assert "median=2.000e+00" in summary
