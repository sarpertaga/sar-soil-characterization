from soilgeo.acquisition.s1_timeseries import (
    build_date_windows,
    date_to_season_index,
)


def test_build_date_windows_count_and_span():
    windows = build_date_windows("2023-01-01", "2023-12-31", step_days=12)
    # ~365/12 ≈ 31 windows
    assert 28 <= len(windows) <= 32
    # first window starts on start date
    assert windows[0][0] == "2023-01-01"
    # each window end >= start
    for s, e in windows:
        assert e >= s


def test_build_date_windows_no_overshoot():
    windows = build_date_windows("2023-01-01", "2023-01-20", step_days=12)
    # last window must not pass the end date
    assert windows[-1][1] <= "2023-01-20"


def test_date_to_season_index_classifies_months():
    windows = build_date_windows("2023-01-01", "2023-12-31", step_days=12)
    wet_idx, dry_idx = date_to_season_index(windows, wet_months=[1, 2, 3], dry_months=[7, 8, 9])
    # all wet indices map to a Jan–Mar start month
    from datetime import date
    for i in wet_idx:
        assert date.fromisoformat(windows[i][0]).month in (1, 2, 3)
    for i in dry_idx:
        assert date.fromisoformat(windows[i][0]).month in (7, 8, 9)
    assert len(wet_idx) > 0 and len(dry_idx) > 0
