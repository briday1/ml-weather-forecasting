from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_dataset
from _nexrad_archive import ArchiveObject


def _scans(day: date, count: int = 60) -> tuple[ArchiveObject, ...]:
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    return tuple(
        ArchiveObject(
            url=f"https://example.test/{day}/{index}",
            filename=f"TLX_N0B_{day:%Y_%m_%d}_{index:06d}",
            size=1000 + index,
            checksum=None,
            scan_time=start + timedelta(minutes=2 * index),
        )
        for index in range(count)
    )


def test_candidate_windows_are_exact_native_consecutive_sequences():
    day = date(2024, 5, 1)
    scans = _scans(day)

    candidates = build_dataset.candidate_windows(
        {day: scans},
        frames_per_example=30,
        minimum_duration_minutes=45,
        maximum_duration_minutes=90,
    )

    assert len(candidates) == 31
    assert candidates[0].frames == scans[:30]
    assert candidates[-1].frames == scans[-30:]
    assert candidates[0].duration_minutes == 58
    assert len({frame.filename for frame in candidates[0].frames}) == 30


def test_seeded_selection_holds_out_dates_and_never_reuses_a_scan():
    days = tuple(date(2024, 5, index) for index in range(1, 5))
    candidates = build_dataset.candidate_windows(
        {day: _scans(day) for day in days},
        frames_per_example=30,
        minimum_duration_minutes=45,
        maximum_duration_minutes=90,
    )

    selected = build_dataset.select_examples(
        candidates,
        count=4,
        validation_percent=25,
        seed=42,
        maximum_per_date=1,
    )
    repeated = build_dataset.select_examples(
        candidates,
        count=4,
        validation_percent=25,
        seed=42,
        maximum_per_date=1,
    )

    assert selected == repeated
    training_dates = {
        candidate.requested_date
        for split_name, candidate in selected
        if split_name == "training"
    }
    validation_dates = {
        candidate.requested_date
        for split_name, candidate in selected
        if split_name == "validation"
    }
    assert len(training_dates) == 3
    assert validation_dates == {days[-1]}
    assert training_dates.isdisjoint(validation_dates)
    filenames = [
        frame.filename for _, candidate in selected for frame in candidate.frames
    ]
    assert len(filenames) == len(set(filenames))


def test_selection_fails_instead_of_duplicating_or_interpolating_frames():
    day = date(2024, 5, 1)
    candidates = build_dataset.candidate_windows(
        {day: _scans(day, 29)},
        frames_per_example=30,
        minimum_duration_minutes=45,
        maximum_duration_minutes=90,
    )

    assert candidates == ()
