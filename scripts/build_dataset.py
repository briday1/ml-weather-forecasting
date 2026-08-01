"""Generate one single-site forecast dataset backed by a shared data lake."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
from _nexrad_archive import (
    NEXRAD_BASE_URL,
    STATION_CATALOG_URL,
    USER_AGENT,
    ArchiveObject,
    NexradStation,
    current_stations,
    discover_site_day,
    normalize_site,
)
from nexrad_viewer.formats.nexrad import read_level3_radial
from requests.adapters import HTTPAdapter


@dataclass(frozen=True)
class CandidateWindow:
    identifier: str
    requested_date: date
    frames: tuple[ArchiveObject, ...]
    duration_minutes: float

    @property
    def filenames(self) -> frozenset[str]:
        return frozenset(frame.filename for frame in self.frames)


@dataclass(frozen=True)
class CatalogDay:
    requested_date: date
    native_scans: int
    eligible_windows: int
    retained_candidates: int
    error: str = ""


def _date_value(value: str) -> date:
    return date.fromisoformat(value)


def _dates(start: date, stop: date) -> tuple[date, ...]:
    if start > stop:
        raise ValueError("start-date must not follow end-date")
    return tuple(
        start + timedelta(days=index) for index in range((stop - start).days + 1)
    )


def _catalog_candidate_pool(
    site: str,
    days: tuple[date, ...],
    *,
    workers: int,
    frames_per_example: int,
    minimum_duration_minutes: float,
    maximum_duration_minutes: float,
    maximum_per_date: int,
    seed: int,
) -> tuple[
    tuple[CandidateWindow, ...],
    tuple[CatalogDay, ...],
    tuple[dict[str, str], ...],
]:
    retained = []
    day_log = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(discover_site_day, site, day): day for day in days}
        for completed, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                scans = future.result()
            except Exception as error:  # noqa: BLE001
                detail = str(error) or type(error).__name__
                errors.append(
                    {
                        "date": day.isoformat(),
                        "error": detail,
                    }
                )
                day_log.append(CatalogDay(day, 0, 0, 0, detail))
                print(f"Catalog error for {site} {day}: {detail}")
                continue
            eligible = list(
                candidate_windows(
                    {day: scans},
                    frames_per_example=frames_per_example,
                    minimum_duration_minutes=minimum_duration_minutes,
                    maximum_duration_minutes=maximum_duration_minutes,
                )
            )
            random.Random(seed * 10_000 + day.toordinal()).shuffle(eligible)
            used_files: set[str] = set()
            day_candidates = []
            for candidate in eligible:
                if candidate.filenames & used_files:
                    continue
                day_candidates.append(candidate)
                used_files.update(candidate.filenames)
                if len(day_candidates) == maximum_per_date:
                    break
            retained.extend(day_candidates)
            day_log.append(
                CatalogDay(
                    day,
                    len(scans),
                    len(eligible),
                    len(day_candidates),
                )
            )
            if completed % 100 == 0 or completed == len(days):
                print(
                    f"Cataloged {completed:,}/{len(days):,} UTC dates · "
                    f"{len(retained):,} retained datetime candidates"
                )
    return (
        tuple(sorted(retained, key=lambda item: item.frames[0].scan_time)),
        tuple(sorted(day_log, key=lambda item: item.requested_date)),
        tuple(sorted(errors, key=lambda item: item["date"])),
    )


def candidate_windows(
    catalog: dict[date, tuple[ArchiveObject, ...]],
    *,
    frames_per_example: int,
    minimum_duration_minutes: float,
    maximum_duration_minutes: float,
) -> tuple[CandidateWindow, ...]:
    """Enumerate every eligible consecutive native-frame window."""
    candidates = []
    for requested_date, scans in catalog.items():
        for start in range(max(0, len(scans) - frames_per_example + 1)):
            frames = scans[start : start + frames_per_example]
            duration = (
                frames[-1].scan_time - frames[0].scan_time
            ).total_seconds() / 60.0
            if not minimum_duration_minutes <= duration <= maximum_duration_minutes:
                continue
            candidates.append(
                CandidateWindow(
                    identifier=(
                        f"{frames[0].scan_time:%Y%m%dT%H%M%S}-"
                        f"{frames[-1].scan_time:%H%M%S}"
                    ),
                    requested_date=requested_date,
                    frames=frames,
                    duration_minutes=duration,
                )
            )
    return tuple(candidates)


def _choose(
    candidates: tuple[CandidateWindow, ...],
    *,
    count: int,
    seed: int,
    maximum_per_date: int,
    used_files: set[str],
) -> tuple[CandidateWindow, ...]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    per_date = {}
    selected = []
    for candidate in shuffled:
        if per_date.get(candidate.requested_date, 0) >= maximum_per_date:
            continue
        if candidate.filenames & used_files:
            continue
        selected.append(candidate)
        used_files.update(candidate.filenames)
        per_date[candidate.requested_date] = (
            per_date.get(candidate.requested_date, 0) + 1
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} non-overlapping examples satisfied a request "
            f"for {count}; widen the date or duration range"
        )
    return tuple(sorted(selected, key=lambda item: item.frames[0].scan_time))


def select_examples(
    candidates: tuple[CandidateWindow, ...],
    *,
    count: int,
    validation_percent: int,
    seed: int,
    maximum_per_date: int,
) -> tuple[tuple[str, CandidateWindow], ...]:
    """Select reproducible examples with a chronological date holdout."""
    eligible_dates = sorted({candidate.requested_date for candidate in candidates})
    if len(eligible_dates) < 2:
        raise ValueError("Date-holdout splitting requires at least two eligible dates")
    validation_date_count = max(
        1,
        round(len(eligible_dates) * validation_percent / 100),
    )
    validation_dates = set(eligible_dates[-validation_date_count:])
    validation_count = max(1, round(count * validation_percent / 100))
    training_count = count - validation_count
    if training_count < 1:
        raise ValueError("Dataset must contain at least one training example")
    training_pool = tuple(
        candidate
        for candidate in candidates
        if candidate.requested_date not in validation_dates
    )
    validation_pool = tuple(
        candidate
        for candidate in candidates
        if candidate.requested_date in validation_dates
    )
    used_files: set[str] = set()
    training = _choose(
        training_pool,
        count=training_count,
        seed=seed,
        maximum_per_date=maximum_per_date,
        used_files=used_files,
    )
    validation = _choose(
        validation_pool,
        count=validation_count,
        seed=seed + 1,
        maximum_per_date=maximum_per_date,
        used_files=used_files,
    )
    return tuple(
        [("training", candidate) for candidate in training]
        + [("validation", candidate) for candidate in validation]
    )


def _lake_path(
    site: str,
    product: str,
    frame: ArchiveObject,
) -> Path:
    return (
        Path("nexrad-level3")
        / product
        / site
        / frame.scan_time.date().isoformat()
        / frame.filename
    )


def build_manifest(
    *,
    identifier: str,
    title: str,
    description: str,
    site: NexradStation,
    start_date: date,
    end_date: date,
    selections: tuple[tuple[str, CandidateWindow], ...],
    catalog_errors: tuple[dict[str, str], ...],
    frames_per_example: int,
    minimum_duration_minutes: float,
    maximum_duration_minutes: float,
    seed: int,
    maximum_per_date: int,
    validation_percent: int,
    radius_km: float,
    input_frames: int,
    grid_width: int,
    colormap: str,
    lake_root_relative: str,
) -> dict[str, object]:
    examples = []
    for index, (split_name, candidate) in enumerate(selections):
        examples.append(
            {
                "identifier": (
                    f"{site.identifier}-{candidate.identifier}-{index + 1:03d}"
                ),
                "split": split_name,
                "frames": [
                    {
                        "scan_time": frame.scan_time.isoformat(),
                        "remote_url": frame.url,
                        "filename": frame.filename,
                        "size_bytes": frame.size,
                        "checksum": frame.checksum,
                        "lake_path": _lake_path(
                            site.identifier,
                            "N0B",
                            frame,
                        ).as_posix(),
                    }
                    for frame in candidate.frames
                ],
            }
        )
    return {
        "schema_version": 1,
        "identifier": identifier,
        "title": title,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lake_root": lake_root_relative,
        "source": {
            "provider": "NOAA NEXRAD Open Data",
            "archive_base_url": NEXRAD_BASE_URL,
            "station_catalog_url": STATION_CATALOG_URL,
            "catalog_errors": list(catalog_errors),
        },
        "request": {
            "site": site.identifier,
            "product": "N0B",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "radius_km": radius_km,
        },
        "site": {
            "identifier": site.identifier,
            "name": site.name,
            "state": site.state,
            "latitude_deg": site.latitude_deg,
            "longitude_deg": site.longitude_deg,
        },
        "selection": {
            "requested_examples": len(selections),
            "frames_per_example": frames_per_example,
            "minimum_duration_minutes": minimum_duration_minutes,
            "maximum_duration_minutes": maximum_duration_minutes,
            "random_seed": seed,
            "non_overlapping": True,
            "maximum_examples_per_date": maximum_per_date,
        },
        "split": {
            "strategy": "chronological-date-holdout",
            "validation_percent": validation_percent,
        },
        "display": {
            "input_frames": input_frames,
            "radius_km": radius_km,
            "grid_width": grid_width,
            "colormap": colormap,
        },
        "examples": examples,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_indexes(
    dataset_root: Path,
    payload: dict[str, object],
    candidates: tuple[CandidateWindow, ...],
    selections: tuple[tuple[str, CandidateWindow], ...],
    day_log: tuple[CatalogDay, ...],
) -> None:
    selected = {
        candidate.identifier: split_name for split_name, candidate in selections
    }
    _atomic_csv(
        dataset_root / "catalog_days.csv",
        (
            "date",
            "native_scans",
            "eligible_windows",
            "retained_datetime_candidates",
            "error",
        ),
        (
            {
                "date": item.requested_date.isoformat(),
                "native_scans": item.native_scans,
                "eligible_windows": item.eligible_windows,
                "retained_datetime_candidates": item.retained_candidates,
                "error": item.error,
            }
            for item in day_log
        ),
    )
    _atomic_csv(
        dataset_root / "candidate_windows.csv",
        (
            "candidate",
            "date",
            "start",
            "end",
            "duration_minutes",
            "frames",
            "selected",
            "split",
        ),
        (
            {
                "candidate": candidate.identifier,
                "date": candidate.requested_date.isoformat(),
                "start": candidate.frames[0].scan_time.isoformat(),
                "end": candidate.frames[-1].scan_time.isoformat(),
                "duration_minutes": round(candidate.duration_minutes, 3),
                "frames": len(candidate.frames),
                "selected": "yes" if candidate.identifier in selected else "no",
                "split": selected.get(candidate.identifier, ""),
            }
            for candidate in candidates
        ),
    )
    _atomic_csv(
        dataset_root / "examples.csv",
        (
            "example",
            "split",
            "start",
            "end",
            "duration_minutes",
            "frames",
        ),
        (
            {
                "example": example["identifier"],
                "split": example["split"],
                "start": example["frames"][0]["scan_time"],
                "end": example["frames"][-1]["scan_time"],
                "duration_minutes": round(
                    (
                        datetime.fromisoformat(example["frames"][-1]["scan_time"])
                        - datetime.fromisoformat(example["frames"][0]["scan_time"])
                    ).total_seconds()
                    / 60,
                    3,
                ),
                "frames": len(example["frames"]),
            }
            for example in payload["examples"]
        ),
    )
    _atomic_csv(
        dataset_root / "frames.csv",
        (
            "example",
            "split",
            "frame",
            "scan_time",
            "filename",
            "size_bytes",
            "lake_path",
            "remote_url",
        ),
        (
            {
                "example": example["identifier"],
                "split": example["split"],
                "frame": position,
                "scan_time": frame["scan_time"],
                "filename": frame["filename"],
                "size_bytes": frame["size_bytes"],
                "lake_path": frame["lake_path"],
                "remote_url": frame["remote_url"],
            }
            for example in payload["examples"]
            for position, frame in enumerate(example["frames"])
        ),
    )


def _download(
    payload: dict[str, object],
    lake_root: Path,
    workers: int,
) -> tuple[str, ...]:
    frames = {
        frame["lake_path"]: frame
        for example in payload["examples"]
        for frame in example["frames"]
    }

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = HTTPAdapter(
        pool_connections=workers,
        pool_maxsize=workers,
        max_retries=2,
    )
    session.mount("https://", adapter)

    def fetch(item):
        relative, frame = item
        destination = lake_root / Path(relative).parent
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / frame["filename"]
        expected_size = int(frame["size_bytes"])
        if target.is_file() and target.stat().st_size == expected_size:
            return target
        temporary = target.with_name(f".{target.name}.part")
        digest = hashlib.md5()  # NOAA supplies an MD5 object ETag.
        with session.get(
            frame["remote_url"],
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 256):
                    if block:
                        stream.write(block)
                        digest.update(block)
                stream.flush()
                os.fsync(stream.fileno())
        if temporary.stat().st_size != expected_size:
            raise OSError(f"Incomplete download for {target.name}")
        checksum = frame.get("checksum")
        if checksum and checksum != f"md5:{digest.hexdigest()}":
            raise OSError(f"Checksum mismatch for {target.name}")
        os.replace(temporary, target)
        return target

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in frames.items()}
        for completed, future in enumerate(as_completed(futures), start=1):
            relative = futures[future]
            try:
                path = future.result()
            except Exception as error:  # noqa: BLE001
                errors.append(relative)
                print(
                    f"Data-lake error {completed}/{len(futures)} "
                    f"for {relative}: {error}"
                )
            else:
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        f"Data lake {completed:,}/{len(futures):,} · latest {path.name}"
                    )
    session.close()
    return tuple(sorted(errors))


def _sample_frame(
    path: Path,
    *,
    radius_km: float,
    azimuth_rays: int,
    range_gates: int,
) -> np.ndarray:
    scan = read_level3_radial(path)
    radial_order = np.argsort(np.asarray(scan.azimuth_center_deg))
    radial_indexes = radial_order[
        np.linspace(
            0,
            len(radial_order) - 1,
            num=azimuth_rays,
            dtype=np.int64,
        )
    ]
    in_range = np.flatnonzero(np.asarray(scan.ground_range_centers_km) <= radius_km)
    gate_indexes = in_range[
        np.linspace(
            0,
            len(in_range) - 1,
            num=range_gates,
            dtype=np.int64,
        )
    ]
    codes = scan.level_codes[np.ix_(radial_indexes, gate_indexes)]
    reflectivity = np.full(codes.shape, np.nan, dtype=np.float16)
    measured = codes >= 2
    reflectivity[measured] = (
        scan.header.minimum_value_dbz
        + (codes[measured].astype(np.float32) - 2) * scan.header.value_increment_dbz
    ).astype(np.float16)
    return reflectivity


def _prepare_tensor(
    payload: dict[str, object],
    dataset_root: Path,
    lake_root: Path,
    *,
    workers: int,
    azimuth_rays: int = 180,
    range_gates: int = 64,
) -> Path:
    examples = payload["examples"]
    identifiers = tuple(str(example["identifier"]) for example in examples)
    destination = dataset_root / "tensors" / f"polar-{azimuth_rays}x{range_gates}.npy"
    metadata_path = destination.with_suffix(".json")
    if destination.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("examples", ())) == identifiers:
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    shape = (
        len(examples),
        int(payload["selection"]["frames_per_example"]),
        azimuth_rays,
        range_gates,
    )
    values = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )
    radius_km = float(payload["request"]["radius_km"])

    def prepare(item):
        index, example = item
        try:
            frames = np.stack(
                [
                    _sample_frame(
                        lake_root / frame["lake_path"],
                        radius_km=radius_km,
                        azimuth_rays=azimuth_rays,
                        range_gates=range_gates,
                    )
                    for frame in example["frames"]
                ]
            )
            return index, frames, None
        except Exception as error:  # noqa: BLE001
            return index, None, str(error) or type(error).__name__

    failures = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for completed, (index, frame_values, error) in enumerate(
                executor.map(prepare, enumerate(examples)),
                start=1,
            ):
                if frame_values is None:
                    values[index] = np.nan
                    failures[identifiers[index]] = error
                else:
                    values[index] = frame_values
                if completed % 25 == 0 or completed == len(examples):
                    print(
                        f"Prepared native-polar tensor {completed:,}/{len(examples):,}"
                    )
        values.flush()
        del values
        os.replace(temporary, destination)
        _atomic_json(
            metadata_path,
            {
                "schema_version": 1,
                "dataset": payload["identifier"],
                "shape": list(shape),
                "dtype": "float16",
                "azimuth_rays": azimuth_rays,
                "range_gates": range_gates,
                "examples": list(identifiers),
                "failed_examples": failures,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_summary(
    payload: dict[str, object],
    dataset_root: Path,
    lake_root: Path,
) -> None:
    examples = payload["examples"]
    frames = {
        frame["lake_path"]: frame for example in examples for frame in example["frames"]
    }
    local = sum((lake_root / relative).is_file() for relative in frames)
    split_counts = {
        name: sum(example["split"] == name for example in examples)
        for name in ("training", "validation")
    }
    _atomic_json(
        dataset_root / "dataset-summary.json",
        {
            "schema_version": 1,
            "identifier": payload["identifier"],
            "title": payload["title"],
            "description": payload["description"],
            "site": payload["site"]["identifier"],
            "product": payload["request"]["product"],
            "start_date": payload["request"]["start_date"],
            "end_date": payload["request"]["end_date"],
            "first_timestamp": examples[0]["frames"][0]["scan_time"],
            "examples": len(examples),
            "training": split_counts["training"],
            "validation": split_counts["validation"],
            "frames_per_example": payload["selection"]["frames_per_example"],
            "radius_km": payload["request"]["radius_km"],
            "local_coverage": round(100 * local / max(1, len(frames)), 2),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Randomly select non-overlapping single-site native sequences and "
            "write one manifest-defined forecast dataset."
        )
    )
    result.add_argument("--id", required=True)
    result.add_argument("--title", required=True)
    result.add_argument("--description", default="")
    result.add_argument("--site", required=True)
    result.add_argument("--start-date", required=True, type=_date_value)
    result.add_argument("--end-date", required=True, type=_date_value)
    result.add_argument("--examples", type=int, default=6)
    result.add_argument("--frames-per-example", type=int, default=30)
    result.add_argument("--minimum-duration-minutes", type=float, default=45)
    result.add_argument("--maximum-duration-minutes", type=float, default=90)
    result.add_argument("--seed", type=int, default=20240520)
    result.add_argument("--maximum-examples-per-date", type=int, default=1)
    result.add_argument("--validation-percent", type=int, default=20)
    result.add_argument("--radius-km", type=float, default=120)
    result.add_argument("--input-frames", type=int, default=15)
    result.add_argument("--grid-width", type=int, default=384)
    result.add_argument("--colormap", default="NEXRAD")
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--download", action="store_true")
    return result


def build(args: argparse.Namespace) -> Path:
    identifier = args.id.strip()
    if not identifier or Path(identifier).name != identifier:
        raise ValueError("id must be a plain directory name")
    if args.workers < 1 or args.examples < 2:
        raise ValueError("workers must be positive and examples must be at least two")
    if args.frames_per_example < 2:
        raise ValueError("frames-per-example must be at least two")
    if not 1 <= args.input_frames < args.frames_per_example:
        raise ValueError("input-frames must leave at least one Y frame")
    if not 1 <= args.validation_percent <= 50:
        raise ValueError("validation-percent must be between 1 and 50")
    if args.maximum_examples_per_date < 1:
        raise ValueError("maximum-examples-per-date must be positive")
    if args.radius_km <= 0 or args.grid_width < 32:
        raise ValueError("radius-km and grid-width must be positive")
    site_id = normalize_site(args.site)
    station_lookup = {station.identifier: station for station in current_stations()}
    if site_id not in station_lookup:
        raise ValueError(f"Unknown current NEXRAD site: {site_id}")
    site = station_lookup[site_id]
    days = _dates(args.start_date, args.end_date)
    candidates, day_log, catalog_errors = _catalog_candidate_pool(
        site_id,
        days,
        workers=args.workers,
        frames_per_example=args.frames_per_example,
        minimum_duration_minutes=args.minimum_duration_minutes,
        maximum_duration_minutes=args.maximum_duration_minutes,
        maximum_per_date=args.maximum_examples_per_date,
        seed=args.seed,
    )
    selections = select_examples(
        candidates,
        count=args.examples,
        validation_percent=args.validation_percent,
        seed=args.seed,
        maximum_per_date=args.maximum_examples_per_date,
    )

    data_root = args.data_root.expanduser().resolve()
    dataset_root = data_root / "datasets" / identifier
    lake_root = data_root / "lake"
    dataset_root.mkdir(parents=True, exist_ok=True)
    relative_lake = os.path.relpath(lake_root, dataset_root)
    payload = build_manifest(
        identifier=identifier,
        title=args.title,
        description=args.description,
        site=site,
        start_date=args.start_date,
        end_date=args.end_date,
        selections=selections,
        catalog_errors=catalog_errors,
        frames_per_example=args.frames_per_example,
        minimum_duration_minutes=args.minimum_duration_minutes,
        maximum_duration_minutes=args.maximum_duration_minutes,
        seed=args.seed,
        maximum_per_date=args.maximum_examples_per_date,
        validation_percent=args.validation_percent,
        radius_km=args.radius_km,
        input_frames=args.input_frames,
        grid_width=args.grid_width,
        colormap=args.colormap,
        lake_root_relative=relative_lake,
    )
    _atomic_json(dataset_root / "dataset.json", payload)
    _write_indexes(dataset_root, payload, candidates, selections, day_log)
    print(
        f"Wrote {dataset_root}: {len(selections)} examples, "
        f"{args.frames_per_example} native frames each, "
        f"{len(candidates):,} eligible candidates"
    )
    if args.download:
        failures = _download(payload, lake_root, args.workers)
        if failures:
            print(
                f"{len(failures)} data-lake objects remain incomplete; "
                "rerun the command to retry"
            )
        else:
            _prepare_tensor(
                payload,
                dataset_root,
                lake_root,
                workers=args.workers,
            )
            _write_summary(payload, dataset_root, lake_root)
    return dataset_root


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
