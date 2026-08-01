"""Generate the ten-year KTLX corpus and its training-ready tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_dataset import build
from build_dataset import parser as dataset_parser


def main() -> None:
    wrapper = argparse.ArgumentParser(
        description=(
            "Select 1,000 reproducible KTLX datetime sequences across ten years "
            "and materialize their native Level III files in the shared lake."
        )
    )
    wrapper.add_argument(
        "--catalog-only",
        action="store_true",
        help="Write dataset/candidate metadata without downloading native scans",
    )
    wrapper.add_argument("--data-root", type=Path, default=Path("data"))
    wrapper.add_argument("--workers", type=int, default=32)
    options = wrapper.parse_args()
    arguments = [
        "--id",
        "ktlx-reflectivity-120km-2016-2025-v1",
        "--title",
        "KTLX reflectivity · 1,000 datetime sequences · 2016–2025",
        "--description",
        (
            "One thousand reproducibly selected KTLX datetime examples across "
            "ten years. Each example contains 30 consecutive unique native N0B "
            "scans spanning 45–90 minutes within 120 km of the radar."
        ),
        "--site",
        "KTLX",
        "--start-date",
        "2016-01-01",
        "--end-date",
        "2025-12-31",
        "--examples",
        "1000",
        "--frames-per-example",
        "30",
        "--minimum-duration-minutes",
        "45",
        "--maximum-duration-minutes",
        "90",
        "--seed",
        "20160101",
        "--maximum-examples-per-date",
        "12",
        "--validation-percent",
        "20",
        "--radius-km",
        "120",
        "--input-frames",
        "15",
        "--grid-width",
        "384",
        "--colormap",
        "NEXRAD",
        "--data-root",
        str(options.data_root),
        "--workers",
        str(options.workers),
    ]
    if not options.catalog_only:
        arguments.append("--download")
    dataset = build(dataset_parser().parse_args(arguments))
    print(f"Starter corpus ready: {dataset}")


if __name__ == "__main__":
    main()
