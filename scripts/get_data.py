"""Stage 1 — download NOAA NEXRAD data and prepare a Cartesian tensor.

Run from the repository root:  python scripts/get_data.py
"""

import sys

from generate_data import main as generate
from prepare_cartesian import main as prepare_cartesian

if __name__ == "__main__":
    generate()
    if "--catalog-only" not in sys.argv:
        prepare_cartesian()
