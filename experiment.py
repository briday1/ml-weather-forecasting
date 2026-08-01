# %% Experiment settings
"""Run embedding training, inspection, prediction, validation, and plotting.

Prefer running the clearly named scripts individually:

    python scripts/train_embedding.py
    python scripts/inspect_embeddings.py
    python scripts/train.py
    python scripts/validate.py
    python scripts/plot.py

Edit settings in those scripts. This file remains as a notebook-style shortcut
and deliberately does not download data.
"""

import runpy
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

missing = [name for name in ("numpy", "plotly", "torch") if find_spec(name) is None]
if missing:
    project_python = ROOT / ".venv" / "bin" / "python"
    raise RuntimeError(
        f"This Python environment is missing: {', '.join(missing)}.\n"
        f"Run with the project environment instead:\n  {project_python} {__file__}"
    )

# %% Learn the spatial embedding
runpy.run_path(str(SCRIPTS / "train_embedding.py"), run_name="__main__")

# %% Build the observed-versus-reconstructed embedding report
runpy.run_path(str(SCRIPTS / "inspect_embeddings.py"), run_name="__main__")

# %% Train the frozen-embedding predictor
runpy.run_path(str(SCRIPTS / "train.py"), run_name="__main__")

# %% Evaluate
runpy.run_path(str(SCRIPTS / "validate.py"), run_name="__main__")

# %% Plot
runpy.run_path(str(SCRIPTS / "plot.py"), run_name="__main__")
