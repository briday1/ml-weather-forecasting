# %% Experiment settings
"""Run isolated embedding, forecast, and validation experiments by radius.

The same examples and 64×64 model shape are used at every radius so the runs
are directly comparable. Each subprocess gets its own tensor and output paths;
no checkpoint or validation result is shared between radii.
"""

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
DATASET = ROOT / "data/datasets/ktlx-reflectivity-120km-2016-2025-v1"
OUTPUT_ROOT = ROOT / "outputs"
RADII_KM = (120, 230, 460)
GRID_SIZE = 64
DEVICE = "auto"

missing = [name for name in ("numpy", "plotly", "torch") if find_spec(name) is None]
if missing:
    project_python = ROOT / ".venv" / "bin" / "python"
    raise RuntimeError(
        f"This Python environment is missing: {', '.join(missing)}.\n"
        f"Run with the project environment instead:\n  {project_python} {__file__}"
    )

results = []
for radius_km in RADII_KM:
    slug = f"radius-{radius_km:g}km"
    run_directory = OUTPUT_ROOT / slug
    run_directory.mkdir(parents=True, exist_ok=True)
    tensor_name = f"cartesian-z-{radius_km:g}km-{GRID_SIZE}x{GRID_SIZE}.npy"
    tensor = DATASET / "tensors" / tensor_name
    autoencoder = run_directory / "autoencoder.pt"
    model = run_directory / "model.pt"
    metrics = run_directory / "validation.json"
    manifest = run_directory / "experiment.json"
    environment = {
        **os.environ,
        "ML_WEATHER_RADIUS_KM": f"{radius_km:g}",
        "ML_WEATHER_GRID_SIZE": str(GRID_SIZE),
        "ML_WEATHER_TENSOR": tensor_name,
        "ML_WEATHER_AUTOENCODER": str(autoencoder),
        "ML_WEATHER_MODEL": str(model),
        "ML_WEATHER_METRICS": str(metrics),
        "ML_WEATHER_DEVICE": DEVICE,
    }
    configuration = {
        "radius_km": radius_km,
        "grid_size": GRID_SIZE,
        "tensor": tensor_name,
        "dataset": str(DATASET.relative_to(ROOT)),
        "autoencoder": str(autoencoder.relative_to(ROOT)),
        "model": str(model.relative_to(ROOT)),
        "validation": str(metrics.relative_to(ROOT)),
    }
    manifest.write_text(
        json.dumps({**configuration, "status": "running"}, indent=2),
        encoding="utf-8",
    )

    # %% Prepare radius data
    if tensor.is_file() and tensor.with_suffix(".json").is_file():
        print(f"Resume {radius_km:g} km: using {tensor}", flush=True)
    else:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_cartesian.py")],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    # %% Train embedding
    if autoencoder.is_file():
        print(f"Resume {radius_km:g} km: using {autoencoder}", flush=True)
    else:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "train_embedding.py")],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    # %% Train forecast
    if model.is_file():
        print(f"Resume {radius_km:g} km: using {model}", flush=True)
    else:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "train.py")],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    # %% Evaluate
    if metrics.is_file():
        print(f"Resume {radius_km:g} km: using {metrics}", flush=True)
    else:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "validate.py")],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    validation = json.loads(metrics.read_text(encoding="utf-8"))
    result = {**configuration, "status": "complete", "metrics": validation}
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    results.append(result)

# %% Plot / return aggregate results
aggregate = OUTPUT_ROOT / "experiment-results.json"
aggregate.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"Experiment results: {aggregate.resolve()}")
