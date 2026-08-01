"""Stage 5 — compare validation truth with learned forecasts."""

import json
from pathlib import Path

import numpy as np
import torch
from forecast import (
    TENSOR_NAME,
    choose_device,
    load_checkpoint,
    load_auxiliary,
    load_data,
    normalized_to_dbz,
    sequence_tensors,
)
from radar_explorer import write_radar_comparison

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
MODEL = Path("outputs/model.pt")
OUTPUT = Path("outputs/validation-explorer.html")
VALIDATION_EXAMPLES = 20
DEVICE = "auto"

device = choose_device(DEVICE)
checkpoint, model = load_checkpoint(MODEL, device)
radar, _ = load_data(DATASET)
auxiliary = load_auxiliary(DATASET, checkpoint["input_frames"])
if checkpoint.get("grid_size") != radar.shape[-1]:
    raise ValueError("The model checkpoint does not match the Cartesian tensor")
indexes = checkpoint["split"]["validation"]
if VALIDATION_EXAMPLES is not None:
    indexes = indexes[:VALIDATION_EXAMPLES]

metadata_path = DATASET / "tensors" / Path(TENSOR_NAME).with_suffix(".json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
labels = [metadata["examples"][int(index)] for index in indexes]
truth, forecast = [], []
for index in indexes:
    example = torch.from_numpy(np.array(radar[int(index)], dtype=np.float32)).unsqueeze(0)
    features, targets, _ = sequence_tensors(example, checkpoint["input_frames"])
    with torch.inference_mode():
        context = torch.from_numpy(auxiliary[int(index)]).unsqueeze(0).to(device)
        prediction = model(features.to(device), context).cpu()
    truth.append(normalized_to_dbz(targets)[0].numpy())
    forecast.append(normalized_to_dbz(prediction)[0].numpy())

write_radar_comparison(
    left=truth,
    right=forecast,
    labels=labels,
    axis_km=np.asarray(metadata["x_km"]),
    output=OUTPUT,
    title="Validation forecast",
    left_title="Observed validation truth",
    right_title="Learned forecast",
    frame_label="Forecast frame",
    show_histograms=True,
)
print(f"Validation examples: {len(labels)}")
print(f"Explorer: {OUTPUT.resolve()}")
