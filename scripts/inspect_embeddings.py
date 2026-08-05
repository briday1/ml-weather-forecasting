"""Browse observed radar frames beside autoencoder reconstructions."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from forecast import (
    AUXILIARY_FEATURES,
    TENSOR_NAME,
    choose_device,
    load_data,
    normalized_to_dbz,
    sequence_tensors,
)
from model import ForecastModel
from radar_explorer import write_radar_comparison

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
AUTOENCODER = Path(os.environ.get("ML_WEATHER_AUTOENCODER", "outputs/autoencoder.pt"))
OUTPUT = Path(
    os.environ.get("ML_WEATHER_EMBEDDING_REPORT", "outputs/embedding-report.html")
)
VALIDATION_EXAMPLES = 20
DEVICE = os.environ.get("ML_WEATHER_DEVICE", "auto")

device = choose_device(DEVICE)
checkpoint = torch.load(AUTOENCODER, map_location=device, weights_only=False)
if checkpoint.get("model_type") != ForecastModel.AUTOENCODER_TYPE:
    raise ValueError("The autoencoder is outdated; run scripts/train_embedding.py")
model = ForecastModel(
    checkpoint["input_frames"],
    checkpoint["output_frames"],
    checkpoint["embedding_channels"],
    checkpoint["motion_skip"],
    checkpoint["motion_channels"],
    checkpoint.get("auxiliary_features", AUXILIARY_FEATURES),
).to(device)
model.frame_encoder.load_state_dict(checkpoint["frame_encoder_state"])
model.frame_decoder.load_state_dict(checkpoint["frame_decoder_state"])
model.motion_encoder.load_state_dict(checkpoint["motion_encoder_state"])
model.motion_decoder.load_state_dict(checkpoint["motion_decoder_state"])
model.eval()

radar, _ = load_data(DATASET)
if checkpoint.get("grid_size") != radar.shape[-1]:
    raise ValueError("The autoencoder resolution does not match the Cartesian tensor")
indexes = np.asarray(checkpoint["split"]["validation"], dtype=np.int64)
if VALIDATION_EXAMPLES is not None:
    indexes = indexes[:VALIDATION_EXAMPLES]

metadata_path = DATASET / "tensors" / Path(TENSOR_NAME).with_suffix(".json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
labels = [metadata["examples"][int(index)] for index in indexes]
observed, reconstructed, details, maes = [], [], [], []
for index in indexes:
    example = torch.from_numpy(np.array(radar[int(index)], dtype=np.float32)).unsqueeze(
        0
    )
    features, _, _ = sequence_tensors(example, checkpoint["input_frames"])
    with torch.inference_mode():
        reconstruction = model.reconstruct(features.to(device)).cpu()
    observed_dbz = normalized_to_dbz(features)[0]
    reconstructed_dbz = normalized_to_dbz(reconstruction)[0]
    mae = float((reconstructed_dbz - observed_dbz).abs().mean())
    observed.append(observed_dbz.numpy())
    reconstructed.append(reconstructed_dbz.numpy())
    maes.append(mae)
    details.append(f"reconstruction MAE {mae:.2f} dBZ")

write_radar_comparison(
    left=observed,
    right=reconstructed,
    labels=labels,
    axis_km=np.asarray(metadata["x_km"]),
    output=OUTPUT,
    title="Embedding reconstruction",
    left_title="Observed Cartesian radar",
    right_title="Encoder → decoder reconstruction",
    frame_label="Observed frame",
    title_details=details,
    show_histograms=True,
)
print(f"Validation examples: {len(labels)}")
print(f"Mean reconstruction MAE: {np.mean(maes):.2f} dBZ")
print(f"Explorer: {OUTPUT.resolve()}")
