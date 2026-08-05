"""Stage 3 — report final metrics on the untouched test split."""

import json
import os
from pathlib import Path

import torch
from forecast import (
    choose_device,
    load_auxiliary,
    load_checkpoint,
    load_data,
    loader,
    normalized_to_linear_z,
    sequence_tensors,
)

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
MODEL = Path(os.environ.get("ML_WEATHER_MODEL", "outputs/model.pt"))
METRICS = Path(os.environ.get("ML_WEATHER_METRICS", "outputs/validation.json"))
BATCH_SIZE = 8
DEVICE = os.environ.get("ML_WEATHER_DEVICE", "auto")

device = choose_device(DEVICE)
checkpoint, model = load_checkpoint(MODEL, device)
radar, _ = load_data(DATASET)
auxiliary = load_auxiliary(DATASET, checkpoint["input_frames"])
if checkpoint.get("grid_size") != radar.shape[-1]:
    raise ValueError("The model checkpoint does not match the Cartesian tensor")
forecast_squared = forecast_absolute = forecast_valid = 0.0
reconstruction_squared = reconstruction_absolute = reconstruction_valid = 0.0
with torch.inference_mode():
    for batch, context in loader(
        radar,
        checkpoint["split"]["test"],
        BATCH_SIZE,
        auxiliary=auxiliary,
    ):
        features, targets, mask = sequence_tensors(batch, checkpoint["input_frames"])
        features = features.to(device)
        mask = mask.to(device)
        forecast_difference = normalized_to_linear_z(
            model(features, context.to(device))
        ) - normalized_to_linear_z(targets.to(device))
        forecast_squared += float((forecast_difference.square() * mask).sum().cpu())
        forecast_absolute += float((forecast_difference.abs() * mask).sum().cpu())
        forecast_valid += float(mask.sum().cpu())

        observed = features[:, : checkpoint["input_frames"]]
        observed_mask = torch.ones_like(observed)
        reconstruction_difference = normalized_to_linear_z(
            model.reconstruct(features)
        ) - normalized_to_linear_z(observed)
        reconstruction_squared += float(
            (reconstruction_difference.square() * observed_mask).sum().cpu()
        )
        reconstruction_absolute += float(
            (reconstruction_difference.abs() * observed_mask).sum().cpu()
        )
        reconstruction_valid += float(observed_mask.sum().cpu())

forecast_mae_z = forecast_absolute / forecast_valid
forecast_rmse_z = (forecast_squared / forecast_valid) ** 0.5
reconstruction_mae_z = reconstruction_absolute / reconstruction_valid
reconstruction_rmse_z = (reconstruction_squared / reconstruction_valid) ** 0.5


def linear_z_to_dbz(value: float) -> float:
    return 10.0 * torch.log10(torch.tensor(max(value, 1e-12))).item()


results = {
    "forecast_mae_linear_z": forecast_mae_z,
    "forecast_rmse_linear_z": forecast_rmse_z,
    "forecast_mae_dbz_equivalent": linear_z_to_dbz(forecast_mae_z),
    "forecast_rmse_dbz_equivalent": linear_z_to_dbz(forecast_rmse_z),
    "reconstruction_mae_linear_z": reconstruction_mae_z,
    "reconstruction_rmse_linear_z": reconstruction_rmse_z,
    "reconstruction_mae_dbz_equivalent": linear_z_to_dbz(reconstruction_mae_z),
    "reconstruction_rmse_dbz_equivalent": linear_z_to_dbz(reconstruction_rmse_z),
}
METRICS.parent.mkdir(parents=True, exist_ok=True)
METRICS.write_text(json.dumps(results, indent=2), encoding="utf-8")


print(
    f"Forecast MAE:       {forecast_mae_z:.3e} linear Z "
    f"({linear_z_to_dbz(forecast_mae_z):.2f} dBZ equivalent)"
)
print(f"Metrics: {METRICS.resolve()}")
print(
    f"Forecast RMSE:      {forecast_rmse_z:.3e} linear Z "
    f"({linear_z_to_dbz(forecast_rmse_z):.2f} dBZ equivalent)"
)
print(
    f"Reconstruction MAE: {reconstruction_mae_z:.3e} linear Z "
    f"({linear_z_to_dbz(reconstruction_mae_z):.2f} dBZ equivalent)"
)
print(
    f"Reconstruction RMSE: {reconstruction_rmse_z:.3e} linear Z "
    f"({linear_z_to_dbz(reconstruction_rmse_z):.2f} dBZ equivalent)"
)
