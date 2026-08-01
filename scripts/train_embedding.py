"""Stage 2 — learn and verify the Cartesian spatial representation."""

import random
from pathlib import Path

import numpy as np
import torch
from forecast import (
    AUXILIARY_FEATURES,
    choose_device,
    load_data,
    loader,
    run_reconstruction_epoch,
    split_indexes,
)
from model import ForecastModel

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
AUTOENCODER = Path("outputs/autoencoder.pt")
INPUT_FRAMES = 25
EMBEDDING_CHANNELS = 48
MOTION_SKIP = 2
MOTION_CHANNELS = 32
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.001
SEED = 20260729
LIMIT = None
DEVICE = "auto"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
radar, usable = load_data(DATASET)
split = split_indexes(usable, SEED, LIMIT)
device = choose_device(DEVICE)
output_frames = radar.shape[1] - INPUT_FRAMES
model = ForecastModel(
    INPUT_FRAMES,
    output_frames,
    EMBEDDING_CHANNELS,
    MOTION_SKIP,
    MOTION_CHANNELS,
).to(device)

# Only encoder/decoder parameters participate. Latent dynamics is untouched.
autoencoder_parameters = [
    *model.frame_encoder.parameters(),
    *model.frame_decoder.parameters(),
    *model.motion_encoder.parameters(),
    *model.motion_decoder.parameters(),
]
optimizer = torch.optim.Adam(autoencoder_parameters, lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
train_batches = loader(radar, split["train"], BATCH_SIZE, shuffle=True, seed=SEED)
validation_batches = loader(radar, split["validation"], BATCH_SIZE)
best_validation, best_epoch = float("inf"), 0
best_encoder = best_decoder = None
best_motion_encoder = best_motion_decoder = None

for epoch in range(1, EPOCHS + 1):
    train_loss = run_reconstruction_epoch(
        model, train_batches, device, INPUT_FRAMES, optimizer
    )
    validation_loss = run_reconstruction_epoch(
        model, validation_batches, device, INPUT_FRAMES
    )
    scheduler.step(validation_loss)
    if validation_loss < best_validation:
        best_validation, best_epoch = validation_loss, epoch
        best_encoder = {
            name: value.detach().cpu().clone()
            for name, value in model.frame_encoder.state_dict().items()
        }
        best_decoder = {
            name: value.detach().cpu().clone()
            for name, value in model.frame_decoder.state_dict().items()
        }
        best_motion_encoder = {
            name: value.detach().cpu().clone()
            for name, value in model.motion_encoder.state_dict().items()
        }
        best_motion_decoder = {
            name: value.detach().cpu().clone()
            for name, value in model.motion_decoder.state_dict().items()
        }
    print(
        f"Epoch {epoch:>3}/{EPOCHS} "
        f"train_reconstruction={train_loss:.6f} "
        f"validation_reconstruction={validation_loss:.6f}"
    )

if any(
    value is None
    for value in (
        best_encoder,
        best_decoder,
        best_motion_encoder,
        best_motion_decoder,
    )
):
    raise RuntimeError("Embedding training completed without a checkpoint")
AUTOENCODER.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_type": ForecastModel.AUTOENCODER_TYPE,
        "frame_encoder_state": best_encoder,
        "frame_decoder_state": best_decoder,
        "motion_encoder_state": best_motion_encoder,
        "motion_decoder_state": best_motion_decoder,
        "input_frames": INPUT_FRAMES,
        "output_frames": output_frames,
        "embedding_channels": EMBEDDING_CHANNELS,
        "motion_skip": MOTION_SKIP,
        "motion_channels": MOTION_CHANNELS,
        "auxiliary_features": AUXILIARY_FEATURES,
        "grid_size": int(radar.shape[-1]),
        "seed": SEED,
        "split": split,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
    },
    AUTOENCODER,
)
print(f"Best epoch: {best_epoch} validation={best_validation:.6f}")
print(f"Saved: {AUTOENCODER.resolve()}")
