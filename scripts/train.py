"""Stage 3 — freeze the embedding model and train future prediction only."""

import random
from pathlib import Path

import numpy as np
import torch
from forecast import (
    AUXILIARY_FEATURES,
    choose_device,
    load_auxiliary,
    load_data,
    loader,
    run_epoch,
)
from model import ForecastModel

DATASET = Path("data/datasets/ktlx-reflectivity-120km-2016-2025-v1")
AUTOENCODER = Path("outputs/autoencoder.pt")
MODEL = Path("outputs/model.pt")
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = "auto"

device = choose_device(DEVICE)
autoencoder = torch.load(AUTOENCODER, map_location=device, weights_only=False)
if autoencoder.get("model_type") != ForecastModel.AUTOENCODER_TYPE:
    raise ValueError("The autoencoder is outdated; run scripts/train_embedding.py")

INPUT_FRAMES = autoencoder["input_frames"]
OUTPUT_FRAMES = autoencoder["output_frames"]
EMBEDDING_CHANNELS = autoencoder["embedding_channels"]
MOTION_SKIP = autoencoder["motion_skip"]
MOTION_CHANNELS = autoencoder["motion_channels"]
SEED = autoencoder["seed"]
split = autoencoder["split"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
radar, _ = load_data(DATASET)
auxiliary = load_auxiliary(DATASET, INPUT_FRAMES)
if autoencoder.get("grid_size") != radar.shape[-1]:
    raise ValueError(
        "The autoencoder was trained at a different resolution; "
        "run scripts/train_embedding.py"
    )
model = ForecastModel(
    INPUT_FRAMES,
    OUTPUT_FRAMES,
    EMBEDDING_CHANNELS,
    MOTION_SKIP,
    MOTION_CHANNELS,
    AUXILIARY_FEATURES,
).to(device)
model.frame_encoder.load_state_dict(autoencoder["frame_encoder_state"])
model.frame_decoder.load_state_dict(autoencoder["frame_decoder_state"])
model.motion_encoder.load_state_dict(autoencoder["motion_encoder_state"])
model.motion_decoder.load_state_dict(autoencoder["motion_decoder_state"])

# The representation is fixed. Gradients pass through the decoder into latent
# dynamics, but encoder/decoder weights cannot change during prediction training.
for parameter in model.frame_encoder.parameters():
    parameter.requires_grad_(False)
for parameter in model.frame_decoder.parameters():
    parameter.requires_grad_(False)
for parameter in model.motion_encoder.parameters():
    parameter.requires_grad_(False)
for parameter in model.motion_decoder.parameters():
    parameter.requires_grad_(False)

predictor_parameters = [
    *model.advection_encoder.parameters(),
    *model.advection_context.parameters(),
    *model.flow_head.parameters(),
    *model.source_head.parameters(),
]
optimizer = torch.optim.Adam(predictor_parameters, lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
train_batches = loader(
    radar, split["train"], BATCH_SIZE, auxiliary=auxiliary, shuffle=True, seed=SEED
)
validation_batches = loader(
    radar, split["validation"], BATCH_SIZE, auxiliary=auxiliary
)
train_losses, validation_losses = [], []
best_validation, best_epoch, best_state = float("inf"), 0, None

print("Encoder and decoder: frozen")
print(f"Predictor parameters: {sum(p.numel() for p in predictor_parameters):,}")
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(model, train_batches, device, INPUT_FRAMES, optimizer)
    validation_loss = run_epoch(model, validation_batches, device, INPUT_FRAMES)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    scheduler.step(validation_loss)
    if validation_loss < best_validation:
        best_validation, best_epoch = validation_loss, epoch
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    print(
        f"Epoch {epoch:>3}/{EPOCHS} "
        f"train_forecast={train_loss:.6f} "
        f"validation_forecast={validation_loss:.6f}"
    )

if best_state is None:
    raise RuntimeError("Prediction training completed without a checkpoint")
MODEL.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_type": ForecastModel.MODEL_TYPE,
        "model_state": best_state,
        "input_frames": INPUT_FRAMES,
        "output_frames": OUTPUT_FRAMES,
        "embedding_channels": EMBEDDING_CHANNELS,
        "motion_skip": MOTION_SKIP,
        "motion_channels": MOTION_CHANNELS,
        "auxiliary_features": AUXILIARY_FEATURES,
        "grid_size": int(radar.shape[-1]),
        "seed": SEED,
        "split": split,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "autoencoder_path": str(AUTOENCODER),
    },
    MODEL,
)
print(f"Best epoch: {best_epoch} validation={best_validation:.6f}")
print(f"Saved: {MODEL.resolve()}")
