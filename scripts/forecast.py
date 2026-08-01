"""Shared pieces for the small train/validate/plot scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from model import ForecastModel
from torch.utils.data import DataLoader, Dataset

LOWER_DBZ = -32.0
UPPER_DBZ = 95.0
TENSOR_NAME = "cartesian-z-64x64.npy"
MAX_LINEAR_Z = 10.0 ** (UPPER_DBZ / 10.0)
MIN_DISPLAY_Z = 10.0 ** (LOWER_DBZ / 10.0)
LOG_LINEAR_Z_MAX = float(np.log1p(MAX_LINEAR_Z))
STRONG_ECHO_DBZ = 50.0
STRONG_ECHO_NORMALIZED = float(
    np.log1p(10.0 ** (STRONG_ECHO_DBZ / 10.0)) / LOG_LINEAR_Z_MAX
)
AUXILIARY_FEATURES = 11


class RadarSequences(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        indexes: np.ndarray,
        auxiliary: np.ndarray | None = None,
    ) -> None:
        self.values, self.indexes, self.auxiliary = values, indexes, auxiliary

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, position: int) -> torch.Tensor:
        value = np.array(self.values[int(self.indexes[position])], dtype=np.float32)
        radar = torch.from_numpy(value)
        if self.auxiliary is None:
            return radar
        return radar, torch.from_numpy(self.auxiliary[int(self.indexes[position])])


def load_data(dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open the generated tensor and return it with its usable row indexes."""
    tensor_path = dataset / "tensors" / TENSOR_NAME
    metadata_path = tensor_path.with_suffix(".json")
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {tensor_path}; run scripts/prepare_cartesian.py first"
        )
    radar = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failed = set(metadata.get("failed_examples", {}))
    indexes = np.array(
        [i for i, identifier in enumerate(metadata["examples"]) if identifier not in failed],
        dtype=np.int64,
    )
    return radar, indexes


def load_auxiliary(dataset: Path, input_frames: int) -> np.ndarray:
    """Build calendar plus observed and forecast cadence features."""
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    by_identifier = {example["identifier"]: example for example in manifest["examples"]}
    tensor_metadata = json.loads(
        (dataset / "tensors" / TENSOR_NAME).with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for identifier in tensor_metadata["examples"]:
        frames = by_identifier[identifier]["frames"]
        times = [datetime.fromisoformat(frame["scan_time"]) for frame in frames]
        observed = times[:input_frames]
        intervals = np.diff([time.timestamp() for time in observed]) / 60.0
        last = observed[-1]
        hour = last.hour + last.minute / 60.0 + last.second / 3600.0
        day = last.timetuple().tm_yday - 1 + hour / 24.0
        future = times[input_frames:]
        forecast_intervals = np.diff(
            [last.timestamp(), *[time.timestamp() for time in future]]
        ) / 60.0
        rows.append(
            [
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * day / 365.25),
                np.cos(2 * np.pi * day / 365.25),
                float(intervals.mean() / 10.0),
                float(intervals[-1] / 10.0),
                *[float(interval / 10.0) for interval in forecast_intervals],
            ]
        )
    values = np.asarray(rows, dtype=np.float32)
    if values.shape[1] != AUXILIARY_FEATURES:
        raise ValueError(
            f"Expected {AUXILIARY_FEATURES} auxiliary features; found {values.shape[1]}"
        )
    return values


def split_indexes(
    indexes: np.ndarray, seed: int, limit: int | None = None
) -> dict[str, np.ndarray]:
    """Make the reproducible 70/15/15 split used by every stage."""
    selected = indexes[: min(limit, len(indexes))] if limit is not None else indexes.copy()
    if len(selected) < 3:
        raise ValueError("At least three usable examples are required")
    np.random.default_rng(seed).shuffle(selected)
    train_stop = round(len(selected) * 0.70)
    validation_stop = train_stop + round(len(selected) * 0.15)
    split = {
        "train": selected[:train_stop],
        "validation": selected[train_stop:validation_stop],
        "test": selected[validation_stop:],
    }
    if any(len(values) == 0 for values in split.values()):
        raise ValueError("The selected data leaves an empty split")
    return split


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal (MPS) is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if requested not in {"mps", "cuda", "cpu"}:
        raise ValueError('DEVICE must be "auto", "mps", "cuda", or "cpu"')
    return torch.device(requested)


def sequence_tensors(batch: torch.Tensor, input_frames: int):
    """Log-compress finite physical linear reflectivity for the network."""
    if not torch.isfinite(batch).all():
        raise ValueError("Cartesian tensor contains non-finite values; rebuild it")
    if (batch < 0).any():
        raise ValueError("Linear reflectivity tensor contains negative values")
    inputs, targets = batch[:, :input_frames], batch[:, input_frames:]
    inputs = torch.log1p(inputs.clamp_max(MAX_LINEAR_Z)) / LOG_LINEAR_Z_MAX
    targets = torch.log1p(targets.clamp_max(MAX_LINEAR_Z)) / LOG_LINEAR_Z_MAX
    return inputs, targets, torch.ones_like(targets)


def normalized_to_linear_z(values: torch.Tensor) -> torch.Tensor:
    """Convert network-space values back to physical linear reflectivity."""
    return torch.expm1(values.clamp(0.0, 1.0) * LOG_LINEAR_Z_MAX)


def normalized_to_dbz(values: torch.Tensor) -> torch.Tensor:
    """Convert network-space values to dBZ for visualization."""
    linear_z = normalized_to_linear_z(values)
    return 10.0 * torch.log10(linear_z.clamp_min(MIN_DISPLAY_Z))


def masked_huber(
    prediction,
    target,
    mask,
    *,
    echo_weight=True,
    importance=None,
    strong_echo_boost=0.0,
):
    difference = prediction - target
    # Above weak-echo levels, log1p(Z) is approximately linear in dBZ.
    delta = 5.0 / UPPER_DBZ
    absolute = difference.abs()
    loss = torch.where(
        absolute <= delta,
        0.5 * difference.square(),
        delta * (absolute - 0.5 * delta),
    )
    # Clear-air pixels are abundant. Give stronger echoes more influence so a
    # smooth empty forecast cannot dominate training.
    weights = mask
    if echo_weight:
        importance = target if importance is None else importance
        # Squaring concentrates extra weight on organized strong echoes while
        # retaining a baseline contribution from weak echo and clear air.
        weights = weights * (1.0 + 8.0 * importance.clamp(0.0, 1.0).square())
        if strong_echo_boost:
            weights = weights * (
                1.0
                + strong_echo_boost
                * (importance >= STRONG_ECHO_NORMALIZED).to(weights.dtype)
            )
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def histogram_progression_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Match each future frame's cumulative reflectivity histogram."""
    thresholds = prediction.new_tensor(
        [
            np.log1p(10.0 ** (dbz / 10.0)) / LOG_LINEAR_Z_MAX
            for dbz in (10, 20, 30, 40, 50, 60)
        ]
    )
    predicted_curve = torch.sigmoid(
        (prediction[:, :, None] - thresholds[None, None, :, None, None]) / 0.02
    ).mean(dim=(-2, -1))
    target_curve = torch.sigmoid(
        (target[:, :, None] - thresholds[None, None, :, None, None]) / 0.02
    ).mean(dim=(-2, -1))
    # Square root expands rare high-reflectivity coverage so small storm cores
    # are not numerically drowned out by broad weak-echo coverage.
    difference = torch.sqrt(predicted_curve + 1e-6) - torch.sqrt(
        target_curve + 1e-6
    )
    threshold_weights = prediction.new_tensor([1, 1, 1, 2, 4, 6])
    return (difference.square() * threshold_weights[None, None]).mean()


def histogram_divergence_loss(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Jensen–Shannon divergence between per-frame soft dBZ histograms."""
    histogram_lower_dbz = -20.0
    histogram_upper_dbz = 70.0
    # Nine broad 10 dBZ bands: [-20,-10), ..., [60,70].
    dbz_bins = np.arange(
        histogram_lower_dbz + 5.0,
        histogram_upper_dbz,
        10.0,
    )
    bins = reconstruction.new_tensor(
        [np.log1p(10.0 ** (dbz / 10.0)) / LOG_LINEAR_Z_MAX for dbz in dbz_bins]
    )

    def distribution(values: torch.Tensor) -> torch.Tensor:
        distance = values[:, :, None] - bins[None, None, :, None, None]
        membership = torch.softmax(-distance.square() / (2 * 0.025**2), dim=2)
        lower = values.new_tensor(
            np.log1p(10.0 ** (histogram_lower_dbz / 10.0)) / LOG_LINEAR_Z_MAX
        )
        upper = values.new_tensor(
            np.log1p(10.0 ** (histogram_upper_dbz / 10.0)) / LOG_LINEAR_Z_MAX
        )
        included = ((values >= lower) & (values <= upper)).to(values.dtype)
        counts = (membership * included[:, :, None]).sum(dim=(-2, -1))
        included_count = included.sum(dim=(-2, -1))
        total = included_count.unsqueeze(-1).clamp_min(1.0)
        distribution = counts / total
        # An entirely empty in-range frame gets a tiny uniform distribution;
        # full-field reconstruction still supervises its actual values.
        empty = included_count.unsqueeze(-1) == 0
        uniform = torch.full_like(distribution, 1.0 / distribution.shape[2])
        return torch.where(empty, uniform, distribution).clamp_min(1e-8)

    reconstructed = distribution(reconstruction)
    actual = distribution(target)

    def js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (first + second)
        first_kl = (first * (first / midpoint).log()).sum(dim=2)
        second_kl = (second * (second / midpoint).log()).sum(dim=2)
        return 0.5 * (first_kl + second_kl)

    full_js = js_divergence(reconstructed, actual)

    # Reweight and renormalize bins so rare high-reflectivity discrepancies
    # cannot be hidden by the clear-air/background peak.
    bin_weights = 1.0 + 12.0 * bins.square()
    reconstructed_weighted = reconstructed * bin_weights[None, None]
    actual_weighted = actual * bin_weights[None, None]
    reconstructed_weighted /= reconstructed_weighted.sum(dim=2, keepdim=True)
    actual_weighted /= actual_weighted.sum(dim=2, keepdim=True)
    strong_echo_js = js_divergence(reconstructed_weighted, actual_weighted)

    # Cumulative coverage retains absolute storm area, which normalization in
    # the weighted histogram would otherwise discard.
    thresholds = reconstruction.new_tensor(
        [
            np.log1p(10.0 ** (dbz / 10.0)) / LOG_LINEAR_Z_MAX
            for dbz in (10, 20, 30, 40, 50, 60)
        ]
    )
    reconstructed_coverage = torch.sigmoid(
        (reconstruction[:, :, None] - thresholds[None, None, :, None, None])
        / 0.02
    ).mean(dim=(-2, -1))
    actual_coverage = torch.sigmoid(
        (target[:, :, None] - thresholds[None, None, :, None, None]) / 0.02
    ).mean(dim=(-2, -1))
    coverage_weights = reconstruction.new_tensor([1, 1, 1, 2, 4, 6])
    coverage_error = (
        (
            torch.sqrt(reconstructed_coverage + 1e-6)
            - torch.sqrt(actual_coverage + 1e-6)
        ).square()
        * coverage_weights[None, None]
    ).mean(dim=2)
    return (full_js + strong_echo_js + 2.0 * coverage_error).mean()


def loader(
    radar, indexes, batch_size, *, auxiliary=None, shuffle=False, seed=0
):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        RadarSequences(radar, indexes, auxiliary),
        batch_size=min(batch_size, len(indexes)),
        shuffle=shuffle,
        generator=generator,
    )


def run_epoch(model, batches, device, input_frames, optimizer=None) -> float:
    """Train or evaluate forecasting only."""
    model.train(optimizer is not None)
    total = count = 0.0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for batch in batches:
            radar_batch, auxiliary = batch
            features, targets, mask = sequence_tensors(radar_batch, input_frames)
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            auxiliary = auxiliary.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(features, auxiliary)
            reflectivity_loss = masked_huber(prediction, targets, mask)

            # Penalize incorrect changes between frames, including the jump
            # from the final observation to the first forecast.
            last_observed = features[:, input_frames - 1 : input_frames]
            last_valid = torch.ones_like(last_observed)
            prediction_sequence = torch.cat((last_observed, prediction), dim=1)
            target_sequence = torch.cat((last_observed, targets), dim=1)
            validity_sequence = torch.cat((last_valid, mask), dim=1)
            motion_mask = validity_sequence[:, 1:] * validity_sequence[:, :-1]
            motion_importance = torch.maximum(
                target_sequence[:, 1:], target_sequence[:, :-1]
            )
            motion_loss = masked_huber(
                prediction_sequence[:, 1:] - prediction_sequence[:, :-1],
                target_sequence[:, 1:] - target_sequence[:, :-1],
                motion_mask,
                importance=motion_importance,
                strong_echo_boost=12.0,
            )

            # Preserve sharper spatial boundaries instead of minimizing error
            # with diffuse blobs.
            horizontal_mask = mask[..., :, 1:] * mask[..., :, :-1]
            vertical_mask = mask[..., 1:, :] * mask[..., :-1, :]
            horizontal_importance = torch.maximum(
                targets[..., :, 1:], targets[..., :, :-1]
            )
            vertical_importance = torch.maximum(
                targets[..., 1:, :], targets[..., :-1, :]
            )
            horizontal_loss = masked_huber(
                prediction[..., :, 1:] - prediction[..., :, :-1],
                targets[..., :, 1:] - targets[..., :, :-1],
                horizontal_mask,
                importance=horizontal_importance,
            )
            vertical_loss = masked_huber(
                prediction[..., 1:, :] - prediction[..., :-1, :],
                targets[..., 1:, :] - targets[..., :-1, :],
                vertical_mask,
                importance=vertical_importance,
            )
            histogram_loss = histogram_progression_loss(prediction, targets)
            loss = (
                reflectivity_loss
                + 1.00 * motion_loss
                + 0.10 * (horizontal_loss + vertical_loss)
                + 0.25 * histogram_loss
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            valid = float(mask.sum().cpu())
            total += float(loss.detach().cpu()) * valid
            count += valid
    count = max(1.0, count)
    return total / count


def run_reconstruction_epoch(
    model, batches, device, input_frames, optimizer=None
) -> float:
    """Train or evaluate only the spatial encoder and shared decoder."""
    model.train(optimizer is not None)
    total = count = 0.0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for batch in batches:
            features, _, _ = sequence_tensors(batch, input_frames)
            features = features.to(device)
            observed = features[:, :input_frames]
            observed_mask = torch.ones_like(observed)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model.reconstruct(features)
            appearance_loss = masked_huber(reconstruction, observed, observed_mask)
            histogram_loss = histogram_divergence_loss(reconstruction, observed)
            motion_target = model.motion_differences(features)
            motion_importance = torch.maximum(
                features[:, model.motion_skip :],
                features[:, : -model.motion_skip],
            )
            motion_loss = masked_huber(
                model.reconstruct_motion(features),
                motion_target,
                torch.ones_like(motion_target),
                importance=motion_importance,
                strong_echo_boost=12.0,
            )
            loss = appearance_loss + motion_loss + 0.10 * histogram_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            valid = float(observed_mask.sum().cpu())
            total += float(loss.detach().cpu()) * valid
            count += valid
    return total / max(1.0, count)


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != ForecastModel.MODEL_TYPE:
        raise ValueError(
            "This checkpoint predates the spatial embedding model. "
            "Retrain it with: python scripts/train.py"
        )
    model = ForecastModel(
        checkpoint["input_frames"],
        checkpoint["output_frames"],
        checkpoint["embedding_channels"],
        checkpoint["motion_skip"],
        checkpoint["motion_channels"],
        checkpoint["auxiliary_features"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model
