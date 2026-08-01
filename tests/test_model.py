from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model import ForecastModel


def test_spatial_embedding_and_forecast_shapes():
    model = ForecastModel(input_frames=25, output_frames=5, motion_skip=2)
    frame = torch.rand(2, 1, 64, 64)
    observations = frame.expand(-1, 25, -1, -1).clone()

    embedding = model.embed(observations)
    reconstruction = model.reconstruct(observations)
    forecast = model(observations, torch.zeros(2, 11))

    assert embedding.shape == (2, 48, 25, 8, 8)
    assert reconstruction.shape == (2, 25, 64, 64)
    assert forecast.shape == (2, 5, 64, 64)
    expected_persistence = observations[:, 24:25].expand_as(forecast)
    assert torch.allclose(forecast, expected_persistence, atol=1e-6)
