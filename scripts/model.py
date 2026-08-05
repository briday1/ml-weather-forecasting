"""Frame-wise spatial encoder, spatiotemporal dynamics, and shared decoder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _normalized_dbz_threshold(dbz: float) -> float:
    maximum_z = 10.0 ** (95.0 / 10.0)
    return float(
        torch.log1p(torch.tensor(10.0 ** (dbz / 10.0)))
        / torch.log1p(torch.tensor(maximum_z))
    )


class ConvGRUCell(nn.Module):
    """A recurrent unit whose state and gates retain spatial layout."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        combined = input_channels + hidden_channels
        self.gates = nn.Conv2d(combined, hidden_channels * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(combined, hidden_channels, kernel_size=3, padding=1)

    def forward(self, value: Tensor, hidden: Tensor) -> Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((value, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((value, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class ForecastModel(nn.Module):
    """Encode frames spatially, evolve them through time, and decode radar."""

    AUTOENCODER_TYPE = "appearance-motion-autoencoder-25-v5"
    MODEL_TYPE = "explicit-advection-residual-predictor-v16"

    def __init__(
        self,
        input_frames: int = 25,
        output_frames: int = 5,
        embedding_channels: int = 48,
        motion_skip: int = 2,
        motion_channels: int = 32,
        auxiliary_features: int = 11,
    ) -> None:
        super().__init__()
        if not 1 <= motion_skip < input_frames:
            raise ValueError("motion_skip must be between 1 and input_frames - 1")
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.motion_skip = motion_skip
        self.motion_channels = motion_channels
        self.auxiliary_features = auxiliary_features

        # The same encoder processes every normalized log1p(linear Z) frame.
        # Prototype resolution: (1, 64, 64) -> (48, 8, 8) per timestamp.
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                32,
                embedding_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1),
        )

        # This encoder sees only signed skip-frame differences. It cannot use
        # absolute reflectivity as a shortcut for learning motion.
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(24, motion_channels, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )
        self.motion_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                motion_channels, 24, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(24, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

        # The recurrent predictor receives only consecutive latent differences.
        # Absolute latent states never enter the learned motion path.
        self.latent_dynamics = nn.ModuleDict(
            {
                "cell": ConvGRUCell(embedding_channels, embedding_channels),
                "delta": nn.Conv2d(
                    embedding_channels, embedding_channels, kernel_size=3, padding=1
                ),
            }
        )
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_features, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, embedding_channels),
        )
        histogram_thresholds = torch.tensor(
            [_normalized_dbz_threshold(value) for value in (10, 20, 30, 40, 50, 60)]
        )
        self.register_buffer("histogram_thresholds", histogram_thresholds)
        self.histogram_encoder = nn.GRU(
            input_size=len(histogram_thresholds) * 2,
            hidden_size=embedding_channels,
            batch_first=True,
        )
        # An untrained predictor begins as persistence, not random motion.
        nn.init.zeros_(self.latent_dynamics["delta"].weight)
        nn.init.zeros_(self.latent_dynamics["delta"].bias)

        # Forecast displacement explicitly.  Treating intensity differences as
        # ordinary latent features allowed the network to minimize loss by
        # fading echoes in place.  This branch can only see the complete stack
        # of skip-frame differences and must emit an x/y displacement field for
        # every future frame.
        difference_frames = input_frames - motion_skip
        self.advection_encoder = nn.Sequential(
            nn.Conv2d(difference_frames, 48, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.advection_context = nn.Linear(auxiliary_features, 48)
        self.flow_head = nn.Conv2d(48, output_frames * 2, 3, padding=1)
        # A small source/sink term handles genuine storm growth and decay after
        # advection.  Starting at zero makes the initial model pure persistence.
        self.source_head = nn.Conv2d(48, output_frames, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)

        # One shared decoder maps any latent frame back into radar space.
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def embed(self, observations: Tensor) -> Tensor:
        """Return (batch, channel, time, latent-ray, latent-range)."""
        frames = observations.unsqueeze(2)
        batch, time, channels, rays, gates = frames.shape
        encoded = self.frame_encoder(
            frames.reshape(batch * time, channels, rays, gates)
        )
        _, latent_channels, latent_rays, latent_gates = encoded.shape
        return encoded.reshape(
            batch, time, latent_channels, latent_rays, latent_gates
        ).permute(0, 2, 1, 3, 4)

    def decode(self, embedding: Tensor) -> Tensor:
        """Decode a sequence of spatial embeddings into radar frames."""
        batch, channels, time, latent_rays, latent_gates = embedding.shape
        frames = embedding.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, latent_rays, latent_gates
        )
        decoded = self.frame_decoder(frames)
        return decoded.reshape(batch, time, decoded.shape[-2], decoded.shape[-1])

    def motion_differences(self, observations: Tensor) -> Tensor:
        return (
            observations[:, self.motion_skip :] - observations[:, : -self.motion_skip]
        )

    def embed_motion(self, observations: Tensor) -> Tensor:
        """Return (batch, motion-channel, time, latent-y, latent-x)."""
        differences = self.motion_differences(observations)
        batch, time, height, width = differences.shape
        encoded = self.motion_encoder(
            differences.reshape(batch * time, 1, height, width)
        )
        return encoded.reshape(
            batch, time, encoded.shape[1], encoded.shape[2], encoded.shape[3]
        ).permute(0, 2, 1, 3, 4)

    def reconstruct_motion(self, observations: Tensor) -> Tensor:
        differences = self.motion_differences(observations)
        embedding = self.embed_motion(observations)
        batch, channels, time, height, width = embedding.shape
        decoded = self.motion_decoder(
            embedding.permute(0, 2, 1, 3, 4).reshape(
                batch * time, channels, height, width
            )
        )
        return decoded.reshape(batch, time, *differences.shape[-2:])

    def reconstruct(self, observations: Tensor) -> Tensor:
        return self.decode(self.embed(observations))

    def evolve(self, embedding: Tensor) -> Tensor:
        raise RuntimeError(
            "Use evolve_from_observations so motion is encoded explicitly"
        )

    def evolve_from_observations(
        self, observations: Tensor, appearance: Tensor, auxiliary: Tensor
    ) -> Tensor:
        differences = (
            appearance[:, :, self.motion_skip :] - appearance[:, :, : -self.motion_skip]
        )
        hidden = torch.zeros_like(differences[:, :, 0])
        cell = self.latent_dynamics["cell"]
        for frame in range(differences.shape[2]):
            hidden = cell(differences[:, :, frame], hidden)
        hidden = hidden + self.auxiliary_encoder(auxiliary)[:, :, None, None]
        histogram = torch.sigmoid(
            (
                observations[:, :, None]
                - self.histogram_thresholds[None, None, :, None, None]
            )
            / 0.02
        ).mean(dim=(-2, -1))
        histogram_change = torch.cat(
            (torch.zeros_like(histogram[:, :1]), histogram[:, 1:] - histogram[:, :-1]),
            dim=1,
        )
        _, histogram_hidden = self.histogram_encoder(
            torch.cat((histogram, histogram_change), dim=-1)
        )
        hidden = hidden + histogram_hidden[0][:, :, None, None]

        recent_steps = min(4, differences.shape[2])
        previous_difference = (
            differences[:, :, -recent_steps:].mean(dim=2) / self.motion_skip
        )
        previous = appearance[:, :, -1]
        empty_difference = torch.zeros_like(differences[:, :, 0])
        future = []
        for _ in range(self.output_frames):
            hidden = cell(empty_difference, hidden)
            previous_difference = previous_difference + self.latent_dynamics["delta"](
                hidden
            )
            previous = previous + previous_difference
            future.append(previous)
        return torch.stack(future, dim=2)

    def forward(self, observations: Tensor, auxiliary: Tensor) -> Tensor:
        differences = self.motion_differences(observations)
        features = self.advection_encoder(differences)
        features = features + self.advection_context(auxiliary)[:, :, None, None]
        batch, _, height, width = observations.shape
        flows = torch.tanh(self.flow_head(features)).reshape(
            batch, self.output_frames, 2, height, width
        )
        # At prototype resolution, allow up to eight grid cells of travel per
        # normal observed interval. Future cadence explicitly scales that move.
        cadence = (
            auxiliary[:, 6 : 6 + self.output_frames] / auxiliary[:, 5:6].clamp_min(0.1)
        ).clamp(0.25, 4.0)
        flows = flows * (8.0 * cadence[:, :, None, None, None])
        sources = 0.08 * torch.tanh(self.source_head(features))

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=observations.device),
            torch.linspace(-1.0, 1.0, width, device=observations.device),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
        last = observations[:, -1:]
        future = []
        for frame in range(self.output_frames):
            # grid_sample specifies where output pixels sample their input;
            # subtracting the learned forward flow transports echoes forward.
            flow = flows[:, frame]
            grid_offset = torch.stack(
                (
                    2.0 * flow[:, 0] / max(width - 1, 1),
                    2.0 * flow[:, 1] / max(height - 1, 1),
                ),
                dim=-1,
            )
            sampled = F.grid_sample(
                last,
                base_grid - grid_offset,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            # Subtract grid_sample's finite-precision identity pass so exactly
            # zero flow remains bit-for-bit persistence.
            identity = F.grid_sample(
                last,
                base_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            advected = last + sampled - identity
            last = (advected + sources[:, frame : frame + 1]).clamp(0.0, 1.0)
            future.append(last[:, 0])
        return torch.stack(future, dim=1)
