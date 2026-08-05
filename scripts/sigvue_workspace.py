"""Browse embedding and forecast validation pairs in SigVue."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw, ImageFont
from plotly import colors as plotly_colors
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from sigvue import (
    UI,
    Batch,
    BatchDestination,
    BatchResult,
    CapabilityChoice,
    DataResource,
    DiscoveryColumn,
    Reader,
    Segment,
    Workspace,
)
from sigvue.helpers import WorkspaceConfig

from scripts.forecast import (
    AUXILIARY_FEATURES,
    LOWER_DBZ,
    TENSOR_NAME,
    UPPER_DBZ,
    choose_device,
    load_auxiliary,
    load_checkpoint,
    load_data,
    normalized_to_dbz,
    sequence_tensors,
)
from scripts.model import ForecastModel

EMBEDDING = "embedding"
FORECAST = "forecast"
RENDER_GIF = "render-side-by-side-gif"
RENDER_ALL_EMBEDDINGS = "render-all-embedding-gifs"
RENDER_ALL_VALIDATION = "render-all-validation-gifs"
RENDER_ALL = "render-all-gifs"
NEXRAD_COLORSCALE = (
    (0.00, "#646464"),
    (0.15, "#04e9e7"),
    (0.28, "#019ff4"),
    (0.40, "#0300f4"),
    (0.48, "#02fd02"),
    (0.58, "#01c501"),
    (0.66, "#008e00"),
    (0.72, "#fdf802"),
    (0.78, "#e5bc00"),
    (0.84, "#fd9500"),
    (0.89, "#fd0000"),
    (0.94, "#d40000"),
    (0.97, "#bc0000"),
    (1.00, "#f800fd"),
)
plotly_colors.sequential.NEXRAD = plotly_colors.sample_colorscale(
    [list(stop) for stop in NEXRAD_COLORSCALE],
    [index / 100 for index in range(101)],
    colortype="rgb",
)
REFLECTIVITY_COLORMAPS = (
    "NEXRAD",
    "Turbo",
    "Viridis",
    "Cividis",
    "Plasma",
    "Inferno",
    "Magma",
    "Jet",
    "Rainbow",
    "Portland",
    "Hot",
)


@dataclass(frozen=True)
class ResultReference:
    radius_km: float
    kind: str
    index: int
    label: str


@dataclass(frozen=True)
class Comparison:
    reference: ResultReference
    left: np.ndarray
    right: np.ndarray
    left_title: str
    right_title: str
    detail: str
    axis_km: np.ndarray
    scan_times: tuple[datetime, ...]
    end_time: datetime
    radius_km: float


@dataclass(frozen=True)
class ComparisonFrame:
    comparison: Comparison
    index: int


class ResultStore:
    """Lazily load tensors and models while exposing every held-out result."""

    def __init__(
        self,
        dataset: Path,
        autoencoder: Path,
        model: Path,
        device_name: str,
        tensor_name: str = TENSOR_NAME,
        radius_km: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.autoencoder_path = autoencoder
        self.model_path = model
        self.tensor_name = tensor_name
        self.device = choose_device(device_name)
        metadata_path = dataset / "tensors" / Path(tensor_name).with_suffix(".json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_times: dict[str, list[datetime]] = {}
        with (dataset / "frames.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                frame_times.setdefault(row["example"], []).append(
                    datetime.fromisoformat(row["scan_time"])
                )
        self.frame_times = {label: tuple(times) for label, times in frame_times.items()}
        self.radius_km = float(radius_km or self.metadata["radius_km"])
        self.radar, _ = load_data(dataset, tensor_name)
        checkpoint = torch.load(autoencoder, map_location="cpu", weights_only=False)
        self.indexes = tuple(int(index) for index in checkpoint["split"]["validation"])
        self.labels = tuple(self.metadata["examples"][index] for index in self.indexes)
        self.axis_km = np.asarray(self.metadata["x_km"], dtype=np.float32)
        self._autoencoder_cache = None
        self._forecast_cache = None

    def references(self) -> tuple[ResultReference, ...]:
        return tuple(
            ResultReference(self.radius_km, kind, index, label)
            for kind in (EMBEDDING, FORECAST)
            for index, label in zip(self.indexes, self.labels, strict=True)
        )

    def _autoencoder(self):
        if self._autoencoder_cache is not None:
            return self._autoencoder_cache
        checkpoint = torch.load(
            self.autoencoder_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("model_type") != ForecastModel.AUTOENCODER_TYPE:
            raise ValueError(
                "The autoencoder is outdated; run scripts/train_embedding.py"
            )
        model = ForecastModel(
            checkpoint["input_frames"],
            checkpoint["output_frames"],
            checkpoint["embedding_channels"],
            checkpoint["motion_skip"],
            checkpoint["motion_channels"],
            checkpoint.get("auxiliary_features", AUXILIARY_FEATURES),
        ).to(self.device)
        for name in (
            "frame_encoder",
            "frame_decoder",
            "motion_encoder",
            "motion_decoder",
        ):
            getattr(model, name).load_state_dict(checkpoint[f"{name}_state"])
        model.eval()
        self._autoencoder_cache = checkpoint, model
        return self._autoencoder_cache

    def _forecast(self):
        if self._forecast_cache is not None:
            return self._forecast_cache
        checkpoint, model = load_checkpoint(self.model_path, self.device)
        auxiliary = load_auxiliary(
            self.dataset, checkpoint["input_frames"], self.tensor_name
        )
        self._forecast_cache = checkpoint, model, auxiliary
        return self._forecast_cache

    def open(self, reference: ResultReference) -> Comparison:
        example = torch.from_numpy(
            np.array(self.radar[reference.index], dtype=np.float32)
        ).unsqueeze(0)
        if reference.kind == EMBEDDING:
            checkpoint, model = self._autoencoder()
            features, _, _ = sequence_tensors(example, checkpoint["input_frames"])
            with torch.inference_mode():
                prediction = model.reconstruct(features.to(self.device)).cpu()
            left = normalized_to_dbz(features)[0].numpy()
            right = normalized_to_dbz(prediction)[0].numpy()
            detail = f"reconstruction MAE {np.mean(np.abs(right - left)):.2f} dBZ"
            titles = ("Observed Cartesian radar", "Encoder → decoder reconstruction")
            all_times = self.frame_times[reference.label]
            scan_times = all_times[: left.shape[0]]
            end_time = all_times[left.shape[0]]
        elif reference.kind == FORECAST:
            checkpoint, model, auxiliary = self._forecast()
            features, targets, _ = sequence_tensors(example, checkpoint["input_frames"])
            with torch.inference_mode():
                context = torch.from_numpy(auxiliary[reference.index]).unsqueeze(0)
                prediction = model(
                    features.to(self.device), context.to(self.device)
                ).cpu()
            left = normalized_to_dbz(targets)[0].numpy()
            right = normalized_to_dbz(prediction)[0].numpy()
            detail = f"forecast MAE {np.mean(np.abs(right - left)):.2f} dBZ"
            titles = ("Observed validation truth", "Learned forecast")
            all_times = self.frame_times[reference.label]
            scan_times = all_times[-left.shape[0] :]
            end_time = scan_times[-1] + (scan_times[-1] - scan_times[-2])
        else:
            raise ValueError(f"Unknown result kind: {reference.kind}")
        return Comparison(
            reference,
            left,
            right,
            *titles,
            detail,
            self.axis_km,
            scan_times,
            end_time,
            self.radius_km,
        )


class ExperimentStore:
    """Combine independently trained radius experiments into one browser."""

    def __init__(self, stores: tuple[ResultStore, ...]) -> None:
        if not stores:
            raise ValueError("At least one completed radius experiment is required")
        self.stores = {store.radius_km: store for store in stores}

    def references(self) -> tuple[ResultReference, ...]:
        return tuple(
            reference
            for radius in sorted(self.stores)
            for reference in self.stores[radius].references()
        )

    def open(self, reference: ResultReference) -> Comparison:
        return self.stores[reference.radius_km].open(reference)


def _describe(reference: ResultReference) -> DataResource:
    match = re.search(r"(\d{8}T\d{6})", reference.label)
    timestamp = (
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        if match
        else None
    )
    title = (
        "Embedding reconstruction"
        if reference.kind == EMBEDDING
        else "Forecast validation"
    )
    return DataResource(
        identifier=f"{reference.radius_km:g}km-{reference.kind}-{reference.index}",
        title=f"{title} · {reference.label}",
        source=reference,
        subtitle=f"Validation example {reference.index}",
        timestamp=timestamp,
        tags=(reference.kind, "validation", "KTLX", "reflectivity"),
        summary={
            "date": timestamp.isoformat() if timestamp else None,
            "result": reference.kind,
            "index": reference.index,
        },
        navigation_path=(
            f"{reference.radius_km:g} km",
            "Embeddings" if reference.kind == EMBEDDING else "Validation",
        ),
    )


def _segments(comparison: Comparison) -> tuple[Segment, ...]:
    origin = comparison.scan_times[0]
    stops = (*comparison.scan_times[1:], comparison.end_time)
    return tuple(
        Segment(
            str(index),
            (timestamp - origin).total_seconds(),
            (stop - timestamp).total_seconds(),
            timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        for index, (timestamp, stop) in enumerate(
            zip(comparison.scan_times, stops, strict=True)
        )
    )


def _duration(comparison: Comparison) -> float:
    return (comparison.end_time - comparison.scan_times[0]).total_seconds()


def _read_frame(comparison: Comparison, segment: Segment) -> ComparisonFrame:
    return ComparisonFrame(comparison, int(segment.identifier))


def comparison_figure(
    frame: ComparisonFrame,
    theme: str = "dark",
    colormap: str = "NEXRAD",
    limits: tuple[float, float] = (LOWER_DBZ, UPPER_DBZ),
) -> go.Figure:
    comparison, index = frame.comparison, frame.index
    canvas_color = "#10252d" if theme == "dark" else "white"
    boundary_color = "#e7f1f3" if theme == "dark" else "#13212b"
    radius_km = comparison.radius_km
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=(comparison.left_title, comparison.right_title)
    )
    for column, values in ((1, comparison.left), (2, comparison.right)):
        visible = np.where(
            (values[index] >= limits[0]) & (values[index] <= limits[1]),
            values[index],
            np.nan,
        )
        figure.add_trace(
            go.Heatmap(
                x=comparison.axis_km,
                y=comparison.axis_km,
                z=visible,
                coloraxis="coloraxis",
                zsmooth=False,
                hovertemplate="x=%{x:.1f} km<br>y=%{y:.1f} km<br>%{z:.1f} dBZ<extra></extra>",
            ),
            row=1,
            col=column,
        )
    axis = {
        "range": [-radius_km - 5, radius_km + 5],
        "constrain": "domain",
    }
    figure.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=canvas_color,
        plot_bgcolor=canvas_color,
        title=(
            f"{comparison.scan_times[index]:%Y-%m-%d %H:%M:%S UTC} · "
            f"{comparison.detail}"
        ),
        coloraxis={
            "colorscale": colormap,
            "cmin": LOWER_DBZ,
            "cmax": UPPER_DBZ,
            "colorbar": {"title": "dBZ"},
        },
        xaxis=axis,
        yaxis={**axis, "scaleanchor": "x"},
        xaxis2=axis,
        yaxis2={**axis, "scaleanchor": "x2"},
        shapes=[
            {
                "type": "circle",
                "xref": xref,
                "yref": yref,
                "x0": -radius_km,
                "x1": radius_km,
                "y0": -radius_km,
                "y1": radius_km,
                "line": {"color": boundary_color, "width": 2},
            }
            for xref, yref in (("x", "y"), ("x2", "y2"))
        ],
        margin={"t": 90, "b": 45, "l": 55, "r": 75},
    )
    return figure


def view(frame: ComparisonFrame, ui: UI) -> None:
    comparison = frame.comparison
    colormap = ui.colormap(
        "reflectivity_colormap",
        label="Colormap",
        default="NEXRAD",
        options=REFLECTIVITY_COLORMAPS,
        group="Reflectivity display",
    )
    limits = ui.limits(
        "reflectivity_limits_dbz",
        label="Visible reflectivity range (dBZ)",
        default=(LOWER_DBZ, UPPER_DBZ),
        minimum=LOWER_DBZ,
        maximum=UPPER_DBZ,
        step=1.0,
        group="Reflectivity display",
    )
    ui.stat(
        "Result", "Embedding" if comparison.reference.kind == EMBEDDING else "Forecast"
    )
    ui.stat("Radius", f"{comparison.radius_km:g} km")
    ui.stat("Example", comparison.reference.label)
    ui.stat(
        "Scan time",
        f"{comparison.scan_times[frame.index]:%Y-%m-%d %H:%M:%S UTC}",
    )
    ui.stat("Error", comparison.detail)
    with ui.tab("Comparison", columns=(0.22, 0.78)):
        with ui.group("column"):
            ui.place_parameters(
                "reflectivity_colormap",
                "reflectivity_limits_dbz",
                label="Display",
            )
        ui.plot(
            lambda: comparison_figure(frame, ui.theme, colormap, limits),
            key="comparison",
            axis_navigation="bounded",
        )


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=1)
def _nexrad_rgb() -> np.ndarray:
    colors = sample_colorscale(
        [list(stop) for stop in NEXRAD_COLORSCALE],
        np.linspace(0, 1, 256),
        colortype="rgb",
    )
    return np.asarray(
        [
            [
                int(part)
                for part in color.removeprefix("rgb(").removesuffix(")").split(", ")
            ]
            for color in colors
        ],
        dtype=np.uint8,
    )


def _radar_image(values: np.ndarray, size: int = 512) -> Image.Image:
    indexes = np.rint(
        np.clip(
            (np.nan_to_num(values, nan=LOWER_DBZ) - LOWER_DBZ)
            / (UPPER_DBZ - LOWER_DBZ),
            0,
            1,
        )
        * 255
    ).astype(np.uint8)
    image = Image.fromarray(_nexrad_rgb()[indexes], mode="RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def render_comparison_gif(
    comparison: Comparison,
    target: str | Path,
    *,
    frame_duration_ms: int,
    cancel: Callable[[], None] | None = None,
) -> Path:
    destination = Path(target).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    title_font, detail_font = _font(19), _font(14)
    for index in range(comparison.left.shape[0]):
        if cancel:
            cancel()
        canvas = Image.new("RGB", (1024, 590), (8, 17, 23))
        canvas.paste(_radar_image(comparison.left[index]), (0, 78))
        canvas.paste(_radar_image(comparison.right[index]), (512, 78))
        draw = ImageDraw.Draw(canvas)
        for left in (0, 512):
            draw.ellipse(
                (left + 2, 80, left + 509, 587),
                outline="white",
                width=2,
            )
        draw.text((12, 8), comparison.reference.label, fill="white", font=title_font)
        draw.text((12, 38), comparison.left_title, fill="#38bdf8", font=detail_font)
        draw.text((524, 38), comparison.right_title, fill="#fb7185", font=detail_font)
        draw.text(
            (12, 566),
            f"{comparison.scan_times[index]:%Y-%m-%d %H:%M:%S UTC} · radius {comparison.radius_km:g} km · {comparison.detail} · {LOWER_DBZ:g} to {UPPER_DBZ:g} dBZ",
            fill="white",
            font=detail_font,
        )
        frames.append(canvas)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".gif",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        frames[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        temporary.replace(destination)
    finally:
        for frame in frames:
            frame.close()
        temporary.unlink(missing_ok=True)
    return destination


class ComparisonGifBatch(Batch[Comparison]):
    item_actions = (CapabilityChoice(RENDER_GIF, "Render side-by-side GIF"),)
    workspace_actions = (
        CapabilityChoice(RENDER_ALL_EMBEDDINGS, "Render all embedding GIFs"),
        CapabilityChoice(RENDER_ALL_VALIDATION, "Render all validation GIFs"),
        CapabilityChoice(RENDER_ALL, "Render all GIFs"),
    )

    def __init__(self, output_root: Path, frame_duration_ms: int) -> None:
        if isinstance(frame_duration_ms, bool) or frame_duration_ms < 20:
            raise ValueError("GIF frame duration must be at least 20 ms")
        self.output_root = output_root
        self.frame_duration_ms = frame_duration_ms

    def _filename(self, resource: DataResource) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", resource.source.label.lower()).strip("-")
        return f"{slug}-side-by-side.gif"

    @staticmethod
    def _radius_directory(resource: DataResource) -> str:
        return f"{resource.source.radius_km:g}km"

    @staticmethod
    def _category_directory(resource: DataResource) -> str:
        return "embedding" if resource.source.kind == EMBEDDING else "forecasting"

    def _relative_path(self, resource: DataResource) -> Path:
        return (
            Path(self._radius_directory(resource))
            / self._category_directory(resource)
            / self._filename(resource)
        )

    @staticmethod
    def _selected(resources, action):
        if action == RENDER_ALL:
            return tuple(resources)
        kind = EMBEDDING if action == RENDER_ALL_EMBEDDINGS else FORECAST
        return tuple(resource for resource in resources if resource.source.kind == kind)

    def item_destination(self, resource, request):
        return BatchDestination(
            self.output_root
            / self._radius_directory(resource)
            / self._category_directory(resource),
            (self._filename(resource),),
            "Side-by-side GIF is ready",
        )

    def workspace_destination(self, resources, request):
        selected = self._selected(resources, request.action)
        summaries = {
            RENDER_ALL_EMBEDDINGS: "All embedding GIFs are ready",
            RENDER_ALL_VALIDATION: "All validation GIFs are ready",
            RENDER_ALL: "All embedding and validation GIFs are ready",
        }
        return BatchDestination(
            self.output_root,
            tuple(sorted({self._radius_directory(resource) for resource in selected})),
            summaries[request.action],
        )

    def run_item(self, resource, source_data, request, directory):
        output = render_comparison_gif(
            source_data,
            directory / self._filename(resource),
            frame_duration_ms=self.frame_duration_ms,
            cancel=request.raise_if_cancelled,
        )
        return BatchResult((output,), "Rendered looping side-by-side GIF")

    def run_workspace(self, resources, open_resource, request, directory):
        selected = self._selected(resources, request.action)
        outputs = request.each(
            selected,
            lambda resource: render_comparison_gif(
                open_resource(resource),
                directory / self._relative_path(resource),
                frame_duration_ms=self.frame_duration_ms,
                cancel=request.raise_if_cancelled,
            ),
        )
        return BatchResult(
            outputs,
            f"Rendered {len(outputs)} looping side-by-side GIFs",
        )


def create_workspace(config) -> Workspace:
    values = WorkspaceConfig(config)
    dataset = values.path("dataset")
    device = values.string("device", "auto")
    experiments_root = values.path("experiments_root")
    project_root = experiments_root.parent
    stores = []
    for manifest_path in sorted(experiments_root.glob("radius-*/experiment.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        autoencoder = project_root / manifest["autoencoder"]
        model = project_root / manifest["model"]
        tensor = dataset / "tensors" / manifest["tensor"]
        if autoencoder.is_file() and model.is_file() and tensor.is_file():
            stores.append(
                ResultStore(
                    dataset,
                    autoencoder,
                    model,
                    device,
                    manifest["tensor"],
                    float(manifest["radius_km"]),
                )
            )
    if not stores:
        stores.append(
            ResultStore(
                dataset,
                values.path("autoencoder"),
                values.path("model"),
                device,
            )
        )
    store = ExperimentStore(tuple(stores))
    reader = Reader(store.references, store.open, describe=_describe).segmented(
        _read_frame,
        duration=_duration,
        segments=_segments,
        playback_rate=4.0,
        playback_step=1,
        time_unit="min",
    )
    return Workspace(
        identifier="ml-weather-forecasting",
        name="ML Weather Forecasting",
        description="Browse every held-out KTLX embedding reconstruction and forecast validation pair.",
        reader=reader,
        view=view,
        batch=ComparisonGifBatch(
            values.path("output_root"),
            values.integer("gif_frame_duration_ms", 250),
        ),
        category="weather forecasting",
        tags=("KTLX", "radar", "embeddings", "validation", "machine learning"),
        discovery_columns=(
            DiscoveryColumn("date", "Date", "datetime"),
            DiscoveryColumn("result", "Result"),
            DiscoveryColumn("index", "Dataset index", "number"),
        ),
    )


__all__ = ["Comparison", "ResultStore", "create_workspace", "render_comparison_gif"]
