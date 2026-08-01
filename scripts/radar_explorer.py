"""One shared Plotly UI for comparing two sets of Cartesian radar frames."""

import json

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from forecast import LOWER_DBZ, UPPER_DBZ
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


def finite_dbz(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        values, nan=LOWER_DBZ, posinf=UPPER_DBZ, neginf=LOWER_DBZ
    ).clip(LOWER_DBZ, UPPER_DBZ)


def write_radar_comparison(
    *,
    left: list[np.ndarray],
    right: list[np.ndarray],
    labels: list[str],
    axis_km: np.ndarray,
    output: Path,
    title: str,
    left_title: str,
    right_title: str,
    frame_label: str,
    title_details: list[str] | None = None,
    radius_km: float = 120.0,
    show_histograms: bool = False,
) -> None:
    """Write dropdown + frame-slider comparisons for paired radar sequences."""
    if not left or len(left) != len(right) or len(left) != len(labels):
        raise ValueError("left, right, and labels must have the same nonzero length")
    left = [finite_dbz(values) for values in left]
    right = [finite_dbz(values) for values in right]
    expected = left[0].shape
    if any(values.shape != expected for values in (*left, *right)):
        raise ValueError("Every radar sequence must have the same shape")
    if len(expected) != 3 or expected[1:] != (len(axis_km), len(axis_km)):
        raise ValueError("Expected frame × y × x arrays matching axis_km")
    details = title_details or ["" for _ in labels]
    if len(details) != len(labels):
        raise ValueError("title_details must match labels")

    def full_title(index: int) -> str:
        suffix = f" · {details[index]}" if details[index] else ""
        return f"{title} · {labels[index]}{suffix}"

    if show_histograms:
        figure = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(left_title, right_title, "Reflectivity distribution"),
            row_heights=[0.72, 0.28],
            vertical_spacing=0.13,
            horizontal_spacing=0.08,
        )
    else:
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(left_title, right_title),
            horizontal_spacing=0.08,
        )
    histogram_edges = np.arange(-20.0, 70.0 + 10.0, 10.0)
    histogram_centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])

    def histogram(values: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=histogram_edges)
        return counts / max(1, counts.sum())

    traces_per_example = 4 if show_histograms else 2
    for column, values, side in (
        (1, left[0], left_title),
        (2, right[0], right_title),
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis_km,
                y=axis_km,
                z=values[0],
                name=f"{labels[0]} {side}",
                coloraxis="coloraxis",
                zsmooth=False,
                hovertemplate=(
                    "x=%{x:.1f} km<br>y=%{y:.1f} km<br>"
                    "%{z:.1f} dBZ<extra></extra>"
                ),
                showlegend=False,
            ),
            1,
            column,
        )
    if show_histograms:
        for values, side, color in (
            (left[0], left_title, "#38bdf8"),
            (right[0], right_title, "#fb7185"),
        ):
            figure.add_trace(
                go.Bar(
                    x=histogram_centers,
                    y=histogram(values[0]),
                    name=side,
                    marker={"color": color},
                    opacity=0.65,
                    hovertemplate=(
                        f"%{{x:.1f}} dBZ<br>%{{y:.3%}}<extra>{side}</extra>"
                    ),
                ),
                2,
                1,
            )

    frame_count = expected[0]
    animation_frames = []
    for example_index, (left_values, right_values) in enumerate(zip(left, right)):
        for frame in range(frame_count):
            data = [
                go.Heatmap(z=left_values[frame]),
                go.Heatmap(z=right_values[frame]),
            ]
            animation_frames.append(
                go.Frame(
                    name=f"{example_index}:{frame + 1}",
                    traces=[0, 1],
                    data=data,
                )
            )
    figure.frames = animation_frames
    dropdown = [
        {"label": label, "method": "skip", "args": [index]}
        for index, label in enumerate(labels)
    ]

    shapes = [
        {
            "type": "circle",
            "xref": xref,
            "yref": yref,
            "x0": -radius_km,
            "x1": radius_km,
            "y0": -radius_km,
            "y1": radius_km,
            "line": {"color": "rgba(255,255,255,0.9)", "width": 2},
        }
        for xref, yref in (("x", "y"), ("x2", "y2"))
    ]
    xaxis = {
        "range": [-radius_km - 5, radius_km + 5],
        "title": "East–west distance (km)",
        "gridcolor": "rgba(255,255,255,0.12)",
        "zerolinecolor": "rgba(255,255,255,0.3)",
    }
    yaxis = {
        "range": [-radius_km - 5, radius_km + 5],
        "title": "North–south distance (km)",
        "scaleanchor": "x",
        "scaleratio": 1,
        "gridcolor": "rgba(255,255,255,0.12)",
        "zerolinecolor": "rgba(255,255,255,0.3)",
    }
    figure.update_layout(
        template="plotly_dark",
        barmode="overlay",
        title={"text": full_title(0), "x": 0.5},
        coloraxis={
            "colorscale": "Turbo",
            "cmin": LOWER_DBZ,
            "cmax": UPPER_DBZ,
            "colorbar": {"title": "Reflectivity<br>(dBZ)"},
        },
        xaxis=xaxis,
        yaxis=yaxis,
        xaxis2={**xaxis, "anchor": "y2"},
        yaxis2={**yaxis, "scaleanchor": "x2", "anchor": "x2"},
        shapes=shapes,
        height=980 if show_histograms else 780,
        margin={"t": 145, "b": 290},
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": dropdown,
                "active": 0,
                "x": 0,
                "y": 1.18,
                "bgcolor": "#172033",
                "bordercolor": "#94a3b8",
                "font": {"color": "white"},
            }
        ],
        sliders=[
            {
                "active": 0,
                "activebgcolor": "#38bdf8",
                "bgcolor": "#334155",
                "bordercolor": "#94a3b8",
                "font": {"color": "#f8fafc"},
                "currentvalue": {"prefix": f"{frame_label}: "},
                "pad": {"t": 25},
                "y": 0.04,
                "steps": [
                    {
                        "label": str(frame + 1),
                        "method": "animate",
                        "value": frame + 1,
                        "args": [
                            [f"0:{frame + 1}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame in range(frame_count)
                ],
            },
            {
                "active": 0,
                "activebgcolor": "#38bdf8",
                "bgcolor": "#334155",
                "bordercolor": "#94a3b8",
                "font": {"color": "#f8fafc"},
                "currentvalue": {"prefix": "Minimum dBZ: "},
                "y": -0.10,
                "steps": [
                    {
                        "label": str(value),
                        "method": "skip",
                        "value": value,
                    }
                    for value in range(int(LOWER_DBZ), 51, 5)
                ],
            },
            {
                "active": int((UPPER_DBZ - 20) // 5),
                "activebgcolor": "#38bdf8",
                "bgcolor": "#334155",
                "bordercolor": "#94a3b8",
                "font": {"color": "#f8fafc"},
                "currentvalue": {"prefix": "Maximum dBZ: "},
                "y": -0.24,
                "steps": [
                    {
                        "label": str(value),
                        "method": "skip",
                        "value": value,
                    }
                    for value in range(20, int(UPPER_DBZ) + 1, 5)
                ],
            },
        ],
    )
    if show_histograms:
        figure.update_xaxes(title_text="Reflectivity (dBZ)", row=2, col=1)
        figure.update_yaxes(
            title_text="Fraction of Cartesian cells",
            tickformat=".1%",
            row=2,
            col=1,
        )
    min_values = list(range(int(LOWER_DBZ), 51, 5))
    max_values = list(range(20, int(UPPER_DBZ) + 1, 5))
    positions = np.linspace(0.0, 1.0, 128)
    fixed_colors = sample_colorscale("Turbo", positions, colortype="rgb")
    visibility_scales = {}
    for minimum in min_values:
        for maximum in max_values:
            scale = []
            for position, color in zip(positions, fixed_colors):
                dbz = LOWER_DBZ + position * (UPPER_DBZ - LOWER_DBZ)
                alpha = 1.0 if minimum <= dbz <= maximum else 0.0
                rgb = color.removeprefix("rgb(").removesuffix(")")
                scale.append([float(position), f"rgba({rgb},{alpha})"])
            visibility_scales[f"{minimum},{maximum}"] = scale
    histogram_data = (
        [
            [
                [histogram(values[frame]).tolist() for frame in range(frame_count)]
                for values in side
            ]
            for side in (left, right)
        ]
        if show_histograms
        else []
    )

    post_script = f"""
    (() => {{
      const plot = document.getElementById('radar-comparison');
      const scales = {json.dumps(visibility_scales)};
      const labels = {json.dumps(labels)};
      const titles = {json.dumps([full_title(index) for index in range(len(labels))])};
      const frameCount = {frame_count};
      const histogramData = {json.dumps(histogram_data)};
      const showHistograms = {str(show_histograms).lower()};
      let minimum = {int(LOWER_DBZ)};
      let maximum = {int(UPPER_DBZ)};
      let selected = 0;
      plot.on('plotly_buttonclicked', event => {{
        selected = labels.indexOf(event.button.label);
        const steps = Array.from({{length: frameCount}}, (_, index) => ({{
          label: String(index + 1),
          method: 'animate',
          args: [[`${{selected}}:${{index + 1}}`], {{
            mode: 'immediate',
            frame: {{duration: 0, redraw: true}},
            transition: {{duration: 0}}
          }}]
        }}));
        Plotly.relayout(plot, {{
          'title.text': titles[selected],
          'sliders[0].steps': steps,
          'sliders[0].active': 0
        }});
        Plotly.animate(plot, [`${{selected}}:1`], {{
          mode: 'immediate', frame: {{duration: 0, redraw: true}}
        }});
        if (showHistograms) {{
          Plotly.restyle(plot, {{y: [histogramData[0][selected][0]]}}, [2]);
          Plotly.restyle(plot, {{y: [histogramData[1][selected][0]]}}, [3]);
        }}
      }});
      plot.on('plotly_sliderchange', event => {{
        const prefix = event.slider.currentvalue.prefix || '';
        if (showHistograms && prefix.startsWith('{frame_label}')) {{
          const frame = Number(event.step.value) - 1;
          Plotly.restyle(plot, {{y: [histogramData[0][selected][frame]]}}, [2]);
          Plotly.restyle(plot, {{y: [histogramData[1][selected][frame]]}}, [3]);
        }}
        if (prefix.startsWith('Minimum')) minimum = Number(event.step.value);
        if (prefix.startsWith('Maximum')) maximum = Number(event.step.value);
        Plotly.relayout(plot, {{
          'coloraxis.colorscale': scales[`${{minimum}},${{maximum}}`]
        }});
      }});
    }})();
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output,
        include_plotlyjs=True,
        auto_play=False,
        div_id="radar-comparison",
        post_script=post_script,
    )
