# NEXRAD Forecast Experiment

A small, script-first weather forecasting experiment using real NOAA NEXRAD
Level III reflectivity. The workflow has four explicit stages:

| Stage | Command | Result |
|---|---|---|
| Get data | `python scripts/get_data.py` | Downloads NOAA scans and builds the Cartesian tensor |
| Convert existing data | `python scripts/prepare_cartesian.py` | Converts an existing polar tensor without downloading |
| Train embedding | `python scripts/train_embedding.py` | Learns encoder/decoder reconstruction |
| Inspect embedding | `python scripts/inspect_embeddings.py` | Verifies reconstruction and latent structure |
| Train predictor | `python scripts/train.py` | Freezes the embedding and learns future prediction |
| Validate | `python scripts/validate.py` | Reports MAE and RMSE on the untouched test set |
| Plot | `python scripts/plot.py` | Writes an interactive Plotly validation explorer |
| Browse | `ml-weather-forecasting` | Opens every embedding and forecast validation result in SigVue |

Each script has a short settings block at the top. Shared loading and model
code lives in `scripts/forecast.py`; it is support code, not another workflow
step.

As a convenience, `python experiment.py` runs isolated 120, 230, and 460 km
experiments. For each radius it prepares a distinct 64×64 Cartesian tensor,
trains a distinct embedding and forecast model, and writes distinct validation
metrics beneath `outputs/radius-<N>km/`. The combined return value is also
written to `outputs/experiment-results.json`. Edit `RADII_KM` in
`experiment.py` to change the matrix.

The actual PyTorch network is defined in `scripts/model.py`. It is a spatial
encoder/decoder with an explicit learned embedding:

```text
Each frame: one normalized log1p(linear Z) channel on a 64 × 64 Cartesian grid
→ shared spatial Conv2D encoder
→ per-frame embedding: 48 × 8 × 8
→ sequence embedding: 48 channels × 25 times × 8 × 8 space
→ skip-2 latent differences: E3−E1, E4−E2, …, E25−E23
→ signed radar differences → separately trained motion encoder
→ ConvGRU reads ordered motion embeddings and estimates motion changes
→ autoregressive latent rollout anchored to the final observed embedding
→ shared spatial ConvTranspose2D decoder
→ 15 radar forecast frames
```

The prototype embedding compresses each observed frame into 3,072 learned
spatial features while retaining an 8×8 layout. Representation learning and prediction
are deliberately separate. `train_embedding.py` trains only the encoder and
decoder using reconstruction. After inspection, `train.py` freezes both and
trains only the deep latent-dynamics network using forecast loss. The predictor
therefore cannot spend its capacity improving reconstruction or alter the
established representation. Call `model.embed(features)` to retrieve the latent
tensor directly.

The learned predictor never receives absolute embeddings. With `MOTION_SKIP=2`,
its learned path receives only embeddings of signed radar differences.
Forecasting starts from the final observed
embedding and a recent mean per-frame latent velocity, then rolls forward autoregressively. The ConvGRU
learns corrections to that velocity. With its correction head at zero, the
baseline is constant-velocity extrapolation; a stationary sequence remains
exact persistence. Radar-space residual correction prevents autoencoder error
from changing the final observed anchor.

Prediction loss combines reflectivity error, frame-to-frame motion error, and
spatial-gradient error. This discourages both motionless persistence and blurry
average blobs. All three terms apply additional squared weighting to stronger
echoes, so motion and boundaries around storm cores matter substantially more
than changes in clear air.

Motion has an additional severe-echo rule: if either endpoint of a displacement
is at least 50 dBZ, that motion receives a 12× multiplicative boost on top of
the continuous reflectivity weighting. The motion encoder is pretrained to
reconstruct signed skip-frame radar differences and then frozen for forecasting.

The predictor also receives the observed progression of cumulative reflectivity
histograms. For every input frame it measures the fraction of the Cartesian
domain exceeding 10, 20, 30, 40, 50, and 60 dBZ, plus the frame-to-frame change
in those fractions. A GRU encodes all 25 histogram steps into the forecast
context. Forecast loss matches the same exceedance curves for all five future
frames, with increasing weight on 40–60 dBZ coverage. This separates “how much
storm intensity should exist” from the motion model's “where should it go.”

Without downloading anything else, prediction also uses eleven manifest-derived
context values: sine/cosine of UTC hour, sine/cosine of day of year, mean
observed scan spacing, most recent scan spacing, and all five future intervals
(last observation→forecast 1 through forecast 4→forecast 5). A small MLP turns these values into a context embedding that
conditions the ConvGRU hidden state. Calendar values are cyclic, and cadence
values are scaled in ten-minute units. Temperature is not fabricated; it can be
added later as another auxiliary value when a real aligned source is available.

Train and inspect the representation before training prediction:

```bash
python scripts/train_embedding.py
python scripts/inspect_embeddings.py
```

This writes `outputs/embedding-report.html`: a validation-example
dropdown with observed and reconstructed Cartesian radar side by side, a frame
scrubber, per-example reconstruction MAE, and the true/reconstructed dBZ
histograms for the selected frame. It intentionally does not attempt
to visualize or project the latent space.

Autoencoder loss includes pixel/structure reconstruction, signed motion-
difference reconstruction, and a differentiable Jensen–Shannon divergence
between the true and reconstructed per-frame reflectivity histograms. Histogram
error therefore backpropagates through the decoder into the spatial embedding;
it is not merely a reporting diagnostic.

Histogram reconstruction combines the full-distribution divergence, a
high-reflectivity-weighted divergence, and cumulative area errors above
10–60 dBZ. Its loss weight is 0.10. Both reconstruction and validation explorers
use the same shared renderer and show overlaid true/predicted histogram bars for
the currently selected example and frame.

Histogram divergence and displayed histogram bars include only reflectivity
between `-20` and `70 dBZ`. The `-32 dBZ` zero/background mass and values above
70 dBZ do not participate in histogram normalization or matching. Full-field
pixel reconstruction remains responsible for values outside that interval. Both
training and reports use nine broad 10 dBZ-wide histogram bands.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 1. Get data

```bash
python scripts/get_data.py
```

This selects 1,000 thirty-frame KTLX sequences from 2016–2025, downloads the
native scans, crops them to 120 km, and writes both the archival native-polar
tensor and the model-ready Cartesian tensor:

```text
data/datasets/ktlx-reflectivity-120km-2016-2025-v1/
├── dataset.json
└── tensors/
    ├── polar-180x64.npy
    ├── polar-180x64.json
    ├── cartesian-z-64x64.npy
    └── cartesian-z-64x64.json
```

The prototype model uses only `cartesian-z-64x64.npy`, shaped
`example × 30 frames × 64 y cells × 64 x cells`. Each cell spans 3.75 km, and
the tensor has four times fewer spatial values than the 128×128 version.
Conversion is nearest-bin resampling followed by `Z = 10^(dBZ/10)`. The tensor
stores float32 physical linear reflectivity Z. NaNs and cells outside the 120 km
circle become exactly `Z = 0`. Before entering the network, Z is transformed
with normalized `log1p(Z)` to control its enormous dynamic range. Predictions
are converted back to dBZ only for metrics and plots. Downloads are
resumable. To build only the catalog and manifest, run:

```bash
python scripts/get_data.py --catalog-only
```

If the polar tensor already exists, do not download again. Convert it once:

```bash
python scripts/prepare_cartesian.py
```

## 2. Learn and inspect the embedding

```bash
python scripts/train_embedding.py
python scripts/inspect_embeddings.py
```

Do not move on just because reconstruction loss decreased. Inspect the animated
observed-versus-reconstructed frames in
`outputs/embedding-report.html`. The
best encoder and decoder are stored in `outputs/autoencoder.pt`.

## 3. Train prediction

```bash
python scripts/train.py
```

This loads `outputs/autoencoder.pt`, freezes its encoder and decoder, and trains
only the deep latent-dynamics network. It observes 25 frames and predicts the
following 5 frames. It
reuses the exact 70/15/15 split established during embedding training and saves
the best predictor to `outputs/model.pt`.

## 4. Validate

```bash
python scripts/validate.py
```

This loads the saved model and evaluates it once on the held-out test examples.
It computes MAE and RMSE in physical linear reflectivity Z first, then reports
both linear-Z values and their `10·log10(error)` dBZ equivalents. It does not
compute per-pixel errors in dBZ space.

## 5. Watch the learned forecast

Install the project once, then launch its SigVue application:

```bash
pip install -e .
ml-weather-forecasting
```

The browser groups results by data radius first, then exposes every held-out
example under **Embeddings** for observed-versus-reconstructed frames and
**Validation** for truth versus learned predictions. Completed radius runs are
discovered from their `experiment.json` manifests. Select an item and use the frame
playback controls to inspect it. Unlike the legacy self-contained HTML files,
the SigVue application loads one selected result at a time, so the full
validation split remains practical to browse.

Each item has a **Render side-by-side GIF** batch action. The command-line
equivalent is:

```bash
ml-weather-forecasting batch \
  --workspace ml-weather-forecasting \
  --item forecast-16 \
  --action render-side-by-side-gif \
  --output outputs/exported-gifs
```

Use `ml-weather-forecasting batch --list` to see the available item IDs. Batch
GIFs loop forever and progress through the frames with observation/truth on the
left and reconstruction/forecast on the right. The workspace-wide action
renders the complete collection to `outputs/gifs`.

### Legacy standalone explorer

```bash
python scripts/plot.py
```

Open `outputs/validation-explorer.html` in a browser. Choose a validation
example from the dropdown, then scrub through the predicted frames.
The explorer reads the same Cartesian tensor used during training and shows
truth beside forecast in east/west and north/south kilometers. A 120 km ring
marks the data boundary.
Hover over a point for its position and reflectivity. Edit
`VALIDATION_EXAMPLES` in `scripts/plot.py` to control how many examples appear
in the dropdown; including more examples makes the self-contained HTML larger.

The model learns ordinary Cartesian spatial neighborhoods rather than treating
native polar ray/gate indexes as if they were a rectangular physical grid.
