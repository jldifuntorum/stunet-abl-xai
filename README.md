# STUNet — Atmospheric boundary-layer wind prediction and explainability

Python implementation of spatiotemporal U-Net wind-speed prediction and explainability analyses accompanying *Atmospheric stability shapes neural network sensitivity across height and time in near surface wind prediction* (submitted to *Communications Earth & Environment*).

The repository provides the STUNet architecture, preprocessing, training, evaluation, baseline models, Integrated Gradients, occlusion, ROI attribution, and numerical source-data plotting. It is a minimal Python release with no notebooks or large data files.

## Data

The processed LES input fields and numerical figure source data are available at **[Zenodo dataset](https://doi.org/10.5281/zenodo.23753082)**. Place the 12 wind-speed arrays in a local `data/` directory.

| Regime key | Near-surface target | Predictor fields |
|---|---|---|
| `neutral` | `neutral_wspeed_4.5.npy` | `neutral_wspeed_{49.5,103.5,148.5}.npy` |
| `transitional` | `transitional_wspeed_4.5.npy` | `transitional_wspeed_{49.5,103.5,148.5}.npy` |
| `convective` | `convective_wspeed_4.5.npy` | `convective_wspeed_{49.5,103.5,148.5}.npy` |

Arrays are expected in `(time, y, x)` order. The predictor axis is ordered **150, 100, 50 m** (sampled at 148.5, 103.5, 49.5 m), followed by four temporal frames from oldest to most recent. The target is wind-speed magnitude at 4.5 m.

## Installation

Use Python 3.10–3.12 and an appropriate PyTorch build for the local machine.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Training and evaluation

The reference workflow uses four input frames separated by 4 s; a **chronological 70/15/15 train/validation/test split**; and standardization estimated from training input frames only. For 3,600 source time steps, this produces 2,511 training, 538 validation and 539 test sequences.

```bash
python -m stunet.cli train --data data --regime neutral --checkpoint checkpoints/neutral.pt
python -m stunet.cli evaluate --data data --regime neutral --checkpoint checkpoints/neutral.pt
```

Replace `neutral` with `transitional` or `convective` for the other stability regimes. The `--spacing` option controls the input-frame spacing for temporal sampling experiments. Model names are `stunet`, `stunet50`, `stunet50_tm`, `unet`, `tinycnn` and `linear`.

## Explainability

```bash
python -m stunet.cli xai --data data --regime neutral --checkpoint checkpoints/neutral.pt --output results/ig_neutral
python -m stunet.cli roi --data data --regime neutral --checkpoint checkpoints/neutral.pt --bounds 100 164 120 184 --output results/roi_neutral
```

The XAI routines compute 16-step Integrated Gradients and height–lag occlusion; ROI attribution uses 32 integration steps. Occlusion reports **unperturbed prediction minus mean-replaced prediction**. Signed ROI attribution is reported at each input height, lag and pixel.

## Figures from source data

Download `Figure_Source_Data.zip` from the Zenodo dataset and run:

```bash
python scripts/plot_source_data.py --archive Figure_Source_Data.zip --output figures
```

The plotting utility produces numerical summaries of the height–lag attribution, temporal-sampling performance, and cross-spacing attribution (Figures 3, 5 and 6).

## Implementation details

The release implements chronological splitting and training-input standardization as specified in the revised manuscript. The execution-specific historical loader was not retained; archived model weights are separate experiment artifacts. The CLI also provides `--normalization per_field` and `--normalization none` for checkpoint compatibility.

## Citation and license

The source LES study is Aksamit, Katurji & Zhang (2024), *Journal of Geophysical Research: Atmospheres*, [doi:10.1029/2024JD041490](https://doi.org/10.1029/2024JD041490). Cite the Zenodo input dataset and the versioned software DOI once the GitHub release is archived. Code is distributed under the MIT license.
