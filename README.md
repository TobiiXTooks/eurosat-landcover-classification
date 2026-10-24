# EuroSAT Land Cover Classification

Classifying Sentinel-2 satellite images from the [EuroSAT](https://github.com/phelber/eurosat)
dataset with deep learning. The project compares a **CNN trained from scratch** with
**transfer learning** on pretrained backbones (ResNet-50, EfficientNet-B0) on two tasks:

- **Multiclass**: 10 land-use classes (AnnualCrop, Forest, HerbaceousVegetation, Highway,
  Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- **Binary**: vegetation vs. non-vegetation

Every model is trained and evaluated on the same stratified 70/15/15 split (seed 42), which keeps
the comparison fair. A further campaign measures how much labelled data each approach needs.

Team project for *Deep Learning und Big Data* at TH Deggendorf (summer 2026) by
Andreas Glashauser, Patrick Hübler and Tobias Litke.

## Results

Test-set accuracy, trained on the full training split:

| Model | Multiclass | Binary |
|---|---:|---:|
| Scratch CNN (0.39 M params) | 94.8 % | 97.4 % |
| ResNet-50, frozen backbone | 93.7 % | – |
| EfficientNet-B0, fine-tuned | 97.3 % | 99.4 % |
| ResNet-50, fine-tuned (15 M trainable params) | **98.2 %** | **99.4 %** |

**Data efficiency**: multiclass test accuracy when only a limited number of training images per
class is used:

| Images per class | Scratch CNN | ResNet-50 |
|---:|---:|---:|
| 50 | 70.2 % | 92.8 % |
| 200 | 84.3 % | 95.6 % |
| all | 94.8 % | 98.2 % |

Fine-tuning a pretrained backbone gives the best results throughout. Its lead is largest when
training data is scarce: with only 50 images per class, ResNet-50 beats the scratch CNN by more than
20 percentage points.

The scratch CNN configuration came from an ablation study (E0–E5). SGD with momentum, augmentation
and regularisation scored best, and the tuned model reaches a test macro-F1 of 95.2 % (multiclass)
and 98.0 % (binary). A depthwise-separable variant has 8× fewer parameters but lost about
4.5 points of validation macro-F1.

Each run's full numbers, plots, confusion matrices and misclassified examples are in
[`outputs/experiments/`](outputs/experiments/), and the LaTeX report is in [`report/`](report/).

## Project structure

```text
src/cnn/                  RGB pipeline (installable `cnn` package): data, models, training, metrics, plots
configs/                  YAML configs for single runs and experiment campaigns
outputs/experiments/      campaign results: course_practical, scratch_vs_transfer, data_efficiency, efficientnet_b0
with_help/EfficientNet/   EfficientNet-B0 transfer learning
from_scratch_only_numpy/  CNN implemented from scratch in plain NumPy
lib/                      shared code for the multispectral data loader
notebooks/                data exploration and transfer-learning experiments
report/                   LaTeX report and figures
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

Download `EuroSAT_RGB.zip` (plus `EuroSAT_MS.zip` for the EfficientNet script) from the
[EuroSAT repository](https://github.com/phelber/eurosat) and put the files in `data/`.

## Usage

```bash
# Prepare data once: extract, stratified split, normalisation stats
uv run python -m cnn.prepare_data --zip data/EuroSAT_RGB.zip

# Train a single model
uv run python -m cnn.train --task multiclass --config configs/scratch_cnn.yaml
uv run python -m cnn.train --task binary     --config configs/scratch_cnn.yaml

# Re-evaluate a finished run
uv run python -m cnn.evaluate --run-dir outputs/runs/<run_id>

# Run an experiment campaign (ablation + final models)
uv run python -m cnn.experiments --config configs/experiments/course_practical.yaml

# EfficientNet-B0 transfer learning
uv run python with_help/EfficientNet/train_efficientnet_b0.py --task all
```

Add `--smoke` to `cnn.train` or `cnn.experiments` for a tiny CPU-only test run.

Every run saves its resolved config, split reference, best checkpoint, metrics (JSON and CSV) and
plots (loss/accuracy curves, confusion matrix, misclassified grid) under `outputs/`. Model
checkpoints are not committed, but training recreates them.
