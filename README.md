# DualNeuron

![DualNeuron Logo](logo.png)

**DualNeuron** is the codebase accompanying our paper:

> **Dual-feature selectivity enables bidirectional coding in visual cortical neurons**  
> Franke K.\*, Karantzas N.\*, Willeke K., Diamantaki M., Ramakrishnan K., Bedel H.A., Elumalai P., Restivo K., Fahey P., Nealley C., Shinn T., Garcia G., Patel S., *et al.*  
> bioRxiv (2025)  
> [https://doi.org/10.1101/2025.07.16.665209](https://doi.org/10.1101/2025.07.16.665209)

The data required to reproduce our results is available on Dryad:  
[https://doi.org/10.5061/dryad.q573n5tx3](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5tx3)

---

## Overview

We discovered that many neurons in visual cortex exhibit **dual-feature selectivity**—they respond strongly to preferred features while being systematically suppressed by distinct non-preferred features around elevated baseline firing rates. This **bidirectional coding** strategy appears conserved across species (macaque and mouse) and visual areas (from V1 to V4).

This package provides tools to:

- **Load digital twin models**: Pretrained neural predictive models (deep learning models trained to predict neural responses from images) for macaque V1 and V4
- **Screen large image datasets**: Identify most-activating (MAIs) and least-activating (LAIs) natural images for each neuron
- **Synthesize optimal stimuli**: Generate most-exciting inputs (MEIs) and least-exciting inputs (LEIs) via gradient-based optimization. The synthesis algorithm uses ideas from [https://github.com/serre-lab/Horama](https://github.com/serre-lab/Horama) for optimization in the frequency domain, though the implementation has been extended with different transforms and alternative constraints.
- **Compute semantic axes**: Use DreamSim embeddings to analyze semantic relationships between high and low activation poles
- **Visualize and analyze**: Plot activation curves, population statistics, and optimization trajectories

## Key Concepts

### Bidirectional Neural Coding

Traditional views characterize neurons by what excites them. Our work reveals that neurons also have a **low pole**—stimuli that systematically suppress activity below baseline. The full response range spans from:

- **High Pole (MEI/MAI)**: Stimuli maximizing neural response
- **Low Pole (LEI/LAI)**: Stimuli minimizing neural response (maximal suppression)

### Digital Twin Models

We trained neural predictive models on recordings from:

| Model | Area | Neurons | Input Size | Backbone |
|-------|------|---------|------------|----------|
| `V1GrayTaskDriven` | Macaque V1 | 458 | 93×93 grayscale | ConvNeXtV2-Atto |
| `V4ColorTaskDriven` | Macaque V4 | 394 | 100×100 RGB | ResNet50 L2-robust |
| `V4GrayTaskDriven` | Macaque V4 | 1244 | 100×100 grayscale | ResNet50 L2-robust |

Each model uses **ensemble averaging** (5-10 models) for robust predictions and **Gaussian readouts** for spatial pooling.

## Installation

**Requirements:** Python ≥3.10

```bash
# Install uv if you don't have it:
#   https://docs.astral.sh/uv/getting-started/installation/

# Create the virtual environment and install the package with all dependencies
# (including a GPU-compatible torch from the configured CUDA 12.1 index)
uv sync

# Activate it (or prefix commands with `uv run`)
source .venv/bin/activate
```

## Configuration

Create a `.env` file in the repository root (copy `.env.example`):

```bash
cp .env.example .env
```

```bash
HF_TOKEN=your_huggingface_token                 # Hugging Face token (to download ImageNet)
DATA_DIR=/path/to/your/data                     # Root data directory (see layout below)
IMAGENET_CACHE_DIR=${DATA_DIR}/datasets         # Hugging Face ImageNet cache
RENDERED_DIR=${DATA_DIR}/datasets/rendered      # Rendered-scene archives (batch_*.zip)
MODELS_DIR=${DATA_DIR}/models                   # Cached model weights (e.g. DreamSim)
ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS  # Regenerated screening/synthesis/DreamSim outputs
LOGS_DIR=./logs                                 # Progress logs (one self-rewriting line per run)
PAPER_FIG_DIR=./figs                            # Saved figures
```

Directories are created on demand by the scripts that write into them; `LOGS_DIR`
and `PAPER_FIG_DIR` are gitignored.

`DATA_DIR` is the root under which the ImageNet cache, the cached model weights,
and the Dryad data live:

```
DATA_DIR/
├── datasets/                  # Hugging Face ImageNet cache (IMAGENET_CACHE_DIR)
│   └── rendered/              # rendered-scene archives batch_*.zip (RENDERED_DIR)
├── models/                    # cached model weights, e.g. DreamSim (MODELS_DIR)
├── DUAL-FEATURE-ANALYSIS/     # regenerated outputs (ANALYSIS_DIR); see "Saved-file layout"
│   ├── v4/
│   └── v1/
└── dryad/                     # optional: the published Dryad release
```

### Getting ImageNet

ImageNet is **not** redistributed on Dryad (its license does not allow it), so you
download it yourself from Hugging Face. One-time setup:

1. Create a Hugging Face account and request access to the gated
   [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
   dataset (accept its terms on that page).
2. Create an access token and put it in `.env` as `HF_TOKEN`.
3. The first screening run downloads ImageNet into `DATA_DIR/datasets` using the
   token; afterwards it loads from that cache and the token is no longer needed.

The download is handled transparently by `ImagenetImages` / `screen_activations`
via `datasets.load_dataset("ILSVRC/imagenet-1k", token=..., cache_dir=DATA_DIR/datasets)`.
The token is read from `HF_TOKEN` in `.env` automatically if you do not pass it
explicitly.

### Getting the Dryad data

The neural responses, sorted indices, MEIs/LEIs, and rendered scenes are released
on Dryad: [https://doi.org/10.5061/dryad.q573n5tx3](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5tx3).
Dryad does not allow anonymous scripted downloads, so download the files you need
from that page in a browser:

- the rendered-scene archives `batch_*.zip` go in **`RENDERED_DIR`** (default
  `${DATA_DIR}/datasets/rendered`) — `RenderedImages` reads them directly, no unzip needed
- the published `.npz` (ordered responses/indices, MEIs/LEIs) are **optional**: this
  pipeline regenerates its own into `ANALYSIS_DIR` (see "Reproducing the paper"), so you
  only need them if you want to start from the released results rather than recompute them

## Usage

### Loading Digital Twin Models

```python
from dualneuron.twins.nets import V4ColorTaskDriven, V1GrayTaskDriven, load_model

# Load V4 color model (ensemble of 5 models)
v4_model = V4ColorTaskDriven(ensemble=True, centered=True)

# Load V1 grayscale model
v1_model = V1GrayTaskDriven(ensemble=True, centered=True)

# Or use the unified loader with layer extraction
model = load_model(
    architecture='v4',      # 'v1', 'v4', 'v4g', or standard architectures
    layer=None,             # Extract from specific layer (optional)
    ensemble=True,
    centered=True,          # Center readout for MEI synthesis
    device='cuda'
)
```

### Screening Large Image Datasets

Identify which natural images most/least activate each neuron:

```python
import os
from dotenv import load_dotenv
from dualneuron.screening.run import screen_activations

load_dotenv()  # reads HF_TOKEN and DATA_DIR from .env
data_dir = os.path.join(os.environ["DATA_DIR"], "datasets")  # ImageNet cache

# Screen ImageNet to find MAIs/LAIs (token read from .env if omitted)
sorted_responses, sorted_indices = screen_activations(
    data_dir=data_dir,
    token=os.getenv("HF_TOKEN"),
    split='train',
    dataset="imagenet",     # or "rendered" for synthetic scenes
    model='v4',             # 'v1', 'v4', or 'v4g'
    batch_size=128,
    device='cuda'
)

# sorted_indices[:, neuron_id][:10]  → LAIs (lowest 10)
# sorted_indices[:, neuron_id][-10:] → MAIs (highest 10)
```

### Synthesizing MEIs and LEIs

Generate optimal stimuli via gradient ascent:

```python
from dualneuron.synthesis.ascend import fourier_ascending, pixel_ascending

# For V4 (color): Fourier-parameterized synthesis with natural priors
result = fourier_ascending(
    objective_function=lambda x: model(x)[:, neuron_id].mean(),
    magnitude_path='natural_rgb.npy',  # Natural image frequency prior
    total_steps=128,
    learning_rate=1.0,
    values_range=(-2.0, 2.0),
    target_norm=40.0,
    device='cuda',
    verbose=True
)

# For V1 (grayscale): Direct pixel optimization
result = pixel_ascending(
    objective_function=lambda x: model(x)[:, neuron_id].mean(),
    image_size=93,
    channels=1,
    total_steps=128,
    learning_rate=0.05,
    target_norm=12.0,
    device='cuda'
)

mei = result['image']           # Synthesized image
alpha = result['alpha']         # Saliency/transparency map
activation = result['activation']  # Final activation value
```

### Batch Generation of MEIs/LEIs

Generate MEIs and LEIs for all neurons:

```python
# Run one area per process (each on its own GPU); resumable, one npz per neuron:
#   CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4
#   CUDA_VISIBLE_DEVICES=1 python -m dualneuron.synthesis.generate --area v1

from dualneuron.synthesis.generate import generate

generate(
    area="v4",           # "v4" or "v1"
    num_seeds=10,        # random initializations per neuron
    neurons=None,        # default: the well-predicted set (correlation-to-average > 0.4)
)
# -> ANALYSIS_DIR/v4/synthesis/v4_neuron{id:04d}.npz
#    (mei/lei image, alpha, and activation for each seed)
```

### Semantic Analysis with DreamSim

Compute semantic axes between activation poles:

```python
from dualneuron.dream.axis import semantic_axis
from dualneuron.dream.sim import embeddings
from dreamsim import dreamsim

# Load DreamSim model
dreamsim_model, _ = dreamsim(pretrained=True, device='cuda')

# Compute semantic axis from MAIs to LAIs
axis = semantic_axis(
    images1=mai_images,    # High-activating images
    images2=lai_images,    # Low-activating images
    dreamsim_model=dreamsim_model
)

# Use axis to guide synthesis toward semantic concepts
result = fourier_ascending(
    objective_function=objective,
    simulation_function=dreamsim_model.embed,
    simulation_axis=axis,
    simulation_weight=0.5,  # Weight for semantic guidance
    ...
)
```

### Visualization

```python
from dualneuron.synthesis.visualize import plot_poles, blend, sequence_animation
from dualneuron.screening.visualize import plot_neuron_activation, plot_neuron_poles

# Plot MEI/LEI pair with activation curves
plot_poles(
    images=[lei_image, mei_image],
    activations=[lei_activations, mei_activations]
)

# Blend image with saliency map
blended = blend(image, alpha, mean=0.45, std=0.25)

# Animate optimization trajectory
animation = sequence_animation(
    imgs=all_step_images,
    activities=all_step_activations,
    title="MEI Synthesis"
)

# Plot sorted activation curve for a neuron
plot_neuron_activation(neuron_id=42, resp_dir="responses/", response_stats=stats)
```

## Package Structure

```
dualneuron/
├── utils.py                # Shared helpers: env_dir, ensure_dir, RewriteLine (logs),
│                           #   well_predicted_neurons (corr>0.4), sparse_split (skewness<2)
├── twins/                  # Digital twin neural predictive models
│   ├── nets.py            # Model loaders (V1GrayTaskDriven, V4ColorTaskDriven, EnsembleModel)
│   ├── activations.py     # Activation extraction utilities
│   ├── V1GrayTaskDriven/  # V1 weights + mask.npy (RF) + correlations.npy
│   ├── V4ColorTaskDriven/ # V4 color weights + mask.npy + correlations.npy
│   └── V4GrayTaskDriven/  # V4 grayscale weights & metadata
│
├── screening/              # Large-scale image screening (MAIs/LAIs)
│   ├── run.py             # screen_activations; --member i for a single ensemble member
│   ├── sets.py            # ImageNet & rendered dataset loaders / transforms
│   ├── utils.py           # Statistics (Gini coefficient, adaptive sampling)
│   └── visualize.py       # Population & single-neuron visualizations
│
├── synthesis/              # Stimulus optimization (MEIs/LEIs)
│   ├── ascend.py          # Fourier (V4) & pixel (V1) gradient ascent
│   ├── generate.py        # Per-neuron MEI/LEI generation (resumable)
│   ├── mask.py            # Build the RF mask from the MEIs/LEIs -> twins/{model}/mask.npy
│   ├── ops.py             # Image ops (create_crops, create_neural_crops, norm, ...)
│   ├── visualize.py       # Optimization trajectory visualization
│   └── priors/            # Natural image magnitude spectra (natural_{gray,rgb}.npy)
│
├── dream/                  # DreamSim embedding analysis
│   ├── sim.py             # DreamSim embedding extraction (fp16, per-area defaults)
│   ├── subset.py          # Build the per-area ImageNet embedding subset
│   ├── similarity.py      # MAI/LAI coherence d-prime (Fig 6), 2D similarity space (Fig 10)
│   └── axis.py            # Semantic axis computation (synthesis guidance)
│
└── figures/                # Paper figure generation
    └── make_fig_dreamsim.py  # Fig 6 (coherence d-prime) + Fig 10 (2D similarity space)
```

## Reproducing the paper — pipeline, runs, and status

The analyses form one dependency chain; each stage's output feeds the next:

**synthesis → acquire mask → screening → DreamSim → similarity (d-prime + R² vs. sparsity) → figures**

1. **Synthesis** (`synthesis/`). Gradient ascent on the centered ensemble twins produces,
   per well-predicted neuron, a most-exciting input (MEI) and a least-exciting input (LEI):
   V4 in the Fourier phase domain (`fourier_ascending`, natural-amplitude prior), V1 in
   pixels (`pixel_ascending`), 10 seeds each, ℓ2-constrained (40 V4 / 12 V1). One npz per neuron.
2. **Acquire mask** (`synthesis/mask.py`). Each area's receptive-field mask is built from the
   **mean alpha over all its synthesized MEIs/LEIs**: threshold at the ~77.5th percentile (the
   RF core), then Gaussian-soften the binary edge (σ ≈ 1.3). Stored at
   `dualneuron/twins/{model}/mask.npy`, it is the shared input to both screening and DreamSim,
   so each evaluates exactly the retinotopic region its neuron drives. The script reproduces the
   shipped masks (correlation 0.996 V4 / 0.992 V1; only a ~1-px edge ring differs).
3. **Screening** (`screening/`). Every image is RF-masked (bg 0) and ℓ2-normalized, run through
   the twins, and sorted per neuron to give MAIs/LAIs. Sources: 200,000 rendered scenes and the
   full 1,281,167 ImageNet-1k train images. Use the **ensemble** (default) or a single
   **member** (`--member i`).
4. **DreamSim** (`dream/sim.py`, `dream/subset.py`). Each image is RF-masked (neutral-gray
   bg 0.45), contrast-normalized, and embedded into the 1792-d DreamSim ensemble space
   (penultimate layer, unit-norm, fp16). Rendered = all 200k; ImageNet = a subset from
   `subset.py` (every neuron's K=15 MAIs+LAIs ∪ a 200k uniform sample), passed via `--indices_path`.
5. **Similarity** (`dream/similarity.py`). Embeddings are **globally centered** before every
   cosine (à la Franke — this removes the common-mode and is what gives d-prime/R² their range).
   **Fig 6 d-prime** (within-MAI / within-LAI cosine coherence vs. random, ddof = 1) and **Fig 10
   2D similarity space** (each image's mean cosine to a neuron's 15 MAI / 15 LAI poles → degree-1
   linear-fit R², with random-pole controls), over **all** well-predicted neurons and related to
   each neuron's **skewness** so R²/d-prime form a continuous spectrum across sparsity (skewness =
   2 is only a soft boundary; `utils.sparse_split`). The linear model is CV-validated — a degree-2
   surface adds only a median ~1% R². V4 non-sparse R² ≈ 0.28 (rendered), matching Franke's 0.23.
6. **Figures** (`figures/make_fig_dreamsim.py`). Renders Fig 6 (population coherence distributions
   + d-prime scatter) and Fig 10 (example 2D spaces with the activity-gradient arrow and its 1D
   projection, the R² histogram, and the R²-vs-control scatter) for both areas × datasets into
   `PAPER_FIG_DIR`.

### Paper → code

| Paper | Code |
|---|---|
| Fig 1 — twins + inclusion (corr-to-avg > 0.4) | `twins/nets.py`, `utils.well_predicted_neurons` |
| Fig 2 — sparseness (skewness < 2) | `utils.sparse_split` (on the ImageNet screening) |
| Figs 3–5 — MEIs/LEIs + MAIs/LAIs | `synthesis/generate.py` + `screening/run.py` |
| RF mask (shared by screening + DreamSim) | `synthesis/mask.py` |
| Fig 6 — DreamSim d-prime | `dream/sim.py` + `dream/similarity.py` |
| Fig 10 — 2D similarity space, R² vs. sparsity | `dream/similarity.py` |
| Figs 6 & 10 — plotting | `figures/make_fig_dreamsim.py` |

### Commands (install → analysis)

```bash
uv sync                                    # env + GPU torch (cu121)

# Synthesis — one area per GPU; resumable
CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4
CUDA_VISIBLE_DEVICES=1 python -m dualneuron.synthesis.generate --area v1

# Acquire mask — from the synthesized MEIs/LEIs (-> twins/{model}/mask.npy; prints corr vs shipped)
python -m dualneuron.synthesis.mask --area v4
python -m dualneuron.synthesis.mask --area v1

# Screening — ensemble (drop --member); a single member with --member i
CUDA_VISIBLE_DEVICES=0 python -m dualneuron.screening.run --model v4 --dataset rendered --num_workers 4
CUDA_VISIBLE_DEVICES=0 python -m dualneuron.screening.run --model v4 --dataset imagenet --num_workers 4

# DreamSim — build the ImageNet subset (ImageNet only), then embed
python -m dualneuron.dream.subset --area v4
CUDA_VISIBLE_DEVICES=1 python -m dualneuron.dream.sim --dataset rendered --area v4 --num_workers 4
CUDA_VISIBLE_DEVICES=1 python -m dualneuron.dream.sim --dataset imagenet --area v4 --num_workers 4 \
    --indices_path "$ANALYSIS_DIR/v4/v4_dreamsim_imagenet_indices.npy"

# Similarity — Fig 6 + Fig 10 (per model × dataset; saves {model}_similarity_{dataset}.npz)
python -m dualneuron.dream.similarity --model v4 --dataset rendered --output "$ANALYSIS_DIR/v4/v4_similarity_rendered.npz"

# Figures — Fig 6 + Fig 10 for both areas × datasets, into PAPER_FIG_DIR
python -m dualneuron.figures.make_fig_dreamsim
```

Per-area transforms (crop, channels, grayscale, contrast norm) are set automatically from
`--area`/`--model` — e.g. crop 200 (V4) / 167 (V1) so the RF mask aligns with the screening.

### Saved-file layout (anticipated names)

```
ANALYSIS_DIR/
├── v4/
│   ├── v4_ensemble_rendered_ordered_{responses,indices}.npz   # screening (ensemble)
│   ├── v4_ensemble_imagenet_ordered_{responses,indices}.npz
│   ├── v4_member0_imagenet_ordered_{responses,indices}.npz    # single-member screening
│   ├── v4_dreamsim_rendered_embeddings.npz                    # DreamSim (fp16, 1792-d)
│   ├── v4_dreamsim_imagenet_embeddings.npz
│   ├── v4_dreamsim_imagenet_indices.npy                       # ImageNet subset (subset.py)
│   ├── v4_similarity_{rendered,imagenet}.npz                  # similarity: per-neuron d-prime, R², controls, skewness
│   └── synthesis/
│       └── v4_neuron{id:04d}.npz                              # MEI/LEI: image, alpha, activation × 10 seeds
└── v1/                                                        # same scheme (grayscale, crop 167)
```

The RF mask is written back to `dualneuron/twins/{model}/mask.npy` (`synthesis/mask.py`), and the
Fig 6 / Fig 10 PDFs to `PAPER_FIG_DIR` (`dreamsim_dprime_{dataset}.pdf`,
`dreamsim_similarity_{area}_{dataset}.pdf`, plus the `mask_reconstruction.pdf` QC).

General scheme: `{area}_{run}_{dataset}_{kind}`, with `run ∈ {ensemble, member{i}}` and
`dataset ∈ {rendered, imagenet}`. Logs mirror it under `LOGS_DIR/`
(`{area}_{run}_{dataset}.log`, `{area}_synthesis.log`, `{area}_dreamsim_{dataset}.log`).

### Equipment, concurrency, and observed times

Hardware: 5 × 24 GB GPUs; a 100 GiB-RAM / 4-CPU-core cgroup. We run **one area per GPU** and use
`--num_workers 4` — the data loaders are JPEG-decode-bound, so without workers the GPU starves
and runtimes are ~4× longer.

- **Screening (observed):** V4 rendered (200k, ensemble) ≈ **20 min**; V4 ImageNet (1.28M, ensemble,
  4 workers) ≈ **1 h 9 min**; V1 ImageNet ≈ **1 h**; V4 ImageNet member-0 ≈ **1 h 9 min**.
- **Synthesis (observed rates):** V4 ≈ 213 s/neuron, V1 ≈ 141 s/neuron at 10 seeds → ≈ **12–13 h**
  (V4, 205 neurons) / ≈ **17 h** (V1, 445). Long runs are detached (`setsid`) to survive disconnects.
- **DreamSim (observed):** rendered (200k, ensemble) ≈ **35–40 min** per area; ImageNet subset
  (~205k V4 / ~209k V1) ≈ **45–47 min** per area. **Similarity** (`similarity.py`) is CPU-only,
  ≈ **2–4 min** per model × dataset; **mask** (`synthesis/mask.py`) ≈ **15–20 s** per area.

**Concurrency — and a memory caveat we hit.** Two ImageNet screenings at once **OOM-killed** the
cgroup: each streams ~140 GB of JPEGs into page cache, pushing past 100 GiB. So ImageNet runs are
paced. The DreamSim rendered passes, by contrast, we ran **two areas concurrently** safely — but
only after checking `memory.stat`: `memory.current` sits near 100 GiB because of **cold
`inactive_file` cache** (~85 GiB), while real (anon) usage is only ~7 GiB. Judge headroom by the
anon / inactive-file split in `memory.stat`, not the headline `memory.current`.

### Status

- **Done:** twins + inclusion; `sparse_split` (V4 160 non-sparse / 45 sparse; V1 312 / 133);
  screening (ensemble V4+V1 rendered+ImageNet, V4 member-0 ImageNet); synthesis (V4+V1 MEIs/LEIs);
  RF masks reproduced from the MEIs/LEIs (`synthesis/mask.py`, corr 0.996 V4 / 0.992 V1); DreamSim
  embeddings (V4+V1, rendered+ImageNet); similarity (`dream/similarity.py`) → Fig 6 d-prime + Fig 10
  R² with centered embeddings, ddof-1 d-prime, 15-image poles and degree-1 controls; figures
  (`figures/make_fig_dreamsim.py`). The 2D similarity-space model is CV-validated as linear
  (degree-2 adds a median ~1% R²), and V4 non-sparse R² ≈ 0.28 (rendered) matches Franke's 0.23.
- **To follow:** Fig 2c (predicted vs recorded skewness), Fig 7 (test-set verification), Fig 9
  (baseline firing rate), Fig 11 (population shared selectivity), the Fig 4/5 per-neuron panels,
  and Fig 8 (independent-evaluator / member cross-check).

## Data Availability

The full dataset (25 GB) supporting our findings is available at:  
[https://doi.org/10.5061/dryad.q573n5tx3](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5tx3)

This includes:
- 200,000 synthetically rendered scenes (236×236 PNG)
- MEIs and LEIs for all neurons (V1 and V4)
- Sorted ImageNet indices (MAIs/LAIs)
- Predicted activation profiles
- Baseline firing rates and reliability metrics

ImageNet itself is **not** included (license restrictions). See
[Getting ImageNet](#getting-imagenet) to download it via Hugging Face.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{franke2025dual,
  title={Dual-feature selectivity enables bidirectional coding in visual cortical neurons},
  author={Franke, Katrin and Karantzas, Nikolaos and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.16.665209}
}
```

## License

MIT

## Authors

Nikos Karantzas