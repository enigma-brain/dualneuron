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

We discovered that many neurons in visual cortex exhibit **dual-feature selectivity** — they respond strongly to preferred features while being systematically suppressed by distinct non-preferred features around elevated baseline firing rates. This **bidirectional coding** strategy appears conserved across species (macaque and mouse) and visual areas (V1 to V4).

This package lets you, for **any twin of choice**:

- **Load / train digital twins** — neural predictive models for macaque V1 and V4, per `(area, backbone)`.
- **Screen large image datasets** — find most-/least-activating images (MAIs/LAIs), in an RF-masked or a full-field regime.
- **Synthesize optimal stimuli** — most-/least-exciting inputs (MEIs/LEIs) by gradient ascent (frequency domain for V4, pixels for V1; ideas from [Horama](https://github.com/serre-lab/Horama), extended with different transforms/constraints).
- **Compute semantic axes** — DreamSim embeddings relating high/low activation poles.
- **Visualize and analyze** — activation curves, population statistics, optimization trajectories, and the paper figures.

## Key Concepts

### Bidirectional neural coding
Traditional views characterize neurons by what excites them. Our work reveals a **low pole** — stimuli that systematically suppress activity below baseline. The response range spans:
- **High pole (MEI/MAI):** stimuli maximizing the response.
- **Low pole (LEI/LAI):** stimuli minimizing it (maximal suppression).

### One organizing principle: `(area, backbone)`
Everything in the pipeline — training, screening, synthesis, DreamSim, figures — is keyed by an `(area, backbone)` **twin**. This single convention runs end to end:

- Every command takes **`--area {v4,v1} --backbone {resnet,dino,convnext,data_driven}`** (both required).
- Every output lives under **`{area}/{backbone}/`** — for cached features, trained weights, analysis results, figures, and logs alike.
- A central registry, [`dualneuron/twins/registry.py`](dualneuron/twins/registry.py), is the **single source of truth**: given `(area, backbone)` it resolves the model, its geometry/normalization, its screening/synthesis constants, and where every artifact is read/written.

### The twins

| `--area` | `--backbone` | Model | Neurons | Input | Core | Staged? |
|---|---|---|---|---|---|---|
| `v4` | `resnet` | color V4 | 394 | 100×100 RGB | ResNet50 L2-robust (frozen) | ✅ shipped |
| `v4` | `data_driven` | color V4, data-driven | 394 | 100×100 RGB | ResNet50 (`resnet50_l2_eps0_1`) | ✅ shipped |
| `v4` | `dino` | color V4 | 394 | 224×224 RGB | DINOv3 ViT-B/16 block 4 (frozen) | trained |
| `v1` | `convnext` | grayscale V1 | 458 | 93×93 gray | ConvNeXtV2-atto (fine-tuned) | ✅ shipped |
| `v1` | `dino` | grayscale V1 | 458 | 224×224 gray | DINOv3 ViT-B/16 block 1 (fine-tuned) | trained |

The **shipped** twins (`v4/resnet` = `V4ColorTaskDriven`, `v4/data_driven` = `V4ColorDataDriven`, `v1/convnext` = `V1GrayTaskDriven`) come with weights and `correlations.npy` under `dualneuron/twins/`, plus `mask.npy` for all but `v4/data_driven`; the DINO twins are produced by [training](#training-digital-twins) (the gated DINOv3 weights are not redistributed). Each twin uses **ensemble averaging** (5 members) and a Gaussian readout.

> **Staged twins are read-only.** The shipped weights / `correlations.npy` / `mask.npy` under `dualneuron/twins/` are never modified. Anything you train or regenerate is written to `TRAINED_MODELS_DIR/{area}/{backbone}/` (weights + correlations) or `ANALYSIS_DIR/{area}/{backbone}/` (a regenerated mask), never back into `twins/`.

`v4/data_driven` (`dualneuron/twins/V4ColorDataDriven/`) is the **data-driven** counterpart of the
task-driven color-V4 twin: same architecture, same input geometry (3×100×100, mean 113.5 / std 59.58)
and the same 394 V4 neurons in the same order, differing only in its trained weights. It ships with
weights + `correlations.npy` (mean 0.411, median 0.419, 205 of 394 neurons above the 0.4 threshold,
recomputed on the recorded test set with the same procedure as `eval_ensemble`), but **no** `mask.npy`
— run `synthesis.generate` then `synthesis.mask` to produce one, as for any non-staged twin.

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync                              # env + GPU torch (CUDA 12.1 index)
source .venv/bin/activate            # or prefix commands with `uv run`
```

> **Use `uv sync`.** `nnfabrik` (0.2.2) imports `from datajoint.schemas import Schema`, removed in DataJoint 2.2, so the twins fail to import on DataJoint ≥ 2.2. The lockfile pins `datajoint<2.2` (2.1.1). If you install manually, keep that constraint: `pip install "datajoint<2.2"`.

## Configuration

Copy `.env.example` to `.env` and set the paths:

```bash
HF_TOKEN=your_huggingface_token                 # Hugging Face token (to download ImageNet)
DATA_DIR=/path/to/your/data                     # Root data directory (see layout below)
IMAGENET_CACHE_DIR=${DATA_DIR}/datasets         # Hugging Face ImageNet cache
RENDERED_DIR=${DATA_DIR}/datasets/rendered      # Rendered-scene archives (batch_*.zip)
EXPERIMENT_DIR=${DATA_DIR}/datasets/experiment  # Recordings + stimuli, per area (see below)
MODELS_DIR=${DATA_DIR}/models                   # Cached model weights (DreamSim; gated DINOv3 in dinov3/)
ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS  # Regenerated screening/synthesis/DreamSim outputs
FEATURES_DIR=${DATA_DIR}/features               # Cached frozen-core features for training
TRAINED_MODELS_DIR=${DATA_DIR}/trained_models   # User-trained twin ensembles
LOGS_DIR=./logs                                 # Progress logs
PAPER_FIG_DIR=./figs                            # Saved figures
```

Directories are created on demand by the scripts that write into them. Every twin-specific location is
`{area}/{backbone}/`:

```
DATA_DIR/
├── datasets/
│   ├── rendered/                       # rendered-scene archives batch_*.zip (RENDERED_DIR)
│   └── experiment/                     # recordings + stimuli (EXPERIMENT_DIR)
│       ├── v4/{trials,images}          #   V4 session pickles + presented stimuli {id:06d}.npy
│       └── v1/{trials,images}          #   V1 session pickles + presented stimuli
├── models/dinov3/                      # gated DINOv3 backbone (hubconf repo + converted weights)
├── DUAL-FEATURE-ANALYSIS/              # regenerated outputs (ANALYSIS_DIR)
│   ├── v4/{resnet,dino}/               #   screening / dreamsim / similarity / mask / synthesis/
│   └── v1/{convnext,dino}/
├── features/                           # cached training inputs (FEATURES_DIR)
│   ├── v4/{resnet,dino}/               #   {train,test}_features_*.npy  (frozen cores)
│   └── v1/{convnext,dino}/             #   {train,test}_images_*.npy    (fine-tuned cores)
└── trained_models/                     # user-trained ensembles (TRAINED_MODELS_DIR)
    ├── v4/{resnet,dino}/               #   {area}_{backbone}_{1..5}.pth.tar + correlations.npy + mask.npy
    └── v1/{convnext,dino}/
```

`logs/` and `figs/` mirror this: `LOGS_DIR/{area}/{backbone}/<stage>.log` and
`PAPER_FIG_DIR/{area}/{backbone}/<figure>.pdf`.

### Getting ImageNet
ImageNet is **not** redistributed on Dryad. One-time setup:
1. Request access to the gated [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset.
2. Put your token in `.env` as `HF_TOKEN`.
3. The first screening run downloads ImageNet into `DATA_DIR/datasets` using the token; afterwards it loads from that cache.

### Getting the Dryad data
The rendered scenes, sorted indices, and MEIs/LEIs are on Dryad
([doi:10.5061/dryad.q573n5tx3](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5tx3)); download in a browser (Dryad blocks scripted downloads):
- rendered archives `batch_*.zip` → **`RENDERED_DIR`** (read directly, no unzip).
- the published `.npz` are **optional** — this pipeline regenerates its own into `ANALYSIS_DIR`.

### Getting the gated DINOv3 backbone (DINO twins only)
Needed only to train/run the DINO twins. One-time setup under `MODELS_DIR/dinov3`:
1. Accept the license and clone the hubconf repo:
   ```bash
   git clone https://github.com/facebookresearch/dinov3 $MODELS_DIR/dinov3/facebookresearch_dinov3_main
   ```
2. Accept the weights license ([`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)), then convert:
   ```bash
   python -m dualneuron.training.convert_dinov3_weights
   ```

## Loading a twin

```python
from dualneuron.twins.nets import load_model

# staged twins (weights_dir=None -> GitHub-staged):
m = load_model("v4", ensemble=True, centered=False)                                    # v4/resnet
m = load_model("v4_data_driven", ensemble=True, centered=False)                        # v4/data_driven
m = load_model("v1", ensemble=True, centered=False)                                    # v1/convnext
# trained twins (weights_dir=None -> TRAINED_MODELS_DIR/{area}/{backbone}):
m = load_model("v4_dino", ensemble=True)                                               # v4/dino
m = load_model("v1_dino", ensemble=True)                                               # v1/dino
# or point at any trained ensemble:
m = load_model("v4", ensemble=True, weights_dir=".../trained_models/v4/resnet")
```

`centered=True` sets the readout to image center (for MEI synthesis); `centered=False` keeps the learned
receptive-field positions (for predicting recorded responses).

The per-twin loaders can also be imported directly. They take the same arguments as `load_model`, but
return the model where nnvision built it (core on `cuda:0`, readout on CPU), so call `.eval().to(device)`
yourself — `load_model` does that for you:

```python
from dualneuron.twins.nets import V4ColorDataDriven

m = V4ColorDataDriven(ensemble=True, centered=False).eval().to("cuda")   # (batch, 3, 100, 100) -> (batch, 394)
```

## Training digital twins

Twins are trained per `(area, backbone)`. Two regimes, chosen automatically from the twin:

- **Frozen core** (`v4/resnet`, `v4/dino`): the core is frozen, so its feature maps are extracted **once**
  to `FEATURES_DIR/{area}/{backbone}/` and only the readout (+ a BatchNorm) is trained on the cache.
- **Fine-tuned core** (`v1/convnext`, `v1/dino`): the backbone is fine-tuned end-to-end (DINO tunes only
  its stem + blocks up to the read-out block, via a truncated forward), so the fixed **input images** are
  cached instead and the whole thing trains on them.

```bash
python -m dualneuron.training.run --area v4 --backbone resnet
python -m dualneuron.training.run --area v4 --backbone dino     # needs the gated DINOv3 backbone
python -m dualneuron.training.run --area v1 --backbone convnext
python -m dualneuron.training.run --area v1 --backbone dino
# multi-GPU: one member per GPU (parent caches once, pool trains, then aggregates)
python -m dualneuron.training.run --area v4 --backbone dino --gpus 0,1,2,3,4
# evaluate a trained ensemble (single-trial + correlation-to-average vs the twin's correlations.npy)
python -m dualneuron.training.eval_ensemble --area v4 --backbone dino
```

Each run writes `{area}_{backbone}_{1..5}.pth.tar` + `correlations.npy` + `mask.npy` to
`TRAINED_MODELS_DIR/{area}/{backbone}/`. Heads mirror the area's task-driven twin (V4 → BatchNorm→ReLU;
V1 → BatchNorm→GELU) with output `ELU(x−1)+1`; loss is a NaN-masked Poisson NLL + `gamma·mean|readout|`
(`gamma_readout` per twin: V4 = 3.0, V1 = 10). Logs go to `LOGS_DIR/{area}/{backbone}/`.

## Reproducing the paper — the pipeline

Two tracks meet in the figures. The **model track** is one dependency chain per twin:

**RF → norms → synthesis → mask → screening → DreamSim → similarity → figures**

The **recorded track** compares a twin's predictions to the macaque recordings (`data/recordings.py`,
reading `EXPERIMENT_DIR/{area}`): Fig 1c accuracy and Fig 7 verification need only the twin + recordings;
Fig 2 (skewness) and Fig 10 (population) also consume the screening output.

0. **RF + norms** (`synthesis/rf.py`, `screening/norms.py`). The ℓ2 constant that bounds synthesis and
   rescales masked screening is **measured per twin**, not inherited. `rf.py` estimates the population
   receptive field from the twin's **input gradients** — one backward pass per neuron on the centered
   ensemble, averaged over natural stimuli, thresholded by the same recipe as `synthesis/mask.py`. It
   needs no MEIs, which is what breaks the circularity: the norm must be measured over the region a
   synthesized stimulus occupies, but that region would otherwise only be known after synthesis. On
   `v4/staged` it reproduces the shipped mask at corr **0.9945** in seconds, versus the 12–17 h the
   alpha-derived mask costs. `norms.py` then measures ‖x·m‖₂ over the twin's recorded **training**
   stimuli and takes `registry.NORM_PERCENTILE` (default 2.56, a choice — inspect
   `figures/make_fig_norms.py`) as the constant. Consumers read it through
   `registry.resolve_synth_norm` / `resolve_screen_norm`, which fall back to the `TwinSpec` literals,
   so a twin behaves exactly as before until its norm has been measured.
1. **Synthesis** (`synthesis/generate.py`). Gradient ascent on the **centered** ensemble produces, per
   well-predicted neuron, an MEI and an LEI (V4 in the Fourier phase domain with a natural-amplitude
   prior; V1 in pixels), 10 seeds each, ℓ2-constrained. One npz per neuron. The ℓ2 rescale is a scalar
   multiply, so on its own it can carry the stimulus outside the twin's `values_range`; passing that
   range to `ops.change_norm` satisfies the norm **and** the bounds together. Rows the range does not
   bind take a closed-form path identical in value and gradient to the plain rescale, so the constraint
   alters the optimization only where it actually binds.
2. **Mask** (`synthesis/mask.py`). Each twin's RF mask is the **mean alpha over its MEIs/LEIs**,
   thresholded (~77.5th pct) and Gaussian-softened. Written to the **non-staged**
   `ANALYSIS_DIR/{area}/{backbone}/mask.npy` (for a shipped twin it reproduces the staged mask, reported
   as a QC comparison — the staged mask is never overwritten). Readers (screening, DreamSim,
   neuron-strips) resolve the mask through `registry.mask_path`, which prefers the regenerated `axis`
   mask, then the regenerated `free` one, then the twin's shipped `twins/<folder>/mask.npy`. So a
   shipped twin can be screened before its own MEIs exist, and its regenerated mask silently takes over
   the moment it is built. A trained twin has no shipped mask, so it must run synthesis + mask first.
3. **Screening** (`screening/run.py`). Two regimes:
   - **`--field masked`** (default): each image is RF-masked (bg 0) and ℓ2-normed, run through the twin,
     sorted per neuron → MAIs/LAIs. This is the paper's MAI/LAI regime.
   - **`--field full`**: **full-field natural** — no mask, no ℓ2 (only the twin's z-score + crop/resize).
     It needs **no mask**, so it can run first (e.g. on a freshly trained twin), and gives the natural
     population responses used to place neurons on their MAI↔LAI axes. Pair with `--n_sample 200000` for
     a fixed uniform ImageNet subset.
   Use the ensemble (default) or a single member (`--member i`).
4. **DreamSim** (`dream/sim.py`, `dream/subset.py`). Each image is RF-masked (neutral-gray bg 0.45),
   contrast-normalized, and embedded into the 1792-d DreamSim ensemble space. Rendered = all 200k;
   ImageNet = a per-twin subset (`subset.py`, `--indices_path`).
5. **Similarity** (`dream/similarity.py`). Globally centered cosine → **Fig 6 d′** (within-MAI/LAI
   coherence vs random) and **Fig 9 2D similarity space** (R² of activity vs the MAI/LAI cosine plane),
   over all well-predicted neurons, related to each neuron's skewness (`registry.sparse_split`).
6. **Figures** (`figures/`). Per-twin PDFs to `PAPER_FIG_DIR/{area}/{backbone}/` (see the table).

### Paper → code

| Paper figure | Produced by | Regime |
|---|---|---|
| **Fig 1c** — prediction accuracy (single-trial + corr-to-avg) | `figures/make_fig_accuracy.py` | recordings + twin (`centered=False`) |
| **Fig 2** — sparseness continuum (model vs recorded skewness) | `figures/make_fig_sparsity.py` | screening + recordings |
| **Figs 3–5** — MEIs/LEIs + MAIs/LAIs | `synthesis/generate.py` → `synthesis/mask.py` → `screening/run.py` | synthesis + masked screening (`centered=True`) |
| **Figs 3–5** — plotting | `figures/neuron_strips.py` + `figures/make_fig_mei_lei.py` | — |
| **Fig 6** — DreamSim MAI/LAI d′ (per twin) | `dream/sim.py` + `dream/similarity.py` → `figures/make_fig_dreamsim.py` | DreamSim |
| **Fig 7** — recorded-response verification | `figures/make_fig_verify_data.py` | recordings + twin (`centered=False`) |
| **Fig 9** — 2D similarity space, R² vs sparsity | `dream/similarity.py` → `figures/make_fig_dreamsim.py` | DreamSim |
| **Fig 10** — population shared selectivity | `figures/make_fig_population.py` | ImageNet screening + `subject_id` |
| **Suppl. Fig 4** — simulated simple/complex cells | `figures/make_fig_simulated.py` | rendered scenes (Gabor; no twin) |

**Two evaluation regimes — keep them straight.** Figures that predict *recorded* responses to the actual
stimuli (Fig 1c, Fig 7) use the twin with **learned readout positions** (`centered=False`) and the
training transform (crop → resize → z-score; no RF mask / ℓ2). The *screening / MAI-LAI* figures (Fig 2
model side, 6, 9, 10) use the **centered, RF-masked, ℓ2-normed** screening (`centered=True`).

### Commands (per twin)

Every command takes `--area --backbone`. Example for `v4/resnet`; swap in any twin:

```bash
# RF (gradient-estimated, seconds) then the twin's own L2 constant, + the figure to choose the percentile
python -m dualneuron.synthesis.rf          --area v4 --backbone resnet
python -m dualneuron.screening.norms       --area v4 --backbone resnet
python -m dualneuron.figures.make_fig_norms --area v4 --backbone resnet

# Synthesis (one twin per GPU; resumable) + mask
CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4 --backbone resnet
python -m dualneuron.synthesis.mask --area v4 --backbone resnet

# Screening — masked (MAI/LAI regime), ensemble; add --member i for one member
python -m dualneuron.screening.run --area v4 --backbone resnet --dataset rendered --num_workers 4
python -m dualneuron.screening.run --area v4 --backbone resnet --dataset imagenet --num_workers 4
# Screening — full-field natural (no mask, needs none), 200k ImageNet subset (population axis)
python -m dualneuron.screening.run --area v4 --backbone resnet --dataset imagenet --field full --n_sample 200000

# DreamSim — build the ImageNet subset, then embed both datasets
python -m dualneuron.dream.subset --area v4 --backbone resnet
python -m dualneuron.dream.sim --area v4 --backbone resnet --dataset rendered --num_workers 4
python -m dualneuron.dream.sim --area v4 --backbone resnet --dataset imagenet --num_workers 4 \
    --indices_path "$ANALYSIS_DIR/v4/resnet/dreamsim_imagenet_indices.npy"

# Similarity (Fig 6 + Fig 9 inputs; auto-saves similarity_{dataset}.npz)
python -m dualneuron.dream.similarity --area v4 --backbone resnet --dataset rendered

# Figures (into PAPER_FIG_DIR/{area}/{backbone}); recorded-track ones also read EXPERIMENT_DIR
python -m dualneuron.figures.make_fig_accuracy      --area v4 --backbone resnet   # Fig 1c
python -m dualneuron.figures.make_fig_sparsity      --area v4 --backbone resnet   # Fig 2
python -m dualneuron.figures.neuron_strips          --area v4 --backbone resnet   # Figs 4-5 (both datasets)
python -m dualneuron.figures.make_fig_mei_lei       --area v4 --backbone resnet   # Figs 4-5
python -m dualneuron.figures.make_fig_dreamsim      --area v4 --backbone resnet   # Fig 6 + Fig 9
python -m dualneuron.figures.make_fig_verify_data   --area v4 --backbone resnet   # Fig 7
python -m dualneuron.figures.make_fig_population     --area v4 --backbone resnet   # Fig 10
python -m dualneuron.figures.make_fig_simulated                                    # Suppl. Fig 4 (no twin)
```

Per-twin geometry (crop, channels, grayscale, contrast norm, RF mask) is resolved automatically from the
registry — e.g. crop 200 (V4) / 167 (V1), 224px inputs for the DINO twins.

### Saved-file layout

**One tree, three roots.** Every artifact lives at `{ROOT}/{area}/{backbone}/<stage…>`, where `ROOT` is
`ANALYSIS_DIR` (data), `LOGS_DIR` (logs) or `PAPER_FIG_DIR` (figures). A stage's relative path is
**identical under all three**, so filenames stay bare and the folder carries the identity. Dataset-scoped
stages nest under `{dataset}/`; model-intrinsic ones (RF, norms, synthesis) sit at the twin root.

```
{ROOT}/{area}/{backbone}/
├── rf/                            mask.npy, sensitivity.npy   | rf.log
├── norms/{split}/                 norms.npz                   | norms.log   | norms.pdf
├── {dataset}/screening/{field}/   responses.npz, indices.npz  | screening.log
│   └── member{i}/                 …                             (single-member run)
├── {dataset}/dreamsim/            embeddings.npz, indices.npy, similarity.npz
└── synthesis/{variant}/           mask.npy                    | mask.log, generate.log
    └── output/                    neuron{id:04d}.npz            (MEI/LEI: image, alpha, activation ×seeds)
```

`{field}` is `masked` or `full`; `{variant}` is `free` or `axis`. Paths are resolved only through
`twins/registry.py` (`screening_path`, `norms_path`, `rf_mask_path`, `synthesis_*`, `log_path`,
`fig_path`) — no stage picks its own output location. `make_fig_simulated` has no twin and stays flat in
`PAPER_FIG_DIR/`.

### Equipment, concurrency, observed times

Hardware: 5 × 24 GB GPUs; a 100 GiB-RAM / 4-CPU-core cgroup. Run **one twin per GPU** and use
`--num_workers 4` (default) — the data loaders are JPEG-decode-bound and 4 workers saturate the 4 cores.

- **Screening:** V4 rendered (200k, ensemble) ≈ 20 min; V4/V1 ImageNet (1.28M, ensemble, 4 workers)
  ≈ 1 h; full-field ImageNet (200k subset) ≈ 20–25 min.
- **Synthesis:** ≈ 213 s/neuron (V4) / 141 s/neuron (V1) at 10 seeds → ≈ 12–17 h per twin. Detach long
  runs (`setsid`) to survive disconnects.
- **DreamSim:** ≈ 35–47 min per twin × dataset. **Similarity** (CPU) ≈ 2–4 min; **mask** ≈ 15–20 s.

**Concurrency caveat:** two ImageNet screenings at once can OOM the cgroup (each streams ~140 GB of JPEGs
into page cache). Pace ImageNet runs; judge headroom by the anon / inactive-file split in `memory.stat`,
not the headline `memory.current`.

### Status

- **Model track (v4/resnet, v1/convnext):** synthesis + masks (reproduce the shipped masks, corr 0.996
  V4 / 0.992 V1), screening (rendered + ImageNet), DreamSim, similarity → **Fig 6** d′ + **Fig 9** 2D
  space, **Figs 4-5** strips. `sparse_split`: V4 160/45, V1 312/133.
- **Recorded track (V4):** **Fig 1c** (recomputed corr-to-avg matches `correlations.npy`, r = 0.9997,
  n > 0.4 = 207), **Fig 2** (model-vs-recorded skewness r = 0.68), **Fig 7**, **Fig 10**, **Suppl. Fig 4**.
- **L2 constants:** measured per twin rather than inherited (`synthesis/rf.py` → `screening/norms.py`,
  read via `registry.resolve_synth_norm` / `resolve_screen_norm`, literals as fallback). Calibrated at
  `NORM_PERCENTILE` = 2.56, where the shipped V4 value (40) and V1 value (12) fall on their own
  RF-masked training distributions (p2.56 / p1.65). Computed so far: `v4/staged` → 38.92.
  `values_range` now matches each twin's physical `(0-mean)/std, (255-mean)/std` exactly.
- **DINO twins:** `v4/dino` trained (5 members); `v1/dino` has 1 of 5 — the rest were lost to a GPU
  fault, see the incident note in the training logs. Their L2 constants have not been measured yet.
- **To follow:** V1 recorded panels need V1's canonical `SESSION_ORDER` (in `data/recordings.py`); Fig 2e,f
  baseline firing; Fig 8 independent evaluator; Fig 11 mouse; a deliberate cross-twin comparison figure.

## Package structure

```
dualneuron/
├── utils.py                # env_dir, ensure_dir, RewriteLine (twin-aware helpers live in twins/registry)
├── twins/
│   ├── registry.py         # (area, backbone) registry: geometry, arch, paths, correlations/mask, neuron selection
│   ├── nets.py             # loaders: V4ColorTaskDriven, V4ColorDataDriven, V1GrayTaskDriven, V4ColorDino, V1GrayDino, load_model
│   ├── dino.py             # DINOv3 model classes (frozen or truncated fine-tuned core + Gaussian readout)
│   ├── activations.py      # activation-extraction utilities
│   ├── V4ColorTaskDriven/, V1GrayTaskDriven/, V4GrayTaskDriven/   # shipped weights + mask.npy + correlations.npy
│   └── V4ColorDataDriven/  # shipped weights + correlations.npy (data-driven color V4; no mask.npy)
├── screening/              # run.py (--field masked|full, --n_sample), norms.py (the twin's L2 constant
│                           #   from its RF-masked training stimuli); sets.py, utils.py, visualize.py
├── synthesis/              # ascend.py, generate.py, mask.py (RF from MEI alphas), rf.py (RF from input
│                           #   gradients, no synthesis needed), ops.py, visualize.py, priors/
├── dream/                  # sim.py, subset.py, similarity.py, axis.py  (DreamSim analyses)
├── data/recordings.py      # load_sessions / build_response_matrix (per area; SESSION_ORDER aligns neurons)
├── training/               # config.py, dataset.py, features.py, trainer.py, run.py, eval_ensemble.py, convert_dinov3_weights.py
├── figures/                # make_fig_*.py, neuron_strips.py  (per-twin PDFs)
└── analysis/pca.py
```

## Data Availability

The full dataset (25 GB) is at [doi:10.5061/dryad.q573n5tx3](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5tx3):
200,000 rendered scenes, MEIs/LEIs (V1 & V4), sorted ImageNet indices, predicted activation profiles,
baseline firing rates and reliability metrics. ImageNet itself is not included (license); see
[Getting ImageNet](#getting-imagenet).

## Citation

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
