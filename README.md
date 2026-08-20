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

- Every command takes **`--area {v4,v1} --backbone {staged,resnet,dino,convnext,data_driven}`** (both required).
- Every output lives under **`{area}/{backbone}/`** — for cached features, trained weights, analysis results, figures, and logs alike.
- A central registry, [`dualneuron/twins/registry.py`](dualneuron/twins/registry.py), is the **single source of truth**: given `(area, backbone)` it resolves the model, its geometry/normalization, its screening/synthesis constants, and where every artifact is read/written.

### The twins

| `--area` | `--backbone` | Model | Neurons | Well-pred. | Input | Core | Weights |
|---|---|---|---|---|---|---|---|
| `v4` | `staged` | color V4, task-driven | 394 | 205 | 100×100 RGB | ResNet50 L2-robust (frozen) | ✅ shipped |
| `v4` | `data_driven` | color V4, data-driven | 394 | 205 | 100×100 RGB | ResNet50 (`resnet50_l2_eps0_1`) | ✅ shipped |
| `v4` | `resnet` | color V4, retrained | 394 | 206 | 100×100 RGB | ResNet50 L2-robust (frozen) | trained |
| `v4` | `dino` | color V4 | 394 | 205 | 224×224 RGB | DINOv3 ViT-B/16 block 4 (frozen) | trained |
| `v1` | `staged` | grayscale V1, task-driven | 458 | 445 | 93×93 gray | ConvNeXtV2-atto (fine-tuned) | ✅ shipped |
| `v1` | `convnext` | grayscale V1, retrained | 458 | 423 | 93×93 gray | ConvNeXtV2-atto (fine-tuned) | trained |
| `v1` | `dino` | grayscale V1 | 458 | 438 | 224×224 gray | DINOv3 ViT-B/16 block 1 (fine-tuned) | trained |

**`staged` is the backbone name of a shipped twin**, not an adjective: `v4/staged` = `V4ColorTaskDriven`, `v1/staged` = `V1GrayTaskDriven`, `v4/data_driven` = `V4ColorDataDriven`. These carry weights and `correlations.npy` under `dualneuron/twins/` (plus `mask.npy` for all but `v4/data_driven`), and are the twins whose numbers the paper reports.

`v4/resnet` and `v1/convnext` are **our retrained ensembles of the same architectures**, written to `TRAINED_MODELS_DIR`; they are not the shipped twins, and their well-predicted sets differ slightly. The DINO twins are likewise produced by [training](#training-digital-twins) (the gated DINOv3 weights are not redistributed). Every twin uses **ensemble averaging** (5 members) and a Gaussian readout. "Well-pred." counts neurons with training correlation > 0.4, per that twin's own `correlations.npy`.

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

> **No git dependencies.** The twin architecture used to come from `nnvision`, installed from a branch, which dragged in `nnfabrik`, `mei`, `datajoint`, `neuralpredictors`, `ptrnets` and `CORnet` — and forced a `datajoint<2.2` pin, because `nnfabrik` 0.2.2 imports a `Schema` symbol DataJoint 2.2 removed. That architecture now lives in [`dualneuron/twins/layers.py`](dualneuron/twins/layers.py), so every dependency resolves from PyPI. `python -m dualneuron.twins.verify` checks that the twins it builds are still the ones the shipped weights expect.

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
│   ├── v4/{staged,data_driven,resnet,dino}/   # rf / norms / screening / dreamsim / synthesis/
│   └── v1/{staged,convnext,dino}/
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
m = load_model("v4", ensemble=True, centered=False)                                    # v4/staged
m = load_model("v4_data_driven", ensemble=True, centered=False)                        # v4/data_driven
m = load_model("v1", ensemble=True, centered=False)                                    # v1/staged
# trained twins (weights_dir=None -> TRAINED_MODELS_DIR/{area}/{backbone}):
m = load_model("v4_dino", ensemble=True)                                               # v4/dino
m = load_model("v1_dino", ensemble=True)                                               # v1/dino
# or point at any trained ensemble:
m = load_model("v4", ensemble=True, weights_dir=".../trained_models/v4/resnet")
```

`centered=True` sets the readout to image center (for MEI synthesis); `centered=False` keeps the learned
receptive-field positions (for predicting recorded responses).

The per-twin loaders can also be imported directly. They take the same arguments as `load_model`, but
return the model where the builder left it (core on `cuda:0`, readout on CPU), so call `.eval().to(device)`
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

**full-field screening → RF → norms → synthesis → mask → masked screening → DreamSim → similarity → figures**

The full-field screening leads because it needs no mask, so it can run on a twin that has nothing yet;
it supplies the population axes that `--mode axis` synthesis conditions on. RF and norms are
model-intrinsic and only need the twin plus its recorded stimuli, so they can run alongside it.

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

   **Two modes.** `--mode free` (default) is the original per-neuron activation ascent — the published
   MEIs/LEIs. `--mode axis` instead folds the drive into the neuron's **natural population axis**
   `a = mean(z_MAI) − mean(z_LAI)`, taken over the well-predicted subspace of a **full-field** screening,
   and ascends the cosine to `±a`. Because the axis carries the neuron's own component, aligning to it
   drives the neuron; the objective is bounded, so no ℓ2 constraint is needed (`--target_norm none`).
   The axis is redrawn **per seed** from a random subsample (`--axis_sample`) of the neuron's extreme
   pools (`--axis_pool`), so a neuron's seeds sample the invariances of its poles rather than
   re-deriving one fixed direction. `--axis_dataset {imagenet,rendered}` chooses the corpus the poles
   come from. Each npz additionally records, per seed and pole, the drawn image ids
   (`axis_{mai,lai}_ids` — with a per-seed rng the axis is not recoverable from the twin alone), the
   cosine actually achieved (`{mei,lei}_cos`) and the response's percentile in that neuron's screened
   distribution (`{mei,lei}_percentile`) — so a run carries the evidence of whether it achieved what it
   optimized for.

   **Subsetting.** A mask needs only enough neurons to average a receptive field over, not the whole
   population. `--n_neurons N` takes a reproducible random subset of the well-predicted set, **nested**
   in `N`: the 15 chosen now are contained in the 50 chosen later, so raising `N` continues from the
   neurons already written instead of redrawing (`registry.sampled_neurons`). `--total_steps` and
   `--target_norm` override the method defaults for a run without touching them, so `free` mode keeps
   exactly the settings that produced the published MEIs.
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
     sorted per neuron → MAIs/LAIs. This is the paper's MAI/LAI regime, and it screens the **entire**
     corpus — ImageNet 1,281,167 and rendered 200,000. Do **not** pass `--n_sample` here.
   - **`--field full`**: **full-field natural** — no mask, no ℓ2 (only the twin's z-score + crop/resize).
     It needs **no mask**, so it can run first (e.g. on a freshly trained twin), and gives the natural
     population responses the axes are built from. Pair with `--n_sample 200000` for a fixed uniform
     ImageNet subset — `--n_sample` is a full-field device, and it subsets **only** when the corpus is
     larger than the request (so it is a no-op on the 200k rendered set).
   Use the ensemble (default) or a single member (`--member i`).
   The axis machinery reads the full-field screening through a cached (image × support-neuron) matrix;
   build it once per twin with `python -m dualneuron.dream.axis --area … --backbone … --dataset …`,
   which writes `context.npy` + `context_meta.npz` beside that screening. Reconstructing it costs a full
   pass over both npz (≈21 s); the cache is reopened as a memmap and an axis draw then faults in only
   the rows it sampled (≈0.2 s), which is what makes a per-seed axis affordable.
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

Every command takes `--area --backbone`. Example for `v4/staged` (the shipped task-driven twin); swap in any twin:

```bash
# RF (gradient-estimated, seconds) then the twin's own L2 constant, + the figure to choose the percentile
python -m dualneuron.synthesis.rf          --area v4 --backbone staged
python -m dualneuron.screening.norms       --area v4 --backbone staged
python -m dualneuron.figures.make_fig_norms --area v4 --backbone staged

# Screening — full-field natural first (no mask, needs none), 200k ImageNet subset (population axis)
python -m dualneuron.screening.run --area v4 --backbone staged --dataset imagenet --field full --n_sample 200000
python -m dualneuron.screening.run --area v4 --backbone staged --dataset rendered --field full --n_sample 200000
# Cache the (image x support-neuron) matrix the axes are drawn from
python -m dualneuron.dream.axis --area v4 --backbone staged --dataset imagenet

# Synthesis (one twin per GPU; resumable) + mask.
# 15 nested neurons x 10 seeds is enough for the mask; raise --n_neurons later to continue.
CUDA_VISIBLE_DEVICES=0 python -m dualneuron.synthesis.generate --area v4 --backbone staged \
    --mode axis --n_neurons 15 --neuron_seed 0 --num_seeds 10 \
    --axis_pool 50 --axis_sample 5 --axis_dataset imagenet --total_steps 256 --target_norm none
python -m dualneuron.synthesis.mask --area v4 --backbone staged --variant axis

# Screening — masked (MAI/LAI regime), ensemble, WHOLE corpus; add --member i for one member
python -m dualneuron.screening.run --area v4 --backbone staged --dataset rendered --num_workers 4
python -m dualneuron.screening.run --area v4 --backbone staged --dataset imagenet --num_workers 4

# DreamSim — build the ImageNet subset, then embed both datasets
python -m dualneuron.dream.subset --area v4 --backbone staged
python -m dualneuron.dream.sim --area v4 --backbone staged --dataset rendered --num_workers 4
python -m dualneuron.dream.sim --area v4 --backbone staged --dataset imagenet --num_workers 4 \
    --indices_path "$ANALYSIS_DIR/v4/staged/imagenet/dreamsim/indices.npy"

# Similarity (Fig 6 + Fig 9 inputs; saves {dataset}/dreamsim/similarity.npz)
python -m dualneuron.dream.similarity --area v4 --backbone staged --dataset rendered
python -m dualneuron.dream.similarity --area v4 --backbone staged --dataset imagenet

# Figures (into PAPER_FIG_DIR/{area}/{backbone}); recorded-track ones also read EXPERIMENT_DIR
python -m dualneuron.figures.make_fig_accuracy      --area v4 --backbone staged   # Fig 1c
python -m dualneuron.figures.make_fig_sparsity      --area v4 --backbone staged   # Fig 2
python -m dualneuron.figures.neuron_strips          --area v4 --backbone staged   # Figs 4-5 (both datasets)
python -m dualneuron.figures.make_fig_mei_lei       --area v4 --backbone staged   # Figs 4-5
python -m dualneuron.figures.make_fig_dreamsim      --area v4 --backbone staged   # Fig 6 + Fig 9
python -m dualneuron.figures.make_fig_verify_data   --area v4 --backbone staged   # Fig 7
python -m dualneuron.figures.make_fig_population     --area v4 --backbone staged   # Fig 10
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
├── {dataset}/screening/{field}/   responses.npz, indices.npz  | screening.log, context.log
│   │                              context.npy, context_meta.npz  (axis cache; full-field only)
│   └── member{i}/                 …                             (single-member run)
├── {dataset}/dreamsim/            embeddings.npz, indices.npy, similarity.npz | dreamsim.log, similarity.log
└── synthesis/{variant}/           mask.npy                    | mask.log, generate.log
    └── output/                    neuron{id:04d}.npz            (MEI/LEI: image, alpha, activation ×seeds)
```

An `axis`-variant npz carries the same `{mei,lei}_{image,alpha,activation}` as `free`, plus the
provenance and self-scoring of the axis run: `mode`, `axis_dataset`, `axis_field`, the per-seed drawn
image ids `axis_{mai,lai}_ids`, the achieved cosines `{mei,lei}_cos`, and the response percentiles
`{mei,lei}_percentile`. A `free` npz is byte-identical to what it was before axis mode existed.

`{field}` is `masked` or `full`; `{variant}` is `free` or `axis`. Paths are resolved only through
`twins/registry.py` (`screening_path`, `norms_path`, `rf_mask_path`, `synthesis_*`, `log_path`,
`fig_path`) — no stage picks its own output location. `make_fig_simulated` has no twin and stays flat in
`PAPER_FIG_DIR/`.

### Equipment, concurrency, observed times

Hardware: 5 × 24 GB GPUs; a 100 GiB-RAM cgroup with no CPU quota (`cpu.max = max`, 32 cores visible).
Run **one twin per GPU**. `--num_workers 4` is the default; the loaders are bound by reads from the
image store rather than by CPU, so on a networked store more workers mainly buy more outstanding I/O.

- **Screening:** rendered (200k, whole corpus) ≈ 20 min; ImageNet masked (1.28M, whole corpus) ≈ 3–9 h
  at 100px depending on how much of the corpus is in page cache, ≈ 5 h at 224px; full-field ImageNet
  (200k subset) ≈ 20–25 min.
- **Synthesis:** `free` over the whole well-predicted set ≈ 213 s/neuron (V4) / 141 s/neuron (V1) at 10
  seeds → ≈ 12–17 h per twin. `axis` at `--n_neurons 15 --total_steps 256` ≈ 350 s/neuron at 100px and
  ≈ 950 s/neuron at 224px → ≈ 1.5–4 h per twin, which is all the mask needs.
- **DreamSim:** ≈ 35–47 min per twin × dataset. **Similarity** (CPU) ≈ 2–4 min; **mask** ≈ 1–100 s;
  **RF** and **norms** ≈ seconds to minutes.

Detach long runs (`setsid`) to survive disconnects.

**Concurrency:** twins screening the *same* corpus should run **together, not in sequence**. They read
the same files in the same order, so the first warms the page cache and the rest read from RAM — one
cold pass serves all of them, and the followers converge to the leader's batch index. Measured on the
1.28M ImageNet masked pass: ≈ 2 s/batch cold versus ≈ 0.24 s/batch warm, and four V4 twins finished in
≈ 8 h against ≈ 36 h serialized. Page cache is shared between identical corpora, so it does not multiply;
watch worker anon growth in `memory.stat` rather than the headline `memory.current`, most of which is
reclaimable cache.

### Status

- **All 7 twins:** trained (5 members each where trained), gradient **RF**, measured **L2 norm**, and
  **full-field** screening on both ImageNet (200k subset) and rendered (200k), with the axis **context
  cache** built for both corpora.
- **All 4 V4 twins:** axis synthesis (15 nested neurons × 10 seeds), **axis mask**, and **masked**
  screening of both corpora at full size (ImageNet 1,281,167 / rendered 200,000). The `v4/staged` axis
  mask reproduces the shipped mask at corr **0.9955** — from 15 neurons, versus 0.996 from the full
  205-neuron `free` set, which is the evidence that 15 neurons suffice for the mask.
- **Model track (v4/staged, v1/staged):** synthesis + masks (reproduce the shipped masks, corr 0.996
  V4 / 0.992 V1), screening (rendered + ImageNet), DreamSim, similarity → **Fig 6** d′ + **Fig 9** 2D
  space, **Figs 4-5** strips. `sparse_split`: V4 160/45, V1 312/133.
- **Recorded track (V4):** **Fig 1c** (recomputed corr-to-avg matches `correlations.npy`, r = 0.9997;
  n > 0.4 = 207 vs the shipped 205 — V4's distribution is centred at 0.407, so 15 neurons sit within
  0.01 of the threshold and four cross it), **Fig 2** (model-vs-recorded skewness r = 0.68), **Fig 7**,
  **Fig 10**, **Suppl. Fig 4**. V1 reproduces its `correlations.npy` at r = 0.9997 with 445/445.
- **L2 constants:** measured per twin rather than inherited (`synthesis/rf.py` → `screening/norms.py`,
  read via `registry.resolve_synth_norm` / `resolve_screen_norm`, literals as fallback). Calibrated at
  `NORM_PERCENTILE` = 2.56, where the shipped V4 value (40) and V1 value (12) fall on their own
  RF-masked training distributions (p2.56 / p1.65). All seven measured: `v4/staged` 38.92,
  `v4/resnet` 38.97, `v4/data_driven` 38.58, `v4/dino` 88.13, `v1/staged` 13.76, `v1/convnext` 13.75,
  `v1/dino` 34.88. The 224px twins scale as √N over their larger masked support.
  `values_range` now matches each twin's physical `(0-mean)/std, (255-mean)/std` exactly.
- **DINO twins:** `v4/dino` and `v1/dino` both trained (5 members each).
- **To follow:** V1 recorded panels need V1's canonical `SESSION_ORDER` (in `data/recordings.py`); Fig 2e,f
  baseline firing; Fig 8 independent evaluator; Fig 11 mouse; a deliberate cross-twin comparison figure.

## Package structure

```
dualneuron/
├── utils.py                # env_dir, ensure_dir, RewriteLine (twin-aware helpers live in twins/registry)
├── twins/
│   ├── registry.py         # (area, backbone) registry: geometry, arch, paths, correlations/mask,
│   │                       #   resolved L2 norms, neuron selection (well_predicted / sampled_neurons)
│   ├── nets.py             # loaders: V4ColorTaskDriven, V4ColorDataDriven, V1GrayTaskDriven, V4ColorDino, V1GrayDino, load_model
│   ├── dino.py             # DINOv3 model classes (frozen or truncated fine-tuned core + Gaussian readout)
│   ├── activations.py      # activation-extraction utilities
│   ├── V4ColorTaskDriven/, V1GrayTaskDriven/, V4GrayTaskDriven/   # shipped weights + mask.npy + correlations.npy
│   └── V4ColorDataDriven/  # shipped weights + correlations.npy (data-driven color V4; no mask.npy)
├── screening/              # run.py (--field masked|full, --n_sample), norms.py (the twin's L2 constant
│                           #   from its RF-masked training stimuli); sets.py, utils.py, visualize.py
├── synthesis/              # ascend.py, generate.py (--mode free|axis), mask.py (RF from MEI alphas),
│                           #   rf.py (RF from input gradients, no synthesis needed), ops.py,
│                           #   visualize.py, priors/
├── dream/                  # sim.py, subset.py, similarity.py (DreamSim analyses); axis.py — the
│                           #   population-axis machinery + its cached (image x neuron) context

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
