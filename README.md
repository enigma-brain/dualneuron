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
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

## Configuration

Create a `.env` file in the repository root (copy `.env.example`):

```bash
cp .env.example .env
```

```bash
HF_TOKEN=your_huggingface_token   # Hugging Face token (needed to download ImageNet)
DATA_DIR=/path/to/your/data/      # Root data directory (see layout below)
MODELS_DIR=${DATA_DIR}/models     # Where model weights (e.g. DreamSim) are cached
```

`DATA_DIR` is the root under which the ImageNet cache, the cached model weights,
and the Dryad data live:

```
DATA_DIR/
├── datasets/          # ImageNet, downloaded automatically by Hugging Face on first run
├── models/            # cached model weights (e.g. DreamSim)
└── dryad/             # data from the Dryad release
    ├── rendered/      # rendered-scene archives batch_001.zip ... batch_020.zip
    └── *.npz          # ordered responses/indices, MEIs/LEIs
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
from that page in a browser and place them under `DATA_DIR/dryad`:

- the 20 rendered-scene archives `batch_001.zip ... batch_020.zip` go in `DATA_DIR/dryad/rendered/`
- the `.npz` files (e.g. `v4_rendered_ordered_responses.npz`, `v4_meis.npz`) go in `DATA_DIR/dryad/`

`RenderedImages` reads the rendered scenes directly from the `batch_*.zip` archives
in `DATA_DIR/dryad/rendered/`, so there is no need to unzip them.

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
from dualneuron.synthesis.generate import generate_poles

# Generate both poles for V1 and V4 neurons
generate_poles(
    output_dir="results/",
    num_seeds=5,          # Multiple random initializations
    v1_neurons=458,       # Number of V1 neurons (or list of IDs)
    v4_neurons=394        # Number of V4 neurons (or list of IDs)
)
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
├── twins/                  # Digital twin neural predictive models
│   ├── nets.py            # Model loaders (V1GrayTaskDriven, V4ColorTaskDriven, etc.)
│   ├── activations.py     # Activation extraction utilities
│   ├── V1GrayTaskDriven/  # V1 model weights & metadata
│   ├── V4ColorTaskDriven/ # V4 color model weights & metadata
│   └── V4GrayTaskDriven/  # V4 grayscale model weights & metadata
│
├── screening/              # Large-scale image screening
│   ├── run.py             # Main screening function
│   ├── sets.py            # ImageNet & rendered dataset loaders
│   ├── utils.py           # Statistics (Gini coefficient, adaptive sampling)
│   └── visualize.py       # Population & single-neuron visualizations
│
├── synthesis/              # Stimulus optimization
│   ├── ascend.py          # Fourier & pixel gradient ascent methods
│   ├── generate.py        # Batch MEI/LEI generation
│   ├── ops.py             # Image operations (crops, noise, normalization)
│   ├── visualize.py       # Optimization trajectory visualization
│   └── priors/            # Natural image magnitude spectra
│       ├── natural_gray.npy
│       └── natural_rgb.npy
│
└── dream/                  # Semantic embedding analysis
    ├── axis.py            # Semantic axis computation
    ├── sim.py             # DreamSim embedding extraction
    └── similarity.py      # MAI/LAI coherence (Fig 6) and 2D similarity space (Fig 10)
```

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