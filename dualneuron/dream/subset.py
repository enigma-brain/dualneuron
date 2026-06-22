"""
Build the per-area ImageNet subset for DreamSim embedding.

Embedding all ~1.28M ImageNet images through the DreamSim ensemble is impractical, so
we embed a subset that still contains every neuron's extremes: the union of each
well-predicted neuron's top-k MAIs and bottom-k LAIs (from the ensemble ImageNet
screening), plus a uniform sample of the remaining (non-extreme) images. The result is
saved as a .npy of global image indices for sim.py's --indices_path.
"""
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from dualneuron.utils import env_dir, ensure_dir, well_predicted_neurons

ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
_AREA_MODELS = {"v1": "V1GrayTaskDriven", "v4": "V4ColorTaskDriven"}


def build_imagenet_subset(area, k=15, n_sample=200000, seed=0, total=1281167,
                          ordered_indices_path=None):
    """
    Indices for the per-area ImageNet DreamSim subset.

    Union of each well-predicted neuron's top-k MAIs and bottom-k LAIs (the extremes,
    from the ensemble ImageNet screening) with n_sample images drawn uniformly from
    the remaining, non-extreme images.

    Args:
        area: "v1" or "v4".
        k: Number of MAIs and of LAIs per neuron to include as extremes. Default: 15.
        n_sample: Uniform-sample size over the non-extreme images. Default: 200000.
        seed: RNG seed for the uniform sample. Default: 0.
        total: Number of screened ImageNet images (the index space). Default: 1281167.
        ordered_indices_path: Path to the screening indices npz. Default:
            ANALYSIS_DIR/{area}/{area}_ensemble_imagenet_ordered_indices.npz.

    Returns:
        dict: {
            "subset": sorted unique indices (extremes + sample),
            "extremes": sorted extreme indices,
            "sample": the (sorted) uniform sample of non-extremes,
        }
    """
    neurons = well_predicted_neurons(_AREA_MODELS[area])
    if ordered_indices_path is None:
        ordered_indices_path = os.path.join(
            ANALYSIS_DIR, area, f"{area}_ensemble_imagenet_ordered_indices.npz"
        )
    ordered = np.load(ordered_indices_path)

    extremes = set()
    for neuron in neurons:
        order = ordered[f"unit_{int(neuron)}"]   # image indices sorted ascending by response
        extremes.update(int(i) for i in order[:k])     # bottom-k = least activating (LAIs)
        extremes.update(int(i) for i in order[-k:])    # top-k = most activating (MAIs)
    extremes = np.array(sorted(extremes), dtype=np.int64)

    rng = np.random.RandomState(seed)
    keep = np.ones(total, dtype=bool)
    keep[extremes] = False
    pool = np.where(keep)[0]
    sample = rng.choice(pool, size=n_sample, replace=False)

    return {
        "subset": np.union1d(extremes, sample),   # sorted, unique
        "extremes": extremes,
        "sample": np.sort(sample),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the per-area ImageNet DreamSim subset indices")
    parser.add_argument("--area", type=str, required=True, choices=["v1", "v4"])
    parser.add_argument("--k", type=int, default=15, help="top-k MAIs and bottom-k LAIs per neuron")
    parser.add_argument("--n_sample", type=int, default=200000, help="uniform sample of non-extreme images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=1281167, help="number of screened ImageNet images")
    parser.add_argument("--output_path", type=str, default=None,
                        help="default ANALYSIS_DIR/{area}/{area}_dreamsim_imagenet_indices.npy")
    args = parser.parse_args()

    res = build_imagenet_subset(args.area, k=args.k, n_sample=args.n_sample,
                                seed=args.seed, total=args.total)
    out = args.output_path or os.path.join(
        ANALYSIS_DIR, args.area, f"{args.area}_dreamsim_imagenet_indices.npy"
    )
    ensure_dir(os.path.dirname(out))
    np.save(out, res["subset"])
    print(f"{args.area}: {len(res['extremes'])} extremes + {len(res['sample'])} sampled "
          f"= {len(res['subset'])} total -> {out}")
