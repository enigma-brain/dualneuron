import torch
import numpy as np

from dualneuron.twins import registry


def population_axis(area, backbone, neurons=None, k=20, dataset="imagenet",
                    field="full", weights_dir=None, exclude_target=True, eps=1e-8):
    """
    Per-neuron population axis a_i = z_bar(MAI) - z_bar(LAI): the difference of the z-scored
    population-response centroids of a neuron's most- and least-activating natural images.

    The axis lives ONLY in the WELL-PREDICTED subspace of the population: the neurons whose training
    correlation-to-average exceeds 0.4 for this twin (``registry.well_predicted_neurons``, from the
    model's own ``correlations.npy``). Poorly-predicted / dead units are excluded from the space, the
    z-score, the centroids, and (at synthesis time) the cosine -- they would otherwise contribute
    unreliable, noise-dominated dimensions to the population-context direction.

    Computed from the FULL-FIELD screening (no mask / no L2 -- the natural regime): each screened
    image has a full population response, so for neuron i we take its top-k / bottom-k images (by i's
    own response), z-score the support neurons' responses over the whole screened set, difference the
    two centroids, ZERO the i-th component (so the axis carries only the surrounding population's
    context, not the target neuron itself), and unit-normalize. This is the same construction as the
    paper's neuron axis, reused to regularize MEI/LEI synthesis toward the natural population manifold.

    The full-field ordered files store, per neuron, its responses sorted ascending and the global
    image id at each rank; the population vectors are reconstructed by inverting those sorts.

    Args:
        area, backbone: the twin (its full-field screening must exist -- run
            ``screening.run --field full`` first).
        neurons: neuron ids to build axes for; default all well-predicted neurons.
        k: number of MAIs and of LAIs per centroid. Default: 20.
        dataset: screening dataset. Default: "imagenet".
        field: screening regime; "full" (full-field) for the natural population axis. Default: "full".
        weights_dir: override for the twin's ``correlations.npy`` (defining the support). Default:
            staged / ``TRAINED_MODELS_DIR`` per the registry.
        exclude_target: if True (default) zero the target's own component so the axis carries only the
            surrounding population's context (the regularizer form). If False, KEEP it -- the full
            MAI-LAI direction incl. the target (a_full), whose target component is dominant; used to
            "fold" the drive into a single bounded cosine objective.
        eps: numerical stabilizer.

    Returns:
        dict with (P = number of well-predicted / support neurons):
            "neurons": (n,) neuron ids the axes are built for,
            "support": (P,) global ids of the well-predicted neurons spanning the axis space (sorted),
            "axis":    (n, P) float32 unit axes in neuron order (row r is a_{neurons[r]} over the
                       support, the target's own component zeroed),
            "mean":    (P,) per-support-neuron full-field response mean (to z-score at synthesis time),
            "std":     (P,) per-support-neuron full-field response std.
    """
    resp = np.load(registry.screening_path(area, backbone, "ensemble", dataset, "responses", field=field))
    idx = np.load(registry.screening_path(area, backbone, "ensemble", dataset, "indices", field=field))

    # The axis space = the well-predicted neurons (training correlation > 0.4), per twin.
    support = np.sort(registry.well_predicted_neurons(area, backbone, weights_dir=weights_dir))
    pos = {int(u): c for c, u in enumerate(support)}       # global id -> column in the support space

    # Reconstruct the (image x support-neuron) response matrix by scattering each neuron's sorted
    # responses back to their images (row = screened image id, resolved by searchsorted).
    row_ids = np.unique(idx[f"unit_{int(support[0])}"])
    mat = np.empty((len(row_ids), len(support)), dtype=np.float32)
    for c, u in enumerate(support):
        mat[np.searchsorted(row_ids, idx[f"unit_{int(u)}"]), c] = resp[f"unit_{int(u)}"]

    mean = mat.mean(0)
    std = mat.std(0) + eps
    zmat = (mat - mean) / std

    neurons = [int(n) for n in (support if neurons is None else neurons)]
    axes = np.zeros((len(neurons), len(support)), dtype=np.float32)
    for r, n in enumerate(neurons):
        order = idx[f"unit_{n}"]                            # image ids for neuron n, ascending response
        mai = np.searchsorted(row_ids, order[-k:])
        lai = np.searchsorted(row_ids, order[:k])
        a = zmat[mai].mean(0) - zmat[lai].mean(0)
        if exclude_target and n in pos:
            a[pos[n]] = 0.0                                 # exclude the target from its own axis
        axes[r] = a / (np.linalg.norm(a) + eps)
    return {"neurons": np.array(neurons), "support": support,
            "axis": axes, "mean": mean, "std": std}


def semantic_axis(
    images1, 
    images2, 
    dreamsim_model, 
    device='cuda'
):
    """
    Compute semantic axis as the difference between centroids of two image sets.
    
    Args:
        images1: Tensor (N, C, H, W) or list of tensors - first image set (e.g., MAIs)
        images2: Tensor (M, C, H, W) or list of tensors - second image set (e.g., LAIs)
        dreamsim_model: DreamSim model
        device: Device to use
    
    Returns:
        axis: Unit vector pointing from centroid of images1 to centroid of images2
    """
    
    def embed_set(images):
        """Embed a set of images and return mean embedding."""
        if not isinstance(images, (list, tuple)):
            images = [images[i] for i in range(len(images))]
        
        embeddings = []
        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(device)
                emb = dreamsim_model.embed(img).flatten().cpu()
                embeddings.append(emb)
        
        return torch.stack(embeddings).mean(dim=0)
    
    # Compute centroids
    centroid1 = embed_set(images1)
    centroid2 = embed_set(images2)
    # Axis from centroid1 → centroid2
    axis = centroid2 - centroid1
    # Normalize to unit vector
    axis = axis / (axis.norm() + 1e-8)
    return axis.numpy()