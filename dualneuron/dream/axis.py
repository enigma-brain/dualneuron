import torch
import numpy as np

from dualneuron.twins import registry


def population_context(area, backbone, neurons=None, dataset="imagenet", field="full",
                       weights_dir=None, eps=1e-8):
    """Everything the axes are built from, loaded ONCE.

    Reading the screening npz and rebuilding the (image x support-neuron) matrix is the expensive
    part; the axis itself is a couple of means over rows of it. Separating them is what makes a
    *per-seed* axis affordable -- :func:`sampled_axis` can then be called once per (neuron, seed)
    without touching disk again.

    Args:
        area, backbone: The twin (its ``field`` screening must exist).
        neurons: Neuron ids whose ranked image order to precompute; default the whole support.
        dataset, field: Screening the axis is built from ("full" = the natural, unmasked regime).
        weights_dir: Override for the twin's ``correlations.npy`` (which defines the support).
        eps: Numerical stabilizer for the z-score.

    Returns:
        dict with ``zmat`` ((images, P) z-scored responses), ``support`` ((P,) global ids),
        ``pos`` (global id -> column), ``row_ids`` ((images,) screened image ids, sorted),
        ``orders`` ({neuron: image ids ascending by that neuron's response}), ``neurons``,
        and the ``mean``/``std`` used for the z-score.
    """
    resp = np.load(registry.screening_path(area, backbone, dataset, "responses", field=field))
    idx = np.load(registry.screening_path(area, backbone, dataset, "indices", field=field))

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

    neurons = np.array([int(n) for n in (support if neurons is None else neurons)])
    orders = {int(n): idx[f"unit_{int(n)}"] for n in neurons}
    return {"zmat": zmat, "support": support, "pos": pos, "row_ids": row_ids,
            "orders": orders, "neurons": neurons, "mean": mean, "std": std}


def sampled_axis(ctx, neuron, pool=100, n_sample=15, rng=None, exclude_target=False, eps=1e-8,
                 return_ids=False):
    """One neuron's population axis, from a random subsample of its extreme images.

    The centroids are taken over ``n_sample`` images drawn without replacement from the neuron's
    top-``pool`` and bottom-``pool`` full-field responses, rather than over a fixed extreme set. With
    a different ``rng`` per synthesis seed this makes the axis itself vary across seeds, so a neuron's
    MEIs/LEIs sample the invariances of its high/low poles instead of re-deriving one fixed direction.

    Args:
        ctx: A :func:`population_context` dict.
        neuron: Target neuron id (must be in ``ctx["orders"]``).
        pool: Size of the extreme pool at each pole to draw from. Default: 100.
        n_sample: Images drawn from each pool for the centroid; ``None`` (or >= ``pool``) uses the
            whole pool, which reproduces the fixed-extreme behaviour. Default: 15.
        rng: ``np.random.RandomState`` for the draw; ``None`` uses the global RNG.
        exclude_target: Zero the target's own component, leaving only the surrounding population's
            context (the regularizer form). False keeps it -- the full MAI-LAI direction used to fold
            the drive into a single bounded cosine.
        eps: Numerical stabilizer.
        return_ids: Also return the drawn image ids. With a per-seed ``rng`` the axis is not
            recoverable from the twin alone, so a caller that persists its results should record
            them alongside.

    Returns:
        ``(P,)`` float32 unit axis over ``ctx["support"]``; with ``return_ids``, the tuple
        ``(axis, mai_ids, lai_ids)`` where the id arrays are the drawn high/low image ids.
    """
    n = int(neuron)
    order = ctx["orders"][n]                                # ascending by this neuron's response
    hi_pool, lo_pool = order[-pool:], order[:pool]
    if n_sample is not None and n_sample < len(hi_pool):
        draw = (rng or np.random).choice
        hi = draw(hi_pool, size=n_sample, replace=False)
        lo = draw(lo_pool, size=n_sample, replace=False)
    else:
        hi, lo = hi_pool, lo_pool

    zmat, row_ids = ctx["zmat"], ctx["row_ids"]
    a = zmat[np.searchsorted(row_ids, hi)].mean(0) - zmat[np.searchsorted(row_ids, lo)].mean(0)
    if exclude_target and n in ctx["pos"]:
        a[ctx["pos"][n]] = 0.0                              # exclude the target from its own axis
    axis = (a / (np.linalg.norm(a) + eps)).astype(np.float32)
    return (axis, np.asarray(hi), np.asarray(lo)) if return_ids else axis


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