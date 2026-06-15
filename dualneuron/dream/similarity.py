"""
DreamSim similarity analyses relating each neuron's most/least activating images
to its predicted activity (paper Figs 6 and 10).

Both analyses operate on DreamSim embeddings (from dualneuron.dream.sim.embeddings)
and the per-neuron ordered responses/indices produced by screening (the Dryad
{area}_{dataset}_ordered_{responses,indices}.npz files, keyed unit_{id}). The
ensemble DreamSim embeddings are L2-normalized, so cosine similarity is just the
dot product.

    coherence_dprime    -> Fig 6: are a neuron's MAIs (and LAIs) more self-similar
                           than random images? (d-prime per neuron)
    similarity_space_2d -> Fig 10: does activity vary linearly with similarity to
                           the MAI and LAI? (2D-space regression R^2 + controls)
"""
import numpy as np
from scipy.stats import skew


def _row_of(indices):
    """
    Map each global image index to its row in the embeddings array.

    Args:
        indices (np.ndarray): (M,) global image index for each embedding row,
            as returned by embeddings().

    Returns:
        dict: {global_index: row} for the M embedded images.
    """
    return {int(g): i for i, g in enumerate(np.asarray(indices))}


def _activity_by_image(responses_sorted, indices_sorted, n_images):
    """
    Reconstruct per-image activity from an ordered (sorted) response/index pair.

    The ordered npz stores, per neuron, responses sorted ascending and the dataset
    index at each sorted position. This inverts that to activity indexed by image.

    Args:
        responses_sorted (np.ndarray): (N,) responses sorted ascending.
        indices_sorted (np.ndarray): (N,) dataset index at each sorted position.
        n_images (int): Number of images (max index + 1).

    Returns:
        np.ndarray: (n_images,) activity indexed by image.
    """
    activity = np.full(n_images, np.nan, dtype=np.float64)
    activity[np.asarray(indices_sorted)] = np.asarray(responses_sorted)
    return activity


def _dprime(within, across):
    """
    Discriminability d' between a within-set and an across-(random) distribution.

    d' = (mean_within - mean_across) / sqrt(0.5 * (var_within + var_across)).

    Args:
        within (np.ndarray): Cosine similarities within an image set.
        across (np.ndarray): Cosine similarities of that set to random images.

    Returns:
        float: d-prime (higher = the set is more self-similar than to random).
    """
    within, across = np.asarray(within), np.asarray(across)
    denom = np.sqrt(0.5 * (within.var(ddof=1) + across.var(ddof=1))) + 1e-12
    return float((within.mean() - across.mean()) / denom)


def _linfit_r2(X, y):
    """
    R^2 of an ordinary least-squares fit of y on the columns of X (intercept added).

    Args:
        X (np.ndarray): (n, p) predictors.
        y (np.ndarray): (n,) target.

    Returns:
        float: Coefficient of determination, 1 - SS_res / SS_tot.
    """
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def coherence_dprime(
    embeddings,
    indices,
    ordered_responses,
    ordered_indices,
    neurons=None,
    k=10,
    n_random=10,
    seed=0,
):
    """
    Per-neuron coherence of MAIs/LAIs in DreamSim space (paper Fig 6).

    For each neuron, the k most-activating (MAI) and k least-activating (LAI) images
    are compared in DreamSim embedding space: the within-MAI and within-LAI pairwise
    cosine similarities are contrasted with the similarities of those images to
    n_random randomly selected images, summarized by d-prime. A random-set control
    (a random k-set vs random images) is also returned and should be ~0.

    Embeddings are assumed L2-normalized (DreamSim ensemble), so cosine = dot product.
    Every MAI/LAI/random image used must be present in the embedded set (`indices`).

    Args:
        embeddings (np.ndarray): (M, D) L2-normalized DreamSim embeddings.
        indices (np.ndarray): (M,) global image index for each embedding row.
        ordered_responses (Mapping): unit_{id} -> (N,) responses sorted ascending
            (used only to flag non-sparse neurons by skewness).
        ordered_indices (Mapping): unit_{id} -> (N,) image indices sorted by ascending
            response. Last k = MAIs, first k = LAIs.
        neurons (list, optional): Neuron ids to use. Default: all in ordered_indices.
        k (int): Number of MAIs/LAIs per pole. Default: 10.
        n_random (int): Number of random reference images. Default: 10.
        seed (int): RNG seed for the random references. Default: 0.

    Returns:
        dict with keys:
            - 'neurons': (n,) neuron ids
            - 'dprime_mai': (n,) within-MAI vs random d-prime
            - 'dprime_lai': (n,) within-LAI vs random d-prime
            - 'dprime_control': (n,) random-set vs random d-prime (~0 expected)
            - 'skewness': (n,) response skewness
            - 'non_sparse': (n,) bool, skewness < 2
    """
    embeddings = np.asarray(embeddings)
    row = _row_of(indices)
    pool = np.asarray(indices)
    rng = np.random.default_rng(seed)

    if neurons is None:
        neurons = sorted(int(key.split('_')[1]) for key in ordered_indices.keys())

    def emb_rows(global_idx):
        rows = []
        for g in np.asarray(global_idx):
            g = int(g)
            if g not in row:
                raise ValueError(f"image {g} is not in the embedded set; embed it first")
            rows.append(row[g])
        return embeddings[rows]

    def within_pairwise(E):
        C = E @ E.T
        return C[np.triu_indices(len(E), k=1)]

    dprime_mai, dprime_lai, dprime_control = [], [], []
    skews, non_sparse = [], []
    for n in neurons:
        order = np.asarray(ordered_indices[f'unit_{n}'])
        mai = emb_rows(order[-k:])
        lai = emb_rows(order[:k])
        rand = emb_rows(rng.choice(pool, size=n_random, replace=False))
        ctrl = emb_rows(rng.choice(pool, size=k, replace=False))
        ctrl_rand = emb_rows(rng.choice(pool, size=n_random, replace=False))
        dprime_mai.append(_dprime(within_pairwise(mai), (mai @ rand.T).ravel()))
        dprime_lai.append(_dprime(within_pairwise(lai), (lai @ rand.T).ravel()))
        dprime_control.append(_dprime(within_pairwise(ctrl), (ctrl @ ctrl_rand.T).ravel()))
        s = float(skew(np.asarray(ordered_responses[f'unit_{n}'])))
        skews.append(s)
        non_sparse.append(s < 2.0)

    return {
        'neurons': np.array(neurons),
        'dprime_mai': np.array(dprime_mai),
        'dprime_lai': np.array(dprime_lai),
        'dprime_control': np.array(dprime_control),
        'skewness': np.array(skews),
        'non_sparse': np.array(non_sparse),
    }


def similarity_space_2d(
    embeddings,
    indices,
    ordered_responses,
    ordered_indices,
    neurons=None,
    seed=0,
):
    """
    Per-neuron 2D similarity space and its explained variance (paper Fig 10).

    For each neuron, the single most-activating (MAI) and least-activating (LAI) image
    define two axes; every embedded image gets x = cosine(image, LAI) and
    y = cosine(image, MAI) in DreamSim space. A linear regression of the neuron's
    predicted activity on (x, y) gives R^2 = variance explained. Three controls refit
    R^2 after replacing the reference image(s) with random ones: both (C1), MAI (C2),
    LAI (C3). Neurons are classified non-sparse if response skewness < 2.

    Embeddings are assumed L2-normalized so cosine = dot product. Each neuron's MAI
    and LAI must be present in the embedded set (for the rendered set this is automatic
    when all scenes are embedded).

    Args:
        embeddings (np.ndarray): (M, D) L2-normalized DreamSim embeddings.
        indices (np.ndarray): (M,) global image index for each embedding row.
        ordered_responses (Mapping): unit_{id} -> (N,) responses sorted ascending.
        ordered_indices (Mapping): unit_{id} -> (N,) image indices at each sorted position.
        neurons (list, optional): Neuron ids to use. Default: all in ordered_indices.
        seed (int): RNG seed for the control reference images. Default: 0.

    Returns:
        dict with keys:
            - 'neurons': (n,) neuron ids
            - 'r2': (n,) R^2 of activity ~ (cos-to-LAI, cos-to-MAI)
            - 'r2_control_both' / 'r2_control_mai' / 'r2_control_lai': (n,) control R^2
            - 'skewness': (n,) response skewness
            - 'non_sparse': (n,) bool, skewness < 2
    """
    embeddings = np.asarray(embeddings)
    img_idx = np.asarray(indices)
    row = _row_of(indices)
    n_images = int(img_idx.max()) + 1
    rng = np.random.default_rng(seed)

    if neurons is None:
        neurons = sorted(int(key.split('_')[1]) for key in ordered_indices.keys())

    def emb_of(global_idx):
        g = int(global_idx)
        if g not in row:
            raise ValueError(f"image {g} is not in the embedded set; embed it first")
        return embeddings[row[g]]

    r2, r2_both, r2_mai, r2_lai = [], [], [], []
    skews, non_sparse = [], []
    for n in neurons:
        order = np.asarray(ordered_indices[f'unit_{n}'])
        resp = np.asarray(ordered_responses[f'unit_{n}'])
        act = _activity_by_image(resp, order, n_images)[img_idx]

        mai, lai = emb_of(order[-1]), emb_of(order[0])
        rand_mai, rand_lai = emb_of(rng.choice(img_idx)), emb_of(rng.choice(img_idx))

        x, y = embeddings @ lai, embeddings @ mai            # cos to LAI (x), to MAI (y)
        xr, yr = embeddings @ rand_lai, embeddings @ rand_mai

        r2.append(_linfit_r2(np.column_stack([x, y]), act))
        r2_both.append(_linfit_r2(np.column_stack([xr, yr]), act))   # C1: both random
        r2_mai.append(_linfit_r2(np.column_stack([x, yr]), act))     # C2: MAI random, keep LAI
        r2_lai.append(_linfit_r2(np.column_stack([xr, y]), act))     # C3: LAI random, keep MAI

        s = float(skew(resp))
        skews.append(s)
        non_sparse.append(s < 2.0)

    return {
        'neurons': np.array(neurons),
        'r2': np.array(r2),
        'r2_control_both': np.array(r2_both),
        'r2_control_mai': np.array(r2_mai),
        'r2_control_lai': np.array(r2_lai),
        'skewness': np.array(skews),
        'non_sparse': np.array(non_sparse),
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    DATA_DIR = os.getenv("DATA_DIR")
    parser = argparse.ArgumentParser(description="DreamSim coherence (Fig 6) and 2D similarity space (Fig 10)")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="npz with 'embeddings' and 'indices' (from dream.sim.embeddings)")
    parser.add_argument("--model", type=str, default="v4", help="v1 or v4")
    parser.add_argument("--dataset", type=str, default="rendered", help="rendered or imagenet")
    parser.add_argument("--dryad_dir", type=str, default=None,
                        help="dir with {model}_{dataset}_ordered_{responses,indices}.npz (default DATA_DIR/dryad)")
    parser.add_argument("--output", type=str, default=None, help="npz to save the results")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random references")
    args = parser.parse_args()

    dryad_dir = args.dryad_dir or os.path.join(DATA_DIR, "dryad")
    emb = np.load(args.embeddings)
    embeddings, indices = emb["embeddings"], emb["indices"]
    ordered_responses = np.load(os.path.join(dryad_dir, f"{args.model}_{args.dataset}_ordered_responses.npz"))
    ordered_indices = np.load(os.path.join(dryad_dir, f"{args.model}_{args.dataset}_ordered_indices.npz"))

    coh = coherence_dprime(embeddings, indices, ordered_responses, ordered_indices, seed=args.seed)
    sp = similarity_space_2d(embeddings, indices, ordered_responses, ordered_indices, seed=args.seed)

    cns = coh['non_sparse']
    print(f"Fig 6  d-prime (median, non-sparse n={int(cns.sum())}): "
          f"MAI={np.nanmedian(coh['dprime_mai'][cns]):.3f} "
          f"LAI={np.nanmedian(coh['dprime_lai'][cns]):.3f} "
          f"control={np.nanmedian(coh['dprime_control'][cns]):.3f}")
    ns = sp['non_sparse']
    print(f"Fig 10 R^2 (mean): non-sparse={np.nanmean(sp['r2'][ns]):.3f} (n={int(ns.sum())}) "
          f"sparse={np.nanmean(sp['r2'][~ns]):.3f} | control(both)={np.nanmean(sp['r2_control_both']):.3f}")

    if args.output is not None:
        np.savez(args.output,
                 **{f"coh_{key}": val for key, val in coh.items()},
                 **{f"sp_{key}": val for key, val in sp.items()})
        print(f"saved {args.output}")
