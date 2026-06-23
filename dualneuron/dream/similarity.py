"""
DreamSim similarity analyses relating each neuron's most/least activating images
to its predicted activity (paper Figs 6 and 10).

Both analyses operate on DreamSim embeddings (from dualneuron.dream.sim.embeddings)
and the per-neuron ordered responses/indices produced by screening (the
{area}_ensemble_{dataset}_ordered_{responses,indices}.npz files, keyed unit_{id}).

Following Franke et al., every cosine similarity is computed on *globally centered*
embeddings: the mean embedding over all images is subtracted first, then each vector
is renormalized. DreamSim's ensemble embeddings are natively unit-norm, so centering
removes the large common-mode component shared by all natural images and is what gives
the d-prime and R^2 their dynamic range. After centering+renormalizing, cosine
similarity is again just the dot product.

    coherence_dprime    -> Fig 6: are a neuron's MAIs (and LAIs) more self-similar
                           than random images? (d-prime per neuron)
    similarity_space_2d -> Fig 10: does activity vary with similarity to the MAI and
                           LAI poles? (2D-space regression R^2 + random-pole controls)
"""
import numpy as np
from scipy.stats import skew
from tqdm import tqdm


def _center_and_normalize(embeddings):
    """
    Globally center the embeddings, then L2-normalize each row (Franke et al.).

    Subtracts the mean embedding across all images (removing the common-mode
    component shared by natural images), then renormalizes each row to unit length
    so that a dot product is the cosine similarity in the centered space.

    Args:
        embeddings (np.ndarray): (M, D) embeddings (DreamSim ensemble, unit-norm).

    Returns:
        np.ndarray: (M, D) float32 centered, row-normalized embeddings.
    """
    E = np.asarray(embeddings, dtype=np.float32)
    E = E - E.mean(axis=0, keepdims=True)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E


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

    d' = (mean_within - mean_across) / sqrt(0.5 * (var_within + var_across)), using
    the unbiased (ddof=1) sample variance.

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

    Solves the normal equations for the degree-1 model y ~ 1 + X; the returned R^2 is
    identical to sklearn LinearRegression().score (1 - SS_res / SS_tot).

    Args:
        X (np.ndarray): (n, p) predictors.
        y (np.ndarray): (n,) target.

    Returns:
        float: Coefficient of determination, 1 - SS_res / SS_tot.
    """
    A = np.column_stack([np.ones(len(X), dtype=np.float64), np.asarray(X, dtype=np.float64)])
    y = np.asarray(y, dtype=np.float64)
    beta = np.linalg.lstsq(A.T @ A, A.T @ y, rcond=None)[0]
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    yc = y - y.mean()
    ss_tot = float(yc @ yc) + 1e-12
    return 1.0 - ss_res / ss_tot


def _r2_two_predictor(r_y1, r_y2, r_12):
    """
    Exact OLS R^2 for the degree-1 model y ~ 1 + p1 + p2 from pairwise correlations.

    R^2 = (r_y1^2 + r_y2^2 - 2 r_y1 r_y2 r_12) / (1 - r_12^2), the closed form for two
    predictors plus intercept. Identical to _linfit_r2 but vectorizes over neurons:
    the inputs may be arrays (one entry per neuron) with r_12 either scalar (a control
    pole shared across neurons) or a matching array.

    Args:
        r_y1, r_y2: corr(target, predictor 1) and corr(target, predictor 2).
        r_12: corr(predictor 1, predictor 2).

    Returns:
        np.ndarray or float: R^2 per element.
    """
    denom = 1.0 - np.asarray(r_12) ** 2
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return (r_y1 ** 2 + r_y2 ** 2 - 2.0 * r_y1 * r_y2 * r_12) / denom


def coherence_dprime(
    embeddings,
    indices,
    ordered_responses,
    ordered_indices,
    neurons=None,
    k=10,
    n_random=10,
    seed=0,
    progress=None,
    mininterval=10.0,
):
    """
    Per-neuron coherence of MAIs/LAIs in centered DreamSim space (paper Fig 6).

    For each neuron, the k most-activating (MAI) and k least-activating (LAI) images
    are compared in centered DreamSim space: the within-MAI and within-LAI pairwise
    cosine similarities are contrasted with the similarities of those images to
    n_random random images, summarized by d-prime. A random-vs-random control
    (cosine of one random set to a second, vs a third random set to a fourth) is also
    returned and should be ~0.

    Embeddings are centered and renormalized internally (see _center_and_normalize),
    so cosine = dot product. Every MAI/LAI image used must be present in `indices`.

    Args:
        embeddings (np.ndarray): (M, D) DreamSim embeddings (unit-norm).
        indices (np.ndarray): (M,) global image index for each embedding row.
        ordered_responses (Mapping): unit_{id} -> (N,) responses sorted ascending
            (used only to flag non-sparse neurons by skewness).
        ordered_indices (Mapping): unit_{id} -> (N,) image indices sorted by ascending
            response. Last k = MAIs, first k = LAIs.
        neurons (list, optional): Neuron ids to use. Default: all in ordered_indices.
        k (int): Number of MAIs/LAIs per pole. Default: 10.
        n_random (int): Size of each random reference set. Default: 10.
        seed (int): RNG seed for the random references. Default: 0.
        progress (file-like, optional): If given, a tqdm bar is written to it.
        mininterval (float): Min seconds between progress updates. Default: 10.0.

    Returns:
        dict with keys:
            - 'neurons': (n,) neuron ids
            - 'dprime_mai': (n,) within-MAI vs MAI-to-random d-prime
            - 'dprime_lai': (n,) within-LAI vs LAI-to-random d-prime
            - 'dprime_control': (n,) random-vs-random d-prime (~0 expected)
            - 'skewness': (n,) response skewness
            - 'non_sparse': (n,) bool, skewness < 2
    """
    E = _center_and_normalize(np.asarray(embeddings))
    row = _row_of(indices)
    n_emb = len(E)
    rng = np.random.default_rng(seed)

    if neurons is None:
        neurons = sorted(int(key.split('_')[1]) for key in ordered_indices.keys())
    neurons = [int(n) for n in neurons]

    def emb_rows(global_idx):
        rows = []
        for g in np.asarray(global_idx):
            g = int(g)
            if g not in row:
                raise ValueError(f"image {g} is not in the embedded set; embed it first")
            rows.append(row[g])
        return E[rows]

    def within_pairwise(M):
        C = M @ M.T
        return C[np.triu_indices(len(M), k=1)]

    dprime_mai, dprime_lai, dprime_control = [], [], []
    skews, non_sparse = [], []
    iterator = neurons if progress is None else tqdm(
        neurons, file=progress, mininterval=mininterval, ncols=100, desc="coherence d'")
    for n in iterator:
        order = np.asarray(ordered_indices[f'unit_{n}'])
        mai = emb_rows(order[-k:])
        lai = emb_rows(order[:k])
        # Four independent random reference sets; r1 is reused for the MAI/LAI-to-random
        # comparisons, (r1, r2) and (r3, r4) form the random-vs-random control.
        r1 = E[rng.choice(n_emb, n_random, replace=False)]
        r2 = E[rng.choice(n_emb, n_random, replace=False)]
        r3 = E[rng.choice(n_emb, n_random, replace=False)]
        r4 = E[rng.choice(n_emb, n_random, replace=False)]
        dprime_mai.append(_dprime(within_pairwise(mai), (mai @ r1.T).ravel()))
        dprime_lai.append(_dprime(within_pairwise(lai), (lai @ r1.T).ravel()))
        dprime_control.append(_dprime((r1 @ r2.T).ravel(), (r3 @ r4.T).ravel()))
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
    n_poles=15,
    n_random=15,
    n_iterations=100,
    seed=0,
    progress=None,
    mininterval=10.0,
):
    """
    Per-neuron 2D similarity space and its explained variance (paper Fig 10).

    For each neuron, two axes are defined from the n_poles most-activating (MAI) and
    n_poles least-activating (LAI) images: every embedded image gets x = mean cosine
    to the LAI pole and y = mean cosine to the MAI pole, in centered DreamSim space. A
    degree-1 linear regression of the neuron's predicted activity on (x, y) gives
    R^2 = variance explained.

    Three controls refit R^2 after replacing pole(s) with random images, each as the
    mean cosine to n_random random images, averaged over n_iterations draws (all
    degree-1, matching the headline fit): both poles random (control_both), the true
    MAI kept with the LAI randomized (control_keep_mai), and the true LAI kept with the
    MAI randomized (control_keep_lai). To keep the analysis tractable over hundreds of
    images, the random poles are drawn once per iteration and shared across neurons;
    this leaves each neuron's control distribution unbiased (a common random null)
    while avoiding a separate matmul over all images for every neuron and iteration.

    Embeddings are centered and renormalized internally so cosine = dot product. Each
    neuron's MAI/LAI poles must be present in the embedded set.

    Args:
        embeddings (np.ndarray): (M, D) DreamSim embeddings (unit-norm).
        indices (np.ndarray): (M,) global image index for each embedding row.
        ordered_responses (Mapping): unit_{id} -> (N,) responses sorted ascending.
        ordered_indices (Mapping): unit_{id} -> (N,) image indices at each sorted
            position. Last n_poles = MAI pole, first n_poles = LAI pole.
        neurons (list, optional): Neuron ids to use. Default: all in ordered_indices.
        n_poles (int): Images per pole (mean cosine over these). Default: 15.
        n_random (int): Images per random control pole. Default: 15.
        n_iterations (int): Random draws averaged for each control. Default: 100.
        seed (int): RNG seed for the control poles. Default: 0.
        progress (file-like, optional): If given, tqdm bars are written to it.
        mininterval (float): Min seconds between progress updates. Default: 10.0.

    Returns:
        dict with keys:
            - 'neurons': (n,) neuron ids
            - 'r2': (n,) R^2 of activity ~ (cos-to-LAI, cos-to-MAI)
            - 'r2_control_both' / 'r2_control_keep_mai' / 'r2_control_keep_lai': (n,)
              mean control R^2 (both poles random / true MAI kept / true LAI kept)
            - 'skewness': (n,) response skewness
            - 'non_sparse': (n,) bool, skewness < 2
    """
    E = _center_and_normalize(np.asarray(embeddings))
    img_idx = np.asarray(indices)
    row = _row_of(indices)
    n_images = int(img_idx.max()) + 1
    n_emb = len(E)
    rng = np.random.default_rng(seed)

    if neurons is None:
        neurons = sorted(int(key.split('_')[1]) for key in ordered_indices.keys())
    neurons = [int(n) for n in neurons]
    n_neurons = len(neurons)

    def pole_sim(rows):
        # Mean cosine of every image to the given pole rows. Cosine is linear in the
        # centered, unit-norm embeddings, so the mean over poles is a single matvec:
        # mean_j (E . E[rows_j]) = E . mean_j E[rows_j].
        return E @ E[rows].mean(axis=0)

    def pole_rows(global_idx):
        rows = []
        for g in np.asarray(global_idx):
            g = int(g)
            if g not in row:
                raise ValueError(f"image {g} is not in the embedded set; embed it first")
            rows.append(row[g])
        return np.array(rows)

    # Per neuron: similarity to the MAI pole (Y) and LAI pole (X), and the activity
    # aligned to the embedded images. Computed once and reused for the controls.
    X = np.empty((n_neurons, n_emb), dtype=np.float32)   # cos to LAI pole
    Y = np.empty((n_neurons, n_emb), dtype=np.float32)   # cos to MAI pole
    ACT = np.empty((n_neurons, n_emb), dtype=np.float32)
    skews, non_sparse = [], []
    it_real = enumerate(neurons) if progress is None else tqdm(
        list(enumerate(neurons)), file=progress, mininterval=mininterval, ncols=100,
        desc="similarity poles")
    for j, n in it_real:
        order = np.asarray(ordered_indices[f'unit_{n}'])
        resp = np.asarray(ordered_responses[f'unit_{n}'])
        Y[j] = pole_sim(pole_rows(order[-n_poles:]))
        X[j] = pole_sim(pole_rows(order[:n_poles]))
        ACT[j] = _activity_by_image(resp, order, n_images)[img_idx]
        s = float(skew(resp))
        skews.append(s)
        non_sparse.append(s < 2.0)

    # Headline R^2: activity ~ (cos-to-LAI, cos-to-MAI), exact OLS per neuron.
    r2 = np.array([_linfit_r2(np.column_stack([X[j], Y[j]]), ACT[j]) for j in range(n_neurons)])

    # Controls via the closed-form two-predictor R^2. Center+normalize each per-neuron
    # vector over the image axis so a dot product is a correlation; the true-pole
    # correlations don't depend on the random draw, so precompute them once. The random
    # pole is shared across neurons within an iteration, so each iteration is a handful
    # of matrix-vector products vectorized over all neurons (no per-neuron refit).
    def _cn(A):
        Ac = A - A.mean(axis=-1, keepdims=True)
        return Ac / (np.linalg.norm(Ac, axis=-1, keepdims=True) + 1e-12)

    ACTn = _cn(ACT)
    Xn = _cn(X)
    Yn = _cn(Y)
    r_yX = np.einsum('ij,ij->i', ACTn, Xn)   # corr(activity, true LAI pole)
    r_yY = np.einsum('ij,ij->i', ACTn, Yn)   # corr(activity, true MAI pole)

    acc_both = np.zeros(n_neurons)
    acc_keep_mai = np.zeros(n_neurons)
    acc_keep_lai = np.zeros(n_neurons)
    it_ctrl = range(n_iterations) if progress is None else tqdm(
        range(n_iterations), file=progress, mininterval=mininterval, ncols=100,
        desc="similarity controls")
    for _ in it_ctrl:
        s1n = _cn(pole_sim(rng.choice(n_emb, n_random, replace=False)))
        s2n = _cn(pole_sim(rng.choice(n_emb, n_random, replace=False)))
        r_y1 = ACTn @ s1n        # corr(activity, random pole 1), per neuron
        r_y2 = ACTn @ s2n
        r_X1 = Xn @ s1n          # corr(true LAI pole, random pole 1)
        r_Y1 = Yn @ s1n          # corr(true MAI pole, random pole 1)
        r_12 = float(s1n @ s2n)  # corr(random pole 1, random pole 2)
        acc_both += _r2_two_predictor(r_y1, r_y2, r_12)        # both poles random
        acc_keep_mai += _r2_two_predictor(r_yY, r_y1, r_Y1)    # true MAI kept, LAI randomized
        acc_keep_lai += _r2_two_predictor(r_yX, r_y1, r_X1)    # true LAI kept, MAI randomized

    return {
        'neurons': np.array(neurons),
        'r2': r2,
        'r2_control_both': acc_both / n_iterations,
        'r2_control_keep_mai': acc_keep_mai / n_iterations,
        'r2_control_keep_lai': acc_keep_lai / n_iterations,
        'skewness': np.array(skews),
        'non_sparse': np.array(non_sparse),
    }


def similarity_space_neuron(
    embeddings, indices, ordered_responses, ordered_indices,
    neuron, n_poles=15, n_random=15, seed=0,
):
    """
    Full 2D similarity space of a single neuron, for the example panels of Fig 10.

    Returns every embedded image's coordinates x = mean cosine to the LAI pole and
    y = mean cosine to the MAI pole (centered DreamSim space), the neuron's activity,
    the headline R^2, and the three control (x, y, R^2) variants used in Fig 10e.

    Args:
        embeddings, indices, ordered_responses, ordered_indices: as in
            similarity_space_2d.
        neuron (int): neuron id.
        n_poles (int): images per pole. Default: 15.
        n_random (int): images per random control pole. Default: 15.
        seed (int): RNG seed for the control poles. Default: 0.

    Returns:
        dict: {'x', 'y', 'activity' (each (M,)), 'r2', 'skewness',
            'control_both' / 'control_keep_mai' / 'control_keep_lai': (x, y, r2)}.
    """
    E = _center_and_normalize(np.asarray(embeddings))
    img_idx = np.asarray(indices)
    row = _row_of(indices)
    n_images = int(img_idx.max()) + 1
    rng = np.random.default_rng(seed)

    order = np.asarray(ordered_indices[f'unit_{neuron}'])
    resp = np.asarray(ordered_responses[f'unit_{neuron}'])

    def pole_sim(rows):
        return E @ E[rows].mean(axis=0)

    def rows_of(global_idx):
        return np.array([row[int(g)] for g in np.asarray(global_idx)])

    y = pole_sim(rows_of(order[-n_poles:]))   # cos to MAI pole
    x = pole_sim(rows_of(order[:n_poles]))    # cos to LAI pole
    act = _activity_by_image(resp, order, n_images)[img_idx]
    xr = pole_sim(rng.choice(len(E), n_random, replace=False))
    yr = pole_sim(rng.choice(len(E), n_random, replace=False))

    return {
        'x': x, 'y': y, 'activity': act,
        'r2': _linfit_r2(np.column_stack([x, y]), act),
        'skewness': float(skew(resp)),
        'control_both': (xr, yr, _linfit_r2(np.column_stack([xr, yr]), act)),
        'control_keep_mai': (xr, y, _linfit_r2(np.column_stack([xr, y]), act)),  # LAI random
        'control_keep_lai': (x, yr, _linfit_r2(np.column_stack([x, yr]), act)),  # MAI random
    }


def coherence_neuron(
    embeddings, indices, ordered_indices, neuron, k=10, n_random=10, seed=0,
):
    """
    Cosine-similarity distributions of a single neuron, for the example panel Fig 6b.

    Returns, in centered DreamSim space, the within-MAI and within-LAI pairwise cosine
    similarities and the cosines of those sets to a random reference set, plus the two
    d-prime values.

    Args:
        embeddings, indices, ordered_indices: as in coherence_dprime.
        neuron (int): neuron id.
        k (int): MAIs/LAIs per pole. Default: 10.
        n_random (int): random reference set size. Default: 10.
        seed (int): RNG seed. Default: 0.

    Returns:
        dict: {'within_mai', 'mai_random', 'within_lai', 'lai_random' (cosine arrays),
            'dprime_mai', 'dprime_lai'}.
    """
    E = _center_and_normalize(np.asarray(embeddings))
    row = _row_of(indices)
    rng = np.random.default_rng(seed)
    order = np.asarray(ordered_indices[f'unit_{neuron}'])

    def emb_rows(global_idx):
        return E[[row[int(g)] for g in np.asarray(global_idx)]]

    def within(M):
        return (M @ M.T)[np.triu_indices(len(M), k=1)]

    mai = emb_rows(order[-k:])
    lai = emb_rows(order[:k])
    r1 = E[rng.choice(len(E), n_random, replace=False)]
    within_mai, mai_random = within(mai), (mai @ r1.T).ravel()
    within_lai, lai_random = within(lai), (lai @ r1.T).ravel()
    return {
        'within_mai': within_mai, 'mai_random': mai_random,
        'within_lai': within_lai, 'lai_random': lai_random,
        'dprime_mai': _dprime(within_mai, mai_random),
        'dprime_lai': _dprime(within_lai, lai_random),
    }


def coherence_pooled(embeddings, indices, ordered_indices, neurons, k=10, n_random=10, seed=0):
    """
    Within-MAI/within-LAI and MAI/LAI-to-random cosine similarities pooled across neurons.

    For the population distribution panel of Fig 6: instead of one example neuron, the
    within-pole pairwise cosines and the pole-to-random cosines are concatenated over all
    given neurons, in centered DreamSim space. Embeddings are centered once.

    Args:
        embeddings, indices, ordered_indices: as in coherence_dprime.
        neurons (sequence): neuron ids to pool over (e.g. the non-sparse set).
        k (int): MAIs/LAIs per pole. Default: 10.
        n_random (int): random reference set size per neuron. Default: 10.
        seed (int): RNG seed. Default: 0.

    Returns:
        dict: {'within_mai', 'within_lai', 'mai_random', 'lai_random'} pooled cosine arrays.
    """
    E = _center_and_normalize(np.asarray(embeddings))
    row = _row_of(indices)
    rng = np.random.default_rng(seed)

    def emb_rows(global_idx):
        return E[[row[int(g)] for g in np.asarray(global_idx)]]

    def within(M):
        return (M @ M.T)[np.triu_indices(len(M), k=1)]

    within_mai, within_lai, mai_random, lai_random = [], [], [], []
    for n in neurons:
        order = np.asarray(ordered_indices[f'unit_{int(n)}'])
        mai, lai = emb_rows(order[-k:]), emb_rows(order[:k])
        r1 = E[rng.choice(len(E), n_random, replace=False)]
        within_mai.append(within(mai))
        within_lai.append(within(lai))
        mai_random.append((mai @ r1.T).ravel())
        lai_random.append((lai @ r1.T).ravel())
    return {
        'within_mai': np.concatenate(within_mai),
        'within_lai': np.concatenate(within_lai),
        'mai_random': np.concatenate(mai_random),
        'lai_random': np.concatenate(lai_random),
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    import time
    import argparse
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()
    from dualneuron.utils import (
        env_dir, well_predicted_neurons, _AREA_MODELS, ensure_dir, RewriteLine,
    )

    ANALYSIS_DIR = env_dir("ANALYSIS_DIR")
    LOGS_DIR = env_dir("LOGS_DIR")
    parser = argparse.ArgumentParser(description="DreamSim coherence (Fig 6) and 2D similarity space (Fig 10)")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="npz with 'embeddings' and 'indices' (default ANALYSIS_DIR/{model}/{model}_dreamsim_{dataset}_embeddings.npz)")
    parser.add_argument("--model", type=str, default="v4", help="v1 or v4")
    parser.add_argument("--dataset", type=str, default="rendered", help="rendered or imagenet")
    parser.add_argument("--analysis_dir", type=str, default=None,
                        help="dir with {model}_ensemble_{dataset}_ordered_{responses,indices}.npz (default ANALYSIS_DIR/{model})")
    parser.add_argument("--output", type=str, default=None, help="npz to save the results")
    parser.add_argument("--k", type=int, default=10, help="MAIs/LAIs per pole for d-prime (Fig 6)")
    parser.add_argument("--n_poles", type=int, default=15, help="images per pole for the R^2 space (Fig 10)")
    parser.add_argument("--n_iterations", type=int, default=100, help="random draws averaged per control")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random references")
    parser.add_argument("--log_path", type=str, default=None,
                        help="progress log file (default LOGS_DIR/{model}_{dataset}_similarity.log)")
    parser.add_argument("--log_every", type=float, default=10.0,
                        help="min seconds between progress-line updates")
    args = parser.parse_args()

    if ANALYSIS_DIR is None and (args.analysis_dir is None or args.embeddings is None):
        raise ValueError(
            "ANALYSIS_DIR is not set. Set it in .env (e.g. "
            "ANALYSIS_DIR=${DATA_DIR}/DUAL-FEATURE-ANALYSIS) or pass "
            "--analysis_dir and --embeddings explicitly."
        )
    analysis_dir = args.analysis_dir or os.path.join(ANALYSIS_DIR, args.model)
    emb_path = args.embeddings or os.path.join(
        ANALYSIS_DIR, args.model, f"{args.model}_dreamsim_{args.dataset}_embeddings.npz"
    )
    emb = np.load(emb_path)
    embeddings, indices = emb["embeddings"], emb["indices"]
    ordered_responses = np.load(os.path.join(analysis_dir, f"{args.model}_ensemble_{args.dataset}_ordered_responses.npz"))
    ordered_indices = np.load(os.path.join(analysis_dir, f"{args.model}_ensemble_{args.dataset}_ordered_indices.npz"))

    # Restrict to the well-predicted neurons (inclusion criterion, correlation > 0.4); the
    # ordered npz files hold every screened neuron, but only the well-predicted ones' extremes
    # were embedded for the imagenet subset.
    neurons = [int(n) for n in well_predicted_neurons(_AREA_MODELS[args.model])]

    # Progress log: one line rewritten in place, bracketed by a header and footer
    # (clean in any editor), mirroring the screening / synthesis runs.
    log_path = args.log_path
    if log_path is None and LOGS_DIR is not None:
        log_path = os.path.join(LOGS_DIR, f"{args.model}_{args.dataset}_similarity.log")
    log_file = None
    progress = None
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        log_file = open(log_path, "w")
        log_file.write(
            f"similarity area={args.model} dataset={args.dataset} neurons={len(neurons)} "
            f"k={args.k} n_poles={args.n_poles} iters={args.n_iterations} "
            f"(centered embeddings, ddof=1, degree-1 controls)\n"
        )
        log_file.flush()
        progress = RewriteLine(log_file, log_file.tell())

    start = time.time()
    coh = coherence_dprime(
        embeddings, indices, ordered_responses, ordered_indices,
        neurons=neurons, k=args.k, seed=args.seed,
        progress=progress, mininterval=args.log_every,
    )
    sp = similarity_space_2d(
        embeddings, indices, ordered_responses, ordered_indices,
        neurons=neurons, n_poles=args.n_poles, n_iterations=args.n_iterations,
        seed=args.seed, progress=progress, mininterval=args.log_every,
    )
    elapsed = time.time() - start

    cns = coh['non_sparse']
    line6 = (f"Fig 6  d-prime (median, non-sparse n={int(cns.sum())}): "
             f"MAI={np.nanmedian(coh['dprime_mai'][cns]):.3f} "
             f"LAI={np.nanmedian(coh['dprime_lai'][cns]):.3f} "
             f"control={np.nanmedian(coh['dprime_control'][cns]):.3f}")
    ns = sp['non_sparse']
    line10 = (f"Fig 10 R^2 (mean): non-sparse={np.nanmean(sp['r2'][ns]):.3f} (n={int(ns.sum())}) "
              f"sparse={np.nanmean(sp['r2'][~ns]):.3f} (n={int((~ns).sum())}) | "
              f"control both={np.nanmean(sp['r2_control_both'][ns]):.3f} "
              f"keep_mai={np.nanmean(sp['r2_control_keep_mai'][ns]):.3f} "
              f"keep_lai={np.nanmean(sp['r2_control_keep_lai'][ns]):.3f}")
    print(line6)
    print(line10)

    if log_file is not None:
        log_file.write(line6 + "\n" + line10 + "\n")
        log_file.write(f"done in {elapsed:.0f}s\n")
        log_file.flush()

    if args.output is not None:
        np.savez(args.output,
                 **{f"coh_{key}": val for key, val in coh.items()},
                 **{f"sp_{key}": val for key, val in sp.items()})
        print(f"saved {args.output}")
        if log_file is not None:
            log_file.write(f"saved {args.output}\n")
            log_file.flush()

    if log_file is not None:
        log_file.close()
