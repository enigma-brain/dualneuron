"""Recorded macaque neuronal responses (V4 and V1).

Loads the per-session spike-count pickles in ``EXPERIMENT_DIR/{area}/trials`` and assembles a global
``(n_images, n_neurons)`` recorded-response matrix whose neuron ordering matches the released digital
twin for that area (and therefore ``twins/{Twin}/correlations.npy``): 394 neurons for V4, 458 for V1.

Conventions (matching the digital twins' training pipeline so recorded and predicted responses are
directly comparable):

* The scalar response per (unit, trial) is the spike count summed over time-bins ``SKIP_BINS:`` of
  the 12-bin window; the first bins precede the response (~80 ms latency for V4).
* For test images, responses are averaged over the repeated presentations.
* Sessions are concatenated in :data:`SESSION_ORDER` for the area, which makes the global neuron
  index reproduce that area's published per-neuron correlations (verified r = 0.998 for V4).
* Each neuron is observed only for its own session's images; unobserved entries are ``NaN``.
"""

import os
import pickle
from glob import glob
from typing import List, Tuple

import numpy as np

from dualneuron.utils import env_dir

# Per-area session order: loading sessions in this order makes the concatenated per-neuron index
# match that area's shipped correlations.npy. V4 is verified at full-set reproduction r = 0.998 (it
# equals session-id order except for four subject-34 color_render sessions interleaved as below).
# The V1 order is still to be determined (via the correlations alignment check); until it is set,
# V1 sessions load in filename order (the empty list is a no-op ranking).
SESSION_ORDER = {
    "v4": [
        3763128562108, 3764337848582, 3764941778638, 3765547012994, 3765976554128,
        3766066276625, 3766584166281, 3766672281968, 3766758352269, 3767279962643,
        3773236106262, 3773408979353, 3773928731251, 3774013234389, 3774619256850,
        3785512132205, 3785855344839, 3786637336874, 3789657204175, 3790003257664,
        3790179108293, 3790351107925, 3790611795132, 3790694306738, 3791212493372,
        3790952209225, 3791382890199, 3791300744340, 3791467334221, 3791556653552,
    ],
    "v1": [
        3631896544452, 3632669014376, 3632932714885, 3633364677437, 3634055946316,
        3634142311627, 3634658447291, 3634744023164, 3635178040531, 3635949043110,
        3636034866307, 3636552742293, 3637161140869, 3637248451650, 3637333931598,
        3637760318484, 3637851724731, 3638367026975, 3638456653849, 3638885582960,
        3638373332053, 3638541006102, 3638802601378, 3638973674012, 3639060843972,
        3639406161189, 3640011636703, 3639664527524, 3639492658943, 3639749909659,
        3640095265572, 3631807112901,
    ],
}

# Time-bins dropped before summing spike counts over the 12-bin window, per area: V4 drops the first
# two (~80 ms response latency); V1 sums all 12.
SKIP_BINS = {"v4": 2, "v1": 0}


def _session_area(sessions: List[dict]) -> str:
    """Infer the area of loaded sessions by matching their ids against SESSION_ORDER (each area's
    canonical session list). Sessions always come from load_sessions(area), so this is unambiguous."""
    ids = {int(s["session_id"]) for s in sessions}
    for area, order in SESSION_ORDER.items():
        if ids & set(order):
            return area
    return "v4"


def trials_dir(area: str = "v4", path: str = None) -> str:
    """Resolve the recorded-sessions directory for an area.

    Args:
        area: ``"v4"`` or ``"v1"``.
        path: Explicit directory of session pickles. If None, defaults to
            ``EXPERIMENT_DIR/{area}/trials``.

    Returns:
        str: Directory containing the ``*.pickle`` session files.
    """
    if path is not None:
        return path
    return os.path.join(env_dir("EXPERIMENT_DIR"), area, "trials")


def load_sessions(area: str = "v4", path: str = None) -> List[dict]:
    """Load an area's session pickles, ordered by :data:`SESSION_ORDER` for that area.

    Sessions whose id is not listed in the area's order (e.g. all of V1 until its order is set) are
    kept in filename order.

    Args:
        area: ``"v4"`` or ``"v1"``.
        path: Directory of session pickles. Defaults to ``EXPERIMENT_DIR/{area}/trials``.

    Returns:
        List[dict]: Loaded session dictionaries in the canonical neuron-ordering sequence.
    """
    order = SESSION_ORDER.get(area, [])
    sessions = []
    for p in sorted(glob(os.path.join(trials_dir(area, path), "*.pickle"))):
        with open(p, "rb") as f:
            sessions.append(pickle.load(f))
    rank = {sid: i for i, sid in enumerate(order)}
    sessions.sort(key=lambda s: rank.get(int(s["session_id"]), len(rank)))
    return sessions


def build_response_matrix(
    sessions: List[dict],
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Assemble the global ``(n_images, n_neurons)`` recorded-response matrix.

    The scalar response sums time-bins ``SKIP_BINS:``; for the test split, responses are averaged
    over repeated presentations of each image. Neurons are concatenated in session order, so the
    column index is the global twin neuron index. Area-agnostic given already-ordered ``sessions``.

    Args:
        sessions: Session dictionaries from :func:`load_sessions`.
        split: ``"train"`` or ``"test"``.

    Returns:
        Tuple of:
            * image_ids (np.ndarray): sorted unique image ids, shape ``(n_images,)``.
            * responses (np.ndarray): float32 ``(n_images, n_neurons)``, ``NaN`` where the neuron
              did not see the image.
            * neuron_meta (List[dict]): per-neuron metadata aligned to the response columns.
    """
    if split == "train":
        id_key, resp_key = "training_image_ids", "training_responses"
    elif split == "test":
        id_key, resp_key = "testing_image_ids", "testing_responses"
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    neuron_meta: List[dict] = []
    all_image_ids = set()
    for si, sess in enumerate(sessions):
        for ui in range(len(sess["unit_ids"])):
            neuron_meta.append({
                "global_idx": len(neuron_meta),
                "session_idx": si,
                "session_id": int(sess["session_id"]),
                "subject_id": int(sess["subject_id"]),
                "unit_id": int(sess["unit_ids"][ui]),
                "electrode": int(sess["electrode_nums"][ui]),
            })
        all_image_ids.update(sess[id_key].tolist())

    image_ids = np.array(sorted(all_image_ids))
    id_to_row = {int(img_id): i for i, img_id in enumerate(image_ids)}
    responses = np.full((len(image_ids), len(neuron_meta)), np.nan, dtype=np.float32)

    skip = SKIP_BINS[_session_area(sessions)]
    global_idx = 0
    for sess in sessions:
        spike_counts = sess[resp_key][:, skip:, :].sum(axis=1).astype(np.float32)  # (units, trials)
        sess_ids = sess[id_key]
        n_units = spike_counts.shape[0]
        if split == "test":
            for uid in np.unique(sess_ids):
                mask = sess_ids == uid
                row = id_to_row[int(uid)]
                for ui in range(n_units):
                    responses[row, global_idx + ui] = spike_counts[ui, mask].mean()
        else:
            for j, img_id in enumerate(sess_ids):
                responses[id_to_row[int(img_id)], global_idx:global_idx + n_units] = spike_counts[:, j]
        global_idx += n_units

    return image_ids, responses, neuron_meta


def recorded_responses(
    area: str = "v4",
    split: str = "test",
    path: str = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load the sessions and build the recorded-response matrix in one call.

    Args:
        area: ``"v4"`` or ``"v1"``.
        split: ``"train"`` or ``"test"``.
        path: Directory of session pickles. Defaults to ``EXPERIMENT_DIR/{area}/trials``.

    Returns:
        Same as :func:`build_response_matrix`: ``(image_ids, responses, neuron_meta)``.
    """
    return build_response_matrix(load_sessions(area, path), split=split)
