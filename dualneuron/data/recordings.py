"""Recorded macaque V4 neuronal responses.

Loads the per-session spike-count pickles in ``EXPERIMENT_DIR/all_trials`` and assembles a
global ``(n_images, n_neurons)`` recorded-response matrix whose neuron ordering matches the
released 394-neuron V4 digital twin (and therefore ``twins/V4ColorTaskDriven/correlations.npy``).
Only macaque V4 was recorded in this dataset; there is no V1.

Conventions (matching the digital twin's training pipeline so recorded and predicted responses
are directly comparable):

* The scalar response per (unit, trial) is the spike count summed over time-bins ``2:`` of the
  12-bin window; the first two bins precede the V4 response (~80 ms latency).
* For test images, responses are averaged over the repeated presentations (typically 20).
* Sessions are concatenated in :data:`SESSION_ORDER`, which makes the global neuron index
  reproduce the published per-neuron correlations (verified r = 0.998 on the full set).
* Each neuron is observed only for its own session's images; unobserved entries are ``NaN``.
"""

import os
import pickle
from glob import glob
from typing import List, Tuple

import numpy as np

from dualneuron.utils import env_dir

# Session order of the released 394-neuron V4 model. Loading sessions in this order makes the
# per-neuron index match the shipped correlations.npy (verified: full-set reproduction r = 0.998).
# It equals session-id order except for the four subject-34 color_render sessions (entries 25-28),
# which the model interleaves in this specific order.
SESSION_ORDER = [
    3763128562108, 3764337848582, 3764941778638, 3765547012994, 3765976554128,
    3766066276625, 3766584166281, 3766672281968, 3766758352269, 3767279962643,
    3773236106262, 3773408979353, 3773928731251, 3774013234389, 3774619256850,
    3785512132205, 3785855344839, 3786637336874, 3789657204175, 3790003257664,
    3790179108293, 3790351107925, 3790611795132, 3790694306738, 3791212493372,
    3790952209225, 3791382890199, 3791300744340, 3791467334221, 3791556653552,
]

# First two of the 12 time-bins precede the V4 response (~80 ms latency) and are dropped.
SKIP_BINS = 2


def all_trials_dir(path: str = None) -> str:
    """Resolve the recordings directory.

    Args:
        path: Explicit directory of session pickles. If None, defaults to
            ``EXPERIMENT_DIR/all_trials``.

    Returns:
        str: Path to the directory containing the ``*.pickle`` session files.
    """
    if path is not None:
        return path
    return os.path.join(env_dir("EXPERIMENT_DIR"), "all_trials")


def load_sessions(path: str = None) -> List[dict]:
    """Load all V4 session pickles, ordered by :data:`SESSION_ORDER`.

    Sessions whose id is not listed in :data:`SESSION_ORDER` are appended in filename order.

    Args:
        path: Directory of session pickles. Defaults to ``EXPERIMENT_DIR/all_trials``.

    Returns:
        List[dict]: Loaded session dictionaries in the canonical neuron-ordering sequence.
    """
    sessions = []
    for p in sorted(glob(os.path.join(all_trials_dir(path), "*.pickle"))):
        with open(p, "rb") as f:
            sessions.append(pickle.load(f))
    rank = {sid: i for i, sid in enumerate(SESSION_ORDER)}
    sessions.sort(key=lambda s: rank.get(int(s["session_id"]), len(rank)))
    return sessions


def build_response_matrix(
    sessions: List[dict],
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Assemble the global ``(n_images, n_neurons)`` recorded-response matrix.

    The scalar response sums time-bins ``2:`` (see :data:`SKIP_BINS`); for the test split,
    responses are averaged over repeated presentations of each image. Neurons are concatenated
    in session order, so the column index is the global twin neuron index.

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

    global_idx = 0
    for sess in sessions:
        spike_counts = sess[resp_key][:, SKIP_BINS:, :].sum(axis=1).astype(np.float32)  # (units, trials)
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
    split: str = "test",
    path: str = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load the sessions and build the recorded-response matrix in one call.

    Args:
        split: ``"train"`` or ``"test"``.
        path: Directory of session pickles. Defaults to ``EXPERIMENT_DIR/all_trials``.

    Returns:
        Same as :func:`build_response_matrix`: ``(image_ids, responses, neuron_meta)``.
    """
    return build_response_matrix(load_sessions(path), split=split)
