"""Train a digital-twin readout on cached frozen-core features.

The frozen core's features are extracted once (:mod:`dualneuron.training.features`); this module
trains the (cheap) trainable head on them and saves a 5-member ensemble per ``(area, backbone)`` to
``TRAINED_MODELS_DIR/{area}/{backbone}/`` (loadable via ``twins.nets.load_model(weights_dir=...)``).

The trainable head differs by backbone but follows one contract — ``BatchNorm -> (ReLU) -> Gaussian
readout -> ELU+1`` on top of the frozen feature map:

* ``dino``   — a fresh ``BatchNorm2d`` + a :class:`~dualneuron.twins.layers.FullGaussian2d` readout (no
  backbone needed at train time, since features are cached); saved as
  ``{readout_state_dict, norm_state_dict}`` for ``V4ColorDino``.
* ``resnet`` — the task-driven twin with its frozen robust core, its readout re-initialized and
  ``OutBatchNorm`` reset (= "frozen pretrained core + untrained readout"); saved as the full
  ``state_dict`` for ``V4ColorTaskDriven``.

Both regimes read out through the *same* readout class with the same initialization, so a
resnet-vs-dino comparison is a statement about backbones and not about two heads.

The objective follows nnvision's trainer defaults, which is the closest reconstruction available of
the regime the shipped twins were produced under (nnvision ships several trainer entry points; which
one was used is not recorded here): a NaN-masked Poisson NLL **summed** over observed entries and
scaled by ``sqrt(n_train / batch)``, plus ``gamma_readout`` times the
**summed** readout L1. Optimization is Adam at ``lr=5e-3`` with no gradient clipping, and
ReduceLROnPlateau on the mean validation correlation (factor 0.3, patience 5, absolute threshold
1e-6, LR floor 1e-4), stopping after ``lr_decay_steps`` reductions. The loss reduction and the
regularizer reduction have to move together — see :func:`poisson_loss`.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

from dualneuron.utils import ensure_dir, env_dir
from dualneuron.data.recordings import load_sessions, build_response_matrix
from dualneuron.training.dataset import split_train_val
from dualneuron.training.features import extract_features, cache_images, load_features

load_dotenv()


# ---------------------------------------------------------------------------
#  Loss + metrics  (faithful to the dev pipeline)
# ---------------------------------------------------------------------------

def poisson_loss(pred, target):
    """NaN-masked Poisson NLL, **summed** over observed entries.

    Summed rather than averaged because nnvision's trainers default to ``PoissonLoss(avg=False)``,
    and because it is the reduction the
    summed readout regularizer in :meth:`TrainableTwin.regularizer` is scaled against. The two must
    agree: a mean loss against a summed L1 would put the regularizer ~C*N times too high.

    Two deliberate departures from nnvision's ``PoissonLoss``. The NaN mask is unavoidable: the
    response matrix here has missing entries where nnvision's dataloaders had none. And ``pred`` is
    floored at 1e-3 before the log rather than offset by 1e-12 — ELU+1 is >= 0 but can underflow
    mid-training, and ``log(1e-12)`` produces huge spurious gradients where ``log(1e-3)`` is tame.
    """
    mask = ~torch.isnan(target)
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    nll = pred - target_clean * torch.log(pred.clamp(min=1e-3))
    return (nll * mask).sum()


def corr_per_neuron(preds, targets):
    """Per-neuron Pearson correlation, NaN-aware. Returns ``(n_neurons,)``."""
    n = preds.shape[1]
    corrs = np.full(n, np.nan)
    for i in range(n):
        m = ~np.isnan(targets[:, i])
        if m.sum() < 3:
            continue
        p, t = preds[m, i], targets[m, i]
        if p.std() > 0 and t.std() > 0:
            corrs[i] = np.corrcoef(p, t)[0, 1]
    return corrs


def compute_oracle_correlation(sessions):
    """Leave-one-out oracle correlation per neuron (Walker et al. 2019) — the noise ceiling.

    For each neuron and each repeated test image, correlate each trial's response against the mean of
    that image's other trials, pooled over all image x trial pairs.
    """
    oracle = []
    for sess in tqdm(sessions, desc="oracle (LOO)", leave=False):
        spike_counts = sess["testing_responses"][:, 2:, :].sum(axis=1).astype(np.float32)  # (units, trials)
        test_ids = sess["testing_image_ids"]
        for ui in range(spike_counts.shape[0]):
            loo_means, held_out = [], []
            for uid in np.unique(test_ids):
                idx = np.where(test_ids == uid)[0]
                if len(idx) < 2:
                    continue
                counts = spike_counts[ui, idx]
                total, n = counts.sum(), len(counts)
                for j in range(n):
                    held_out.append(counts[j])
                    loo_means.append((total - counts[j]) / (n - 1))
            a, b = np.array(loo_means), np.array(held_out)
            if len(a) >= 3 and a.std() > 0 and b.std() > 0:
                oracle.append(np.corrcoef(a, b)[0, 1])
            else:
                oracle.append(np.nan)
    return np.array(oracle)


# ---------------------------------------------------------------------------
#  Trainable twin  (frozen pretrained core + untrained readout)
# ---------------------------------------------------------------------------

class TrainableTwin:
    """One ensemble member for an ``(area, backbone)`` twin, with a uniform trainer interface.

    Two regimes, selected by ``config.fine_tune`` (equivalently ``config.cache_kind``):

    * **frozen core** (``cache_kind="features"``) — the head trains on cached frozen-core feature maps:
        - ``dino``   : fresh ``BatchNorm2d -> readout_nonlin -> Gaussian readout -> ELU(x+offset)+1``.
        - ``resnet`` : the staged nnvision twin's frozen robust core with a re-initialized head
          (OutBatchNorm reset, Gaussian readout re-initialized); the frozen core is skipped at train
          time since its features are cached.

    * **fine-tuned core** (``cache_kind="images"``) — the (truncated) backbone + head train
      end-to-end on cached images:
        - ``dino``     : a :class:`DINONeuralPredictor` whose backbone runs/tunes only blocks 0..block.
        - ``convnext`` : the full nnvision ConvNeXt twin (ImageNet backbone, trainable) + fresh readout.

    :meth:`forward` maps the cached input (feature map if frozen, image if fine-tuned) to predicted
    rates; the trainer is agnostic to which. :meth:`save` writes the format the matching loader in
    :mod:`dualneuron.twins.nets` expects. Every twin outputs ``ELU(x + elu_offset) + 1`` and uses the
    ``readout_nonlin`` (ReLU for V4, GELU for V1) so DINO mirrors its area's task-driven head.
    """

    def __init__(self, config, seed, device):
        self.config = config
        self.kind = config.kind
        self.fine_tune = config.fine_tune
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.kind == "dino":
            self._build_dino(config, device)
        elif self.kind == "nnvision":
            self._build_nnvision(config, seed, device)
        else:
            raise ValueError(f"Unknown backbone kind {self.kind!r}.")

    # -- construction ------------------------------------------------------

    def _build_dino(self, config, device):
        import torch.nn as nn
        from dualneuron.twins.dino import DINONeuralPredictor, GaussianReadout, make_nonlinearity
        from dualneuron.twins.layers import FullGaussian2d
        self._offset = config.elu_offset
        if self.fine_tune:
            # Full twin: truncated, grad-enabled backbone + BN + readout_nonlin + readout + ELU.
            self._model = DINONeuralPredictor(
                n_neurons=config.n_neurons, model_name=config.model_name,
                feature_dim=config.feature_dim, spatial_size=config.spatial_size,
                init_mu_range=config.init_mu_range, init_sigma_range=config.init_sigma_range,
                block=config.block, model_dir=config.dino_model_dir,
                readout_type=config.readout_type, readout_nonlin=config.readout_nonlin,
                elu_offset=config.elu_offset, fine_tune=True).to(device)
        else:
            # Head only (frozen backbone's features are cached): fresh BN + nonlin + readout.
            self._model = None
            self.norm = nn.BatchNorm2d(config.feature_dim, momentum=0.1).to(device)
            self.readout_nonlin = make_nonlinearity(config.readout_nonlin).to(device)
            if config.readout_type == "fullgaussian2d":
                # Same class, same init as the task-driven twins' readout: the two backbone
                # families start training from an identical head.
                self.readout = FullGaussian2d(
                    in_shape=(config.feature_dim, config.spatial_size, config.spatial_size),
                    outdims=config.n_neurons, bias=True, init_mu_range=config.init_mu_range,
                    init_sigma=config.init_sigma_range, gauss_type="isotropic").to(device)
            else:
                self.readout = GaussianReadout(
                    config.n_neurons, config.feature_dim, config.spatial_size,
                    config.init_mu_range, config.init_sigma_range).to(device)

    def _build_nnvision(self, config, seed, device):
        from dualneuron.twins.nets import load_model, build_convnext_trainable
        if self.fine_tune:
            # ConvNeXt: fresh ImageNet backbone (trainable) + fresh Gaussian readout.
            self._nnv = build_convnext_trainable(seed, device)
        else:
            # ResNet: staged twin's frozen robust core + re-initialized head (features cached).
            self._nnv = load_model(architecture=config.area, ensemble=False, untrained=False,
                                   centered=False, device=device)
            self._nnv.readout["all_sessions"].initialize()
            self._nnv.core.features.OutBatchNorm.reset_parameters()
            self._nnv.core.features.OutBatchNorm.reset_running_stats()

    # -- forward -----------------------------------------------------------

    def forward(self, x):
        """Predicted rates from the cached input (feature map if frozen, image if fine-tuned)."""
        if self.kind == "dino":
            if self.fine_tune:
                return self._model(x)
            y = self.readout(self.readout_nonlin(self.norm(x)))
            return F.elu(y + self._offset) + 1
        # nnvision
        if self.fine_tune:
            return self._nnv(x, data_key="all_sessions")            # full model on images
        # Frozen: apply the post-core head to cached features (skip the frozen core).
        for name, child in self._nnv.core.features.named_children():
            if name in ("TaskDriven", "backbone"):
                continue
            x = child(x)
        x = self._nnv.readout(x, data_key="all_sessions")
        return F.elu(x + self._nnv.offset) + 1

    # -- params / regularizer / modes --------------------------------------

    def trainable_parameters(self):
        if self.kind == "dino":
            if self.fine_tune:
                return [p for p in self._model.parameters() if p.requires_grad]
            return list(self.norm.parameters()) + list(self.readout.parameters())
        return [p for p in self._nnv.parameters() if p.requires_grad]

    def regularizer(self):
        # gamma_readout * sum(|readout channel weights|), the convention the shipped twins were
        # trained under (nnvision's MultipleFullGaussian2d.regularizer takes feature_l1(average=False)),
        # paired with the summed Poisson loss below. Written once for both backbone families, since
        # both readouts expose feature_l1. config.gamma_readout rather than the value baked into the
        # readout, so --gamma_readout still overrides it.
        ro = ((self._model.readout if self.fine_tune else self.readout) if self.kind == "dino"
              else self._nnv.readout["all_sessions"])
        return self.config.gamma_readout * ro.feature_l1(average=False)

    def train_mode(self):
        if self.kind == "dino":
            if self.fine_tune:
                self._model.train()          # DINOv3Core.train keeps the backbone in eval (grad still flows)
            else:
                self.norm.train(); self.readout.train()
        elif self.fine_tune:
            self._nnv.train()
        else:
            self._nnv.core.features.OutBatchNorm.train()
            self._nnv.readout.train()

    def eval_mode(self):
        if self.kind == "dino":
            if self.fine_tune:
                self._model.eval()
            else:
                self.norm.eval(); self.readout.eval()
        elif self.fine_tune:
            self._nnv.eval()
        else:
            self._nnv.core.features.OutBatchNorm.eval()
            self._nnv.readout.eval()

    # -- persistence -------------------------------------------------------

    def state(self):
        """In-memory CPU snapshot of the trainable state (for keeping the best epoch)."""
        if self.kind == "dino" and self.fine_tune:
            return {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
        if self.kind == "dino":
            return {"norm": {k: v.detach().cpu().clone() for k, v in self.norm.state_dict().items()},
                    "readout": {k: v.detach().cpu().clone() for k, v in self.readout.state_dict().items()}}
        return {k: v.detach().cpu().clone() for k, v in self._nnv.state_dict().items()}

    def load_state(self, snap):
        """Restore a snapshot produced by :meth:`state`."""
        if self.kind == "dino" and self.fine_tune:
            self._model.load_state_dict(snap)
        elif self.kind == "dino":
            self.norm.load_state_dict(snap["norm"])
            self.readout.load_state_dict(snap["readout"])
        else:
            self._nnv.load_state_dict(snap)

    def save(self, path):
        """Write the member weights in the format its loader expects."""
        if self.kind == "dino":
            meta = dict(block=self.config.block, readout_type=self.config.readout_type,
                        readout_nonlin=self.config.readout_nonlin, elu_offset=self.config.elu_offset,
                        fine_tune=self.fine_tune)
            if self.fine_tune:
                torch.save({"model_state_dict": self._model.state_dict(), **meta}, path)
            else:
                torch.save({"readout_state_dict": self.readout.state_dict(),
                            "norm_state_dict": self.norm.state_dict(), **meta}, path)
        else:
            torch.save(self._nnv.state_dict(), path)


def build_trainable_twin(config, seed, device):
    """Build one ensemble member's trainable twin (see :class:`TrainableTwin`)."""
    return TrainableTwin(config, seed, device)


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

class _Log:
    """Tiny logger: print to stdout and (if a path is given) append to a LOGS_DIR file."""

    def __init__(self, path):
        self.fh = None
        if path:
            ensure_dir(Path(path).parent)
            self.fh = open(path, "w")

    def __call__(self, msg):
        print(msg, flush=True)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def _predict_loader(twin, loader, device):
    """Predicted rates over a DataLoader of cached features -> ``(n, n_neurons)``."""
    twin.eval_mode()
    out = []
    with torch.no_grad():
        for feat, _ in loader:
            out.append(twin.forward(feat.to(device, non_blocking=True).float()).detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, twin.config.n_neurons), np.float32)


def _loaders(config, train_feat, train_resp, train_idx, val_idx, test_feat, test_resp):
    """Build train/val/test DataLoaders over the cached features (the unified fast path).

    The cache is served via ``CachedFeatureDataset`` with ``pin_memory=True`` (async ``non_blocking``
    host->device copy) and ``num_workers`` parallel row-gather — the actual bottleneck, since the
    readout head itself is tiny. The in-RAM feature array is shared copy-on-write across workers, so
    even DINO's 22 GB cache is not duplicated per worker. Size-agnostic: identical for ResNet (6.9 GB)
    and DINO (22 GB) — no per-backbone branching.
    """
    from torch.utils.data import DataLoader, Subset
    from dualneuron.training.dataset import CachedFeatureDataset
    nw = config.num_workers
    common = dict(pin_memory=True, num_workers=nw, persistent_workers=nw > 0)
    full_train = CachedFeatureDataset(train_feat, train_resp)
    train_loader = DataLoader(Subset(full_train, list(train_idx)), batch_size=config.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Subset(full_train, list(val_idx)), batch_size=256, shuffle=False, **common)
    test_loader = DataLoader(CachedFeatureDataset(test_feat, test_resp), batch_size=256,
                             shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _train_one(config, seed, train_feat, train_resp, train_idx, val_idx,
               test_feat, test_resp, device, log):
    """Train one ensemble member's head on cached features; return (twin, best_val, test_corr, test_preds)."""
    twin = build_trainable_twin(config, seed, device)
    params = twin.trainable_parameters()
    opt = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    # Absolute threshold and a floor on the LR, as nnvision's trainer configured it: a relative
    # threshold would scale the "is this an improvement?" test with the correlation itself.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=config.lr_decay_factor, patience=config.lr_decay_patience,
        threshold=config.lr_threshold, threshold_mode="abs", min_lr=config.min_lr)

    train_loader, val_loader, test_loader = _loaders(
        config, train_feat, train_resp, train_idx, val_idx, test_feat, test_resp)
    val_targets = train_resp[val_idx]
    # The summed Poisson loss is scaled by sqrt(n_train / batch) so the objective a single batch
    # reports is comparable to the whole training set's -- nnvision's `scale_loss`.
    n_train = len(train_loader.dataset)
    best_val, best_snap, n_decays, prev_lr = -1.0, None, 0, config.lr

    for epoch in range(1, config.max_epochs + 1):
        twin.train_mode()
        ep_loss, n_batches = 0.0, 0
        for feat, t in train_loader:
            f = feat.to(device, non_blocking=True).float()
            t = t.to(device, non_blocking=True)
            loss_scale = math.sqrt(n_train / f.shape[0])
            loss = loss_scale * poisson_loss(twin.forward(f), t) + twin.regularizer()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            n_batches += 1
        avg_loss = ep_loss / max(n_batches, 1)

        val = corr_per_neuron(_predict_loader(twin, val_loader, device), val_targets)
        valid = ~np.isnan(val)
        val_mean = float(val[valid].mean()) if valid.any() else 0.0
        val_med = float(np.median(val[valid])) if valid.any() else 0.0

        sched.step(val_mean)
        lr = opt.param_groups[0]["lr"]
        if lr < prev_lr:
            n_decays += 1
            prev_lr = lr
        if val_mean > best_val:
            best_val, best_snap = val_mean, twin.state()
        log(f"  seed {seed} ep {epoch:3d}  loss={avg_loss:.4f}  val={val_mean:.4f}/{val_med:.4f}  lr={lr:.1e}")
        if n_decays >= config.lr_decay_steps:
            log(f"  seed {seed}: early stop after {n_decays} LR decays (epoch {epoch})")
            break

    if best_snap is not None:
        twin.load_state(best_snap)
    test_preds = _predict_loader(twin, test_loader, device)
    return twin, best_val, corr_per_neuron(test_preds, test_resp), test_preds


def _load_training_data(config, device, rewrite=False, log=None):
    """Load sessions/responses and the cached trainable-part input (extract/cache once if needed).

    ``cache_kind="features"`` (frozen core) caches frozen-core feature maps; ``"images"`` (fine-tuned
    core) caches the transformed input images. The returned ``train_feat``/``test_feat`` are that
    cached input (feature maps or images); :meth:`TrainableTwin.forward` consumes either. ``rewrite``
    forces the cache to be recomputed (e.g. after a transform change); the multi-GPU pool rebuilds the
    cache once up front and leaves this at the default so members reuse it.
    """
    sessions = load_sessions(config.area)
    train_ids, train_resp, _ = build_response_matrix(sessions, "train")
    test_ids, test_resp, _ = build_response_matrix(sessions, "test")
    if config.cache_kind == "images":
        train_feat = load_features(cache_images(config, train_ids, "train", rewrite=rewrite),
                                   log=log, desc="load train images")
        test_feat = load_features(cache_images(config, test_ids, "test", rewrite=rewrite),
                                  log=log, desc="load test images")
    else:
        train_feat = load_features(extract_features(config, train_ids, "train", device=device, rewrite=rewrite),
                                   log=log, desc="load train features")
        test_feat = load_features(extract_features(config, test_ids, "test", device=device, rewrite=rewrite),
                                  log=log, desc="load test features")
    return dict(sessions=sessions, train_ids=train_ids, train_resp=train_resp,
                test_ids=test_ids, test_resp=test_resp, train_feat=train_feat, test_feat=test_feat)


def _member_path(config, seed):
    return os.path.join(config.trained_dir, f"member_{seed}.pth.tar")


def _save_ensemble_correlations(config, ens_preds, recorded_avg, log):
    """Save the ensemble's per-neuron corr-to-average (the verified eval metric).

    The RF mask is NOT written here: non-staged twins regenerate their own mask from their own
    MEIs via ``dualneuron.synthesis.mask`` -> ANALYSIS_DIR/{area}/{backbone}/mask.npy (which is what
    ``registry.mask_path`` reads); staged twins keep their read-only shipped mask.
    """
    from dualneuron.training.eval_ensemble import correlation_to_average
    corr = correlation_to_average(ens_preds, recorded_avg)
    ensure_dir(config.trained_dir)
    np.save(os.path.join(config.trained_dir, "correlations.npy"), corr)
    log(f"ensemble {config.area}/{config.backbone}: corr-to-avg mean={np.nanmean(corr):.4f}  "
        f"median={np.nanmedian(corr):.4f}  n>0.4={int(np.nansum(corr > 0.4))}  -> {config.trained_dir}")
    return corr


def train_member(config, seed, device=None, log_path=None, rewrite=False):
    """Train and save ONE ensemble member (used by the multi-GPU pool; also runnable standalone).

    Extracts features (idempotent) and trains the head for ``seed`` on them; writes
    ``member_{seed}.pth.tar`` under ``trained_dir``. Does NOT write correlations.npy — call
    :func:`aggregate_ensemble` once all members exist. ``rewrite`` forces the input cache to be
    recomputed; the pool passes it only to its up-front cache step, not to the per-member workers.
    """
    device = device or (config.device if torch.cuda.is_available() else "cpu")
    if log_path is None and config.logs_dir:
        log_path = os.path.join(config.logs_dir, config.area, config.backbone, f"seed{seed}.log")
    log = _Log(log_path)
    try:
        log(f"  seed {seed}: loading cached input into RAM ...")
        d = _load_training_data(config, device, rewrite=rewrite, log=log)
        log(f"  seed {seed}: loaded train {tuple(d['train_feat'].shape)} test {tuple(d['test_feat'].shape)}; training ...")
        tr_idx, va_idx = split_train_val(d["train_ids"], config.val_fraction, seed)
        twin, best_val, test_corr, _ = _train_one(
            config, seed, d["train_feat"], d["train_resp"], tr_idx, va_idx,
            d["test_feat"], d["test_resp"], device, log)
        ensure_dir(config.trained_dir)
        path = _member_path(config, seed)
        twin.save(path)
        log(f"  seed {seed}: best_val={best_val:.4f}  test_mean={np.nanmean(test_corr):.4f}  -> {path}")
        return path
    finally:
        log.close()


def aggregate_ensemble(config, device=None, log_path=None):
    """Compute + save ``correlations.npy`` (+ mask) from the already-saved members.

    Used after a parallel run (members trained in separate processes): loads the full ensemble via
    the verified evaluator and writes per-neuron corr-to-average.
    """
    from dualneuron.training.eval_ensemble import predict
    device = device or (config.device if torch.cuda.is_available() else "cpu")
    if log_path is None and config.logs_dir:
        log_path = os.path.join(config.logs_dir, config.area, config.backbone, "aggregate.log")
    log = _Log(log_path)
    try:
        sessions = load_sessions(config.area)
        test_ids, recorded_avg, _ = build_response_matrix(sessions, "test")
        ens_preds = predict(config, config.trained_dir, test_ids, device)
        return _save_ensemble_correlations(config, ens_preds, recorded_avg, log)
    finally:
        log.close()


def train_ensemble(config, seeds=(1, 2, 3, 4, 5), device=None, log_path=None, rewrite=False):
    """Train a 5-member ensemble sequentially and save it to ``TRAINED_MODELS_DIR``.

    Extracts the frozen-core features once, trains each member's head on them (writing
    ``member_{seed}.pth.tar``), then writes the ensemble ``correlations.npy`` and the
    shared ``mask.npy``. For multi-GPU, use :func:`train_member` per process + :func:`aggregate_ensemble`.
    ``rewrite`` forces the shared input cache to be recomputed once before training.
    """
    device = device or (config.device if torch.cuda.is_available() else "cpu")
    if log_path is None and config.logs_dir:
        log_path = os.path.join(config.logs_dir, config.area, config.backbone, "train.log")
    log = _Log(log_path)
    try:
        d = _load_training_data(config, device, rewrite=rewrite, log=log)
        oracle_med = float(np.nanmedian(compute_oracle_correlation(d["sessions"])))
        ensure_dir(config.trained_dir)
        log(f"train {config.area}/{config.backbone}: {config.n_neurons} neurons, "
            f"{len(d['train_ids'])} train / {len(d['test_ids'])} test imgs, oracle median={oracle_med:.4f}")

        member_preds = []
        for seed in seeds:
            tr_idx, va_idx = split_train_val(d["train_ids"], config.val_fraction, seed)
            twin, best_val, test_corr, test_preds = _train_one(
                config, seed, d["train_feat"], d["train_resp"], tr_idx, va_idx,
                d["test_feat"], d["test_resp"], device, log)
            twin.save(_member_path(config, seed))
            member_preds.append(test_preds)
            log(f"  seed {seed}: best_val={best_val:.4f}  test_mean={np.nanmean(test_corr):.4f}")

        return _save_ensemble_correlations(config, np.mean(member_preds, 0), d["test_resp"], log)
    finally:
        log.close()
