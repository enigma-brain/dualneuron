"""Check that the vendored twin architecture still builds the twins it is supposed to.

Run as ``python -m dualneuron.twins.verify``. Exits non-zero on the first failed check.

The architecture in :mod:`dualneuron.twins.layers` was established as bitwise identical to the
``nnvision``/``nnfabrik``/``mei``/``neuralpredictors`` stack by building every twin both ways in one
process and comparing parameters and forward outputs — 38 builds, all identical. That comparison
cannot be repeated once those packages are uninstalled, which is the whole point of vendoring, so
what it established is pinned here instead:

* the shipped weights load with an exact key match, for every staged loader and every member;
* the readout is the one shared :class:`~dualneuron.twins.layers.FullGaussian2d` in all four trained
  twins, so a cross-backbone comparison is not comparing two readout implementations;
* its channel weights initialize to the constant ``1 / channels``, the initialization the shipped
  twins were trained under — a random init here would silently change what training converges to;
* the readout samples its position in train mode and fixes it in eval mode.

What is pinned is structural — key names, shapes, counts, exact init constants — rather than digests
of forward outputs, so a PyTorch upgrade that changes the last bits of a convolution cannot turn this
into a false alarm. The one numeric comparison is internal (an ensemble's output against the mean of
its own members, computed in the same process).
"""

import hashlib

import torch

from dualneuron.twins.layers import FullGaussian2d

# Per staged loader: member count, state_dict size, a hash of the state_dict's key list, and the
# geometry the readout must end up with. Captured from the builds that were verified against nnvision.
STAGED = {
    "V4ColorTaskDriven": dict(members=5, n_keys=177, keys="fd8165b3d9ccf87b",
                              channels=1024, neurons=394, spatial=(7, 7)),
    "V4ColorDataDriven": dict(members=5, n_keys=177, keys="fd8165b3d9ccf87b",
                              channels=1024, neurons=394, spatial=(7, 7)),
    "V4GrayTaskDriven": dict(members=10, n_keys=177, keys="fd8165b3d9ccf87b",
                             channels=1024, neurons=1244, spatial=(7, 7)),
    "V1GrayTaskDriven": dict(members=5, n_keys=37, keys="01e08091841dcb5b",
                             channels=80, neurons=458, spatial=(11, 11)),
}

# Per (area, backbone) as dualneuron.training builds it at training start.
TRAINABLE = {
    ("v4", "resnet"): dict(channels=1024, neurons=394),
    ("v1", "convnext"): dict(channels=80, neurons=458),
    ("v4", "dino"): dict(channels=768, neurons=394),
    ("v1", "dino"): dict(channels=768, neurons=458),
}


def _key_hash(state_dict):
    """Hash of a state_dict's key list — structure, not values, so it is version-stable."""
    return hashlib.sha256("\n".join(state_dict).encode()).hexdigest()[:16]


def _check_readout(readout, channels, neurons, spatial=None, where=""):
    """Assert a readout is the shared class, correctly shaped, and correctly initialized."""
    assert isinstance(readout, FullGaussian2d), (
        f"{where}: readout is {type(readout).__name__}, not FullGaussian2d. Every trained twin must "
        "read out through the one shared class, or a cross-backbone comparison confounds the "
        "backbone with the readout implementation.")
    assert readout.outdims == neurons, f"{where}: {readout.outdims} neurons, expected {neurons}"
    assert readout.in_shape[0] == channels, \
        f"{where}: {readout.in_shape[0]} channels, expected {channels}"
    if spatial is not None:
        assert tuple(readout.in_shape[1:]) == spatial, \
            f"{where}: feature map {tuple(readout.in_shape[1:])}, expected {spatial}"


def check_readout_initialization():
    """The channel weights must initialize to the constant ``1 / channels``, and only that."""
    for channels, neurons in ((1024, 394), (80, 458), (768, 394), (768, 458)):
        ro = FullGaussian2d(in_shape=(channels, 7, 7), outdims=neurons, bias=True,
                            init_mu_range=0.4, init_sigma=0.6, gauss_type="isotropic")
        f = ro.state_dict()["_features"]
        assert f.unique().numel() == 1, (
            f"channel weights init to {f.unique().numel()} distinct values; nnvision's "
            f"FullGaussian2d fills a single constant. A random init changes what training converges "
            f"to and breaks comparability with the shipped twins.")
        assert torch.isclose(f.flatten()[0], torch.tensor(1.0 / channels), atol=0, rtol=1e-12), \
            f"channel weights init to {f.flatten()[0].item()}, expected 1/{channels}"

        sigma = ro.state_dict()["sigma"]
        assert sigma.unique().numel() == 1 and abs(sigma.flatten()[0].item() - 0.6) < 1e-6, \
            "sigma must init to the constant init_sigma"
        assert ro.state_dict()["bias"].abs().max().item() == 0.0, "bias must init to zero"
        assert ro.state_dict()["_mu"].abs().max().item() <= 0.4, "mu must init within init_mu_range"
    print("  [ok] readout initialization: channel weights = 1/channels, sigma constant, bias zero")


def check_readout_sampling():
    """Positions must be sampled in train mode and fixed in eval mode."""
    ro = FullGaussian2d(in_shape=(64, 5, 5), outdims=7, bias=True, init_mu_range=0.4,
                        init_sigma=0.6, gauss_type="isotropic")
    ro.eval()
    assert torch.equal(ro.sample_grid(batch_size=4), ro.sample_grid(batch_size=4)), \
        "eval mode must fix the read position at mu"
    ro.train()
    assert not torch.equal(ro.sample_grid(batch_size=4), ro.sample_grid(batch_size=4)), \
        "train mode must sample the read position from N(mu, sigma)"
    print("  [ok] readout sampling: stochastic in train, deterministic in eval")


def check_staged_loaders():
    """Every staged twin loads its shipped weights with an exact key match."""
    from dualneuron.twins import nets

    for name, expect in STAGED.items():
        loader = getattr(nets, name)
        model = loader(ensemble=True)
        assert len(model.members) == expect["members"], \
            f"{name}: {len(model.members)} members, expected {expect['members']}"

        member = model.members[0]
        sd = member.state_dict()
        assert len(sd) == expect["n_keys"], \
            f"{name}: {len(sd)} state_dict entries, expected {expect['n_keys']}"
        assert _key_hash(sd) == expect["keys"], (
            f"{name}: state_dict key layout changed (hash {_key_hash(sd)}, expected "
            f"{expect['keys']}). The shipped weights are keyed by these names.")
        _check_readout(member.readout["all_sessions"], expect["channels"], expect["neurons"],
                       expect["spatial"], where=name)

        # The ensemble averages its members, and exposes them unaveraged on request.
        n_in = member.core.input_channels if hasattr(member.core, "input_channels") else 1
        side = 93 if expect["neurons"] == 458 else 100
        x = torch.randn(2, n_in, side, side)
        model.eval()
        with torch.no_grad():
            mean = model(x, data_key="all_sessions")
            stack = model(x, data_key="all_sessions", avg=False)
        assert stack.shape == (expect["members"], 2, expect["neurons"]), \
            f"{name}: avg=False gave {tuple(stack.shape)}"
        assert torch.allclose(stack.mean(0), mean, atol=0, rtol=0), \
            f"{name}: ensemble output is not the members' mean"
        assert (mean >= 0).all(), f"{name}: predicted rates must be non-negative (ELU + 1)"
        print(f"  [ok] {name}: {expect['members']} members, {len(sd)} keys, "
              f"rates in [{mean.min():.4f}, {mean.max():.4f}]")


def check_trainable_twins():
    """Every twin dualneuron.training builds reads out through the one shared class."""
    from dualneuron.training.config import TrainConfig
    from dualneuron.training.trainer import TrainableTwin

    for (area, backbone), expect in TRAINABLE.items():
        config = TrainConfig(area=area, backbone=backbone, device="cpu")
        twin = TrainableTwin(config, seed=1, device="cpu")
        if config.kind == "dino":
            readout = twin._model.readout if config.fine_tune else twin.readout
        else:
            readout = twin._nnv.readout["all_sessions"]
        _check_readout(readout, expect["channels"], expect["neurons"],
                       where=f"{area}/{backbone}")

        f = readout.state_dict()["_features"]
        assert f.unique().numel() == 1, (
            f"{area}/{backbone}: readout channel weights are not the constant init. Both backbone "
            "families must start training from the same head.")
        # Both families' readouts expose feature_l1, which is what TrainableTwin.regularizer calls
        # with average=False -- so one regularizer expression covers both.
        assert hasattr(readout, "feature_l1"), \
            f"{area}/{backbone}: readout has no feature_l1; TrainableTwin.regularizer needs it"
        assert torch.isclose(readout.feature_l1(average=False),
                             readout.feature_l1(average=True) * f.numel()), \
            f"{area}/{backbone}: feature_l1 sum/mean are inconsistent"
        print(f"  [ok] {area}/{backbone}: FullGaussian2d, {expect['channels']} channels -> "
              f"{expect['neurons']} neurons, init 1/{expect['channels']}")


def main():
    checks = (check_readout_initialization, check_readout_sampling,
              check_trainable_twins, check_staged_loaders)
    for check in checks:
        print(f"{check.__name__}:")
        check()
    print("\nall checks passed")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    main()
