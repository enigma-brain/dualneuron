"""Download + convert the gated DINOv3 backbone weights (one-time, per variant).

Reproduces the dualneuron-dev procedure, but with every path under the env ``MODELS_DIR/dinov3``:

1. The hubconf repo must be cloned (manually — it is license-gated) to
   ``MODELS_DIR/dinov3/facebookresearch_dinov3_main`` (``git clone https://github.com/facebookresearch/dinov3``).
2. The gated HF weights ``facebook/dinov3-vitb16-pretrain-lvd1689m/model.safetensors`` are fetched
   with ``huggingface_hub`` (same as dev — note the ``.cache/`` it leaves) into
   ``MODELS_DIR/dinov3/hf_download/`` using ``HF_TOKEN`` (requires accepting the license on HF).
3. This script maps the HF ``Dinov3Model`` keys to the Meta ``DinoVisionTransformer`` layout (fusing
   q/k/v into a single qkv) and writes the converted checkpoint to
   ``MODELS_DIR/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-<hash>.pth`` — exactly the path and
   filename :class:`dualneuron.twins.dino.DINOv3Core` loads.

    python -m dualneuron.training.convert_dinov3_weights              # download (if needed) + convert
    python -m dualneuron.training.convert_dinov3_weights --no_download # convert a pre-downloaded file

The key-mapping / QKV-fusion logic is a faithful copy of the dev converter (it is verified correct).
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from dotenv import load_dotenv

from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins.dino import DINOV3_WEIGHT_HASHES

load_dotenv()

HF_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_FILENAME = "model.safetensors"
HUB_MODEL = "dinov3_vitb16"
N_BLOCKS = 12


def _dinov3_dir() -> str:
    """``MODELS_DIR/dinov3`` (the env-set models directory)."""
    models_dir = env_dir("MODELS_DIR")
    if not models_dir:
        raise ValueError("MODELS_DIR is not set in the environment (.env).")
    return os.path.join(models_dir, "dinov3")


def _rename_hf_key(k: str):
    """HF ``Dinov3Model`` key -> Meta ``DinoVisionTransformer`` key (1-to-1 keys only).

    Q/K/V projections are NOT handled here — those need fusion, handled separately.
    """
    if k == "embeddings.cls_token":               return "cls_token"
    if k == "embeddings.mask_token":              return "mask_token"
    if k == "embeddings.register_tokens":         return "storage_tokens"
    if k == "embeddings.patch_embeddings.weight": return "patch_embed.proj.weight"
    if k == "embeddings.patch_embeddings.bias":   return "patch_embed.proj.bias"
    if k == "norm.weight":                        return "norm.weight"
    if k == "norm.bias":                          return "norm.bias"

    if k.startswith("layer."):
        parts = k.split(".")
        b = parts[1]
        sub = ".".join(parts[2:])
        m = {
            "norm1.weight":            f"blocks.{b}.norm1.weight",
            "norm1.bias":              f"blocks.{b}.norm1.bias",
            "norm2.weight":            f"blocks.{b}.norm2.weight",
            "norm2.bias":              f"blocks.{b}.norm2.bias",
            "attention.o_proj.weight": f"blocks.{b}.attn.proj.weight",
            "attention.o_proj.bias":   f"blocks.{b}.attn.proj.bias",
            "layer_scale1.lambda1":    f"blocks.{b}.ls1.gamma",
            "layer_scale2.lambda1":    f"blocks.{b}.ls2.gamma",
            "mlp.up_proj.weight":      f"blocks.{b}.mlp.fc1.weight",
            "mlp.up_proj.bias":        f"blocks.{b}.mlp.fc1.bias",
            "mlp.down_proj.weight":    f"blocks.{b}.mlp.fc2.weight",
            "mlp.down_proj.bias":      f"blocks.{b}.mlp.fc2.bias",
        }
        if sub in m:
            return m[sub]
    return None  # signals "needs fusion" or "unknown"


def _fuse_qkv_block(hf_sd, block_idx, target_dim=2304):
    """Concat q, k, v along the output dim (dim=0) in order [Q, K, V] for one block.

    Meta hubconf reshapes qkv -> (B,N,3,heads,head_dim) -> unbind(dim=2), so output dim is Q,K,V.
    HF omits the K bias (Meta masks it via ``mask_k_bias=True``); we use zeros for that slice.
    """
    qw = hf_sd[f"layer.{block_idx}.attention.q_proj.weight"]
    kw = hf_sd[f"layer.{block_idx}.attention.k_proj.weight"]
    vw = hf_sd[f"layer.{block_idx}.attention.v_proj.weight"]
    qb = hf_sd[f"layer.{block_idx}.attention.q_proj.bias"]
    kb = torch.zeros_like(qb)
    vb = hf_sd[f"layer.{block_idx}.attention.v_proj.bias"]

    weight = torch.cat([qw, kw, vw], dim=0)
    bias = torch.cat([qb, kb, vb], dim=0)
    assert weight.shape[0] == target_dim, f"got {weight.shape}, expected ({target_dim},*)"
    return weight, bias


def download_hf_weights(hf_download_dir: str) -> str:
    """Fetch the gated HF safetensors via huggingface_hub (the dev method), return its path."""
    from huggingface_hub import hf_hub_download
    ensure_dir(hf_download_dir)
    print(f"Downloading {HF_REPO}/{HF_FILENAME} -> {hf_download_dir} …", flush=True)
    path = hf_hub_download(
        repo_id=HF_REPO, filename=HF_FILENAME,
        local_dir=hf_download_dir, token=os.getenv("HF_TOKEN"),
    )
    print(f"  got {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Download + convert gated DINOv3 weights")
    parser.add_argument("--no_download", action="store_true",
                        help="Skip the HF download; convert a pre-downloaded model.safetensors.")
    parser.add_argument("--hub_model", default=HUB_MODEL)
    parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    args = parser.parse_args()

    from safetensors.torch import load_file

    base = _dinov3_dir()
    repo_path = os.path.join(base, "facebookresearch_dinov3_main")
    hf_download_dir = os.path.join(base, "hf_download")
    hf_file = os.path.join(hf_download_dir, HF_FILENAME)
    out_path = os.path.join(
        base, "checkpoints",
        f"{args.hub_model}_pretrain_lvd1689m-{DINOV3_WEIGHT_HASHES[args.hub_model]}.pth")

    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"DINOv3 hubconf repo not found at {repo_path}.\n"
            f"Clone it (license-gated): git clone https://github.com/facebookresearch/dinov3 {repo_path}")

    if not args.no_download and not os.path.isfile(hf_file):
        download_hf_weights(hf_download_dir)
    if not os.path.isfile(hf_file):
        raise FileNotFoundError(
            f"HF weights not found at {hf_file}. Run without --no_download (needs HF_TOKEN and "
            f"accepting the license at https://huggingface.co/{HF_REPO}).")

    print(f"Loading HF weights from {hf_file} …", flush=True)
    hf_sd = load_file(hf_file)
    print(f"  {len(hf_sd)} keys, {sum(v.numel() for v in hf_sd.values()):,} params", flush=True)

    print(f"Building fresh Meta model ({args.hub_model}) …", flush=True)
    model = torch.hub.load(repo_path, args.hub_model, source="local", pretrained=False)
    model.init_weights()  # populates RoPE periods, qkv bias_mask
    target_sd = model.state_dict()
    print(f"  {len(target_sd)} target keys", flush=True)

    converted = dict(target_sd)
    overridden_keys = set()
    unmapped = []

    # 1-to-1 keys
    for k, v in hf_sd.items():
        if "_proj." in k and "attention." in k and "o_proj" not in k:
            continue  # q/k/v handled below
        new_key = _rename_hf_key(k)
        if new_key is None:
            unmapped.append(k)
            continue
        if new_key not in converted:
            print(f"  ! target missing key for {k} -> {new_key}", flush=True)
            unmapped.append(k)
            continue
        if converted[new_key].shape != v.shape:
            if v.numel() == converted[new_key].numel():
                v = v.reshape(converted[new_key].shape)
            else:
                raise ValueError(
                    f"shape mismatch: {k}{tuple(v.shape)} -> "
                    f"{new_key}{tuple(converted[new_key].shape)}")
        converted[new_key] = v
        overridden_keys.add(new_key)

    # Fused QKV per block
    for b in range(args.n_blocks):
        qkv_w, qkv_b = _fuse_qkv_block(hf_sd, b)
        wk, bk = f"blocks.{b}.attn.qkv.weight", f"blocks.{b}.attn.qkv.bias"
        if converted[wk].shape != qkv_w.shape:
            raise ValueError(f"qkv weight shape mismatch at block {b}: "
                             f"{tuple(qkv_w.shape)} vs target {tuple(converted[wk].shape)}")
        converted[wk] = qkv_w
        converted[bk] = qkv_b
        overridden_keys.add(wk)
        overridden_keys.add(bk)

    print(f"  overrode {len(overridden_keys)} target keys from HF", flush=True)
    if unmapped:
        print(f"  WARN: {len(unmapped)} HF keys not mapped: {unmapped[:5]}…", flush=True)

    # Every non-overridden key must be a deterministic buffer from init_weights().
    not_overridden = set(target_sd) - overridden_keys
    expected_init_only = (
        {"rope_embed.periods"} |
        {f"blocks.{b}.attn.qkv.bias_mask" for b in range(args.n_blocks)}
    )
    unexpected = not_overridden - expected_init_only
    if unexpected:
        print(f"  ! UNEXPECTED keys not overridden: {sorted(unexpected)}", flush=True)
        sys.exit(1)
    print(f"  {len(not_overridden)} keys kept from init (deterministic buffers)", flush=True)

    # Hard-verify the load against a fresh model.
    fresh = torch.hub.load(repo_path, args.hub_model, source="local", pretrained=False)
    fresh.init_weights()
    missing, unexpected_keys = fresh.load_state_dict(converted, strict=False)
    if missing or unexpected_keys:
        print(f"  ! load_state_dict check: missing={missing}, unexpected={unexpected_keys}", flush=True)
        sys.exit(1)

    ensure_dir(os.path.dirname(out_path))
    torch.save(converted, out_path)
    print(f"wrote {out_path}", flush=True)
    print(f"  size: {os.path.getsize(out_path) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
