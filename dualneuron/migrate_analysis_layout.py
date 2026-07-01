"""One-time migration of ANALYSIS_DIR from the old flat ``{area}/`` layout to ``{area}/{backbone}/``.

Runs made before the pipeline became ``(area, backbone)``-aware wrote every twin's outputs directly
under ``ANALYSIS_DIR/{area}/`` with an ``{area}_`` filename prefix. Those results are all from each
area's **staged** twin (v4 → resnet, v1 → convnext), so this relocates them into the new layout
``ANALYSIS_DIR/{area}/{backbone}/`` with the redundant prefix stripped (matching how the refactored
pipeline reads/writes), and they are reused rather than recomputed.

Safe + idempotent: it moves within ``ANALYSIS_DIR`` (same filesystem → atomic per-file rename), skips
any target that already exists, only touches ``ANALYSIS_DIR`` (never the read-only ``twins/`` tree),
and leaves unrecognized files in place.

    python -m dualneuron.migrate_analysis_layout            # dry run (prints the planned moves)
    python -m dualneuron.migrate_analysis_layout --apply    # execute
"""
import argparse
import os

from dotenv import load_dotenv
load_dotenv()

from dualneuron.utils import env_dir, ensure_dir
from dualneuron.twins import registry


def staged_backbone(area):
    """The single staged backbone of an area (the one shipped in ``twins/``), or None if not unique.

    The old flat ``{area}/`` files were produced by exactly this twin.
    """
    staged = [b for (a, b) in registry.TWINS if a == area and registry.resolve(a, b).staged_folder]
    return staged[0] if len(staged) == 1 else None


def plan_moves(area, backbone):
    """(src, dst) moves for one area: old flat ``{area}/{area}_*`` → ``{area}/{backbone}/*`` (prefix
    stripped), including the ``synthesis/`` subfolder. Returns the moves and any unrecognized files."""
    ad = env_dir("ANALYSIS_DIR")
    old_dir = os.path.join(ad, area)
    new_dir = registry.analysis_dir(area, backbone)
    prefix = f"{area}_"
    moves, skipped = [], []

    for name in sorted(os.listdir(old_dir)):
        src = os.path.join(old_dir, name)
        if os.path.isdir(src):                       # the new {backbone}/ or synthesis/ subdir
            continue
        if name.startswith(prefix):
            moves.append((src, os.path.join(new_dir, name[len(prefix):])))
        else:
            skipped.append(src)

    old_syn = os.path.join(old_dir, "synthesis")
    if os.path.isdir(old_syn):
        new_syn = os.path.join(new_dir, "synthesis")
        for name in sorted(os.listdir(old_syn)):
            src = os.path.join(old_syn, name)
            if os.path.isfile(src) and name.startswith(prefix):
                moves.append((src, os.path.join(new_syn, name[len(prefix):])))
            elif os.path.isfile(src):
                skipped.append(src)

    return moves, skipped


def main():
    parser = argparse.ArgumentParser(description="Migrate ANALYSIS_DIR to the {area}/{backbone}/ layout")
    parser.add_argument("--apply", action="store_true", help="execute the moves (default: dry run)")
    args = parser.parse_args()

    ad = env_dir("ANALYSIS_DIR")
    if ad is None:
        raise ValueError("ANALYSIS_DIR is not set (see .env).")

    planned = moved = existed = 0
    for area in registry.AREAS:
        old_dir = os.path.join(ad, area)
        if not os.path.isdir(old_dir):
            continue
        backbone = staged_backbone(area)
        if backbone is None:
            print(f"{area}: no unique staged backbone; skipping")
            continue
        moves, skipped = plan_moves(area, backbone)
        print(f"\n{area} -> {area}/{backbone}: {len(moves)} files"
              + (f"  ({len(skipped)} unrecognized, left in place)" if skipped else ""))
        for src, dst in moves:
            rel = f"{os.path.relpath(src, ad)} -> {os.path.relpath(dst, ad)}"
            if os.path.exists(dst):
                print(f"  skip (target exists): {rel}")
                existed += 1
                continue
            print(f"  {'move' if args.apply else 'would move'}: {rel}")
            planned += 1
            if args.apply:
                ensure_dir(os.path.dirname(dst))
                os.rename(src, dst)
                moved += 1
        for src in skipped:
            print(f"  ? unrecognized (kept): {os.path.relpath(src, ad)}")
        # Remove the now-empty old synthesis/ dir on apply.
        old_syn = os.path.join(old_dir, "synthesis")
        if args.apply and os.path.isdir(old_syn) and not os.listdir(old_syn):
            os.rmdir(old_syn)

    if args.apply:
        print(f"\nmoved {moved} files ({existed} skipped, already at target).")
    else:
        print(f"\nplanned {planned} moves ({existed} already at target). Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
