#!/usr/bin/env python3
"""Export rod-packing trajectories to a standalone web viewer.

By default produces a single self-contained index.html with all data
embedded as base64 — open it directly in any browser, no server needed.

Usage
-----
python -m entanglement_optimization.viz.web_export <run_dir> [--output <dir>] [--every N]

Examples
--------
python -m entanglement_optimization.viz.web_export results/N200/2026-03-19_19_Relaxed-N200
python -m entanglement_optimization.viz.web_export .../run_dir --output ~/Desktop/viewer --every 2
"""
import base64
import json
import sys
import argparse
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export rod-packing trajectories to a standalone web viewer."
    )
    p.add_argument("run_dir",
                   help="Path to run directory, e.g. .../2026-03-19_19_Relaxed-N200")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: <run_dir>/web/)")
    p.add_argument("--every", type=int, default=1,
                   help="Keep every Nth frame per AR step (default: 1, keep all)")
    p.add_argument("--entangle-frames", type=int, default=300,
                   help="Use only the first N frames of the entanglement trajectory "
                        "(stride=1, independent of --every; default: 300)")
    p.add_argument("--entangle-traj", type=str, default=None,
                   help="Path to entanglement trajectory.npy to prepend "
                        "(auto-detected from <run_dir>/../trajectory.npy if present)")
    p.add_argument("--entangle-ar", type=int, default=1000,
                   help="AR used during entanglement (sets rod radius, default 1000)")
    return p.parse_args()


def build_standalone(viewer_src: Path, manifest: dict, combined: np.ndarray) -> str:
    """Return viewer HTML with trajectory data embedded as base64."""
    html = viewer_src.read_text()

    b64 = base64.b64encode(combined.tobytes()).decode('ascii')
    inject = (
        "<script>\n"
        f"window.EMBEDDED_DATA = {{\n"
        f"  manifest: {json.dumps(manifest)},\n"
        f"  trajectory: \"{b64}\"\n"
        f"}};\n"
        "</script>\n"
    )
    return html.replace("</head>", inject + "</head>", 1)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    output_dir = Path(args.output).resolve() if args.output else run_dir / "web"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect trajectories: largest AR first (thinnest -> thickest)
    traj_paths = sorted(run_dir.glob("AR*/trajectory.npy"))
    if not traj_paths:
        sys.exit(f"No AR*/trajectory.npy files found under {run_dir}")

    pairs: list[tuple[int, Path]] = []
    for p in traj_paths:
        ar = int(p.parent.stem[2:])
        pairs.append((ar, p))
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Auto-detect entanglement trajectory if not given explicitly
    entangle_traj_path: Path | None = None
    if args.entangle_traj:
        entangle_traj_path = Path(args.entangle_traj)
    else:
        candidate = run_dir.parent / "trajectory.npy"
        if candidate.exists():
            entangle_traj_path = candidate

    print(f"Run dir : {run_dir}")
    print(f"Output  : {output_dir}")
    if entangle_traj_path:
        print(f"Entangle: {entangle_traj_path}")
    print(f"AR steps: {[ar for ar, _ in pairs]}")
    print()

    steps: list[dict] = []
    all_frames: list[np.ndarray] = []

    # Prepend entanglement phase (capped at --entangle-frames, stride=1)
    if entangle_traj_path:
        raw  = np.load(entangle_traj_path)
        raw  = raw[:args.entangle_frames]       # first N frames, stride=1
        traj = raw.astype(np.float32)
        all_frames.append(traj)
        steps.append({
            "label":  "Entangle",
            "type":   "entangle",
            "AR":     args.entangle_ar,
            "frames": int(traj.shape[0]),
            "radius": float(1.0 / (2.0 * args.entangle_ar)),
        })
        print(f"  Entangle: {traj.shape[0]:>5} frames  AR={args.entangle_ar}  radius={steps[-1]['radius']:.6f}")

    for ar, traj_path in pairs:
        raw  = np.load(traj_path)               # (T, N, 6) float64
        traj = raw[::args.every].astype(np.float32)
        all_frames.append(traj)
        steps.append({
            "label":  f"Relax AR{ar}",
            "type":   "relax",
            "AR":     ar,
            "frames": int(traj.shape[0]),
            "radius": float(1.0 / (2.0 * ar)),
        })
        print(f"  AR{ar:>6}: {traj.shape[0]:>5} frames  radius={steps[-1]['radius']:.6f}")

    # Dynamics perturbation trajectories (tight / loose / all)
    dyn_dir = run_dir / "dynamics"
    if dyn_dir.is_dir():
        meta_path = dyn_dir / "metadata.json"
        if meta_path.exists():
            dyn_meta   = json.loads(meta_path.read_text())
            dyn_radius = float(dyn_meta["rod_radius"])
            dyn_ar     = int(dyn_meta["AR"])
        else:
            # Fallback: use the final (smallest) AR step
            dyn_ar     = min(ar for ar, _ in pairs)
            dyn_radius = 1.0 / (2.0 * dyn_ar)

        case_display = {
            "tight": "Perturb tight",
            "loose": "Perturb loose",
            "all":   "Perturb all",
        }
        for case, base_label in case_display.items():
            case_dir = dyn_dir / case
            if not case_dir.is_dir():
                continue

            # New nested layout: dynamics/{case}/mu{friction}/trajectory.npy
            mu_dirs = sorted(
                [d for d in case_dir.iterdir() if d.is_dir() and d.name.startswith("mu")],
                key=lambda d: float(d.name[2:])
            )
            if mu_dirs:
                for mu_dir in mu_dirs:
                    mu        = float(mu_dir.name[2:])
                    traj_path = mu_dir / "trajectory.npy"
                    if not traj_path.exists():
                        continue
                    raw  = np.load(traj_path)
                    traj = raw[::args.every].astype(np.float32)
                    all_frames.append(traj)
                    label = f"{base_label} μ={mu:.4g} (AR{dyn_ar})"
                    steps.append({
                        "label":    label,
                        "type":     "perturb",
                        "case":     case,
                        "friction": mu,
                        "AR":       dyn_ar,
                        "frames":   int(traj.shape[0]),
                        "radius":   dyn_radius,
                    })
                    print(f"  {label}: {traj.shape[0]:>5} frames  radius={dyn_radius:.6f}")
            else:
                # Backward-compat: flat layout dynamics/{case}/trajectory.npy
                traj_path = case_dir / "trajectory.npy"
                if traj_path.exists():
                    raw  = np.load(traj_path)
                    traj = raw[::args.every].astype(np.float32)
                    all_frames.append(traj)
                    label = f"{base_label} (AR{dyn_ar})"
                    steps.append({
                        "label":  label,
                        "type":   "perturb",
                        "case":   case,
                        "AR":     dyn_ar,
                        "frames": int(traj.shape[0]),
                        "radius": dyn_radius,
                    })
                    print(f"  {label}: {traj.shape[0]:>5} frames  radius={dyn_radius:.6f}")

    combined     = np.concatenate(all_frames, axis=0)   # (T_total, N, 6)
    N            = int(combined.shape[1])
    total_frames = int(combined.shape[0])
    data_mb      = combined.nbytes / 1e6
    print(f"\nTotal frames : {total_frames}  ({data_mb:.2f} MB float32)")
    print(f"Rods         : {N}")

    manifest = {"N": N, "total_frames": total_frames, "steps": steps}

    viewer_src = Path(__file__).parent.parent / "viewer" / "index.html"
    if not viewer_src.exists():
        sys.exit(f"viewer/index.html not found at {viewer_src}")

    print("Embedding data…")
    html     = build_standalone(viewer_src, manifest, combined)
    out_html = output_dir / "index.html"
    out_html.write_text(html)
    html_mb  = out_html.stat().st_size / 1e6
    print(f"Standalone   : {out_html}  ({html_mb:.2f} MB)")
    print()
    print("Open directly in any browser — no server needed:")
    print(f"  {out_html}")


if __name__ == "__main__":
    main()
