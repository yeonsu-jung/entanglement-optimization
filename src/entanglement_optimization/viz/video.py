import argparse
import numpy as np
import polyscope as ps
import imageio.v2 as imageio
import tempfile
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a video from trajectory files across all AR steps in a run directory."
    )
    p.add_argument("run_dir", type=str,
                   help="Path to a run directory (e.g. .../2026-03-19_01_Relaxed-N200) "
                        "or a single AR*/trajectory.npy file")
    p.add_argument("--output", type=str, default=None,
                   help="Output video path (default: <run_dir>/video.mp4)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    p.add_argument("--every", type=int, default=1, help="Render every Nth frame per AR (default: 1)")
    return p.parse_args()


def collect_trajectories(run_dir: Path) -> list[tuple[float, Path]]:
    """Return (AR, traj_path) pairs sorted largest AR → smallest (thin → thick)."""
    if run_dir.is_file() and run_dir.suffix == ".npy":
        ar = float(run_dir.parent.stem.split('AR')[1])
        return [(ar, run_dir)]

    traj_paths = sorted(run_dir.glob("AR*/trajectory.npy"))
    if not traj_paths:
        raise FileNotFoundError(f"No AR*/trajectory.npy files found under {run_dir}")

    pairs = []
    for p in traj_paths:
        ar = float(p.parent.stem.split('AR')[1])
        pairs.append((ar, p))

    # largest AR first = thinnest rods first (matches relax.py order)
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


args = parse_args()
run_dir = Path(args.run_dir)
trajectories = collect_trajectories(run_dir)

if run_dir.is_file():
    output_path = args.output or str(run_dir.parent / "video.mp4")
else:
    output_path = args.output or str(run_dir / "video.mp4")

print(f"Found {len(trajectories)} AR trajectory file(s):")
for ar, p in trajectories:
    traj = np.load(p)
    print(f"  AR{int(ar):>6}  {p}  shape={traj.shape}")

ps.init()
ps.set_program_name("trajectory")

with tempfile.TemporaryDirectory() as tmpdir:
    frame_paths = []
    global_frame = 0

    for ar, traj_path in trajectories:
        rod_radius = 1.0 / ar / 2.0
        traj = np.load(traj_path)
        num_frames, num_rods = traj.shape[0], traj.shape[1]
        edges = np.array([[2 * i, 2 * i + 1] for i in range(num_rods)])

        frame_indices = range(0, num_frames, args.every)
        print(f"AR{int(ar)}: rendering {len(frame_indices)} / {num_frames} frames ...")

        for t in frame_indices:
            nodes = traj[t].reshape(-1, 3)
            ps_curve = ps.register_curve_network("packing", nodes, edges)
            ps_curve.set_radius(rod_radius, relative=False)

            frame_path = os.path.join(tmpdir, f"frame_{global_frame:06d}.png")
            ps.screenshot(frame_path)
            frame_paths.append(frame_path)
            global_frame += 1

    total = len(frame_paths)
    print(f"\nTotal frames: {total}  →  {total / args.fps:.1f}s at {args.fps} fps")
    print(f"Writing video to {output_path} ...")

    with imageio.get_writer(output_path, fps=args.fps) as writer:
        for i, frame_path in enumerate(frame_paths):
            writer.append_data(imageio.imread(frame_path))
            if i % 50 == 0:
                print(f"  {i}/{total}", end="\r")

print(f"\nDone. Video saved to {output_path}")
