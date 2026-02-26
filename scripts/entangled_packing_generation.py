import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import numpy as onp
from jax import numpy as jnp

def find_root_dir(target_name: str = "entanglement-optimization") -> Path:
    current_path = Path(__file__).resolve()
    for parent in [current_path.parent, *current_path.parents]:
        if parent.name == target_name:
            return parent
    raise SystemExit(f"Could not find repo root named '{target_name}' from {current_path}")


def parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate entangled packings. If --num-rods is provided, generates a single packing; otherwise uses --num-rods-list."
    )
    ap.add_argument(
        "--num-rods",
        type=int,
        default=None,
        help="Generate for a single N (overrides --num-rods-list).",
    )
    ap.add_argument(
        "--num-rods-list",
        type=str,
        default="10,20,50,100,200,500",
        help="Comma-separated list of N to generate when --num-rods is not provided.",
    )
    ap.add_argument(
        "--random-keys",
        type=str,
        default="56,321,194",
        help="Comma-separated PRNG keys, e.g. '56,321,194'.",
    )
    ap.add_argument("--Nmax", type=int, default=300)
    ap.add_argument("--N-outer", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument(
        "--initial-q",
        type=str,
        default="gathered",
        help="Initial configuration mode passed to create_entangled_rods.",
    )
    ap.add_argument(
        "--rod-diameter",
        type=float,
        default=-1.0,
        help="Rod diameter argument passed through (use -1 to let protocol decide).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output dir (default: <repo>/scripts/outputs).",
    )
    ap.add_argument(
        "--scale-factor",
        type=int,
        default=1,
        help="Scale factor stored in packing_id naming.",
    )
    args = ap.parse_args()

    root_dir = find_root_dir()
    sys.path.append(str(root_dir / "core"))

    from potentials import total_effective_potential
    from protocols import create_entangled_rods
    from transforms import q_to_x

    random_keys = parse_int_list(args.random_keys)
    if len(random_keys) != 3:
        raise SystemExit("--random-keys must contain exactly 3 integers, e.g. '56,321,194'")

    if args.num_rods is not None:
        num_rods_list = [int(args.num_rods)]
    else:
        num_rods_list = parse_int_list(args.num_rods_list)
        if not num_rods_list:
            raise SystemExit("--num-rods-list produced an empty list")

    output_dir = args.output_dir if args.output_dir is not None else (root_dir / "scripts" / "outputs")
    output_folder = output_dir / "entangled_packings"
    output_folder.mkdir(parents=True, exist_ok=True)

    results_per_random_keys = output_folder / f"{random_keys[0]},{random_keys[1]},{random_keys[2]}"
    results_per_random_keys.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()

    for num_rods in num_rods_list:
        save_dir_name = results_per_random_keys / f"N{num_rods}"
        save_dir_name.mkdir(parents=True, exist_ok=True)

        dt_string = now.strftime("%Y-%m-%d_%H")
        packing_id = f"{dt_string}_EntangledPacking-N{num_rods:04d}-Scale{int(args.scale_factor)}"

        packing_dir = results_per_random_keys / packing_id
        packing_dir.mkdir(parents=True, exist_ok=True)

        filename = packing_dir / "qq.npy"
        if filename.exists():
            qq = onp.load(filename)
        else:
            onp.save(filename, [])
            qq = []

        q_entangled = create_entangled_rods(
            num_rods,
            total_effective_potential,
            random_keys,
            rod_diameter=float(args.rod_diameter),
            Nmax=int(args.Nmax),
            N_outer=int(args.N_outer),
            atol=float(args.atol),
            dt=float(args.dt),
            initial_q=str(args.initial_q),
            callback=None,
        )

        np.save(save_dir_name / "q_entangled.npy", q_entangled)

        x_entangled = q_to_x(jnp.array(q_entangled))
        np.savetxt(save_dir_name / "x_entangled_packing.txt", x_entangled.reshape(-1, 6))

        if len(qq) == 0:
            qq = np.array([q_entangled])

        q0 = qq.reshape(-1, num_rods, 5)
        x0 = q_to_x(jnp.array(q0))
        np.savetxt(packing_dir / "x_entangled_packing.txt", x0.reshape(-1, 6))


if __name__ == "__main__":
    main()
    
    
