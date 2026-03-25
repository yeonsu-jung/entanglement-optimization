#!/usr/bin/env python3
"""Extract entangled packings into a standardized folder layout.

For each run directory (e.g. runs/20251220-0034_ENTANGLEONLY_AR10_N200_randomkeys373,471,666),
this script finds the latest entangled output (prefers x_entangled.txt, falls back to x_entangled.npy)
under that run's nested results folder and writes it to:

    results/entangled_packings/N{N}/{randomkeys}/x_entangled.txt

The output is in endpoint format (N,6): (x1,y1,z1,x2,y2,z2).
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


_RANDOMKEYS_RE = re.compile(r"randomkeys(?P<keys>\d+,\d+,\d+)")
_N_RE = re.compile(r"_N(?P<n>\d+)")


def _infer_randomkeys(run_dir: Path) -> str:
    m = _RANDOMKEYS_RE.search(run_dir.name)
    if m:
        return m.group("keys")

    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results dir: {results_dir}")

    candidates = [p.name for p in results_dir.iterdir() if p.is_dir()]
    candidates = [c for c in candidates if re.fullmatch(r"\d+,\d+,\d+", c or "")]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not infer randomkeys from run dir name or unique results/<keys>/ folder. "
        f"run_dir={run_dir} candidates={candidates}"
    )


def _infer_num_rods_from_name(run_dir: Path) -> Optional[int]:
    m = _N_RE.search(run_dir.name)
    if not m:
        return None
    return int(m.group("n"))


def _pick_latest(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate files found")
    return max(paths, key=lambda p: p.stat().st_mtime)


def _find_entangled_file(run_dir: Path, keys: str) -> Path:
    base = run_dir / "results" / keys
    if not base.is_dir():
        raise FileNotFoundError(f"Missing expected results keys dir: {base}")

    # Prefer txt, then npy.
    txt_candidates = list(base.glob("**/x_entangled.txt"))
    txt_candidates = [p for p in txt_candidates if p.is_file()]
    if txt_candidates:
        return _pick_latest(txt_candidates)

    npy_candidates = list(base.glob("**/x_entangled.npy"))
    npy_candidates = [p for p in npy_candidates if p.is_file()]
    if npy_candidates:
        return _pick_latest(npy_candidates)

    raise FileNotFoundError(f"No x_entangled.(txt|npy) found under {base}")


def _load_endpoints(path: Path) -> np.ndarray:
    if path.suffix == ".txt":
        x = np.loadtxt(path)
    elif path.suffix == ".npy":
        x = np.load(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"Expected endpoints shape (N,6), got {x.shape} from {path}")
    return x


def _write_x_entangled(dest: Path, x: np.ndarray, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        raise FileExistsError(f"Destination exists (use --force to overwrite): {dest}")
    np.savetxt(dest, x, fmt="%.18e")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories under ./runs/...",
    )
    parser.add_argument(
        "--num-rods",
        type=int,
        default=None,
        help="Optional: validate that each extracted packing has this N (useful as a safety check)",
    )
    parser.add_argument(
        "--dest-root",
        type=str,
        default="results/entangled_packings",
        help="Destination root (default: results/entangled_packings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files",
    )

    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    dest_root = (repo_root / args.dest_root).resolve()

    failures: List[str] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        try:
            keys = _infer_randomkeys(run_dir)
            src = _find_entangled_file(run_dir, keys)
            x = _load_endpoints(src)

            n_from_name = _infer_num_rods_from_name(run_dir)
            n = n_from_name if n_from_name is not None else int(x.shape[0])

            if x.shape[0] != n:
                raise ValueError(
                    f"N mismatch for {run_dir.name}: inferred N={n} but file has N={x.shape[0]}"
                )

            if args.num_rods is not None and n != args.num_rods:
                raise ValueError(
                    f"Unexpected N for {run_dir.name}: got {n}, expected {args.num_rods}"
                )

            dest = dest_root / f"N{n}" / keys / "x_entangled.txt"
            _write_x_entangled(dest, x, force=args.force)
            print(f"OK  N{n}  {keys}  {src}  ->  {dest}")
        except Exception as e:
            failures.append(f"FAIL {run_dir_str}: {e}")

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
