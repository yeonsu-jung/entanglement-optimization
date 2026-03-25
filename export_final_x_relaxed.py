#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def sph2cart(theta: np.ndarray, phi: np.ndarray, r: float | np.ndarray = 1.0) -> np.ndarray:
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    r = np.asarray(r)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def q_to_x(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape((-1, 5))
    x0 = q[:, :3]
    offsets = sph2cart(q[:, 3], q[:, 4])
    x1 = x0 + offsets
    return np.concatenate([x0, x1], axis=1)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    ar: int
    n: int
    random_keys: str


def parse_run_dir_name(run_dir: Path) -> RunInfo:
    name = run_dir.name
    m_ar = re.search(r"_AR(\d+)_", name)
    m_n = re.search(r"_N(\d+)_", name)
    m_keys = re.search(r"randomkeys(\d+,\d+,\d+)$", name)
    if not (m_ar and m_n and m_keys):
        raise ValueError(f"Cannot parse run folder name: {run_dir}")

    return RunInfo(
        run_dir=run_dir,
        ar=int(m_ar.group(1)),
        n=int(m_n.group(1)),
        random_keys=m_keys.group(1),
    )


def _reshape_by_last_dim(arr: np.ndarray, last_dim: int) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if flat.size % last_dim != 0:
        raise ValueError(f"Array of size {flat.size} not divisible by {last_dim}")
    return flat.reshape((-1, last_dim))


def as_q(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim == 2 and arr.shape[-1] == 5:
        return arr
    if arr.ndim == 1:
        return _reshape_by_last_dim(arr, 5)
    if arr.ndim == 2 and arr.shape[-1] != 5:
        return _reshape_by_last_dim(arr, 5)
    raise ValueError(f"Cannot interpret array as q with shape {arr.shape}")


def as_x(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim == 2 and arr.shape[-1] == 6:
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 3 and arr.shape[1] == 2:
        return arr.reshape((-1, 6))
    if arr.ndim == 1:
        return _reshape_by_last_dim(arr, 6)
    if arr.ndim == 2 and arr.shape[-1] != 6:
        return _reshape_by_last_dim(arr, 6)
    raise ValueError(f"Cannot interpret array as x with shape {arr.shape}")


def find_final_config_file(run_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Return (config_path, label_dir).

    `label_dir` is the directory whose name we use for output filenames when present
    (e.g. `2025-12-19_12_EntangledRelaxedPacking-N0200-AR0100-Scale1`).
    """

    results_dir = run_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results dir: {results_dir}")

    # Prefer relaxed q/x in the most specific run subfolder.
    preferred = [
        "q_relaxed.txt",
        "x_relaxed.txt",
        "q_relaxed.npy",
        "x_relaxed.npy",
        # If relaxation didn't write q_relaxed, qq.npy usually contains the latest iterate.
        "qq.npy",
        # Fallbacks.
        "q_entangled.npy",
        "x_entangled.npy",
    ]

    for fname in preferred:
        matches = sorted(results_dir.rglob(fname))
        if matches:
            cfg = matches[0]
            label_dir = cfg.parent if cfg.parent != results_dir else None
            return cfg, label_dir

    raise FileNotFoundError(f"No known config file found under: {results_dir}")


def _maybe_take_last_frame(arr: np.ndarray, expected_n: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # Common save format: flattened time series of q or x.
        for dim in (5, 6):
            block = expected_n * dim
            if block > 0 and arr.size != block and arr.size % block == 0:
                steps = arr.size // block
                return arr.reshape((steps, expected_n, dim))[-1]
        return arr
    if arr.ndim == 3:
        return arr[-1]
    if arr.ndim == 2 and arr.shape[0] != expected_n and arr.shape[0] % expected_n == 0:
        steps = arr.shape[0] // expected_n
        return arr.reshape((steps, expected_n, arr.shape[1]))[-1]
    return arr


def load_config_as_x(cfg_path: Path, expected_n: int) -> np.ndarray:
    suffix = cfg_path.suffix.lower()
    if suffix == ".txt":
        arr = np.loadtxt(cfg_path)
    elif suffix == ".npy":
        arr = np.load(cfg_path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported config file type: {cfg_path}")

    arr = _maybe_take_last_frame(arr, expected_n)

    name = cfg_path.name
    if name.startswith("x_"):
        return as_x(arr)

    # Otherwise assume q-like
    q = as_q(arr)
    x = np.asarray(q_to_x(q))
    return as_x(x)


def make_output_name(run_info: RunInfo, label_dir: Optional[Path]) -> str:
    if label_dir is not None:
        # Example: 2025-12-19_12_EntangledRelaxedPacking-N0200-AR0100-Scale1
        label = label_dir.name
        # Keep it compact but still informative.
        m_dt = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}", label)
        dt = m_dt.group(0) if m_dt else "unknownDT"
        return f"x_relaxed_{dt}_N{run_info.n:04d}_AR{run_info.ar:04d}_Scale1.txt"

    return f"x_relaxed_N{run_info.n:04d}_AR{run_info.ar:04d}.txt"


def export_runs(run_dirs: Iterable[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        run_info = parse_run_dir_name(run_dir)
        cfg_path, label_dir = find_final_config_file(run_dir)
        x = load_config_as_x(cfg_path, run_info.n)

        if x.shape != (run_info.n, 6):
            raise ValueError(
                f"Unexpected x shape for {run_dir}: got {x.shape}, expected ({run_info.n}, 6).\n"
                f"Config file used: {cfg_path}"
            )

        out_name = make_output_name(run_info, label_dir)
        out_path = out_dir / out_name
        np.savetxt(out_path, x, fmt="%.18e")
        print(f"Wrote {out_path} (from {cfg_path})")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    runs_root = repo_root / "runs"

    run_dirs = [
        runs_root / "20251219-1227_RUN_protocol_AR1000_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR300_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR500_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR10_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR25_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR50_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR100_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR150_N200_randomkeys36,298,312",
        runs_root / "20251219-1227_RUN_protocol_AR200_N200_randomkeys36,298,312",
    ]

    out_dir = repo_root / "data" / "36,298,312"
    export_runs(run_dirs, out_dir)


if __name__ == "__main__":
    main()
