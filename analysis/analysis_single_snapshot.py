#!/usr/bin/env python3
"""
Single-snapshot analysis for a rods configuration.

Given a file containing either:
  - endpoints (Nx6) array: [x1 y1 z1 x2 y2 z2] per rod, e.g., x_relaxed.txt, or
  - configuration (Nx5) array: q parameters per rod

Computes and reports:
  - number of rods (N)
  - minimal pairwise distance (min_d)
  - minimal pairwise angle (min_angle) [deg]
  - number of contacts (#pairs with distance < contact_threshold)
    * also reports per-rod average contacts = 2 * pairs / N
  - total entanglement energy (e)
  - normalized entanglement: -e / (N*(N-1)/2)
  - skewness stats: mean, std, min, max (for a_i in [0,1])
  - skewness width around 0.5: std(a_i - 0.5)

Diameter/contact threshold:
  - If aspect ratio AR is known, a natural diameter is 1/AR.
  - By default, contact_threshold = diameter * contact_factor (default 1.05).
  - You can explicitly pass --diameter or --contact-threshold to override.

Usage examples:
  python analysis_single_snapshot.py --file /path/to/x_relaxed.txt --ar 100 --json-out result.json
  python analysis_single_snapshot.py --file /path/to/q_relaxed.txt --contact-threshold 0.012

This script tries to import from the project modules `transforms` and `potentials`.
If direct imports fail (depending on working directory), it will extend sys.path
to include the repository root and core folder as a fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any

import numpy as np


def _import_project_api():
    """Import required functions via importlib with flexible paths.

    Returns a tuple: (
        x_to_q,
        create_pairs, all_pairwise_distances, all_pairwise_angles,
        all_pairwise_skewness, total_effective_potential
    )
    """
    import importlib

    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..'))
    core_dir = os.path.join(repo_root, 'core')
    for p in [repo_root, core_dir, here]:
        if p not in sys.path:
            sys.path.append(p)

    transforms_mod = None
    potentials_mod = None

    for name in ('transforms', 'core.transforms'):
        try:
            transforms_mod = importlib.import_module(name)
            break
        except ModuleNotFoundError:
            continue
    if transforms_mod is None:
        raise ImportError("Could not import transforms or core.transforms")

    for name in ('potentials', 'core.potentials'):
        try:
            potentials_mod = importlib.import_module(name)
            break
        except ModuleNotFoundError:
            continue
    if potentials_mod is None:
        raise ImportError("Could not import potentials or core.potentials")

    return (
        getattr(transforms_mod, 'x_to_q'),
        getattr(potentials_mod, 'create_pairs'),
        getattr(potentials_mod, 'all_pairwise_distances'),
        getattr(potentials_mod, 'all_pairwise_angles'),
        getattr(potentials_mod, 'all_pairwise_skewness'),
        getattr(potentials_mod, 'total_effective_potential'),
    )


x_to_q, create_pairs, all_pairwise_distances, all_pairwise_angles, all_pairwise_skewness, total_effective_potential = _import_project_api()


@dataclass
class SnapshotMetrics:
    # Basic
    file: str
    N: int
    AR: Optional[float]
    diameter: Optional[float]
    contact_factor: Optional[float]
    contact_threshold: Optional[float]

    # Pairwise distances/angles
    min_distance: float
    min_angle_deg: float

    # Contacts
    num_contact_pairs: Optional[int]
    avg_contacts_per_rod: Optional[float]

    # Entanglement
    total_entanglement: float
    normalized_entanglement: float

    # Skewness
    skewness_mean: float
    skewness_std: float
    skewness_min: float
    skewness_max: float
    skewness_width_sigma_about_half: float
    skewness_count: int


def load_array(path: str) -> np.ndarray:
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        # Single row
        if arr.size == 6:
            arr = arr.reshape(1, 6)
        elif arr.size == 5:
            arr = arr.reshape(1, 5)
    return arr


def infer_ar_from_path(path: str) -> Optional[float]:
    # Matches ...-AR0100-... or ...AR1000...
    m = re.search(r"AR(\d+)", path)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def to_q(arr: np.ndarray) -> np.ndarray:
    """Accept Nx6 endpoints or Nx5 q; return Nx5 q."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[1] == 5:
        return arr
    if arr.shape[1] == 6:
        return x_to_q(arr)
    raise ValueError(f"Unrecognized array width {arr.shape[1]} (expected 5 or 6)")


def compute_metrics(
    q: np.ndarray,
    AR: Optional[float] = None,
    diameter: Optional[float] = None,
    contact_factor: Optional[float] = 1.05,
    contact_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    N = int(q.shape[0])
    pairs = create_pairs(q)

    distances = all_pairwise_distances(pairs)
    angles = all_pairwise_angles(pairs)
    skewness = all_pairwise_skewness(pairs)

    # all_pairwise_skewness returns a tuple of (t_array, u_array). Concatenate for stats.
    if isinstance(skewness, (tuple, list)):
        try:
            skew_concat = np.concatenate([np.array(s) for s in skewness])
        except Exception:
            # Fallback: flatten each then concatenate
            flat_parts = []
            for s in skewness:
                a = np.array(s)
                flat_parts.append(a.reshape(-1))
            skew_concat = np.concatenate(flat_parts)
    else:
        skew_concat = np.array(skewness)

    # Minima
    min_distance = float(np.min(distances)) if distances.size else math.nan
    min_angle_deg = float(np.min(angles) * 180.0 / math.pi) if angles.size else math.nan

    # Contacts threshold precedence: explicit > computed from diameter > computed from AR
    used_contact_threshold: Optional[float] = None
    used_diameter: Optional[float] = diameter
    if contact_threshold is not None:
        used_contact_threshold = float(contact_threshold)
    else:
        if used_diameter is None and AR is not None and AR > 0:
            used_diameter = 1.0 / float(AR)
        if used_diameter is not None and contact_factor is not None:
            used_contact_threshold = float(used_diameter) * float(contact_factor)

    num_contact_pairs: Optional[int] = None
    avg_contacts_per_rod: Optional[float] = None
    if used_contact_threshold is not None:
        num_contact_pairs = int(np.count_nonzero(distances < used_contact_threshold))
        # Each pair contributes 2 contacts across rods
        avg_contacts_per_rod = float(2.0 * num_contact_pairs / max(N, 1))

    # Entanglement
    total_e = float(total_effective_potential(q))
    n_pairs = N * (N - 1) / 2.0
    normalized_e = float(-total_e / n_pairs) if n_pairs > 0 else math.nan

    # Skewness stats a_i in [0,1]
    skew_mean = float(np.mean(skew_concat)) if skew_concat.size else math.nan
    skew_std = float(np.std(skew_concat)) if skew_concat.size else math.nan
    skew_min = float(np.min(skew_concat)) if skew_concat.size else math.nan
    skew_max = float(np.max(skew_concat)) if skew_concat.size else math.nan
    # Width around 0.5 (empirical std)
    skew_width = float(np.std(skew_concat - 0.5)) if skew_concat.size else math.nan

    return {
        'N': N,
        'AR': float(AR) if AR is not None else None,
        'diameter': float(used_diameter) if used_diameter is not None else None,
        'contact_factor': float(contact_factor) if contact_factor is not None else None,
        'contact_threshold': float(used_contact_threshold) if used_contact_threshold is not None else None,
        'min_distance': min_distance,
        'min_angle_deg': min_angle_deg,
        'num_contact_pairs': num_contact_pairs,
        'avg_contacts_per_rod': avg_contacts_per_rod,
        'total_entanglement': total_e,
        'normalized_entanglement': normalized_e,
        'skewness_mean': skew_mean,
        'skewness_std': skew_std,
        'skewness_min': skew_min,
        'skewness_max': skew_max,
        'skewness_width_sigma_about_half': skew_width,
        'skewness_count': int(skew_concat.size),
    }


def print_human(metrics: SnapshotMetrics) -> None:
    def _fmt(x):
        if x is None:
            return '—'
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return str(x)
            return f"{x:.6g}"
        return str(x)

    print("Single-snapshot analysis")
    print("------------------------")
    print(f"file:                  {metrics.file}")
    print(f"N (rods):              {metrics.N}")
    print(f"AR:                    {_fmt(metrics.AR)}")
    print(f"diameter:              {_fmt(metrics.diameter)}")
    print(f"contact_factor:        {_fmt(metrics.contact_factor)}")
    print(f"contact_threshold:     {_fmt(metrics.contact_threshold)}")
    print()
    print(f"min distance:          {_fmt(metrics.min_distance)}")
    print(f"min angle [deg]:       {_fmt(metrics.min_angle_deg)}")
    print(f"# contact pairs:       {_fmt(metrics.num_contact_pairs)}")
    print(f"avg contacts / rod:    {_fmt(metrics.avg_contacts_per_rod)}")
    print()
    print(f"total entanglement e:  {_fmt(metrics.total_entanglement)}")
    print(f"normalized entangl.:   {_fmt(metrics.normalized_entanglement)}")
    print()
    print(f"skewness mean:         {_fmt(metrics.skewness_mean)}")
    print(f"skewness std:          {_fmt(metrics.skewness_std)}")
    print(f"skewness min:          {_fmt(metrics.skewness_min)}")
    print(f"skewness max:          {_fmt(metrics.skewness_max)}")
    print(f"skewness width (|a-0.5| std): {_fmt(metrics.skewness_width_sigma_about_half)}")
    print(f"skewness sample count:    {metrics.skewness_count}")


def main():
    p = argparse.ArgumentParser(description='Analyze a single rods snapshot (x_relaxed Nx6 or q Nx5).')
    p.add_argument('--file', required=True, help='Path to x_relaxed.txt (Nx6) or q_relaxed.txt (Nx5)')
    p.add_argument('--ar', type=float, default=None, help='Aspect ratio (alpha). If omitted, attempts to parse from path (AR####).')
    p.add_argument('--diameter', type=float, default=None, help='Rod diameter; overrides 1/AR if provided.')
    p.add_argument('--contact-factor', type=float, default=1.05, help='Multiplier on diameter to define contact threshold (default 1.05).')
    p.add_argument('--contact-threshold', type=float, default=None, help='Absolute distance threshold for contact; overrides factor*diameter.')
    p.add_argument('--json-out', type=str, default=None, help='Optional path to write metrics as JSON.')
    p.add_argument('--csv-out', type=str, default=None, help='Optional path to write metrics as one-line CSV.')

    args = p.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    arr = load_array(args.file)
    q = to_q(arr)

    # Infer AR from path if not supplied
    AR = args.ar
    if AR is None:
        AR = infer_ar_from_path(args.file)

    info = compute_metrics(
        q,
        AR=AR,
        diameter=args.diameter,
        contact_factor=args.contact_factor,
        contact_threshold=args.contact_threshold,
    )

    metrics = SnapshotMetrics(
        file=args.file,
        N=int(info['N']),
        AR=info['AR'],
        diameter=info['diameter'],
        contact_factor=info['contact_factor'],
        contact_threshold=info['contact_threshold'],
        min_distance=info['min_distance'],
        min_angle_deg=info['min_angle_deg'],
        num_contact_pairs=info['num_contact_pairs'],
        avg_contacts_per_rod=info['avg_contacts_per_rod'],
        total_entanglement=info['total_entanglement'],
        normalized_entanglement=info['normalized_entanglement'],
        skewness_mean=info['skewness_mean'],
        skewness_std=info['skewness_std'],
        skewness_min=info['skewness_min'],
        skewness_max=info['skewness_max'],
        skewness_width_sigma_about_half=info['skewness_width_sigma_about_half'],
        skewness_count=info['skewness_count'],
    )

    print_human(metrics)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    if args.csv_out:
        # Write a one-line CSV with header if file doesn't exist
        header = [
            'file','N','AR','diameter','contact_factor','contact_threshold',
            'min_distance','min_angle_deg','num_contact_pairs','avg_contacts_per_rod',
            'total_entanglement','normalized_entanglement',
            'skewness_mean','skewness_std','skewness_min','skewness_max','skewness_width_sigma_about_half','skewness_count'
        ]
        row = [
            metrics.file, metrics.N, metrics.AR, metrics.diameter, metrics.contact_factor, metrics.contact_threshold,
            metrics.min_distance, metrics.min_angle_deg, metrics.num_contact_pairs, metrics.avg_contacts_per_rod,
            metrics.total_entanglement, metrics.normalized_entanglement,
            metrics.skewness_mean, metrics.skewness_std, metrics.skewness_min, metrics.skewness_max, metrics.skewness_width_sigma_about_half, metrics.skewness_count,
        ]
        need_header = not os.path.exists(args.csv_out) or os.path.getsize(args.csv_out) == 0
        with open(args.csv_out, 'a') as f:
            if need_header:
                f.write(','.join(header) + '\n')
            # Convert None to empty string
            def _cell(v):
                return '' if v is None else (f"{v}" if not isinstance(v, float) else ("nan" if math.isnan(v) else f"{v}"))
            f.write(','.join(_cell(v) for v in row) + '\n')


if __name__ == '__main__':
    main()
