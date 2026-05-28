# entanglement-optimization

JAX-based optimization of entangled rod packings. Generates configurations where
$N$ rigid rods are driven into an entangled state at high aspect ratio and then
sequentially relaxed as the rods inflate, producing dense non-overlapping
packings.

## Install

```bash
pip install -e .            # CPU (default)
pip install -e ".[gpu]"     # adds jax[cuda12] for NVIDIA GPUs
pip install -e ".[viz]"     # adds polyscope + imageio for rendering
pip install -e ".[dev]"     # adds pytest
```

JAX is backend-agnostic — the same code runs on CPU, GPU, or TPU; the `[gpu]`
extra just swaps in the CUDA-12 wheel.

## Entry points

Four CLIs are installed:

| Command         | What it does                                         |
|-----------------|------------------------------------------------------|
| `eo-entangle`   | Drive $N$ thin rods into an entangled state          |
| `eo-relax`      | Relax an entangled state by inflating rod radius     |
| `eo-perturb`    | Perturb a relaxed packing and re-relax               |
| `eo-make-video` | Render an entangled/relaxed trajectory as MP4        |

Run any with `--help` for arguments.

## Package layout

```
src/entanglement_optimization/
├── core/
│   ├── physics.py             # geometry, linking number, potentials
│   ├── fire.py                # FIRE optimizer
│   └── initial_conditions.py  # random rod generation
├── viz/
│   ├── render.py              # polyscope scene rendering
│   ├── video.py               # frame-by-frame MP4 export
│   └── web_export.py          # JSON/base64 export for web viewer
└── cli/
    ├── entangle.py
    ├── relax.py
    ├── perturb.py
    └── make_video.py
```

## Tests

```bash
pytest
```

## Related

Batch runs and downstream analysis live in a companion control-tower repo.
This repo stays focused on the JAX kernels and per-packing CLIs.
