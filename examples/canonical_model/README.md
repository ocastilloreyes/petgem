# Canonical marine CSEM model

A PETGEM 3D controlled-source electromagnetic (CSEM) example: a canonical
marine benchmark with a thin, resistive hydrocarbon layer buried in conductive
marine sediments beneath a seawater column. It is the textbook setup
demonstrating that marine CSEM can detect a thin resistor, and here it
exercises the full PETGEM forward workflow (preprocess --> `fm.csem` -->
postprocess) on a realistic layered earth.

## The model

Four layered materials, meshed as separate physical volumes (`mesh.geo`):

| Layer        | Vol. tag | Depth `z` (m)     | Conductivity      | Resistivity |
|--------------|:--------:|-------------------|-------------------|-------------|
| Water        | 4        | `0 … -1000`       | `3.3333` S/m      | `0.3` Ω·m   |
| Sediments 1  | 3        | `-1000 … -2000`   | `1.0` S/m         | `1` Ω·m     |
| **Oil**      | 2        | `-2000 … -2100`   | `0.01` S/m        | `100` Ω·m   |
| Sediments 2  | 1        | `-2100 … -3500`   | `1.0` S/m         | `1` Ω·m     |

The **Oil** layer is the 100 m-thick resistive target, ~1 km below the seafloor.

**Acquisition**
- Frequency: **2 Hz**.
- Source: one x-directed horizontal electric dipole (HED) at
  `(1750, 1750, -975)` m — 25 m above the seafloor, unit current and length.
- Receivers: an inline seafloor profile of **58 receivers** at `z = -990` m,
  `y = 1750` m, offsets `x ≈ 58 … 3383` m (`receivers.txt`).
- Domain: `x ∈ [58.33 − 4·δ, 3383.33 + 4·δ]`, `y ∈ [0, 3500]`,
  `z ∈ [-3500, 0]` m, with minimum skin depth `δ = 194.811` m.

## Files

| File            | Role                                                                    |
|-----------------|-------------------------------------------------------------------------|
| `mesh.geo`      | Gmsh geometry script (parametric; layers, source/receiver refinement).  |
| `sources.txt`   | Source definition: frequency + dipole row.                              |
| `receivers.txt` | Receiver positions (`x y z` per row).                                   |
| `sigmas.txt`    | Per-material conductivity table (row order = physical tag − 1).         |
| `reference.h5`  | Precomputed reference `Ex` (native PETSc complex Vec, `/reference`).    |
| `postprocess.py`| Compares PETGEM `Ex` to the reference, prints error metrics, plots.     |

## Running

```bash
# 1. Mesh the geometry (skip if mesh.msh is already present).
gmsh -3 examples/canonical_model/mesh.geo -o examples/canonical_model/mesh.msh

# 2. Preprocess: build the input bundle (input.h5) for the chosen order.
python3 utils/preprocess.py \
    -mode forward -order 1 \
    -case_dir examples/canonical_model \
    -mesh_filename mesh.msh \
    -source_filename sources.txt \
    -receiver_filename receivers.txt \
    -sigma_file sigmas.txt

# 3. Forward solve with fm.csem (order via -order; N MPI tasks).
mpirun -n 4 build/fm.csem \
    -options_file examples/canonical_model/params.txt -order 1

# 4. Validate + plot (writes figure_p<order>.png next to the case).
python3 examples/canonical_model/postprocess.py \
    -responses_filename responses_p1.h5 -tolerance 0.03
```

`postprocess.py` reports the NRMSD, relative L2 and MAPE of `|Ex|` against
`reference.h5` and exits non-zero if the NRMSD exceeds the tolerance.

## Reference

> Castillo-Reyes, O., de la Puente, J., & Cela, J. M. (2018). *PETGEM: A
> parallel code for 3D CSEM forward modeling using edge finite elements.*
> Computers & Geosciences, 119, 123–136.
