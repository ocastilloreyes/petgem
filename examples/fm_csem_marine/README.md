# Marine CSEM model

A marine CSEM example for `fm.csem`: a thin resistive layer buried in
conductive sediments beneath 1 km of seawater. It runs the full forward
workflow (preprocess, `fm.csem`, postprocess) and compares the result with a
precomputed reference.

## The model

Four layered materials, meshed as separate physical volumes (`geometry/mesh.geo`):

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
  `(1750, 1750, -975)` m - 25 m above the seafloor, unit current and length.
- Receivers: an inline seafloor profile of **58 receivers** at `z = -990` m,
  `y = 1750` m, offsets `x ≈ 58 … 3383` m (`survey/receivers.txt`).
- Domain: `x ∈ [58.33 - 4·δ, 3383.33 + 4·δ]`, `y ∈ [0, 3500]`,
  `z ∈ [-3500, 0]` m, with minimum skin depth `δ = 194.811` m.

## Directory layout

```
fm_csem_marine/
├── README.md
├── geometry/          mesh and its gmsh source
│   └── mesh.geo           parametric geometry (layers, source/receiver refinement)
├── survey/            transmitter, receivers, conductivities
│   ├── sources.txt        source definition: frequency + dipole row
│   ├── receivers.txt      receiver positions (x y z per row)
│   └── sigmas.txt         per-material conductivity table (row order = tag - 1)
├── configs/           solver options
│   └── params.txt         forward solve (PCBDDC), order 1
├── scripts/           benchmark-specific drivers
│   ├── build_bundles.sh   gmsh + preprocess --> outputs/input.h5
│   ├── run_fm.slurm       run fm.csem
│   └── postprocess.py     compare Ex to the reference, print metrics, plot
└── reference/         provided data and expected result
    └── reference.h5       precomputed reference Ex (PETSc complex Vec, /reference)
```

Generated files go to `outputs/`, which is git-ignored and created by
`build_bundles.sh`. A new run overwrites `outputs/responses_p<order>.h5`.

## Running

Run the commands from the repository root; the solve runs from `examples/fm_csem_marine`.

```bash
make

# 1. Mesh + input bundle (gmsh -> preprocess -> outputs/input.h5)
bash examples/fm_csem_marine/scripts/build_bundles.sh 1

# 2. Forward solve with fm.csem (N MPI tasks; committed options in configs/).
(cd examples/fm_csem_marine && mpirun -n 4 ../../build/fm.csem -options_file configs/params.txt)

# 3. Validate + plot (writes outputs/figure_p<order>.png).
python3 examples/fm_csem_marine/scripts/postprocess.py -tolerance 0.03
```

`postprocess.py` reports the NRMSD, relative L2 error and MAPE of `|Ex|`
against `reference/reference.h5` and exits non-zero if the NRMSD exceeds the
tolerance. On a cluster, submit the solve from `examples/fm_csem_marine` with
`sbatch scripts/run_fm.slurm`.

For another order, run `build_bundles.sh 2` and add
`-output_filename responses_p2` to the solve; `configs/params.txt` fixes the
stem to `responses_p1`.

## Reference

> Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
> code for 3D CSEM forward modeling using edge finite elements*. Computers &
> Geosciences, vol 119: 123-136. ISSN 0098-3004,  Elsevier.
> https://doi.org/10.1016/j.cageo.2018.07.005

