# Marine CSEM model

A PETGEM 3D controlled-source electromagnetic (CSEM) example: a canonical
marine benchmark with a thin, resistive hydrocarbon layer buried in conductive
marine sediments beneath a seawater column. It is the textbook setup
demonstrating that marine CSEM can detect a thin resistor, and here it
exercises the full PETGEM forward workflow (preprocess --> `fm.csem` -->
postprocess) on a realistic layered earth.

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
fm_model/
├── README.md
├── geometry/          mesh and its gmsh source
│   └── mesh.geo           parametric geometry (layers, source/receiver refinement)
├── survey/            transmitter, receivers, conductivities
│   ├── sources.txt        source definition: frequency + dipole row
│   ├── receivers.txt      receiver positions (x y z per row)
│   └── sigmas.txt         per-material conductivity table (row order = tag - 1)
├── configs/           solver options
│   └── params.txt         forward solve (PCBDDC), order 1
├── reference/         provided data and expected result
│   └── reference.h5       precomputed reference Ex (PETSc complex Vec, /reference)
├── scripts/           benchmark-specific drivers
│   ├── build_bundles.sh   gmsh + preprocess --> outputs/input.h5
│   ├── run_forward.slurm  run fm.csem
│   └── postprocess.py     compare Ex to the reference, print metrics, plot
└── outputs/           generated files (bundle, responses, figures, logs) - git-ignored
```

`outputs/` is git-ignored and disposable: a fresh clone starts empty there and
`build_bundles.sh` recreates it. Re-running the forward stage overwrites
`outputs/responses_p<order>.h5`, so copy any run worth keeping elsewhere first.

General, reusable tools live in the PETGEM `utils/` package, not here
(`utils/preprocess.py` builds the solver input bundle from a mesh + survey).

## Running

Run all commands **from the repository root**.

```bash
make

# 1. Build the input bundle (gmsh -> preprocess -> outputs/input.h5).
#    Needs the petgem package; e.g. via the petgem-env container:
#      docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
#        bash examples/fm_model/scripts/build_bundles.sh 1
bash examples/fm_model/scripts/build_bundles.sh 1

# 2. Forward solve with fm.csem (N MPI tasks; committed options in configs/).
mpirun -n 4 build/fm.csem \
    -options_file examples/fm_model/configs/params.txt -order 1

# 3. Validate + plot (writes outputs/figure_p<order>.png).
python3 examples/fm_model/scripts/postprocess.py -tolerance 0.03
```

`postprocess.py` reports the NRMSD, relative L2 and MAPE of `|Ex|` against
`reference/reference.h5` and exits non-zero if the NRMSD exceeds the tolerance.
On a cluster, submit the solve with `sbatch examples/fm_model/scripts/run_forward.slurm`.

## Reference

> Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
> code for 3D CSEM forward modeling using edge finite elements*. Computers &
> Geosciences, vol 119: 123-136. ISSN 0098-3004,  Elsevier.
> https://doi.org/10.1016/j.cageo.2018.07.005

