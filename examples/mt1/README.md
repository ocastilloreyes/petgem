# Trapezoidal hill MT model

A PETGEM 3D magnetotelluric (MT) example: the trapezoidal hill of Nam et al.
(2007), used in Castillo-Reyes et al. (2022), Section 3.1, to validate the MT
routine and to study how far the boundaries must be from the survey. It
exercises the full PETGEM MT workflow (preprocess --> `fm.mt` --> postprocess)
on a model with surface topography.

## The model

Two materials, meshed as separate physical volumes (`geometry/mesh.geo`):

| Material | Vol. tag | Conductivity  | Resistivity  |
|----------|:--------:|---------------|--------------|
| Air      | 1        | `1e-8` S/m    | `1e8` Ω·m    |
| Earth    | 2        | `0.01` S/m    | `100` Ω·m    |

The hill is 450 m high, with a 450 × 450 m top and a 2000 × 2000 m base,
centred at the origin; the flat surface is `z = 0` (z up).

**Acquisition**
- Frequency: **2 Hz** (skin depth `δ ≈ 3.5` km in the earth).
- Polarizations: x and y (both solved in one run).
- Receivers: **41 stations** along `y = 0`, `x = -2000 … 2000` m every 100 m,
  2.1 m below the surface (`survey/receivers.txt`).
- Domain: cube `[-L, L]³` with `L = 2000 + nskin·3500` m, i.e. the boundaries
  are `nskin` skin depths away from the survey (Table 1 of the paper).
- Boundary condition: natural, `n × H` from the 1D field of the lateral faces
  (Eq. 10–11 of the paper).

## Directory layout

```
mt1/
├── README.md
├── geometry/          mesh source
│   └── mesh.geo           parametric geometry (nskin, hmin, hmax)
├── survey/            receivers, frequency, conductivities
│   ├── receivers.txt      receiver positions (x y z per row)
│   ├── mt_frequency.txt   frequency (Hz)
│   └── sigmas.txt         per-material conductivity table (row order = tag - 1)
├── configs/           solver options
│   └── params.txt         forward solve (MUMPS LU), 1D equation of the paper
├── scripts/           benchmark-specific drivers
│   ├── build_bundles.sh   gmsh (or a given mesh) + preprocess --> outputs/input_p<order>_n<nskin>.h5
│   ├── run_mt.slurm       run fm.mt
│   └── postprocess.py     compare rho and phase to the references, print metrics, plot
└── reference/         reference responses at 2 Hz
    ├── emmi3d/            independent 3D solution (Coordinates, Rxy, Ryx, Pxy, Pyx)
    └── petgem_2022.h5     responses published in the paper (p = 1, 2; nskin = 1 … 10)
```

All generated files land in `outputs/`, which is git-ignored.

Phases are compared in the first-quadrant convention of the references,
`φ = mod(-φ_fm.mt, 180°)`; `fm.mt` writes them for `exp(-iωt)` with z up.

## Running

Run the commands from the repository root; the solve runs from `examples/mt1`.

```bash
make

# 1. Build the input bundle for order 2 and 4 skin depths from geometry/mesh.geo.
#    A third argument sets hmin (m) or gives an existing mesh instead, e.g. the
#    meshes of the paper: build_bundles.sh 2 4 /path/to/p2_4skin.msh
bash examples/mt1/scripts/build_bundles.sh 2 4

# 2. Forward solve with fm.mt.
(cd examples/mt1 && mpirun -n 4 ../../build/fm.mt -options_file configs/params.txt \
    -input_filename outputs/input_p2_n4.h5 -output_dir outputs/ -output_filename responses_p2_n4)
#    or on a cluster:
(cd examples/mt1 && sbatch --export=ALL,ORDER=2,NSKIN=4 scripts/run_mt.slurm)

# 3. Validate + plot (writes outputs/figure_p2.png).
python3 examples/mt1/scripts/postprocess.py -order 2
```

`postprocess.py` reads every `outputs/responses_p<order>_n<nskin>.h5` present,
prints the median misfit against both references, and passes when the runs with
`nskin >= 4` are within 2 % in apparent resistivity and 1° in phase of EMMI3D.
The published PETGEM responses reach about 1 % and 0.5° from `nskin = 4`.

## References

- Nam, M.J., Kim, H.J., Song, Y., Lee, T.J., Son, J.S., Suh, J.H., 2007. 3D
  magnetotelluric modelling including surface topography. Geophysical
  Prospecting 55, 277–287.
- Castillo-Reyes, O., Modesto, D., Queralt, P., Marcuello, A., Ledo, J.,
  Amor-Martin, A., de la Puente, J., García-Castillo, L.E., 2022. 3D
  magnetotelluric modeling using high-order tetrahedral Nédélec elements on
  massively parallel computing platforms. Computers & Geosciences 160, 105030.
