# Time-domain CSEM half-space model

A PETGEM 3D time-domain CSEM example: the half-space model of Cai et al.
(2017), Section 5.1. It exercises the full PETGEM time-domain workflow
(preprocess --> `fm.tdcsem` --> postprocess) and compares `Ex` with the
analytical solution of the paper, Eq. (A.8).

## The model

Two materials, meshed as separate physical volumes (`geometry/mesh.geo`):

| Material | Vol. tag | Conductivity  | Resistivity  |
|----------|:--------:|---------------|--------------|
| Earth    | 1        | `0.002` S/m   | `500` Ω·m    |
| Air      | 2        | `1e-8` S/m    | `1e8` Ω·m    |

The surface is `z = 0` (z up).

**Acquisition**
- Source: one x-directed electric dipole at `(-1000, 0, 0)` m, unit current
  and length (`survey/sources.txt`).
- Waveform: unit-area Gaussian pulse centred at `0.5` µs with a standard
  deviation of `0.1` µs, sampled every `5e-8` s (Fig. 3 of the paper,
  `survey/waveform.txt`).
- Receivers: inline, on the surface, at offsets **1000** and **2000** m
  (`survey/receivers.txt`).
- Time window: `0 … 1` s, BDF2 (second-order backward differentiation
  formula), initial time step `5e-8` s, step doubling trial every 50 steps
  (`configs/params.txt`).
- Domain: cube `[-L, L]³` with `L = 25` km.
- Boundary condition: `n × E = 0`.

## Directory layout

```
fm_tdcsem_halfspace/
├── README.md
├── geometry/          mesh source
│   └── mesh.geo           geometry (L = 25 km, hmin = 50 m, hmax = 2.5 km)
├── survey/            transmitter, waveform, receivers, conductivities
│   ├── sources.txt        dipole row (x y z current length dip azimuth)
│   ├── waveform.txt       source waveform (t s(t) per row)
│   ├── receivers.txt      receiver positions (x y z per row)
│   └── sigmas.txt         per-material conductivity table (row order = tag - 1)
├── configs/           time-stepping and solver options
│   └── params.txt         time stepping (BDF2) and MUMPS Cholesky
└── scripts/           benchmark-specific drivers
    ├── build_bundles.sh   gmsh + preprocess --> outputs/input_p<order>.h5
    ├── run_fm.slurm       run fm.tdcsem
    └── postprocess.py     compare Ex with Eq. (A.8), print metrics, plot
```

All generated files land in `outputs/`, which is git-ignored. `postprocess.py`
evaluates the reference, Eq. (A.8).

## Running

Run the commands from the repository root; the solve runs from `examples/fm_tdcsem_halfspace`.

```bash
make

# 1. Build the input bundle for order 2 from geometry/mesh.geo.
bash examples/fm_tdcsem_halfspace/scripts/build_bundles.sh 2

# 2. Forward solve with fm.tdcsem.
(cd examples/fm_tdcsem_halfspace && mpirun -n 4 ../../build/fm.tdcsem -options_file configs/params.txt \
    -input_filename outputs/input_p2.h5 -output_dir outputs/ -output_filename responses_p2)
#    or on a cluster:
(cd examples/fm_tdcsem_halfspace && sbatch scripts/run_fm.slurm)

# 3. Validate + plot (writes outputs/figure_p2.png).
python3 examples/fm_tdcsem_halfspace/scripts/postprocess.py -order 2
```

`postprocess.py` prints, per receiver and decade, the median relative misfit of
`Ex` against Eq. (A.8) from the peak of the analytical response,
`t_peak = μσρ²/10`, and passes when the median misfit over `[t_peak, -t_max]`
is within `-tolerance` (%) for both receivers (defaults `0.3` s and `3` %).

On MareNostrum 5 (56 MPI tasks with 2 cores each, order 2, 552 080 DOFs, MUMPS
Cholesky) the run takes 961 time steps, 19 operator setups and 5 min. The median
misfit over `[t_peak, 0.3]` s is 0.08 % at 1000 m and 0.10 % at 2000 m. From
`0.01` s the response departs from Eq. (A.8): the median misfit is 0.2 % in
`[1e-2, 1e-1)` s, 2 % in `[1e-1, 1)` s and 50 % at `1` s.
Cai et al. (2017), Fig. 6, show the same late-time departure for this domain
with the Dirichlet boundary condition.

## References

- Cai, H., Hu, X., Xiong, B., Auken, E., Han, M., Li, J., 2017. Finite element
  time domain modeling of controlled-source electromagnetic data with a hybrid
  boundary condition. Journal of Applied Geophysics 145, 133–143.
