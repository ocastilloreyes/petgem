# Test cases

Self-contained PETGEM modelling cases used by the test suite and for
manual exploration.  Each subdirectory ships the case-specific inputs
that the centralised `utils/preprocess.py` consumes, plus (where
applicable) a `postprocess.py` that performs case-specific validation
against a reference solution.

## Per-case files

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `mesh_p{N}.msh`   | Gmsh mesh at polynomial order `N` (sometimes `.geo` too). |
| `sources.txt`     | Forward-source list (frequency + dipole rows).             |
| `receivers.txt`   | Receiver positions (`x y z` per row).                      |
| `sigmas.csv`      | Per-material conductivity table (one row per material).    |
| `reference.h5`    | Optional analytical / external reference for validation.  |
| `postprocess.py`  | Optional case-specific validation + plotting.              |
| `README.md`       | Optional case notes.                                        |

Anything else in a case directory (input bundles, params files, response
HDF5s, figures) is generated output and is `.gitignore`-able.

## Generic workflow

```sh
# 1. Preprocess (writes input.h5 + params.txt into the case directory)
python3 utils/preprocess.py \
    -mode forward -nord 1 -case_dir tests/cases/<case> \
    -mesh_filename mesh_p1.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.csv

# 2. Solve
mpirun -n N fm.csem -options_file tests/cases/<case>/params.txt

# 3. Postprocess (only for cases that ship a postprocess.py)
python3 tests/cases/<case>/postprocess.py \
    -case_dir tests/cases/<case> \
    -responses_filename responses_p1_src1.h5
```

## Available cases

| Case          | Materials | Notes                                              |
|---------------|-----------|----------------------------------------------------|
| `csem_model`  | 2         | Reference CSEM validation against ModEM (Ex). Ships an `extrae/` subdirectory with Extrae profiling configs (same mesh / sources / receivers, just instrumented). |
| `unit_cube`   | 1         | Single-material smoke case, multiple nord meshes.  |
| `inverse`     | 4         | Inversion case; ships `observed_data.h5`.          |

The `inverse` case also requires a multi-frequency `sources.txt` consumed
by `im.csem` via the `-source_filename` CLI option (separate from the
single-frequency `sources.txt` baked into the bundle).
