# csem_model

Reference CSEM forward-modeling case for PETGEM validation.

## Layout

- `mesh_p{N}.geo`, `mesh_p{N}.msh`  Gmsh inputs / generated meshes.
- `sigmas.csv`                        Per-material conductivity table.
- `sources.txt`                       Source-frequency + dipole records.
- `receivers.txt`                     Receiver positions (x y z per row).
- `reference.h5`                      ModEM reference Ex (`/reference_real`
                                       + `/reference_imag`).
- `postprocess.py`                    Case-specific validation script
                                       (compares PETGEM Ex against the
                                       reference and emits a figure).

## Workflow

```sh
# 1. Preprocess (writes input.h5 + params.txt into this directory)
python3 utils/preprocess.py \
    -mode forward -nord 1 -case_dir tests/cases/csem_model \
    -mesh_filename mesh_p1.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.csv

# 2. Solve (reads input.h5; writes responses_p1_src1.h5)
mpirun -n 4 fm.csem -options_file tests/cases/csem_model/params.txt

# 3. Postprocess (case-specific validation)
python3 tests/cases/csem_model/postprocess.py \
    -case_dir tests/cases/csem_model \
    -responses_filename responses_p1_src1.h5
```

The pytest integration test under `tests/integration/test_fm_csem_smoke.py`
exercises this workflow automatically when the `fm.csem` binary is
available.
