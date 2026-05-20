# Extrae profiling configuration for csem_model

These files instrument an `fm.csem` run on the csem_model case with the
[Extrae](https://tools.bsc.es/extrae) tracer. The modeling inputs (mesh,
sources, receivers, conductivity) are the same as the parent case — only
the runtime environment and BSC tooling differ, so there is no separate
case directory.

## Files

| File                       | Purpose                                                                  |
|---------------------------|--------------------------------------------------------------------------|
| `extrae.xml`              | Extrae XML configuration (which events to capture, output naming, etc.). |
| `petgem_functions.cfg`    | Function list selecting which PETGEM symbols Extrae should instrument.   |
| `petgem_labels.txt`       | Pretty-name mapping for the captured user events.                         |

## Usage

Drive a normal csem_model forward run with Extrae enabled. The exact
incantation depends on your cluster's Extrae module, but the pattern is:

```sh
# Generate the input bundle and params file as usual.
python3 utils/preprocess.py \
    -mode forward -nord 1 -case_dir tests/cases/csem_model \
    -mesh_filename mesh_p1.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.csv

# Run the kernel under Extrae. EXTRAE_HOME must be set; LD_PRELOAD pulls
# in the MPI tracer. EXTRAE_CONFIG_FILE points at the XML in this folder.
EXTRAE_CONFIG_FILE=tests/cases/csem_model/extrae/extrae.xml \
LD_PRELOAD=$EXTRAE_HOME/lib/libmpitrace.so \
mpirun -n 4 fm.csem -options_file tests/cases/csem_model/params.txt

# Merge the per-rank traces and post-process with paraver afterwards.
${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o petgem.prv
```

`mpi2prv` produces `petgem.prv` / `.pcf` / `.row`, which Paraver opens.
