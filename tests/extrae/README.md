# Extrae profiling / instrumentation config

Runtime configuration for tracing an `fm.csem` run with the
[Extrae](https://tools.bsc.es/extrae) tracer. These files are test/CI
infrastructure — they are consumed by the `extrae-smoke` job in
[`tests-fm-csem.yml`](../../.github/workflows/tests-fm-csem.yml), which
verifies that the *Extrae-instrumented* `fm.csem.extrae` build actually runs
and emits a trace — and they double as the template for manual profiling runs.

## Files

| File                    | Purpose                                                                  |
|-------------------------|--------------------------------------------------------------------------|
| `extrae.xml`            | Extrae XML configuration (events to capture, output naming). Writes `TRACE.*` into the current working directory. |
| `petgem_functions.cfg`  | Function list selecting which PETGEM symbols Extrae should instrument.    |
| `petgem_labels.txt`     | Pretty-name mapping for the captured user events.                        |

## CI smoke

The smoke runs the instrumented binary on the committed
[`unit_cube`](../../examples/unit_cube) bundle and asserts a `TRACE.mpits` / `set-0`
trace is produced:

```sh
export EXTRAE_CONFIG_FILE=$PWD/tests/extrae/extrae.xml
export LD_LIBRARY_PATH=$EXTRAE_HOME/lib:$LD_LIBRARY_PATH
cd "$(mktemp -d)"
mpirun -n 2 build/fm.csem.extrae \
    -input_filename examples/unit_cube/outputs/input.h5 \
    -order 1 -output_dir . -output_filename extrae_smoke \
    -dm_mat_type aij -ksp_type preonly -pc_type jacobi
ls TRACE.mpits set-0        # emitted next to the run
```

## Manual profiling run

```sh
EXTRAE_CONFIG_FILE=tests/extrae/extrae.xml \
LD_PRELOAD=$EXTRAE_HOME/lib/libmpitrace.so \
mpirun -n 4 build/fm.csem -options_file examples/unit_cube/configs/params_p1.txt \
    -input_filename examples/unit_cube/outputs/input.h5 -order 1

# Merge the per-rank traces and post-process with Paraver afterwards.
${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o petgem.prv
```

`mpi2prv` produces `petgem.prv` / `.pcf` / `.row`, which Paraver opens.
