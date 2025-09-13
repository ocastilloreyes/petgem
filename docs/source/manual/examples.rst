Examples
========

Canonical Examples
------------------
- Provided in `tests/canonical_model`
- Includes p=1 and p=2 simulations for a simple layered resistivity model

Running Examples
----------------
.. code-block:: bash

    # Generate mesh
    gmsh tests/canonical_model/mesh.geo -3

    # Generate resistivity model and parameters
    python3 tests/canonical_model/generate_resistivity_model.py
    python3 tests/canonical_model/generate_params_file.py

    # Run PETGEM simulation
    mpirun -n 1 build/csem-kernel -options_file tests/canonical_model/params_nord1.txt
