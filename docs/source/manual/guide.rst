User Guide
==========

Workflow
--------
1. Generate or import a mesh using GMSH
2. Define the subsurface resistivity model
3. Create a parameter file with simulation options using Python tools
4. Run PETGEM kernel for your simulation
5. Post-process the output using Python tools

Parameter files
---------------
- Parameter files define simulation properties, solver options, and mesh details.
- Example: `params_nord1.txt` for canonical test cases.

Pre- and post-processing
------------------------
- Python scripts available for mesh conversion, resistivity model generation, and visualization (VTK format)
- Example scripts located in `tests/canonical_model`

Running PETGEM
---------------
.. code-block:: bash

    mpirun -n 4 build/csem-kernel -options_file tests/canonical_model/params_nord1.txt
