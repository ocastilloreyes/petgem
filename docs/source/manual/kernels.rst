Kernels
=======

PETGEM Kernels
--------------
- `csem-kernel`: solves 3D electromagnetic wave propagation problems using finite element methods.
- Compiled with high-order basis functions (p=1,2,...)
- Supports MPI parallel execution

Usage
-----
.. code-block:: bash

    mpirun -n 2 build/csem-kernel -options_file <parameter_file>
