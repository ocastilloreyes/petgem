.. PETGEM documentation master file.

##################################
Welcome to PETGEM's documentation!
##################################

**PETGEM** (Parallel Exascale Toolkit for Geophysical Electromagnetic Modeling) is an open-source software for large-scale 3D electromagnetic (EM) forward modeling. It implements a high-order edge finite element method on unstructured tetrahedral meshes, enabling accurate simulations for active-source EM problems.

Originally developed in Python, **PETGEM** has been refactored in C and integrated with `PETSc <https://petsc.org/release/>`_ to improve scalability on massively parallel architectures, memory efficiency, and mesh handling. These advances make **PETGEM** well suited for current and future exascale systems.

**PETGEM** has been successfully applied to subsurface exploration in oil & gas, geothermal reservoir characterization, and environmental EM surveys.

Key features
------------
- High-order edge finite element method
- Unstructured tetrahedral mesh support (`Gmsh <http://gmsh.info/>`_)
- Parallel computing with MPI and `PETSc <https://petsc.org/release/>`_
- Python bindings for pre- and post-processing
- Integration with performance analysis tools (`Extrae <https://tools.bsc.es/extrae>`_)


More information
----------------

- `PETGEM GitHub repository <https://github.com/ocastilloreyes/petgem/>`_

- See our publication list at :ref:`publications`

Manual Documentation
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Manual:

   manual/install
   manual/guide
   manual/examples
   manual/publications
   manual/contact


API Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`