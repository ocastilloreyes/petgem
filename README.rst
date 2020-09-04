.. image:: https://github.com/ocastilloreyes/petgem/blob/master/doc/source/_static/figures/petgem_logo.png
   :target: https://github.com/ocastilloreyes/petgem
   :align: center
   :alt: petgem logo

----

.. image:: https://readthedocs.org/projects/emg3d/badge/?version=latest
   :target: http://petgem.bsc.es/
   :alt: Documentation Status
.. image:: https://travis-ci.org/ocastilloreyes/petgem.svg?branch=master
   :target: https://travis-ci.org/ocastilloreyes/petgem
   :alt: Travis-CI
.. image:: https://coveralls.io/repos/github/ocastilloreyes/petgem/badge.svg
   :target: https://coveralls.io/github/ocastilloreyes/petgem?branch=master
   :alt: Coveralls
.. image:: https://app.codacy.com/project/badge/Grade/283b8199432f4daa8526783d6630377d
   :target: https://www.codacy.com/app/ocastilloreyes/petgem
   :alt: Codacy-grade
.. image:: https://img.shields.io/pypi/v/petgem
   :target: https://pypi.org/project/petgem/
   :alt: Pypi-petgem
.. image:: https://img.shields.io/github/v/release/ocastilloreyes/petgem
   :target: https://github.com/ocastilloreyes/petgem/releases
   :alt: GitHub release (latest by date)
.. image:: https://img.shields.io/static/v1?label=Ubuntu&logo=Ubuntu&logoColor=white&message=support&color=success
   :target: https://ubuntu.com/
   :alt: Ubuntu support
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: petgem-license


Parallel Edge-based Tool for Geophysical Electromagnetic Modelling
------------------------------------------------------------------
**petgem** is a Python code for the scalable solution of electromagnetic diffusion fields on tetrahedral
meshes, as these are the easiest to scale-up to very large domains or
arbitrary shape. It supports distributed-memory parallelism through
petsc4py package. **petgem** allow users to specify high-order edge-based variational
forms of H(curl) for the simulation of electromagnetic fields in realistic
3D CSEM surveys with accuracy, reliability and efficiency.

More information
----------------
For more information regarding installation, usage, contributing and bug reports see:

- **Website**: http://petgem.bsc.es/
- **Documentation**: http://petgem.bsc.es/
- **Installation**: http://petgem.bsc.es/
- **Source Code**: https://github.com/ocastilloreyes/petgem
- **Pypi site**: https://pypi.org/project/petgem/
- **Examples**: https://github.com/ocastilloreyes/petgem

Requests and contributions are welcome.

Dependencies
------------

-  A matching version of PETSc\_

-  Python\_ (versions 3.5.2, 3.6.3, 3.6.9, 3.12.0 have been tested).

-  A recent NumPy\_ release.

-  A recent Scipy\_ release.

- A recent Singleton-decorator\_ release.

- A recent Sphinx\_ release.

- A recent texlive\_ release.

- A recent Petsc4py\_ release.

- A recent Mpi4py\_ release.

- A recent h5py\_ release.

Citation
--------
If you publish results for which you used **petgem**, please give credit by citing
`Castillo-Reyes, O. et al. (2019) <https://doi.org/10.1093/gji/ggz285>`_:

  Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J.M. (2019).
  *Parallel 3-D marine controlled-source electromagnetic modelling using high-order
  tetrahedral Nédélec elements*. Geophysical Journal International, Volume 219,
  Issue 1, October 2019, Pages 39–65, https://doi.org/10.1093/gji/ggz285

and `Castillo-Reyes, O. et al. (2018) <https://doi.org/10.1016/j.cageo.2018.07.005>`_:

  Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
  code for 3D CSEM forward modeling using edge finite elements*. Computers &
  Geosciences, vol 119: 123-136. ISSN 0098-3004,  Elsevier.
  https://doi.org/10.1016/j.cageo.2018.07.005


License
-------
**petgem** is developed as open-source under BSD-3 license at Computer Applications
in Science & Engineering of the Barcelona Supercomputing Center - Centro Nacional
de Supercomputación. Please, see the CONDITIONS OF USE described in the LICENSE.rst file.
