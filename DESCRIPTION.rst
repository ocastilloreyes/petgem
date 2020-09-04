**petgem**
==========

**petgem** is a parallel python code for 3D Controlled-Source
Electromagnetic Method (3D CSEM) in geophysics using high-order edge finite
elements (Nédélec finite elements).


Requirements
------------

**petgem** is known to run on various flavors of Linux clusters. Its requirements are:

* `PETSc <https://www.mcs.anl.gov/petsc/>`__ (builded version for **COMPLEX-VALUED NUMBERS**) for the use of direct/iterative parallel solvers
* `Python 3 <https://www.python.org/>`__ (versions 3.5.2, 3.6.3 and 3.6.9 have been tested)
* `Numpy <http://www.numpy.org/>`__ for arrays manipulation
* `Scipy <http://www.scipy.org/>`__ for numerical operations
* `Singleton-decorator <https://pypi.org/project/singleton-decorator/>`_
* `Sphinx <http://www.sphinx-doc.org>`__ and `LaTeX <https://www.latex-project.org/>`__ (textlive) to generate documentation
* `Petsc4py <https://bitbucket.org/petsc/petsc4py>`__ for parallel computations on distributed-memory platforms. It allows the use of parallel direct/iterative solvers from `PETSc <http://www.mcs.anl.gov/petsc/>`_
* `Mpi4py <https://pypi.org/project/mpi4py/>`__ for parallel computations on distributed-memory platforms.
* `h5py <https://pypi.org/project/h5py/>`__ for input/output tasks.

On Linux, consult the package manager of your preference. **petgem** can be
used without any installation by running the kernel from the top-level
directory of the distribution.


Install
-------

* Following commands may require root privileges

* Download `PETSc <https://www.mcs.anl.gov/petsc/>`__ (PETSc 3.7, 3.8, 3.9, and 3.12 have been tested)

* Uncompress the PETSc archive (in this example, using PETSc 3.12.0)::

  $ tar zxvf petsc-3.12.0.tar.gz

* Configure and build PETSc. The configuration options depend on the calculations you want to perform (complex- or real-valued) as well as your compiler/MPI/Blas/Lapack setup. For **petgem** executions, **PETSC MUST BE BUILD FOR COMPLEX-VALUED NUMBERS**. In order to avoid incompatibilities between PETSC, petsc4py and **petgem**, we highly recommend the following configuration lines. Please, visit PETSc website for advanced configuration options. If you have a clean environment (not working MPI/Blas/Lapack), then run::

  $ cd petsc-3.12.0
  $ export PETSC_DIR=$PWD
  $ export PETSC_ARCH=arch-linux2-c-debug

* If you do not want support for MUMPS, run following configure line::

  $ ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran  --download-mpich --download-fblaslapack --with-scalar-type=complex

* If you want support for MUMPS, please add following options to previous configure line::

  $ --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-cmake

* Further, to activate GPUs support, please add following options to previous configure line::

  $ --with-cuda=1 --with_cuda_dir=PATH

  where ``PATH`` is the directory of your CUDA libraries.

* Then, build and install PETSc::

  $ make $PETSC_DIR $PETSC_ARCH all
  $ make $PETSC_DIR $PETSC_ARCH test
  $ make $PETSC_DIR $PETSC_ARCH streams

* Ensure your ``mpicc`` compiler wrapper is on your search path::

  $ export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:${PATH}"

* Ensure you have a Numpy installed::

  $ pip3 install numpy

* And finally, install **petgem** with its dependencies (`Scipy <http://www.scipy.org/>`_ , `Blessings <https://pypi.python.org/pypi/blessings/>`__, `Sphinx <http://www.sphinx-doc.org>`__, `Petsc4py <https://bitbucket.org/petsc/petsc4py>`__) by typing::

  $ pip3 install **petgem**


Citations
---------

If **petgem** been significant to a project that leads to an academic
publication, please acknowledge that fact by citing the project:

* Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J. M. (2019).
  *Parallel 3D marine controlled-source electromagnetic modeling using high-order
  tetrahedral Nédélec elements*. Geophysical Journal International, ggz285,
  vol 219: 39-65. ISSN 0956-540X. https://doi.org/10.1093/gji/ggz285

* Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018).
  ***petgem**: A parallel code for 3D CSEM forward modeling using edge finite
  elements*. Computers & Geosciences, vol 119: 123-136. ISSN 0098-3004,
  Elsevier. https://doi.org/10.1016/j.cageo.2018.07.005

* Castillo-Reyes, O., de la Puente, J., Cela, J.M. (2017).
  *Three-Dimensional CSEM Modelling on Unstructured Tetrahedral Meshes
  Using Edge Finite Elements*, Communications in Computer and
  Information Science, vol 697: 247-256. ISBN 978-3-319-57971-9
  Springer, Cham. https://doi.org/10.1007/978-3-319-57972-6_18
