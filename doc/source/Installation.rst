.. _Installation:

Installation
============

This section describe the requirements and installation steps for PETGEM.

.. _Requeriments:

Requirements
------------

PETGEM is known to run on various flavors of Linux clusters. Its requirements are:

* `PETSc <https://www.mcs.anl.gov/petsc/>`__ (builded version for **COMPLEX-VALUED NUMBERS**) for the use of direct/iterative parallel solvers
* `Python 3 <https://www.python.org/>`__ (versions 3.5.2, 3.6.3 and 3.6.9 have been tested)
* `Numpy <http://www.numpy.org/>`__ for arrays manipulation
* `Scipy <http://www.scipy.org/>`__ for numerical operations
* `Singleton-decorator <https://pypi.org/project/singleton-decorator/>`_
* `Sphinx <http://www.sphinx-doc.org>`__ and `LaTeX <https://www.latex-project.org/>`__ (textlive) to generate documentation
* `Petsc4py <https://bitbucket.org/petsc/petsc4py>`__ for parallel computations on distributed-memory platforms. It allows the use of parallel direct/iterative solvers from `PETSc <http://www.mcs.anl.gov/petsc/>`_
* `Mpi4py <https://pypi.org/project/mpi4py/>`__ for parallel computations on distributed-memory platforms.
* `h5py <https://pypi.org/project/h5py/>`__ for input/output tasks.


On Linux, consult the package manager of your preference. PETGEM can be
used without any installation by running the kernel from the top-level
directory of the distribution.

.. _Install:

Install PETGEM
--------------

* Following commands may require root privileges

* Download `PETSc <https://www.mcs.anl.gov/petsc/>`__ (PETSc 3.7, 3.8, 3.9, and 3.12 have been tested)

* Uncompress the `PETSc <https://www.mcs.anl.gov/petsc/>`__ archive (in this example, using PETSc 3.12.0):

  .. code-block:: bash

    $ tar zxvf petsc-3.12.0.tar.gz

* Configure and build `PETSc <https://www.mcs.anl.gov/petsc/>`__. The configuration options depend on the calculations you want to perform (complex- or real-valued) as well as your compiler/MPI/Blas/Lapack setup. For PETGEM executions, **PETSC MUST BE BUILD FOR COMPLEX-VALUED NUMBERS**. In order to avoid incompatibilities between PETSC, petsc4py and PETGEM, we highly recommend the following configuration lines. Please, visit `PETSc <https://www.mcs.anl.gov/petsc/>`__ website for advanced configuration options. If you have a clean environment (not working MPI/Blas/Lapack), then run:

  .. code-block:: bash

    $ cd petsc-3.12.0
    $ export PETSC_DIR=$PWD
    $ export PETSC_ARCH=arch-linux2-c-debug

* If you do not want support for MUMPS, run following configure line:

  .. code-block:: bash

    $ ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran  --download-mpich --download-fblaslapack --with-scalar-type=complex

* If you want support for MUMPS, please add following options to previous configure line:

  .. code-block:: bash

    $ --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-cmake

* Further, to activate GPUs support, please add following options to previous configure line:

  .. code-block:: bash

    $ --with-cuda=1 --with_cuda_dir=PATH

  where ``PATH`` is the directory of your CUDA libraries.

* Then, build and install `PETSc <https://www.mcs.anl.gov/petsc/>`__:

  .. code-block:: bash

    $ make $PETSC_DIR $PETSC_ARCH all
    $ make $PETSC_DIR $PETSC_ARCH test
    $ make $PETSC_DIR $PETSC_ARCH streams

* Ensure your ``mpicc`` compiler wrapper is on your search path:

  .. code-block:: bash

    $ export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:${PATH}"

* Ensure you have a `Numpy <http://www.numpy.org/>`__ installed:

  .. code-block:: bash

    $ pip3 install numpy

* And finally, install PETGEM with its dependencies (`Scipy <http://www.scipy.org/>`__ , `Singleton-decorator <https://pypi.org/project/singleton-decorator/>`__, `Sphinx <http://www.sphinx-doc.org>`__, `Petsc4py <https://bitbucket.org/petsc/petsc4py>`__, `Mpi4py <https://pypi.org/project/mpi4py/>`__, `h5py <https://pypi.org/project/h5py/>`__) by typing:

  .. code-block:: bash

    $ pip3 install petgem

.. _DownloadingBuildingPETGEM:

Downloading and building PETGEM
-------------------------------

The PETGEM package is available for download at
`Python Package Index (PyPI) <https://pypi.python.org/pypi/petgem/>`__, at
`GitHub <https://github.com/ocastilloreyes/petgem>`__,
and the :ref:`Download` section of this project website.

* Configure and install `PETSc <https://www.mcs.anl.gov/petsc/>`__ (see :ref:`Install` section)

* Ensure you have a `Numpy <http://www.numpy.org/>`__ installed:

  .. code-block:: bash

    $ pip3 install numpy

* Download PETGEM (PETGEM 0.6 have been tested)

* Uncompress the PETGEM archive:

  .. code-block:: bash

    $ tar zxvf petgem-0.6.tar.gz
    $ cd petgem-0.6

* After unpacking the release tarball, the distribution is ready for building. Some environment configuration is needed to inform the `PETSc <https://www.mcs.anl.gov/petsc/>`__ location. As in :ref:`Install` section, you can set the environment variables ``PETSC_DIR`` and ``PETSC_ARCH`` indicating where you have built/installed `PETSc <https://www.mcs.anl.gov/petsc/>`__:

  .. code-block:: bash

    $ export PETSC_DIR=/usr/local/petsc
    $ export PETSC_ARCH=arch-linux2-c-debug

* Alternatively, you can edit the file ``setup.cfg`` and provide the required information below ``[config]`` section:

  .. code-block:: bash

     [config]
     petsc_dir = /usr/local/petsc
     petsc_arch = arch-linux2-c-debug

* Build the distribution by typing:

  .. code-block:: bash

    $ python3 setup.py build

* After building, the distribution is ready for installation (this option may require root privileges):

  .. code-block:: bash

    $ python3 setup.py install


.. _Build documentation:

Build documentation
---------------------

PETGEM is documented in PDF and HTML format using `Sphinx <http://www.sphinx-doc.org>`__ and
`LaTeX <https://www.latex-project.org/>`__. The documentation source
is in the ``doc/`` directory. The following steps summarize how to generate PETGEM documentation.

* Move to the PETGEM doc directory:

  .. code-block:: bash

    $ cd doc

* Generate the PETGEM documentation in HTML format by typing:

  .. code-block:: bash

    $ make html

* Or, if you prefer the PDF format by typing:

  .. code-block:: bash

    $ make latexpdf

* The previous steps will build the documentation in the ``doc/build`` directory. Alternatively, you can modify this path by editing the file ``setup.cfg`` and provide the required information below ``[build_sphinx]`` section:

  .. code-block:: bash

     [build_sphinx]
     source-dir = doc/source
     build-dir  = usr/local/path-build
     all_files  = 1
