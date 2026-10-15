===========
MT modeling
===========

``fm.mt`` computes the 3D magnetotelluric response of a conductivity model at
one frequency: the fields of the x and y polarizations, and the impedance
tensor, apparent resistivity, phase and tipper at the receivers. It uses the
same discretization and solver as ``fm.csem``. There is no source inside the
domain; the excitation enters through a boundary condition built from a 1D
field (see :doc:`method`).

Domain requirements
-------------------
- The domain must be an **axis-aligned box**. Every boundary face is assigned
  to one of the six box faces (top, bottom and four lateral faces); the kernel
  stops if a boundary face is not on the bounding box.
- On the **lateral faces**, the conductivity must vary only with ``z`` and be
  horizontally isotropic (``sigma_x = sigma_y``). The kernel builds the 1D
  profile of the boundary field from these faces, and stops if two lateral
  faces disagree at the same height.
- The boundaries must be far enough from the survey for the 1D boundary field
  to be accurate. In :doc:`mt_examples`, four skin depths or more keep the
  misfit against an independent 3D solution within 2 % and 1°.

Inputs
------
Besides the mesh, ``sigmas.txt`` and the receiver table (one ``x y z`` per
row), ``preprocess.py -mode fm.mt`` takes ``-mt_frequency_filename``, a file
with one frequency in Hz (``#`` comments allowed):

.. code-block::

   # Frequency (Hz)
   2.0

Kernel options
--------------
``fm.mt`` takes the common options of :doc:`overview` and:

- ``-mt_1d_equation <paper|h>`` - equation of the 1D boundary field.
  ``paper`` (default) is Eq. (11) of Castillo-Reyes et al. (2022),
  :math:`H'' + i\omega\mu\sigma H = 0`; ``h`` is
  :math:`(\rho H')' + i\omega\mu H = 0`. See :doc:`method`.
- ``-mt_1d_refine <n>`` - the 1D element size is the smallest 3D edge length
  on the lateral faces of each layer divided by ``n``. Default: 10.

Parameter file
--------------
``preprocess.py -mode fm.mt`` writes a direct-solver file:

.. code-block::

   -input_filename <case_dir>/<input_filename>
   -dm_mat_type aij
   -ksp_type preonly
   -pc_type lu
   -pc_factor_mat_solver_type mumps
   -output_dir <case_dir>/
   -output_filename responses_p<order>

Running
-------
.. code-block:: bash

   mpirun -n 4 build/fm.mt -options_file params.txt

Output
------
``<output_dir>/<output_filename>.h5`` holds the fields of both polarizations
and the impedance, apparent resistivity, phase and tipper at every receiver.
The layout is in :doc:`formats`, and a full example in :doc:`mt_examples`.
