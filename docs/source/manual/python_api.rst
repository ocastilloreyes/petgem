==========
Python API
==========

The package in ``utils/`` (imported as ``petgem`` after ``pip install -e .``)
builds input bundles and reads the bundles and the kernel output. Every
function needs ``petsc4py``. ``utils/preprocess.py`` and
``utils/make_observed.py`` run from a clone without installing it. The file
formats are described in :doc:`formats`.

Example
-------
Read the ``fm.csem`` output of ``examples/fm_csem_marine`` and compare it with the
reference:

.. code-block:: python

   import numpy as np
   import petgem

   bundle = petgem.readBundle("examples/fm_csem_marine/outputs/input.h5")
   resp = petgem.readResponses("examples/fm_csem_marine/outputs/responses_p1.h5", source=1)
   ref = petgem.readVectorH5("examples/fm_csem_marine/reference/reference.h5", "reference")

   x = bundle["receivers"][:, 0]
   ex = np.ravel(resp["Ex"])
   print(x[:3], np.abs(ex[:3]))
   print(petgem.compareMagnitude(ex, np.ravel(ref)))   # nrmsd, rel_l2, mape

Preprocessing
-------------
.. autofunction:: petgem.runPreprocessing

.. autofunction:: petgem.readSigmaTable

.. autofunction:: petgem.readSourcesText

.. autofunction:: petgem.readObservedDataH5

.. autofunction:: petgem.readInvExDat

Reading results
---------------
.. autofunction:: petgem.readBundle

.. autofunction:: petgem.readResponses

.. autofunction:: petgem.readAllResponses

.. autofunction:: petgem.readVectorH5

.. autofunction:: petgem.compareMagnitude
