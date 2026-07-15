==========
Python API
==========

The Python package under ``utils/`` (importable as ``petgem``) handles pre- and
post-processing: it assembles the input bundle from the mesh, model, source, and
receiver files, and reads the bundle and the kernel responses back into NumPy
arrays. These are the functions used by ``utils/preprocess.py`` and by the
per-case ``postprocess.py`` scripts. The file formats are documented in
:doc:`formats`.

Preprocessing
-------------
.. autofunction:: petgem.runPreprocessing

Readers
-------
.. autofunction:: petgem.readBundle

.. autofunction:: petgem.readResponses

.. autofunction:: petgem.readAllResponses

.. autofunction:: petgem.readSigmaTable

.. autofunction:: petgem.readSourcesText

.. autofunction:: petgem.readObservedDataH5

.. autofunction:: petgem.readInvExDat

Postprocessing
--------------
.. autofunction:: petgem.compareMagnitude
