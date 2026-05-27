==================
Python API
==================

**PETGEM** ships a small Python package for pre- and post-processing, imported
as ``petgem``. It assembles the unified input bundle from mesh/model/source
files and reads the bundle and the kernel responses back into NumPy arrays.
These are the functions used by ``utils/preprocess.py`` and the per-case
``postprocess.py`` scripts; the underlying file formats are documented in
:doc:`formats`.

Preprocessing
-------------
.. autofunction:: petgem.runPreprocessing

Readers
-------
.. autofunction:: petgem.readBundle

.. autofunction:: petgem.readResponses

.. autofunction:: petgem.readSigmaTable

.. autofunction:: petgem.readObservedDataH5

.. autofunction:: petgem.readInvExDat
