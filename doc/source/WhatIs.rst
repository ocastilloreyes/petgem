.. _What is PETGEM:

What is PETGEM?
===============
Electromagnetic methods (EM) are an established tool in geophysics, with
application in many areas such as hydrocarbon and mineral exploration,
reservoir monitoring, CO\ :sub:`2` storage characterization, geothermal reservoir
imaging and many others. In particular, the marine Controlled-Source
Electromagnetic method (CSEM) has become an important technique for reducing
ambiguities in data interpretation for hydrocarbon exploration. In order to be
able to predict the EM signature of a given geological structure, modelling
tools provide us with synthetic results which we can then compare to real data.
In particular, if the geology is structurally complex, one might need to use
methods able to cope with such complexity in a natural way by means of, e.g.,
an unstructured mesh representing its geometry. Among the modelling methods
for EM based upon 3D unstructured meshes, the High-order Nédélec Finite Elements (FE),
a type of Edge Finite Elements, offer a good trade-off between accuracy and number
of degrees of freedom, i.e. size of the problem.

In the multi-core and many-core era, parallelization is a crucial issue.
Nédélec FE offer good scalability potential. Its low DOF number make them potentially fast, which is
crucial in the future goal of solving inverse problems which might
involve over 100,000 realizations (e.g. within a inversion routine). However, the state of the art shows a
relative scarcity of robust high-order edge-based codes to simulate these problems.

On top of that, **Parallel Edge-based Tool for Geophysical Electromagnetic
Modelling** (PETGEM) is a `Python <https://www.python.org/>`_ tool
for the scalable solution of EM on tetrahedral meshes, as these are the
easiest to scale-up to very large domains or arbitrary shape. It supports
distributed-memory paralelism through `petsc4py <https://pypi.python.org/pypi/petsc4py>`__
package.

As a result, PETGEM tool allow users to specify high-order edge-based
variational forms of H(curl) for the simulation of electromagnetic fields
in realistic 3D CSEM surveys with accuracy, reliability and efficiency.

PETGEM is developed as open-source under
:download:`BSD-3 <_downloads/BSD-3.pdf>` license at Computer
Applications in Science & Engineering
(`CASE <http://www.bsc.es/computer-applications>`_)
of the Barcelona Supercomputing Center (`BSC <http://www.bsc.es/>`_).
Requests and contributions are welcome.
