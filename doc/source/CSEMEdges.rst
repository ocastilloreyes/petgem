.. _CSEM & Edge elements formulation:

CSEM forward modelling & High-order edge finite element method
==============================================================

The last decade has been a period of rapid growth for electromagnetic
methods (EM) in geophysics, mostly because of their industrial adoption.
In particular, the 3D marine controlled-source electromagnetic (3D CSEM) method
has become an important technique for reducing ambiguities in data
interpretation in hydrocarbon exploration. In order to be able to predict
the EM signature of a given geological structure, modelling tools provide us
with synthetic results which we can then compare to measured data. In particular,
if the geology is structurally complex, one might need to use methods able
to cope with such complexity in a natural way by means of, e.g., an
unstructured mesh representing its geometry. Among the modelling methods
for EM based upon 3D unstructured meshes, the high-order Nédélec Edge Finite Element
Method (HEFEM) offers a good trade-off between accuracy and number of degrees
of freedom, e.g. size of the problem. Furthermore, its divergence-free basis
is very well suited for solving Maxwell’s equation. On top of that, we choose
to support tetrahedral meshes, as these are the easiest to use for very large
domains or complex geometries.

We refer to the following papers for a complete discussion of
marine 3-D CSEM modelling and its problem statement within PETGEM:

* Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J.M. (2019). `Parallel 3-D marine controlled-source electromagnetic modelling using high-order tetrahedral Nédélec elements <https://doi.org/10.1093/gji/ggz285>`_. Geophysical Journal International, Volume 219, Issue 1, October 2019, Pages 39–65.

* Castillo-Reyes, O., de la Puente, Cela, J. M. `PETGEM: A parallel code for 3D CSEM forward modeling using edge finite elements <https://doi.org/10.1016/j.cageo.2018.07.005>`_. Computers & Geosciences, vol 119: 123-136. ISSN 0098-3004. Elsevier.
