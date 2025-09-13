############
About PETGEM
############

Overview
--------
PETGEM (Parallel Exascale Toolkit for Geophysical Electromagnetic Modeling) is an HPC-ready software suite
designed for simulating electromagnetic wave propagation in 3D subsurface models. PETGEM supports
MPI parallelism, unstructured tetrahedral meshes, and high-order vector finite element methods.

Key Features
------------
- High-order vector finite element method for controlled-source EM problems
- Support for unstructured tetrahedral meshes
- Parallel computing with MPI and PETSc
- Python bindings for pre- and post-processing
- Integration with mesh generators (`Gmsh <http://gmsh.info/>`_) and performance tools (`Extrae <https://tools.bsc.es/extrae>`_)

Applications
------------
PETGEM is suitable for:

- Geothermal reservoir exploration
- Oil & gas subsurface imaging
- Environmental EM surveys
- Academic research in geophysics

References
----------
For more information, see:

- `PETGEM GitHub repository <https://github.com/ocastilloreyes/petgem/>`_
- Publications using PETGEM: 
	- Rulff, P., Deleersnyder, W., Castillo-Reyes, O., Carrizo Mascarell, M., King, J. *An evaluation of computational methods in electromagnetic geophysics and their potential for groundwater system imaging.* EGU General Assembly 2025. `DOI: 10.5194/egusphere-egu25-5895 <https://doi.org/10.5194/egusphere-egu25-5895>`_
	- Castillo-Reyes, O., Orihuela García, X., Piña Suárez, X. *Designing a mockup and refactoring code for HPC geo-electromagnetic applications.*  2024 IEEE International Conference on Engineering Veracruz (ICEV). `DOI: 10.1109/ICEV63254.2024.10765935 <https://doi.org/10.1109/ICEV63254.2024.10765935>`_
	- Castillo-Reyes, O., Ledesma-Prol, R.M., Corbo-Camargo, F., Rojas, O. *Geothermal resources in Latin-America and their exploration using electromagnetic methods.* Geothermal Energy. `DOI: 10.1186/s40517-024-00314-5 <https://doi.org/10.1186/s40517-024-00314-5>`_
	- Castillo-Reyes, O., Queralt, P., Piñas-Varas, P., Ledo, J., Rojas, O. *Electromagnetic subsurface imaging in the presence of metallic structures: A review of numerical strategies.*  Surveys in Geophysics. `DOI: 10.1007/s10712-024-09855-7 <https://doi.org/10.1007/s10712-024-09855-7>`_
	- Rulff, P., Castillo-Reyes, O., Koyan, P., Martin, T., Deleersnyder, W., Carrizo Mascarell, M. *Geoelectrical and electromagnetic imaging methods applied to groundwater systems: recent advances and future potentials.*  EGU General Assembly 2024. `DOI: 10.5194/egusphere-egu24-654 <https://doi.org/10.5194/egusphere-egu24-654>`_
	- Castillo-Reyes, O., Rulff, P., Um, E., Amor-Martin, A. *Meshing strategies for 3D geo-electromagnetic modeling in the presence of metallic infrastructure.*  Computational Geosciences. `DOI: 10.1007/s10596-023-10247-w <https://doi.org/10.1007/s10596-023-10247-w>`_
	- Castillo-Reyes, O., Hu, X., Wang, B., Wang, Y., Guo, Z. *Electromagnetic imaging and deep learning for transition to renewable energies: a technology review.* Frontiers in Earth Science. `DOI: 10.3389/feart.2023.1159910 <https://doi.org/10.3389/feart.2023.1159910>`_
	- Castillo-Reyes, O., Amor-Martin, A., Botella, A., Pierre, A., García-Castillo, L.E. *Tailored meshing for parallel 3D electromagnetic modeling using high-order edge elements.* Journal of Computational Science. `DOI: 10.1016/j.jocs.2022.101813 <https://doi.org/10.1016/j.jocs.2022.101813>`_
	- Castillo-Reyes, O., de la Puente, J., Cela, J.M. *HPC geophysical electromagnetics: A synthetic VTI model with complex bathymetry.* Energies, vol. 15:1272. `DOI: 10.3390/en15041272 <https://doi.org/10.3390/en15041272>`_
	- Castillo-Reyes, O., Modesto, D., Queralt, P., Marcuello, A., Ledo, J., Amor-Martin, A., de la Puente, J., García-Castillo, L.E. *3D magnetotelluric modeling using high-order tetrahedral Nédélec elements on massively parallel computing platforms.* Computers & Geosciences. `DOI: 10.1016/j.cageo.2021.105030 <https://doi.org/10.1016/j.cageo.2021.105030>`_
	- Werthmüller, D., Rochlitz, R., Castillo-Reyes, O., Heagy, L. *Towards an open-source landscape for 3-D CSEM modelling.* Geophysical Journal International. `DOI: 10.1093/gji/ggab238 <https://doi.org/10.1093/gji/ggab238>`_
	- Castillo-Reyes, O., Queralt, P., Marcuello, A., Ledo, J. *Land CSEM simulations and experimental test using metallic casing in a geothermal exploration context: Vallès Basin (NE Spain) case study.* IEEE Transactions on Geoscience and Remote Sensing. `DOI: 10.1109/TGRS.2021.3069042 <https://doi.org/10.1109/TGRS.2021.3069042>`_
	- Castillo-Reyes, O., de la Puente, García-Castillo, L.E., Cela, J.M. *Parallel 3D marine controlled-source electromagnetic modeling using high-order tetrahedral Nédélec elements.* Geophysical Journal International. `DOI: 10.1093/gji/ggz285 <https://doi.org/10.1093/gji/ggz285>`_
	- Castillo-Reyes, O., de la Puente, Cela, J.M. *PETGEM: A parallel code for 3D CSEM forward modeling using edge finite elements.* Computers & Geosciences. `DOI: 10.1016/j.cageo.2018.07.005 <https://doi.org/10.1016/j.cageo.2018.07.005>`_