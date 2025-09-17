#!/usr/bin/env python3
"""
*********************************************************************
 Script for preprocessing mesh, resistivity data, and 
 receiver information for PETGEM simulations.
 This script processes a Gmsh-generated mesh by associating 
 resistivity values with each finite element cell and 
 incorporating receiver locations. The resulting datasets 
 are formatted and stored for use in PETGEM forward modeling 
 and inversion workflows.

 Usage:
   python3 preprocessing.py -resistivity_view vtk:canonical_model.vtu

 Notes:
   - The option `-resistivity_view vtk:canonical_model.vtu` exports 
     the resistivity distribution in VTK format, allowing direct 
     visualization with tools such as ParaView.
   - Similarly, the option `-field_view vtk:wham_model.vtu` 
     exports the simulated field data in VTK format for 
     postprocessing and analysis.

 Author: Octavio Castillo-Reyes (UPC/BSC) (octavio.castillo@upc.edu; octavio.castillo@bsc.es)
 Latest update: September 10th, 2025     
*********************************************************************
"""

import sys
import meshio
import petsc4py
import numpy as np
import h5py

petsc4py.init(sys.argv)

from petsc4py import PETSc

# ------------------------------------------------------------------------------
# USER PARAMS
# ------------------------------------------------------------------------------
nord = sys.argv[1]
input_mesh_filename = f"tests/csem_model/mesh_p{nord}.msh"
output_mesh_filename = f"tests/csem_model/resistivity_model_p{nord}.h5"
input_receivers_filename = "tests/csem_model/receivers.txt"
output_receivers_filename = "tests/csem_model/receivers.h5"
numDimensions = 3
sigma_x = np.array([3.3333, 1., 0.01, 1.], dtype=float)
sigma_y = np.array([3.3333, 1., 0.01, 1.], dtype=float)
sigma_z = np.array([3.3333, 1., 0.01, 1.], dtype=float)

# ------------------------------------------------------------------------------
# IMPORT MESH
# ------------------------------------------------------------------------------
mesh = meshio.read(input_mesh_filename)

# Determine number of materials
numMaterials = len(mesh.cells)

# Number of elements and connectivity
numCells = 0
for i in np.arange(numMaterials):
	cells = mesh.cells[i].data
	numCells += np.shape(cells)[0]

cells = mesh.cells[i].data

# Number of vertices and coordinates
coords = mesh.points
numCoords = np.shape(coords)[0]

# ------------------------------------------------------------------------------
# CREATE DM OBJECT
# ------------------------------------------------------------------------------
plex = PETSc.DMPlex().create()
plex.setFromOptions()
plex.createFromCellList(numDimensions, cells, coords)
plex.viewFromOptions("-dm_view")
dim = plex.getDimension()

# Create manually a section with 1 field, dim components on cells
plex.setNumFields(1)
numComp = dim
numDof = [0] * (dim + 1)
numDof[-1] = dim
s = plex.createSection(numComp, numDof)
s.setFieldName(0, "resistivity")
s.setUp()
plex.setSection(s)

# ------------------------------------------------------------------------------
# CREATE RESISTIVITY ARRAY 
# ------------------------------------------------------------------------------
resistivity = np.zeros((numCells, numDimensions), dtype=float)
elemsS = np.copy(mesh.cell_data_dict["gmsh:physical"]["tetra"])
elemsS -= 1

for i in np.arange(numCells):
	resistivity[i,0] = sigma_x[elemsS[i]]	
	resistivity[i,1] = sigma_y[elemsS[i]] 
	resistivity[i,2] = sigma_z[elemsS[i]]


# ------------------------------------------------------------------------------
# STORE DM OBJECT
# ------------------------------------------------------------------------------
v = plex.createGlobalVec()
v.getArray()[:] = resistivity.reshape(-1)[:]
v.viewFromOptions("-resistivity_view")

viewer = PETSc.ViewerHDF5().create(output_mesh_filename, "w")
viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
plex.setName("petgem_mesh")
plex.topologyView(viewer)
plex.labelsView(viewer)
plex.coordinatesView(viewer)
plex.sectionView(viewer, plex)
v.setName("resistivity")
plex.globalVectorView(viewer, plex, v)


# ------------------------------------------------------------------------------
# EXPORT RECEIVERS
# ------------------------------------------------------------------------------
# Here, the receivers are defined as real, but PETSc for PETGEM is configured to 
# use complex scalars. Currently, the conversion from real to complex has not been 
# implemented. Therefore, we store the data as complex, allowing PETGEM to cast it 
# from complex to real when needed.
receivers = np.loadtxt(input_receivers_filename, comments='#')
vector = PETSc.Vec().createWithArray(receivers, comm=PETSc.COMM_SELF)
vector.setName("receivers")
vector.setUp()
viewer = PETSc.Viewer().createHDF5(output_receivers_filename, mode='w', comm=PETSc.COMM_SELF)
vector.view(viewer)
