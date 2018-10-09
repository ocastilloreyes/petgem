#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' **PETGEM** preprocessing. This script provides all common utility
functions and transformer methods to change raw feature data  into a
representation that is suitable for **PETGEM** modelling.
THIS TASK IS ABSOLUTELY SEQUENTIAL.
'''
# ###############################################
# ----------------- Load modules ----------------
import sys
import numpy as np
# ###############################################
# ------------- Load PETGEM modules -------------
from petgem.base.styles import printPetgemHeader
from petgem.base.styles import printPetgemFooter
from petgem.parallel.parallel import printMessage
from petgem.preprocessing.preprocessing import readPreprocessingParams
from petgem.preprocessing.preprocessing import preprocessNodes
from petgem.preprocessing.preprocessing import preprocessingNodalConnectivity
from petgem.preprocessing.preprocessing import preprocessingEdges
from petgem.preprocessing.preprocessing import preprocessingFaces
from petgem.preprocessing.preprocessing import preprocessingConductivityModel
from petgem.preprocessing.preprocessing import preprocessingDataReceivers
from petgem.preprocessing.preprocessing import preprocessingNNZ
from petgem.preprocessing.preprocessing import printPreprocessingSummary


# ###############################################
# COMMENT: PETGEM preprocessing is a sequential task, therefore
# rank == 0 in order to use parallel PETGEM functions

# ###############################################
# ----------- Print header (Master) -------------
rank = 0
printPetgemHeader(rank)

# ###############################################
# ----------------- User input ------------------
printMessage('\nInit', rank)
printMessage('='*75, rank)
input_params = sys.argv
preprocessing = readPreprocessingParams(input_params, rank)

# ###############################################
# ----------- Check and read user input ---------
printMessage('\nData preprocessing', rank)
printMessage('='*75, rank)

# ###############################################
# ----------- Data preprocessing ---------
# Nodal coordinates preprocessing (nodes)
nNodes = preprocessNodes(preprocessing['MESH_FILE'],
                         preprocessing['OUT_DIR'], rank)

# Nodal connectivity preprocessing (elemsN)
nElems = preprocessingNodalConnectivity(preprocessing['MESH_FILE'],
                                        preprocessing['OUT_DIR'], rank)

# Edges connectivity and edge boundaries preprocessing (edges)
nEdges, nDofs = preprocessingEdges(preprocessing['NEDELEC_ORDER'],
                                   preprocessing['MESH_FILE'],
                                   preprocessing['OUT_DIR'], rank)

# Faces connectivity and faces boundaries preprocessing (faces)
nFaces = preprocessingFaces(preprocessing['NEDELEC_ORDER'],
                            preprocessing['MESH_FILE'],
                            preprocessing['OUT_DIR'], rank)

# Conductivity model
preprocessingConductivityModel(preprocessing['MESH_FILE'],
                               preprocessing['MATERIAL_CONDUCTIVITIES'],
                               preprocessing['OUT_DIR'], rank)
# Receiver positions
nReceivers = preprocessingDataReceivers(preprocessing['NEDELEC_ORDER'],
                                        preprocessing['MESH_FILE'],
                                        preprocessing['RECEIVERS_FILE'],
                                        preprocessing['OUT_DIR'], rank)

# Sparsity pattern for parallel matrix allocation
preprocessingNNZ(preprocessing['NEDELEC_ORDER'],
                 preprocessing['MESH_FILE'],
                 preprocessing['OUT_DIR'], rank)

# ###############################################
# ------------------ Summary --------------------
printMessage('\nSummary', rank)
printMessage('='*75, rank)
printPreprocessingSummary(nElems, nNodes, nFaces, nDofs, nReceivers, rank)

# ###############################################
# ----------- Print footer (Master) -------------
printPetgemFooter(rank)
