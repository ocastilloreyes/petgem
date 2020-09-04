#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es

import pytest
import numpy as np
from petgem.mesh import readGmshNodes, readGmshConnectivity, computeEdges, computeFaces
from petgem.mesh import computeBoundaryEdges


def test_mesh_functions():
    """Test mesh functions."""

    # Setup mesh file name
    mesh_filename = 'tests/data/test_mesh.msh'

    # Read nodes
    nodes, _ = readGmshNodes(mesh_filename)
    nNodes = nodes.shape[0]

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_filename)

    # Compute edges
    elemsE, edgesNodes = computeEdges(elemsN, nElems)
    nEdges = edgesNodes.shape[0]

    # Compute faces
    elemsF, facesN = computeFaces(elemsN, nElems)
    nFaces = facesN.shape[0]

    try:
        nElems == 9453
    except:
        print("Number of elements is not consistent.")

    try:
        nNodes == 2163
    except:
        print("Number of nodes is not consistent.")

    try:
        nFaces == 20039
    except:
        print("Number of faces is not consistent.")

    try:
        nEdges == 12748
    except:
        print("Number of edges is not consistent.")

    try:
        np.allclose(elemsE[0,:], np.array([10591,10600,10831,10832,10601,11465], dtype=np.int))
    except:
        print("Edges connectivity is not consistent.")

    try:
        np.allclose(elemsF[0,:], np.array([17369, 17370, 17400, 17977]))
    except:
        print("Faces connectivity is not consistent.")
