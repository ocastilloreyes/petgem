#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es

import pytest
import sys
import numpy as np
from petgem.mesh import readGmshNodes, readGmshConnectivity, computeEdges, computeFaces
from petgem.mesh import computeBoundaryFaces, computeBoundaryEdges


def test_mesh_functions():
    """Test mesh functions."""

    # Setup mesh file name
    mesh_filename = 'tests/data/test_mesh.msh'
    basis_order = 2
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

    # Verify mesh data

    try:
        nElems == 9453
        nNodes == 2163
        nFaces == 20039
        nEdges == 12748
        raise mesh_error
    except mesh_error:
        print("Data mesh is not consistent.")
