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
    assert nElems == 9453, "Wrong number of elements"
    assert nNodes == 2163, "Wrong number of nodes"
    assert nFaces == 20039, "Wrong number of faces"
    assert nEdges == 12748, "Wrong number of edges"
