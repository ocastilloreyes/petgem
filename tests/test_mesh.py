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

    try:
        nElems == 9453
    except mesh_error:
        print("Number of elements is not consistent.")

    try:
        nNodes == 2163
    except mesh_error:
        print("Number of nodes is not consistent.")

    try:
        nFaces == 20039
    except mesh_error:
        print("Number of faces is not consistent.")

    try:
        nEdges == 12748
    except mesh_error:
        print("Number of edges is not consistent.")
