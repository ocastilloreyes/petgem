#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es

import pytest
import numpy as np
from petgem.mesh import readGmshNodes, readGmshConnectivity, computeEdges, computeFaces


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

    try:
        nNodes == 2163

    try:
        nFaces == 20039

    try:
        nEdges == 12748

    try:
        np.allclose(elemsE[0,:], np.array([10591,10600,10831,10832,10601,11465], dtype=np.int))

    try:
        np.allclose(elemsF[0,:], np.array([17369, 17370, 17400, 17977]))
