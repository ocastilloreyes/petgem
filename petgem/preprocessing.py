#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define data preprocessing operations for **PETGEM**.
'''


# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import os
import numpy as np
import h5py
from scipy.spatial import Delaunay
from petsc4py import PETSc

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .common import Print, Timers, measure_all_class_methods, measure_time
from .parallel import MPIEnvironment, createSequentialDenseMatrixWithArray
from .parallel import writeParallelDenseMatrix, createSequentialVectorWithArray
from .parallel import writePetscVector
from .mesh import readGmshNodes, readGmshConnectivity, computeEdges, computeFaces
from .mesh import computeBoundaryFaces, computeBoundaryEdges, computeBoundaries
from .mesh import readGmshPhysicalGroups
from .vectors import invConnectivity
from .hvfem import computeConnectivityDOFS

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
@measure_all_class_methods
class Preprocessing():
    ''' Class for preprocessing.
    '''
    def __init__(self):
        ''' Initialization of a preprocessing class.
        '''
        return

    def run(self, setup):
        ''' Run a preprocessing task

        :param obj setup: inputSetup object.
        :return: None
        '''

        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Preprocessing"].start()

        # Parameters shortcut (for code legibility)
        model = setup.model
        run = setup.run
        output = setup.output

        # Obtain the MPI environment
        parEnv = MPIEnvironment()

        # ---------------------------------------------------------------
        # Import mesh file (gmsh format)
        # ---------------------------------------------------------------
        # Read nodes
        nodes, nNodes = readGmshNodes(model.mesh_file)

        # Read connectivity
        elemsN, nElems = readGmshConnectivity(model.mesh_file)

        # ---------------------------------------------------------------
        # Preprocessing nodal coordinates
        # ---------------------------------------------------------------
        Print.master('     Nodal coordinates')

        # Build coordinates in PETGEM format where each row
        # represent the xyz coordinates of the 4 tetrahedral element
        num_dimensions = 3
        num_nodes_per_element = 4
        data = np.array((nodes[elemsN[:], :]), dtype=np.float)
        data = data.reshape(nElems, num_dimensions*num_nodes_per_element)

        # Get matrix dimensions
        size = data.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

        # Build path to save the file
        out_path = output.directory_scratch + '/nodes.dat'

        if parEnv.rank == 0:
            # Write PETGEM nodes in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing mesh connectivity
        # ---------------------------------------------------------------
        Print.master('     Mesh connectivity')

        # Get matrix dimensions
        size = elemsN.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsN)

        # Build path to save the file
        out_path = output.directory_scratch + '/meshConnectivity.dat'

        if parEnv.rank == 0:
            # Write PETGEM connectivity in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing edges connectivity
        # ---------------------------------------------------------------
        Print.master('     Edges connectivity')

        # Compute edges
        elemsE, edgesNodes = computeEdges(elemsN, nElems)
        nEdges = edgesNodes.shape[0]

        # Get matrix dimensions
        size = elemsE.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsE)

        # Build path to save the file
        out_path = output.directory_scratch + '/edges.dat'

        if parEnv.rank == 0:
            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # Reshape edgesNodes and save
        num_nodes_per_edge = 2
        num_edges_per_element = 6
        data = np.array((edgesNodes[elemsE[:], :]), dtype=np.float)
        data = data.reshape(nElems, num_nodes_per_edge*num_edges_per_element)

        # Get matrix dimensions
        size = data.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

        # Build path to save the file
        out_path = output.directory_scratch + '/edgesNodes.dat'

        if parEnv.rank == 0:
            # Write PETGEM edgesNodes in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing faces connectivity
        # ---------------------------------------------------------------
        Print.master('     Faces connectivity')

        # Compute faces
        elemsF, facesN = computeFaces(elemsN, nElems)
        nFaces = facesN.shape[0]

        # Get matrix dimensions
        size = elemsF.shape
        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsF)

        # Build path to save the file
        out_path = output.directory_scratch + '/faces.dat'

        if parEnv.rank == 0:
            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing faces-edges connectivity
        # ---------------------------------------------------------------
        Print.master('     Faces-edges connectivity')

        N = invConnectivity(elemsF, nFaces)

        if nElems != 1:
            N = np.delete(N, 0, axis=1)

        # Allocate
        facesE = np.zeros((nFaces, 3), dtype=np.int)

        # Compute edges list for each face
        for i in np.arange(nFaces):
            iEle = N[i, 0]
            edgesEle = elemsE[iEle,:]
            facesEle = elemsF[iEle,:]
            kFace = np.where(facesEle == i)[0]
            if kFace == 0:  # Face 1
                facesE[facesEle[kFace],:] = [edgesEle[0], edgesEle[1], edgesEle[2]]
            elif kFace == 1:  # Face 2
                facesE[facesEle[kFace],:] = [edgesEle[0], edgesEle[4], edgesEle[3]]
            elif kFace == 2:  # Face 3
                facesE[facesEle[kFace],:] = [edgesEle[1], edgesEle[5], edgesEle[4]]
            elif kFace == 3:  # Face 4
                facesE[facesEle[kFace],:] = [edgesEle[2], edgesEle[5], edgesEle[3]]

        num_faces_per_element = 4
        num_edges_per_face = 3
        data = np.array((facesE[elemsF[:], :]), dtype=np.float)
        data = data.reshape(nElems, num_faces_per_element*num_edges_per_face)

        # Get matrix dimensions
        size = data.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

        # Build path to save the file
        out_path = output.directory_scratch + '/facesEdges.dat'

        if parEnv.rank == 0:
            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing dofs connectivity
        # ---------------------------------------------------------------
        Print.master('     DOFs connectivity')

        # Compute degrees of freedom connectivity
        dofs, dof_edges, dof_faces, dof_volume, total_num_dofs = computeConnectivityDOFS(elemsE,elemsF,model.basis_order)

        # Get matrix dimensions
        size = dofs.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], dofs)

        # Build path to save the file
        out_path = output.directory_scratch + '/dofs.dat'

        if parEnv.rank == 0:
            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing boundaries
        # ---------------------------------------------------------------
        Print.master('     Boundaries')

        # Compute boundary faces
        bFacesN, bFaces = computeBoundaryFaces(elemsF, facesN)

        # Compute boundary edges
        bEdges = computeBoundaryEdges(edgesNodes, bFacesN)

        # Compute dofs on boundaries
        indx_inner_dofs, indx_boundary_dofs = computeBoundaries(dofs, dof_edges, dof_faces, bEdges, bFaces, model.basis_order);

        # Build PETSc structures
        vector = createSequentialVectorWithArray(indx_boundary_dofs)

        # Build path to save the file
        out_path = output.directory_scratch + '/boundaries.dat'

        if parEnv.rank == 0:
            # Write PETGEM nodes in PETSc format
            writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing sigma model
        # ---------------------------------------------------------------
        Print.master('     Conductivity model')

        # Read element's tag
        elemsS, nElems = readGmshPhysicalGroups(model.mesh_file)

        # Number of materials
        nMaterials = elemsS.max()

        # Build conductivity arrays
        conductivityModel = np.zeros((nElems, 2), dtype=np.float)
        for i in np.arange(nElems):
            # Set horizontal sigma
            conductivityModel[i, 0] = model.sigma_horizontal[np.int(elemsS[i])]
            # Set vertical sigma
            conductivityModel[i, 1] = model.sigma_vertical[np.int(elemsS[i])]

        # Get matrix dimensions
        size = conductivityModel.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], conductivityModel)

        # Build path to save the file
        out_path = output.directory_scratch + '/conductivityModel.dat'

        if parEnv.rank == 0:
            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Preprocessing receivers
        # ---------------------------------------------------------------
        Print.master('     Receivers')

        # Open receivers_file
        fileID = h5py.File(model.receivers_file, 'r')

        # Read receivers
        receivers = fileID.get('data')[()]

        # Number of receivers
        if receivers.ndim == 1:
            nReceivers = 1
        else:
            dim = receivers.shape
            nReceivers = dim[0]

        # Build Delaunay triangulation with nodes
        tri = Delaunay(nodes)

        # Overwrite Delaunay structure with mesh_file connectivity and points
        tri.simplices = elemsN.astype(np.int32)
        tri.vertices = elemsN.astype(np.int32)

        # Find out which tetrahedral element points are in
        recvElems = tri.find_simplex(receivers, bruteforce=True, tol=1.e-12)

        # Find out which tetrahedral element source point is in
        srcElem = tri.find_simplex(model.src_position, bruteforce=True, tol=1.e-12)

        # Determine if all receiver points were found
        idx = np.where(np.logical_or(recvElems>nElems, recvElems<0))[0]

        # If idx is not empty, there are receivers outside the domain
        if idx.size != 0:
            Print.master('        The following receivers were not located and will not be taken into account ' + str(idx))
            # Update number of receivers
            nReceivers = nReceivers - len(idx)

            if nReceivers == 0:
                Print.master('     No receiver has been found. Nothing to do. Aborting')
                exit(-1)

            # Remove idx from receivers matrix
            receivers = np.delete(receivers, idx, axis=0)

            # Remove idx from recvElems
            recvElems = np.delete(recvElems, idx, axis=0)

        # If srcElem is empty, source not located
        if srcElem == 0:
            Print.master('        Source no located in the computational domain. Please, improve the mesh quality')
            exit(-1)

        # Compute number of dofs per element
        num_dof_in_element = np.int(model.basis_order*(model.basis_order+2)*(model.basis_order+3)/2)

        # Allocate
        data_receiver = np.zeros((nReceivers, 53+num_dof_in_element), dtype=np.float)

        # Fill tmp matrix with receiver positions, element coordinates and
        # nodal indexes
        for i in np.arange(nReceivers):
            # If there is one receiver
            if nReceivers == 1:
                # Get index of tetrahedral element (receiver container)
                iEle = recvElems
                # Get dofs of element container
                dofsElement = dofs[iEle]
            # If there are more than one receivers
            else:
                # Get index of tetrahedral element (receiver container)
                iEle = recvElems[i]
                # Get dofs of element container
                dofsElement = dofs[iEle, :]

            # Get indexes of nodes for iand insert
            nodesReceiver = elemsN[iEle, :]
            data_receiver[i, 0:4] = nodesReceiver
            # Get nodes coordinates for i and insert
            coordEle = nodes[nodesReceiver, :]
            coordEle = coordEle.flatten()
            data_receiver[i, 4:16] = coordEle
            # Get indexes of faces for i and insert
            facesReceiver = elemsF[iEle, :]
            data_receiver[i, 16:20] = facesReceiver
            # Get edges indexes for faces in i and insert
            edgesReceiver = facesE[facesReceiver, :]
            edgesReceiver = edgesReceiver.flatten()
            data_receiver[i, 20:32] = edgesReceiver
            # Get indexes of edges for i and insert
            edgesReceiver = elemsE[iEle, :]
            data_receiver[i, 32:38] = edgesReceiver
            # Get node indexes for edges in i and insert
            edgesNodesReceiver = edgesNodes[edgesReceiver, :]
            edgesNodesReceiver = edgesNodesReceiver.flatten()
            data_receiver[i, 38:50] = edgesNodesReceiver
            # Get receiver coordinates
            coordReceiver = receivers[i,: ]
            data_receiver[i, 50:53] = coordReceiver
            # Get dofs for srcElem and insert
            dofsReceiver = dofsElement
            data_receiver[i, 53::] = dofsReceiver

        # Get matrix dimensions
        size = data_receiver.shape

        # Build PETSc structures
        matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data_receiver)

        # Build path to save the file
        out_path = output.directory_scratch + '/receivers.dat'

        if parEnv.rank == 0:
            # Write PETGEM receivers in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

        # Compute number of dofs per element
        num_dof_in_element = np.int(model.basis_order*(model.basis_order+2)*(model.basis_order+3)/2)

        # Build data for source insertion
        vector = np.zeros(50+num_dof_in_element, dtype=np.float)

        # Get indexes of nodes for srcElem and insert
        nodesSource = elemsN[srcElem, :]
        vector[0:4] = nodesSource
        # Get nodes coordinates for srcElem and insert
        coordSource = nodes[nodesSource, :]
        coordSource = coordSource.flatten()
        vector[4:16] = coordSource
        # Get indexes of faces for srcElem and insert
        facesSource = elemsF[srcElem, :]
        vector[16:20] = facesSource
        # Get edges indexes for faces in srcElem and insert
        edgesFace = facesE[facesSource, :]
        edgesFace = edgesFace.flatten()
        vector[20:32] = edgesFace
        # Get indexes of edges for srcElem and insert
        edgesSource = elemsE[srcElem, :]
        vector[32:38] = edgesSource
        # Get node indexes for edges in srcElem and insert
        edgesNodesSource = edgesNodes[edgesSource, :]
        edgesNodesSource = edgesNodesSource.flatten()
        vector[38:50] = edgesNodesSource
        # Get dofs for srcElem and insert
        dofsSource = dofs[srcElem,:]
        vector[50::] = dofsSource

        # Build PETSc structures
        vector = createSequentialVectorWithArray(vector)

        # Build path to save the file
        out_path = output.directory_scratch + '/source.dat'

        if parEnv.rank == 0:
            # Write PETGEM nodes in PETSc format
            writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Sparsity pattern
        # ---------------------------------------------------------------
        # Setup valence for each basis order (adding a small percentage to keep safe)
        valence = np.array([50, 200, 400, 800, 1400, 2500])

        # Build nnz pattern for each row
        nnz = np.full((total_num_dofs), valence[model.basis_order-1], dtype=np.int)

        # Build PETSc structures
        vector = createSequentialVectorWithArray(nnz)

        # Build path to save the file
        out_path = output.directory_scratch + '/nnz.dat'

        if parEnv.rank == 0:
            # Write PETGEM nodes in PETSc format
            writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

        # ---------------------------------------------------------------
        # Print mesh statistics
        # ---------------------------------------------------------------
        Print.master(' ')
        Print.master('  Mesh statistics')
        Print.master('     Number of elements:   {0:12}'.format(str(nElems)))
        Print.master('     Number of faces:      {0:12}'.format(str(nFaces)))
        Print.master('     Number of edges:      {0:12}'.format(str(nEdges)))
        Print.master('     Number of dofs:       {0:12}'.format(str(total_num_dofs)))
        Print.master('     Number of boundaries: {0:12}'.format(str(len(indx_boundary_dofs))))

        # ---------------------------------------------------------------
        # Print data model
        # ---------------------------------------------------------------
        Print.master(' ')
        Print.master('  Model data')
        Print.master('     Number of materials:    {0:12}'.format(str(np.max(elemsS)+1)))
        Print.master('     Vector basis order:     {0:12}'.format(str(model.basis_order)))
        Print.master('     Frequency (Hz):         {0:12}'.format(str(model.frequency)))
        Print.master('     Source position (xyz):  {0:12}'.format(str(model.src_position)))
        Print.master('     Source azimuth:         {0:12}'.format(str(model.src_azimuth)))
        Print.master('     Source dip:             {0:12}'.format(str(model.src_dip)))
        Print.master('     Source current:         {0:12}'.format(str(model.src_current)))
        Print.master('     Source length:          {0:12}'.format(str(model.src_length)))
        Print.master('     Sigma horizontal:       {0:12}'.format(str(model.sigma_horizontal)))
        Print.master('     Sigma vertical:         {0:12}'.format(str(model.sigma_vertical)))
        Print.master('     Number of receivers:    {0:12}'.format(str(nReceivers)))

        # Apply barrier for MPI tasks alignement
        parEnv.comm.barrier()

        # Stop timer
        Timers()["Preprocessing"].stop()


# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################

def unitary_test():
    ''' Unitary test for parallel.py script.
    '''

# ###############################################################
# ################             MAIN             #################
# ###############################################################

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
