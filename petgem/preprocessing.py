#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define data preprocessing operations for **PETGEM**."""


# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
import h5py
import meshio
from scipy.spatial import Delaunay
from petsc4py import PETSc
from scipy.io import savemat

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .common import Print, Timers, measure_all_class_methods
from .parallel import MPIEnvironment, createSequentialDenseMatrixWithArray
from .parallel import writeParallelDenseMatrix, createSequentialVectorWithArray
from .parallel import writePetscVector
from .mesh import computeEdges, computeBoundaryEdges, computeFacesEdges
from .mesh import computeFaces, computeBoundaryFaces
from .mesh import computeBoundaryElements, computeBoundaries, computeFacePlane
from .hvfem import computeConnectivityDOFS


# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
@measure_all_class_methods
class Preprocessing():
    """Class for preprocessing."""

    def __init__(self):
        """Initialization of a preprocessing class."""
        return

    def run(self, inputSetup):
        """Run a preprocessing task.

        :param obj inputSetup: inputSetup object.
        :return: None
        """
        # ---------------------------------------------------------------
        # Obtain the MPI environment
        # ---------------------------------------------------------------
        parEnv = MPIEnvironment()

        # Start timer
        Timers()["Preprocessing"].start()

        # ---------------------------------------------------------------
        # Preprocessing (sequential task)
        # ---------------------------------------------------------------
        if( parEnv.rank == 0 ):
            # Parameters shortcut (for code legibility)
            model = inputSetup.model
            run = inputSetup.run
            output = inputSetup.output
            out_dir = output.get('directory_scratch')
            # Compute number of dofs per element
            basis_order = run.get('nord')
            num_dof_in_element = np.int(basis_order*(basis_order+2)*(basis_order+3)/2)
            if (model.get('mode') == 'csem'):
                mode = 'csem'
            elif (model.get('mode') == 'mt'):
                mode = 'mt'
            # Get data model
            data_model = model.get(mode)

            # ---------------------------------------------------------------
            # Import mesh file
            # ---------------------------------------------------------------
            mesh_file = model.get('mesh')
            # Import mesh
            mesh = meshio.read(mesh_file)
            # Number of elements
            size = mesh.cells[0][1][:].shape
            nElems = size[0]
            # Number of nodes
            size = mesh.points.shape
            nNodes = size[0]

            # ---------------------------------------------------------------
            # Preprocessing nodal coordinates
            # ---------------------------------------------------------------
            Print.master('     Nodal coordinates')

            # Build coordinates in PETGEM format where each row
            # represent the xyz coordinates of the 4 tetrahedral element
            num_dimensions = 3
            num_nodes_per_element = 4
            data = mesh.points[mesh.cells[0][1][:], :]
            data = data.reshape(nElems, num_dimensions*num_nodes_per_element)

            # Get matrix dimensions
            size = data.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

            # Build path to save the file
            out_path = out_dir + '/nodes.dat'

            # Write PETGEM nodes in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            # Remove temporal matrix
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing mesh connectivity
            # ---------------------------------------------------------------
            Print.master('     Mesh connectivity')

            # Get matrix dimensions
            size = mesh.cells[0][1][:].shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], mesh.cells[0][1][:])

            # Build path to save the file
            out_path = out_dir + '/meshConnectivity.dat'

            # Write PETGEM connectivity in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            # Remove temporal matrix
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing edges connectivity
            # ---------------------------------------------------------------
            Print.master('     Edges connectivity')

            # Compute edges
            elemsE, edgesNodes = computeEdges(mesh.cells[0][1][:], nElems)
            nEdges = edgesNodes.shape[0]

            # Get matrix dimensions
            size = elemsE.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsE)

            # Build path to save the file
            out_path = out_dir + '/edges.dat'

            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            # Remove temporal matrix
            del matrix

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
            out_path = out_dir + '/edgesNodes.dat'

            # Write PETGEM edgesNodes in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            # Remove temporal matrix
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing faces connectivity
            # ---------------------------------------------------------------
            Print.master('     Faces connectivity')

            # Compute faces
            elemsF, facesN = computeFaces(mesh.cells[0][1][:], nElems)
            nFaces = facesN.shape[0]

            # Get matrix dimensions
            size = elemsF.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsF)

            # Build path to save the file
            out_path = out_dir + '/faces.dat'

            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            # Remove temporal matrix
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing faces-edges connectivity
            # ---------------------------------------------------------------
            Print.master('     Faces-edges connectivity')

            facesE = computeFacesEdges(elemsF, elemsE, nFaces, nElems)

            num_faces_per_element = 4
            num_edges_per_face = 3
            data = np.array((facesE[elemsF[:], :]), dtype=np.float)
            data = data.reshape(nElems, num_faces_per_element*num_edges_per_face)

            # Get matrix dimensions
            size = data.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

            # Build path to save the file
            out_path = out_dir + '/facesEdges.dat'

            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing dofs connectivity
            # ---------------------------------------------------------------
            Print.master('     DOFs connectivity')

            # Compute degrees of freedom connectivity
            basis_order = run.get('nord')
            dofs, dof_edges, dof_faces, _, total_num_dofs = computeConnectivityDOFS(elemsE,elemsF,basis_order)

            # Get matrix dimensions
            size = dofs.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], dofs)

            # Build path to save the file
            out_path = out_dir + '/dofs.dat'

            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing sigma model
            # ---------------------------------------------------------------
            Print.master('     Conductivity model')

            i_model = data_model.get('sigma')

            if (run.get('conductivity_from_file')):
                Print.master('     Interpolation from file not supported.')
                Print.master('     Using a constant conductivity model.')
                #exit(-1)
                # Add function to interpolate data from file
                # Allocate conductivity array
                conductivityModel = np.ones((nElems, 2), dtype=np.float)
            else:
                # Get physical groups
                elemsS = mesh.cell_data['gmsh:physical'][0]
                elemsS -= np.int(1)     # 0-based indexing

                # Get horizontal sigma
                horizontal_sigma = i_model.get('horizontal')
                vertical_sigma = i_model.get('vertical')

                # Allocate conductivity array
                conductivityModel = np.zeros((nElems, 2), dtype=np.float)

                for i in np.arange(nElems):
                    # Set horizontal sigma
                    conductivityModel[i, 0] = horizontal_sigma[np.int(elemsS[i])]

                    # Set vertical sigma
                    conductivityModel[i, 1] = vertical_sigma[np.int(elemsS[i])]

            # Get matrix dimensions
            size = conductivityModel.shape

            # Build PETSc structures
            matrix = createSequentialDenseMatrixWithArray(size[0], size[1], conductivityModel)

            # Build path to save the file
            out_path = out_dir + '/conductivityModel.dat'

            # Write PETGEM edges in PETSc format
            writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
            del matrix

            # ---------------------------------------------------------------
            # Preprocessing boundaries
            # ---------------------------------------------------------------
            Print.master('     Boundaries')

            # Compute boundary faces
            bFacesN, bFaces, nbFaces = computeBoundaryFaces(elemsF, facesN)

            # Build array with boundary dofs for csem mode (dirichlet BC)
            if (mode == 'csem'):
                # Compute boundary edges
                bEdges = computeBoundaryEdges(edgesNodes, bFacesN)

                # Compute dofs on boundaries
                _, indx_boundary_dofs = computeBoundaries(dofs, dof_edges, dof_faces, bEdges, bFaces, basis_order);

                # Build PETSc structures
                vector = createSequentialVectorWithArray(indx_boundary_dofs)

                # Build path to save the file
                out_path = out_dir + '/boundaries.dat'

                # Write PETGEM nodes in PETSc format
                writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)
                del vector

            elif (mode == 'mt'):
                # Compute to what plane the boundary face belongs
                planeFace = computeFacePlane(mesh.points, bFaces, bFacesN)

                # Compute boundary elements
                bElems, numbElems = computeBoundaryElements(elemsF, bFaces, nFaces)

                if (nbFaces != numbElems):
                    Print.master('     Number of boundary faces is not consistent.')
                    exit(-1)

                # Allocate
                data_boundaries = np.zeros((nbFaces, 53+num_dof_in_element), dtype=np.float)

                # Fill tmp matrix with data for boundary faces
                for i in np.arange(nbFaces):
                    # Get index of tetrahedral element (boundary element)
                    iEle = bElems[i]
                    # Get dofs of element container
                    dofsElement = dofs[iEle, :]

                    # Get indexes of nodes for i-boundary element and insert
                    nodesBoundaryElement = mesh.cells[0][1][iEle,:]
                    data_boundaries[i, 0:4] = nodesBoundaryElement
                    # Get nodes coordinates for i-boundary element and insert
                    coordEle = mesh.points[nodesBoundaryElement, :]
                    coordEle = coordEle.flatten()
                    data_boundaries[i, 4:16] = coordEle
                    # Get indexes of faces for i-boundary element and insert
                    facesBoundaryElement = elemsF[iEle, :]
                    data_boundaries[i, 16:20] = facesBoundaryElement
                    # Get edges indexes for faces in i-boundary element and insert
                    edgesBoundaryFace = facesE[facesBoundaryElement, :]
                    edgesBoundaryFace = edgesBoundaryFace.flatten()
                    data_boundaries[i, 20:32] = edgesBoundaryFace
                    # Get indexes of edges for i-boundary and insert
                    edgesBoundaryElement = elemsE[iEle, :]
                    data_boundaries[i, 32:38] = edgesBoundaryElement
                    # Get node indexes for edges in i-boundary and insert
                    edgesNodesBoundaryElement = edgesNodes[edgesBoundaryElement, :]
                    edgesNodesBoundaryElement = edgesNodesBoundaryElement.flatten()
                    data_boundaries[i, 38:50] = edgesNodesBoundaryElement
                    # Get plane face
                    ifacetype = planeFace[i]
                    data_boundaries[i, 50] = ifacetype
                    # Get global face index
                    localFaceIndex = bFaces[i]
                    data_boundaries[i, 51] = localFaceIndex
                    # Get sigma value
                    sigmaEle = conductivityModel[iEle, 0]
                    data_boundaries[i, 52] = sigmaEle
                    # Get dofs for boundary element and insert
                    dofsBoundaryElement = dofsElement
                    data_boundaries[i, 53::] = dofsBoundaryElement

                # Get matrix dimensions
                size = data_boundaries.shape

                # Build PETSc structures
                matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data_boundaries)

                # Build path to save the file
                out_path = out_dir + '/boundaryElements.dat'

                # Write PETGEM receivers in PETSc format
                writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)
                del matrix
                del data_boundaries


            # ---------------------------------------------------------------
            # Preprocessing receivers
            # ---------------------------------------------------------------
            Print.master('     Receivers')

            # Open receivers_file
            receivers_file = model.get('receivers')
            fileID = h5py.File(receivers_file, 'r')

            # Read receivers
            receivers = fileID.get('data')[()]

            # Number of receivers
            if receivers.ndim == 1:
                nReceivers = 1
            else:
                dim = receivers.shape
                nReceivers = dim[0]

            # Find out which tetrahedral element source point is in (only for csem mode)
            if (mode == 'csem'):
                # Allocate vector to save source data
                data_source = np.zeros(50+num_dof_in_element, dtype=np.float)

                i_model = data_model.get('source')

                # Get source position
                i_source_position = np.asarray(i_model.get('position'), dtype=np.float)

                # Build Delaunay triangulation with nodes
                tri = Delaunay(mesh.points)

                # Overwrite Delaunay structure with mesh_file connectivity and points
                tri.simplices = mesh.cells[0][1][:].astype(np.int32)
                tri.vertices = mesh.cells[0][1][:].astype(np.int32)

                srcElem = tri.find_simplex(i_source_position, bruteforce=True, tol=1.e-12)

                # If srcElem=-1, source not located
                if srcElem < 0:
                    Print.master('        Source no located in the computational domain. Please, verify source position or improve the mesh quality.')
                    exit(-1)

                # Build data for source insertion
                # Get indexes of nodes for srcElem and insert
                nodesSource = mesh.cells[0][1][srcElem,:]
                data_source[0:4] = nodesSource
                # Get nodes coordinates for srcElem and insert
                coordSource = mesh.points[nodesSource, :]
                coordSource = coordSource.flatten()
                data_source[4:16] = coordSource
                # Get indexes of faces for srcElem and insert
                facesSource = elemsF[srcElem, :]
                data_source[16:20] = facesSource
                # Get edges indexes for faces in srcElem and insert
                edgesFace = facesE[facesSource, :]
                edgesFace = edgesFace.flatten()
                data_source[20:32] = edgesFace
                # Get indexes of edges for srcElem and insert
                edgesSource = elemsE[srcElem, :]
                data_source[32:38] = edgesSource
                # Get node indexes for edges in srcElem and insert
                edgesNodesSource = edgesNodes[edgesSource, :]
                edgesNodesSource = edgesNodesSource.flatten()
                data_source[38:50] = edgesNodesSource
                # Get dofs for srcElem and insert
                dofsSource = dofs[srcElem,:]
                data_source[50::] = dofsSource

                # Get matrix dimensions
                size = data_source.shape

                # Build PETSc structures
                vector = createSequentialVectorWithArray(data_source)

                # Build path to save the file
                out_path = out_dir + '/source.dat'

                # Write PETGEM nodes in PETSc format
                writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)
                del vector

            # ---------------------------------------------------------------
            # Sparsity pattern
            # ---------------------------------------------------------------
            # Setup valence for each basis order (adding a small percentage to keep safe)
            valence = np.array([50, 200, 400, 800, 1400, 2500])

            # Build nnz pattern for each row
            nnz = np.full((total_num_dofs), valence[basis_order-1], dtype=np.int)

            # Build PETSc structures
            vector = createSequentialVectorWithArray(nnz)

            # Build path to save the file
            out_path = out_dir + '/nnz.dat'

            # Write PETGEM nodes in PETSc format
            writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

            # ---------------------------------------------------------------
            # Print mesh statistics
            # ---------------------------------------------------------------
            Print.master(' ')
            Print.master('  Mesh statistics')
            Print.master('     Mesh file:            {0:12}'.format(str(model.get('mesh'))))
            Print.master('     Number of elements:   {0:12}'.format(str(nElems)))
            Print.master('     Number of faces:      {0:12}'.format(str(nFaces)))
            Print.master('     Number of edges:      {0:12}'.format(str(nEdges)))
            Print.master('     Number of dofs:       {0:12}'.format(str(total_num_dofs)))
            if (mode == 'csem'):
                Print.master('     Number of boundaries: {0:12}'.format(str(len(indx_boundary_dofs))))

            # ---------------------------------------------------------------
            # Print data model
            # ---------------------------------------------------------------
            Print.master(' ')
            Print.master('  Model data')
            Print.master('     Modeling mode:       {0:12}'.format(str(mode)))
            i_sigma = data_model.get('sigma')

            if (run.get('conductivity_from_file')):
                Print.master('     Conductivity file:   {0:12}'.format(i_sigma.get('file')))
            else:
                Print.master('     Horizontal conductivity:  {0:12}'.format(str(i_sigma.get('horizontal'))))
                Print.master('     Vertical conductivity:    {0:12}'.format(str(i_sigma.get('vertical'))))

            if (mode == 'csem'):
                i_source = data_model.get('source')
                Print.master('     Source:')
                Print.master('      - Frequency (Hz):  {0:12}'.format(str(i_source.get('frequency'))))
                Print.master('      - Position (xyz):  {0:12}'.format(str(i_source.get('position'))))
                Print.master('      - Azimuth:         {0:12}'.format(str(i_source.get('azimuth'))))
                Print.master('      - Dip:             {0:12}'.format(str(i_source.get('dip'))))
                Print.master('      - Current:         {0:12}'.format(str(i_source.get('current'))))
                Print.master('      - Length:          {0:12}'.format(str(i_source.get('length'))))
            else:
                Print.master('     Frequency (Hz):           {0:12}'.format(str(data_model.get('frequency'))))
                Print.master('     Polarization:             {0:12}'.format(str(data_model.get('polarization'))))

            Print.master('     Vector basis order:       {0:12}'.format(str(basis_order)))
            Print.master('     Receivers file:           {0:12}'.format(str(model.get('receivers'))))
            Print.master('     Number of receivers:      {0:12}'.format(str(nReceivers)))
            Print.master('     VTK output:               {0:12}'.format(str(output.get('vtk'))))
            Print.master('     Cuda support:             {0:12}'.format(str(run.get('cuda'))))
            Print.master('     Output directory:         {0:12}'.format(str(output.get('directory'))))
            Print.master('     Scratch directory:        {0:12}'.format(str(output.get('directory_scratch'))))

        # Stop timer
        Timers()["Preprocessing"].stop()

        # Apply barrier for MPI tasks alignement
        parEnv.comm.barrier()

        return

# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################
def unitary_test():
    """Unitary test for preprocessing.py script."""
# ###############################################################
# ################             MAIN             #################
# ###############################################################


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
