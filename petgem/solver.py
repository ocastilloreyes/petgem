#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define functions a 3D CSEM/MT solver using high-order vector finite element method (HEFEM)."""

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
from petsc4py import PETSc
import h5py
import shutil
from mpi4py import MPI

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .common import Print, Timers, measure_all_class_methods
from .parallel import readPetscMatrix, readPetscVector, createParallelMatrix, createParallelVector
from .parallel import MPIEnvironment, createSequentialVector
from .parallel import writePetscVector
from .hvfem import computeJacobian, computeElementOrientation, computeElementalMatrices, computeSourceVectorRotation
from .hvfem import tetrahedronXYZToXiEtaZeta, computeBasisFunctions
from .hvfem import getNormalVector, get2DJacobDet, compute2DGaussPoints
from .hvfem import transform2Dto3DInReferenceElement, getRealFromReference, computeBasisFunctionsReferenceElement
from .hvfem import getFaceByLocalNodes, getNeumannBCface
from .mt1d import eval_MT1D

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
@measure_all_class_methods
class Solver():
    """Class for solver."""

    def __init__(self):
        """Initialization of a solver class."""
        return


    def setup(self, inputSetup):
        """Setup of a solver class.

        :param object inputSetup: user input setup.
        """
        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Setup"].start()

        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        output = inputSetup.output
        out_dir = output.get('directory_scratch')

        Print.master('     Importing files')

        # ---------------------------------------------------------------
        # Obtain the MPI environment
        # ---------------------------------------------------------------
        parEnv = MPIEnvironment()

        # ---------------------------------------------------------------
        # Import files
        # ---------------------------------------------------------------
        # Read nodes coordinates
        input_file = out_dir + '/nodes.dat'
        self.nodes = readPetscMatrix(input_file, communicator=None)

        # elements-nodes connectivity
        input_file = out_dir  + '/meshConnectivity.dat'
        self.elemsN = readPetscMatrix(input_file, communicator=None)

        # elements-edges connectivity
        input_file = out_dir + '/edges.dat'
        self.elemsE = readPetscMatrix(input_file, communicator=None)

        # edges-nodes connectivity
        input_file = out_dir + '/edgesNodes.dat'
        self.edgesNodes = readPetscMatrix(input_file, communicator=None)

        # elements-faces connectivity
        input_file = out_dir + '/faces.dat'
        self.elemsF = readPetscMatrix(input_file, communicator=None)

        # faces-edges connectivity
        input_file = out_dir + '/facesEdges.dat'
        self.facesEdges = readPetscMatrix(input_file, communicator=None)

        # Dofs connectivity
        input_file = out_dir + '/dofs.dat'
        self.dofs = readPetscMatrix(input_file, communicator=None)

        # Conductivity model
        input_file = out_dir + '/conductivityModel.dat'
        self.sigmaModel = readPetscMatrix(input_file, communicator=None)

        # # Receivers
        # input_file = out_dir + '/receivers.dat'
        # self.receivers = readPetscMatrix(input_file, communicator=None)

        # Sparsity pattern (NNZ) for matrix allocation
        input_file = out_dir + '/nnz.dat'
        tmp = readPetscVector(input_file, communicator=None)
        self.nnz = (tmp.getArray().real).astype(PETSc.IntType)

        # Number of dofs (length of nnz correspond to the total number of dofs)
        self.total_num_dofs = tmp.getSizes()[1]     # Get global sizes

        # Depending on modeling mode, load data for source, boundary faces or boundary dofs
        if (model.get('mode') == 'csem'):
            # Boundary dofs for csem mode
            input_file = out_dir + '/boundaries.dat'
            self.boundaries = readPetscVector(input_file, communicator=None)
            # Load source data (master task)
            if parEnv.rank == 0:
                # Read source file
                input_file = out_dir + '/source.dat'
                self.source_data = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

        elif (model.get('mode') == 'mt'):
            # Boundary faces for mt mode
            input_file = out_dir + '/boundaryElements.dat'
            self.boundaries = readPetscMatrix(input_file, communicator=None)

        # Stop timer
        Timers()["Setup"].stop()

        return


    def assembly(self, inputSetup):
        """Assembly a linear system for 3D CSEM/MT based on HEFEM.

        :param object inputSetup: user input setup.
        """
        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Assembly"].start()

        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        run = inputSetup.run
        output = inputSetup.output

        Print.master('     Assembling linear system')

        # ---------------------------------------------------------------
        # Obtain the MPI environment
        # ---------------------------------------------------------------
        parEnv = MPIEnvironment()

        # ---------------------------------------------------------------
        # Define constants
        # ---------------------------------------------------------------
        num_nodes_per_element = 4
        num_edges_per_element = 6
        num_faces_per_element = 4
        num_nodes_per_face    = 3
        num_edges_per_face    = 3
        num_nodes_per_edge    = 2
        num_dimensions        = 3
        basis_order           = run.get('nord')
        num_polarizations     = run.get('num_polarizations')
        num_dof_in_element    = np.int(basis_order*(basis_order+2)*(basis_order+3)/2)
        if (model.get('mode') == 'csem'):
            mode = 'csem'
            data_model = model.get(mode)        # Get data model
            frequency         = data_model.get('source').get('frequency')
        elif (model.get('mode') == 'mt'):
            mode = 'mt'
            data_model = model.get(mode)        # Get data model
            frequency         = data_model.get('frequency')
        omega                 = frequency*2.*np.pi
        mu                    = 4.*np.pi*1e-7
        Const                 = np.sqrt(-1. + 0.j)*omega*mu

        # ---------------------------------------------------------------
        # Get global ranges
        # ---------------------------------------------------------------
        # Ranges over elements
        Istart_elemsE, Iend_elemsE = self.elemsE.getOwnershipRange()

        # ---------------------------------------------------------------
        # Assembly linear system (Left-Hand Side - LHS)
        # ---------------------------------------------------------------
        # Left-hand side
        self.A = createParallelMatrix(self.total_num_dofs, self.total_num_dofs, self.nnz, run.get('cuda'), communicator=None)

        # Compute contributions for all local elements
        for i in np.arange(Istart_elemsE, Iend_elemsE):
            # Get indexes of nodes for i
            nodesEle = (self.elemsN.getRow(i)[1].real).astype(PETSc.IntType)

            # Get coordinates of i
            coordEle = self.nodes.getRow(i)[1].real
            coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

            # Get edges indexes for faces in i
            edgesFace = self.facesEdges.getRow(i)[1].real
            edgesFace = np.reshape(edgesFace, (num_faces_per_element, num_edges_per_face))

            # Get indexes of edges for i
            edgesEle = (self.elemsE.getRow(i)[1].real).astype(PETSc.IntType)

            # Get node indexes for edges in i
            edgesNodesEle = self.edgesNodes.getRow(i)[1].real
            edgesNodesEle = np.reshape(edgesNodesEle, (num_edges_per_element, num_nodes_per_edge))

            # Get conductivity values for i (horizontal and vertical conductivity)
            sigmaEle = self.sigmaModel.getRow(i)[1].real

            # Compute jacobian for i
            jacobian, invjacobian = computeJacobian(coordEle)

            # Compute global orientation for i
            edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

            # Compute elemental matrices (stiffness and mass matrices)
            M, K = computeElementalMatrices(edge_orientation, face_orientation, jacobian, invjacobian, basis_order, sigmaEle)

            # Compute elemental matrix
            Ae = K - Const*M
            Ae = Ae.flatten()

            # Get dofs indexes for i
            dofsEle = (self.dofs.getRow(i)[1].real).astype(PETSc.IntType)

            # Add local contributions to global matrix
            self.A.setValues(dofsEle, dofsEle, Ae, addv=PETSc.InsertMode.ADD_VALUES)

        # Start global LHS assembly
        self.A.assemblyBegin()
        # End global LHS assembly
        self.A.assemblyEnd()

        # ---------------------------------------------------------------
        # Assembly linear system (Right-Hand Side RHS)
        # ---------------------------------------------------------------
        self.b = []
        self.x = []
        for i in np.arange(num_polarizations):
            self.b.append(createParallelVector(self.total_num_dofs, run.get('cuda'), communicator=None))
            self.x.append(createParallelVector(self.total_num_dofs, run.get('cuda'), communicator=None))

        # Assembly RHS for csem mode
        if (mode == 'csem'):
            # Get source parameters
            position = np.asarray(data_model.get('source').get('position'), dtype=np.float)
            azimuth  = data_model.get('source').get('azimuth')
            dip      = data_model.get('source').get('dip')
            current  = data_model.get('source').get('current')
            length   = data_model.get('source').get('length')
            # Compute matrices for source rotation
            sourceRotationVector = computeSourceVectorRotation(azimuth, dip)

            # Total electric field formulation. Set dipole definition
            # x-directed dipole
            Dx = np.array([current*length*1., 0., 0.], dtype=np.float)
            # y-directed dipole
            Dy = np.array([0., current*length*1., 0.], dtype=np.float)
            # z-directed dipole
            Dz = np.array([0., 0., current*length*1.], dtype=np.float)

            # Rotate source and setup electric field
            field = sourceRotationVector[0]*Dx + sourceRotationVector[1]*Dy + sourceRotationVector[2]*Dz

            # Insert source (only master)
            if parEnv.rank == 0:
                # Get source data
                source_data = self.source_data.getArray().real

                # Get indexes of nodes for srcElem
                nodesEle = source_data[0:4].astype(np.int)

                # Get nodes coordinates for srcElem
                coordEle = source_data[4:16]
                coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

                # Get faces indexes for srcElem
                #facesEle = source_data[16:20].astype(np.int)

                # Get edges indexes for faces in srcElem
                edgesFace = source_data[20:32].astype(np.int)
                edgesFace = np.reshape(edgesFace, (num_faces_per_element, num_edges_per_face))

                # Get indexes of edges for srcElem
                edgesEle = source_data[32:38].astype(np.int)

                # Get node indexes for edges in srcElem
                edgesNodesEle = source_data[38:50].astype(np.int)
                edgesNodesEle = np.reshape(edgesNodesEle, (num_edges_per_element, num_nodes_per_edge))

                # Get dofs for srcElem
                dofsSource = source_data[50::].astype(PETSc.IntType)

                # Compute jacobian for srcElem
                jacobian, invjacobian = computeJacobian(coordEle)

                # Compute global orientation for srcElem
                edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

                # Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element)
                XiEtaZeta = tetrahedronXYZToXiEtaZeta(coordEle, position)

                # Compute basis for srcElem
                basis, _ = computeBasisFunctions(edge_orientation, face_orientation, jacobian, invjacobian, basis_order, XiEtaZeta)

                # Compute integral
                rhs_contribution = np.matmul(field, basis[:,:,0])

                # Multiplication by constant value
                rhs_contribution = rhs_contribution * Const

                # Add local contributions to global matrix
                self.b[0].setValues(dofsSource, rhs_contribution, addv=PETSc.InsertMode.ADD_VALUES)

        elif (mode == 'mt'):
            # Ranges over boundary faces or boundary elements
            Istart_boundaryF, Iend_boundaryF = self.boundaries.getOwnershipRange()

            # Compute the two-dimensional gauss points.
            gauss_order = np.int(2)*basis_order
            gaussPoints2D, Wi = compute2DGaussPoints(gauss_order)
            ngaussP = gaussPoints2D.shape[0]

            # Allocate array for interpolation points
            num_local_boundaries = self.boundaries.getLocalSize()
            interpolationPoints = np.zeros([num_local_boundaries[0], ngaussP], dtype=np.float)
            centroid_z_face4 = []
            sigma_face4 = []
            indx_local_face = np.int(0)     # Initialize index of local boundary face

            # Compute local contributions for each boundary face
            for i in np.arange(Istart_boundaryF, Iend_boundaryF):
                boundary_data = self.boundaries.getRow(i)[1].real

                # Get face plane for boundary element
                faceType = boundary_data[50].astype(np.int)

                # Get nodes coordinates for boundary element
                coordEle = boundary_data[4:16]
                coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

                # Get faces indexes for boundary element
                facesEle = boundary_data[16:20].astype(np.int)

                # Get global face index
                faceGlobalIndex = boundary_data[51].astype(np.int)

                # Get sigma for element with boundary face
                sigmaBoundaryElement = boundary_data[52].astype(np.float)

                # Get local index of boundary face
                faceLocalIndex = np.where(facesEle==faceGlobalIndex)[0][0]

                for j in np.arange(ngaussP):
                    # Transform 2D gauss points to 3D in the reference element.
                    gaussPoint3D = transform2Dto3DInReferenceElement(gaussPoints2D[j,:], faceLocalIndex)

                    # This is the real point where the excitation is evaluated.
                    realPoint = getRealFromReference(gaussPoint3D, coordEle)

                    # Save z-component of gauss point
                    interpolationPoints[indx_local_face, j] = realPoint[2]

                # Save centroid only for face 3
                if faceType == 3:
                    nodesInFace = getFaceByLocalNodes(faceLocalIndex)
                    centroid_face4 = np.sum(coordEle[nodesInFace], axis=0)/3.
                    centroid_z_face4.append(centroid_face4[2])
                    sigma_face4.append(sigmaBoundaryElement)

                # Increment index of local boundary face
                indx_local_face += np.int(1)

            # List to numpy arrays
            centroid_z_face4 = np.asarray(centroid_z_face4, dtype=np.float)
            sigma_face4 = np.asarray(sigma_face4, dtype=np.float)

            # Compute the max/min z-coordinate in the domain
            coord_z = []
            for i in np.arange(Istart_elemsE, Iend_elemsE):
                # Get indexes of nodes for i
                nodesEle = (self.elemsN.getRow(i)[1].real).astype(PETSc.IntType)

                # Get coordinates of i
                coordEle = self.nodes.getRow(i)[1].real
                coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

                coord_z.append(coordEle[:,2])

            # Get local max/min
            coord_z = np.asarray(coord_z, dtype=np.float)
            coord_z = coord_z.flatten()
            z_max_local = np.max(coord_z)
            z_min_local = np.min(coord_z)

            # Get global max/min
            za = parEnv.comm.allreduce(z_max_local, op=MPI.MAX)
            zb = parEnv.comm.allreduce(z_min_local, op=MPI.MIN)

            u = eval_MT1D(za, zb, np.float(1), np.float(0), sigma_face4, centroid_z_face4,
                          omega, mu, np.int(1e6), np.int(1), interpolationPoints)

            # For each polarization mode
            for i in np.arange(num_polarizations):
                # Get polarization mode
                tmp = data_model.get('polarization')
                if (tmp[i] == 'x'):
                    polarization_mode = np.int(1)
                elif (tmp[i] == 'y'):
                    polarization_mode = np.int(2)
                else:
                    Print.master('     MT polarization mode not supported.')
                    exit(-1)

                # Compute local contributions for each boundary face
                indx_local_face = np.int(0)     # Initialize index of local boundary face
                for j in np.arange(Istart_boundaryF, Iend_boundaryF):
                    boundary_data = self.boundaries.getRow(j)[1].real

                    # Get indexes of nodes for boundary element
                    nodesEle = boundary_data[0:4].astype(np.int)

                    # Get nodes coordinates for boundary element
                    coordEle = boundary_data[4:16]
                    coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

                    # Get faces indexes for boundary element
                    facesEle = boundary_data[16:20].astype(np.int)

                    # Get edges indexes for boundary element
                    edgesFace = boundary_data[20:32].astype(np.int)
                    edgesFace = np.reshape(edgesFace, (num_faces_per_element, num_edges_per_face))

                    # Get indexes of edges for boundary element
                    edgesEle = boundary_data[32:38].astype(np.int)

                    # Get node indexes for edges in boundary element
                    edgesNodesEle = boundary_data[38:50].astype(np.int)
                    edgesNodesEle = np.reshape(edgesNodesEle, (num_edges_per_element, num_nodes_per_edge))

                    # Get face plane for boundary element
                    faceType = boundary_data[50].astype(np.int)

                    # Get global face index
                    faceGlobalIndex = boundary_data[51].astype(np.int)

                    # Get sigma for element with boundary face
                    #sigmaBoundaryElement = boundary_data[52].astype(np.int)

                    # Get dofs for boundary element
                    dofsBoundaryElement = boundary_data[53::].astype(PETSc.IntType)

                    # Compute jacobian for boundary element
                    _, invjacobian = computeJacobian(coordEle)

                    # Compute global orientation for boundary element
                    edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

                    # Get local index of boundary face
                    faceLocalIndex = np.where(facesEle==faceGlobalIndex)[0][0]

                    # Compute normal
                    normalVector = getNormalVector(faceLocalIndex, invjacobian)

                    # Compute normal unit vector
                    normalUnitVector = normalVector/np.linalg.norm(normalVector)

                    # Compute 2D Jacobian
                    detJacob2D = get2DJacobDet(coordEle, faceLocalIndex)

                    # Allocate array for local contribution
                    rhs_contribution = np.zeros(num_dof_in_element, dtype=np.complex)

                    # Get excitation for boundary face
                    ex, ey, ez = getNeumannBCface(faceType, polarization_mode, u)

                    for k in np.arange(ngaussP):
                        # Transform 2D gauss points to 3D in the reference element.
                        gaussPoint3D = transform2Dto3DInReferenceElement(gaussPoints2D[k,:], faceLocalIndex)

                        # 3D basis functions evaluated on reference element
                        allBasesEvaluated = computeBasisFunctionsReferenceElement(edge_orientation, face_orientation, basis_order, gaussPoint3D)

                        # Same mapping as in mass matrix.
                        allBasesReal = np.matmul(invjacobian,allBasesEvaluated[:,:,0])

                        # Add excitation field
                        ex_g = ex[indx_local_face, k]
                        ey_g = ey[indx_local_face, k]
                        ez_g = ez[indx_local_face, k]
                        excitation_value = np.array([ex_g, ey_g, ez_g], dtype=np.complex)

                        # Allocate
                        integrandTangential = np.zeros(num_dof_in_element, dtype=np.complex)

                        for l in np.arange(num_dof_in_element):
                            iBaseTangential = np.cross(np.cross(normalUnitVector, allBasesReal[:,l]), normalUnitVector)
                            integrandTangential[l] = np.dot(iBaseTangential, excitation_value)

                        rhs_contribution += Wi[k]*integrandTangential*detJacob2D

                    # Multiplication by constant value
                    rhs_contribution = rhs_contribution * Const

                    # Add local contributions to global matrix
                    self.b[i].setValues(dofsBoundaryElement, rhs_contribution, addv=PETSc.InsertMode.ADD_VALUES)

                    # Increment index of local boundary face
                    indx_local_face += np.int(1)

        # Global assembly for each RHS
        for i in np.arange(num_polarizations):
            # Start global RHS assembly
            self.b[i].assemblyBegin()
            # End global RHS assembly
            self.b[i].assemblyEnd()

        # Stop timer
        Timers()["Assembly"].stop()

        return


    def run(self, inputSetup):
        """Run solver for linear systems generated by the HEFEM for a 3D CSEM/MT problem.

        :param object inputSetup: user input setup.
        """
        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        run   = inputSetup.run
        output = inputSetup.output
        out_dir = output.get('directory_scratch')

        # ---------------------------------------------------------------
        # Define constants
        # ---------------------------------------------------------------
        num_polarizations     = run.get('num_polarizations')
        if (model.get('mode') == 'csem'):
            mode = 'csem'
        elif (model.get('mode') == 'mt'):
            mode = 'mt'

        Print.master('     Solving linear system')

        if (mode == 'csem'):
            # Start timer
            Timers()["SetBoundaries"].start()

            # ---------------------------------------------------------------
            # Set dirichlet boundary conditions
            # ---------------------------------------------------------------
            # Ranges over boundaries
            Istart_boundaries, Iend_boundaries = self.boundaries.getOwnershipRange()
            # Boundaries for LHS
            self.A.zeroRowsColumns(np.real(self.boundaries).astype(PETSc.IntType))
            # Boundaries for RHS
            numLocalBoundaries = Iend_boundaries - Istart_boundaries
            self.b[0].setValues(np.real(self.boundaries).astype(PETSc.IntType),
                                np.zeros(numLocalBoundaries, dtype=np.complex),
                                addv=PETSc.InsertMode.INSERT_VALUES)

            # Start global system assembly
            self.A.assemblyBegin()
            self.b[0].assemblyBegin()
            # End global system assembly
            self.A.assemblyEnd()
            self.b[0].assemblyEnd()

            # Stop timer
            Timers()["SetBoundaries"].stop()

        # ---------------------------------------------------------------
        # Solve system
        # ---------------------------------------------------------------
        Timers()["Solver"].start()

        for i in np.arange(num_polarizations):
            # Create KSP: linear equation solver
            ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
            ksp.setOperators(self.A)
            ksp.setFromOptions()
            ksp.solve(self.b[i], self.x[i])
            ksp.destroy()

            # Write vector solution
            out_path = out_dir + '/x' + str(i) + '.dat'
            writePetscVector(out_path, self.x[i], communicator=None)

        Timers()["Solver"].stop()

        return


def unitary_test():
    """Unitary test for solver.py script."""


if __name__ == '__main__':
    unitary_test()
