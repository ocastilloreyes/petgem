#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions a 3D CSEM solver using high-order vector finite element method.
'''

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import sys
import numpy as np
from petsc4py import PETSc
import h5py
import shutil

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .common import Print, Timers, measure_all_class_methods, measure_time
from .parallel import readPetscMatrix, readPetscVector, createParallelMatrix, createParallelVector
from .parallel import MPIEnvironment, createSequentialVector, createParallelVector, createParallelDenseMatrix
from .parallel import writePetscVector
from .hvfem import computeJacobian, computeElementOrientation, computeElementalMatrices, computeSourceVectorRotation
from .hvfem import tetrahedronXYZToXiEtaZeta, computeBasisFunctions

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
@measure_all_class_methods
class Solver():
    ''' Class for solver.
    '''
    def __init__(self):
        ''' Initialization of a solver class.
        '''
        return


    def setup(self, inputSetup):
        ''' Setup of a solver class.

        :param object inputSetup: user input setup.
        '''

        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Setup"].start()

        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        run = inputSetup.run
        output = inputSetup.output

        Print.master('     Importing files')

        # ---------------------------------------------------------------
        # Import files
        # ---------------------------------------------------------------
        # Read nodes coordinates
        input_file = output.directory_scratch + '/nodes.dat'
        self.nodes = readPetscMatrix(input_file, communicator=None)

        # elements-nodes connectivity
        input_file = output.directory_scratch + '/meshConnectivity.dat'
        self.elemsN = readPetscMatrix(input_file, communicator=None)

        # elements-edges connectivity
        input_file = output.directory_scratch + '/edges.dat'
        self.elemsE = readPetscMatrix(input_file, communicator=None)

        # edges-nodes connectivity
        input_file = output.directory_scratch + '/edgesNodes.dat'
        self.edgesNodes = readPetscMatrix(input_file, communicator=None)

        # elements-faces connectivity
        input_file = output.directory_scratch + '/faces.dat'
        self.elemsF = readPetscMatrix(input_file, communicator=None)

        # faces-edges connectivity
        input_file = output.directory_scratch + '/facesEdges.dat'
        self.facesEdges = readPetscMatrix(input_file, communicator=None)

        # Dofs connectivity
        input_file = output.directory_scratch + '/dofs.dat'
        self.dofs = readPetscMatrix(input_file, communicator=None)

        # Boundaries
        input_file = output.directory_scratch + '/boundaries.dat'
        self.boundaries = readPetscVector(input_file, communicator=None)

        # Conductivity model
        input_file = output.directory_scratch + '/conductivityModel.dat'
        self.sigmaModel = readPetscMatrix(input_file, communicator=None)

        # Receivers
        input_file = output.directory_scratch + '/receivers.dat'
        self.receivers = readPetscMatrix(input_file, communicator=None)

        # Sparsity pattern (NNZ) for matrix allocation
        input_file = output.directory_scratch + '/nnz.dat'
        tmp = readPetscVector(input_file, communicator=None)
        self.nnz = (tmp.getArray().real).astype(PETSc.IntType)

        # Number of dofs (length of nnz correspond to the total number of dofs)
        self.total_num_dofs = tmp.getSizes()[1]     # Get global sizes

        # Stop timer
        Timers()["Setup"].stop()

        return


    def assembly(self, inputSetup):
        ''' This function assembly a linear system for 3D CSEM based on vector
        high-order finite element method.

        :param object inputSetup: user input setup.
        '''

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
        # Get ranges
        # ---------------------------------------------------------------
        # Ranges over elements
        Istart_elemsE, Iend_elemsE = self.elemsE.getOwnershipRange()

        # ---------------------------------------------------------------
        # Allocate parallel arrays
        # ---------------------------------------------------------------
        # Left-hand side
        self.A = createParallelMatrix(self.total_num_dofs, self.total_num_dofs, self.nnz, run.cuda, communicator=None)
        # Right-hand side
        self.b = createParallelVector(self.total_num_dofs, run.cuda, communicator=None)
        # X vector
        self.x = createParallelVector(self.total_num_dofs, run.cuda, communicator=None)

        # ---------------------------------------------------------------
        # Assembly linear system (Left-Hand Side - LHS)
        # ---------------------------------------------------------------
        num_nodes_per_element = 4
        num_edges_per_element = 6
        num_faces_per_element = 4
        num_nodes_per_face    = 3
        num_edges_per_face    = 3
        num_nodes_per_edge    = 2
        num_dimensions        = 3
        omega                 = model.frequency*2.*np.pi
        mu                    = 4.*np.pi*1e-7
        Const                 = np.sqrt(-1. + 0.j)*omega*mu

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
            M, K = computeElementalMatrices(edge_orientation, face_orientation, jacobian, invjacobian, model.basis_order, sigmaEle)

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
        # Compute matrices for source rotation
        sourceRotationVector = computeSourceVectorRotation(model)

        # Total electric field formulation. Set dipole definition
        # x-directed dipole
        Dx = np.array([model.src_current*model.src_length*1., 0., 0.], dtype=np.float)
        # y-directed dipole
        Dy = np.array([0., model.src_current*model.src_length*1., 0.], dtype=np.float)
        # % z-directed dipole
        Dz = np.array([0., 0., model.src_current*model.src_length*1.], dtype=np.float)

        # Rotate source and setup electric field
        field = sourceRotationVector[0]*Dx + sourceRotationVector[1]*Dy + sourceRotationVector[2]*Dz

        # Add source (only master)
        if MPIEnvironment().rank == 0:
            # Read source file
            input_file = output.directory_scratch + '/source.dat'
            source_data = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            # Get source data
            source_data = source_data.getArray().real

            # Get indexes of nodes for srcElem
            nodesEle = source_data[0:4].astype(np.int)

            # Get nodes coordinates for srcElem
            coordEle = source_data[4:16]
            coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

            # Get faces indexes for srcElem
            facesEle = source_data[16:20].astype(np.int)

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
            _, invjacobian = computeJacobian(coordEle)

            # Compute global orientation for srcElem
            edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

            # Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element)
            XiEtaZeta = tetrahedronXYZToXiEtaZeta(coordEle, model.src_position)

            # Compute basis for srcElem
            basis = computeBasisFunctions(edge_orientation, face_orientation, invjacobian, model.basis_order, XiEtaZeta)

            # Compute integral
            src_contribution = np.matmul(field, basis[:,:,0])

            # Multiplication by constant value
            src_contribution = src_contribution * Const

            # Add local contributions to global matrix
            self.b.setValues(dofsSource, src_contribution, addv=PETSc.InsertMode.ADD_VALUES)

        # Start global RHS assembly
        self.b.assemblyBegin()
        # End global RHS assembly
        self.b.assemblyEnd()

        # Stop timer
        Timers()["Assembly"].stop()

        return


    def solve(self):
        ''' This function solves a linear system generated by the vector
        high-order finite element method for a 3D CSEM problem.

        '''

        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["SetBoundaries"].start()

        Print.master('     Solving linear system')

        # ---------------------------------------------------------------
        # Set dirichlet boundary conditions
        # ---------------------------------------------------------------
        # Ranges over boundaries
        Istart_boundaries, Iend_boundaries = self.boundaries.getOwnershipRange()
        # Boundaries for LHS
        self.A.zeroRowsColumns(np.real(self.boundaries).astype(PETSc.IntType))
        # Boundaries for RHS
        numLocalBoundaries = Iend_boundaries - Istart_boundaries
        self.b.setValues(np.real(self.boundaries).astype(PETSc.IntType),
                         np.zeros(numLocalBoundaries, dtype=np.complex),
                         addv=PETSc.InsertMode.INSERT_VALUES)

        # Start global system assembly
        self.A.assemblyBegin()
        self.b.assemblyBegin()
        # End global system assembly
        self.A.assemblyEnd()
        self.b.assemblyEnd()

        # Stop timer
        Timers()["SetBoundaries"].stop()

        # ---------------------------------------------------------------
        # Solve system
        # ---------------------------------------------------------------
        Timers()["Solver"].start()
        # Create KSP: linear equation solver
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(self.A)
        ksp.setFromOptions()
        ksp.solve(self.b, self.x)
        iterationNumber = ksp.getIterationNumber()
        ksp.destroy()
        Timers()["Solver"].stop()

        return


    def postprocess(self, inputSetup):
        ''' This function interpolates a given electric field for a vector of receivers
        '''

        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Postprocessing"].start()

        Print.master('     Postprocessing solution')

        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        run = inputSetup.run
        output = inputSetup.output

        # Ranges over receivers
        Istart_receivers, Iend_receivers = self.receivers.getOwnershipRange()

        # Number of receivers
        total_num_receivers = self.receivers.getSize()[0]
        local_num_receivers = Iend_receivers-Istart_receivers

        # Define constants
        num_nodes_per_element = 4
        num_faces_per_element = 4
        num_edges_per_face    = 3
        num_edges_per_element = 6
        num_nodes_per_edge    = 2
        num_dimensions        = 3

        # Compute number of dofs per element
        num_dof_in_element = np.int(model.basis_order*(model.basis_order+2)*(model.basis_order+3)/2)

        # ---------------------------------------------------------------
        # Get dofs-connectivity for receivers
        # ---------------------------------------------------------------
        # Auxiliar arrays
        dofsIdxRecv = np.zeros((local_num_receivers, num_dof_in_element), dtype=PETSc.IntType)

        j = 0
        for i in np.arange(Istart_receivers, Iend_receivers):
            # Get dofs for receiver
            tmp = (self.receivers.getRow(i)[1].real).astype(PETSc.IntType)
            tmp = tmp[53::]
            dofsIdxRecv[j, :] = tmp
            j += 1

        # Gather global solution of x to local vector
        # Sequential vector for gather tasks
        x_local = createSequentialVector(local_num_receivers*num_dof_in_element, run.cuda, communicator=None)

        # Build Index set in PETSc format
        IS_dofs = PETSc.IS().createGeneral(dofsIdxRecv.flatten(), comm=PETSc.COMM_WORLD)

        # Build gather vector
        gatherVector = PETSc.Scatter().create(self.x, IS_dofs, x_local, None)
        # Ghater values
        gatherVector.scatter(self.x, x_local, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

        # ---------------------------------------------------------------
        # Allocate parallel arrays for output
        # ---------------------------------------------------------------
        receiver_fieldX = createParallelVector(total_num_receivers, run.cuda ,communicator=None)
        receiver_fieldY = createParallelVector(total_num_receivers, run.cuda ,communicator=None)
        receiver_fieldZ = createParallelVector(total_num_receivers, run.cuda ,communicator=None)
        receiver_coordinatesX = createParallelVector(total_num_receivers, run.cuda ,communicator=None)
        receiver_coordinatesY = createParallelVector(total_num_receivers, run.cuda ,communicator=None)
        receiver_coordinatesZ = createParallelVector(total_num_receivers, run.cuda ,communicator=None)

        k = 0
        for i in np.arange(Istart_receivers, Iend_receivers):
            # Get receiver data
            receiver_data = self.receivers.getRow(i)[1].real

            # Get indexes of nodes for i
            nodesEle = receiver_data[0:4].astype(np.int)

            # Get nodes coordinates for i
            coordEle = receiver_data[4:16]
            coordEle = np.reshape(coordEle, (num_nodes_per_element, num_dimensions))

            # Get faces indexes for i
            facesEle = receiver_data[16:20].astype(np.int)

            # Get edges indexes for faces in i
            edgesFace = receiver_data[20:32].astype(np.int)
            edgesFace = np.reshape(edgesFace, (num_faces_per_element, num_edges_per_face))

            # Get indexes of edges for i
            edgesEle = receiver_data[32:38].astype(np.int)

            # Get node indexes for edges in i
            edgesNodesEle = receiver_data[38:50].astype(np.int)
            edgesNodesEle = np.reshape(edgesNodesEle, (num_edges_per_element, num_nodes_per_edge))

            # Get receiver coordinates
            coordReceiver = receiver_data[50:53]

            # Get dofs for i
            dofsSource = receiver_data[53::].astype(PETSc.IntType)

            # Compute jacobian for i
            _, invjacobian = computeJacobian(coordEle)

            # Compute global orientation for i
            edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

            # Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element)
            XiEtaZeta = tetrahedronXYZToXiEtaZeta(coordEle, coordReceiver)

            # Compute basis for i
            basis = computeBasisFunctions(edge_orientation, face_orientation, invjacobian, model.basis_order, XiEtaZeta)

            # Get element fields
            local_field = x_local[k*num_dof_in_element : (k * num_dof_in_element) + num_dof_in_element]

            # Initialize variables
            Ex = 0.
            Ey = 0.
            Ez = 0.

            # Interpolate electric field
            for j in np.arange(num_dof_in_element):
                Rfield = np.real(local_field[j]) # Real part
                Ifield = np.imag(local_field[j]) # Imaginary part
                # Exyz[i] = Exyz[i] + real_part*basis + imag_part*basis
                Ex += Rfield*basis[0,j] + np.sqrt(-1. + 0.j)*Ifield*basis[0,j]
                Ey += Rfield*basis[1,j] + np.sqrt(-1. + 0.j)*Ifield*basis[1,j]
                Ez += Rfield*basis[2,j] + np.sqrt(-1. + 0.j)*Ifield*basis[2,j]

            # Increase counter over local vector (x_local)
            k += 1

            # Set total field components for i
            receiver_fieldX.setValue(i, Ex, addv=PETSc.InsertMode.INSERT_VALUES)
            receiver_fieldY.setValue(i, Ey, addv=PETSc.InsertMode.INSERT_VALUES)
            receiver_fieldZ.setValue(i, Ez, addv=PETSc.InsertMode.INSERT_VALUES)

            # Set coordinates for i
            receiver_coordinatesX.setValue(i, coordReceiver[0], addv=PETSc.InsertMode.INSERT_VALUES)
            receiver_coordinatesY.setValue(i, coordReceiver[1], addv=PETSc.InsertMode.INSERT_VALUES)
            receiver_coordinatesZ.setValue(i, coordReceiver[2], addv=PETSc.InsertMode.INSERT_VALUES)

        # Start global assembly
        receiver_fieldX.assemblyBegin()
        receiver_fieldY.assemblyBegin()
        receiver_fieldZ.assemblyBegin()
        receiver_coordinatesX.assemblyBegin()
        receiver_coordinatesY.assemblyBegin()
        receiver_coordinatesZ.assemblyBegin()

        # End global assembly
        receiver_fieldX.assemblyEnd()
        receiver_fieldY.assemblyEnd()
        receiver_fieldZ.assemblyEnd()
        receiver_coordinatesX.assemblyEnd()
        receiver_coordinatesY.assemblyEnd()
        receiver_coordinatesZ.assemblyEnd()

        # Write intermediate results
        out_path = output.directory_scratch + '/tmp_fieldsX.dat'
        writePetscVector(out_path, receiver_fieldX, communicator=None)

        out_path = output.directory_scratch + '/tmp_fieldsY.dat'
        writePetscVector(out_path, receiver_fieldY, communicator=None)

        out_path = output.directory_scratch + '/tmp_fieldsZ.dat'
        writePetscVector(out_path, receiver_fieldZ, communicator=None)

        out_path = output.directory_scratch + '/tmp_receiver_coordinatesX.dat'
        writePetscVector(out_path, receiver_coordinatesX, communicator=None)

        out_path = output.directory_scratch + '/tmp_receiver_coordinatesY.dat'
        writePetscVector(out_path, receiver_coordinatesY, communicator=None)

        out_path = output.directory_scratch + '/tmp_receiver_coordinatesZ.dat'
        writePetscVector(out_path, receiver_coordinatesZ, communicator=None)

        # Write final solution
        if MPIEnvironment().rank == 0:
            input_file = output.directory_scratch + '/tmp_fieldsX.dat'
            electric_fieldsX = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            input_file = output.directory_scratch + '/tmp_fieldsY.dat'
            electric_fieldsY = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            input_file = output.directory_scratch + '/tmp_fieldsZ.dat'
            electric_fieldsZ = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            input_file = output.directory_scratch + '/tmp_receiver_coordinatesX.dat'
            recv_coordX = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            input_file = output.directory_scratch + '/tmp_receiver_coordinatesY.dat'
            recv_coordY = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            input_file = output.directory_scratch + '/tmp_receiver_coordinatesZ.dat'
            recv_coordZ = readPetscVector(input_file, communicator=PETSc.COMM_SELF)

            # Allocate
            data_fields = np.zeros((total_num_receivers, 3), dtype=np.complex)
            data_coordinates = np.zeros((total_num_receivers, 3), dtype=np.float)

            # Loop over receivers
            for i in np.arange(total_num_receivers):
                # Get electric field components
                data_fields[i, 0] = electric_fieldsX.getValue(i)
                data_fields[i, 1] = electric_fieldsY.getValue(i)
                data_fields[i, 2] = electric_fieldsZ.getValue(i)

                # Get coordinates
                data_coordinates[i, 0] = recv_coordX.getValue(i).real
                data_coordinates[i, 1] = recv_coordY.getValue(i).real
                data_coordinates[i, 2] = recv_coordZ.getValue(i).real

            # Write final output
            output_file = output.directory + '/electric_fields.h5'
            fileID = h5py.File(output_file, 'w')

            # Create coordinates dataset
            dset = fileID.create_dataset('electric_fields', data=data_fields)
            dset = fileID.create_dataset('receiver_coordinates', data=data_coordinates)

            # Close file
            fileID.close()

            # Remove temporal directory
            shutil.rmtree(output.directory_scratch)

        # Stop timer
        Timers()["Postprocessing"].stop()



def unitary_test():
    ''' Unitary test for hvfem.py script.
    '''


if __name__ == '__main__':
    unitary_test()
