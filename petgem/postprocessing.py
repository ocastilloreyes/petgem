#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define data postprocessing operations for **PETGEM**."""


# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
import h5py
import meshio
import platform
import shutil
import os
from datetime import datetime
from scipy.spatial import Delaunay
from petsc4py import PETSc

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .common import Print, Timers, measure_all_class_methods
from .parallel import MPIEnvironment, readPetscVector, readPetscMatrix
from .mesh import computeEdges, computeFacesEdges
from .mesh import computeFaces
from .hvfem import computeConnectivityDOFS
from .hvfem import computeJacobian, computeElementOrientation, tetrahedronXYZToXiEtaZeta
from .hvfem import computeBasisFunctions

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
@measure_all_class_methods
class Postprocessing():
    """Class for postprocessing."""

    def __init__(self):
        """Initialization of a postprocessing class."""
        return

    def run(self, inputSetup):
        """Run a postprocessing task.

        :param obj inputSetup: inputSetup object.
        :return: None
        """
        # ---------------------------------------------------------------
        # Obtain the MPI environment
        # ---------------------------------------------------------------
        parEnv = MPIEnvironment()

        # ---------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------
        # Start timer
        Timers()["Postprocessing"].start()

        # Parameters shortcut (for code legibility)
        model = inputSetup.model
        run   = inputSetup.run
        output = inputSetup.output
        out_dir_scratch = output.get('directory_scratch')
        out_dir = output.get('directory')

        # ---------------------------------------------------------------
        # Postprocessing (sequential task)
        # ---------------------------------------------------------------
        if( parEnv.rank == 0 ):
            # ---------------------------------------------------------------
            # Define constants
            # ---------------------------------------------------------------
            num_nodes_per_element = 4
            #num_edges_per_element = 6
            #num_faces_per_element = 4
            #num_nodes_per_face    = 3
            num_edges_per_face    = 3
            #num_nodes_per_edge    = 2
            #num_dimensions        = 3
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
            # Get version code
            version_file = open('petgem/VERSION')
            code_version = version_file.read().strip()
            version_file.close()

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
            # Build data structures for edge elements
            # ---------------------------------------------------------------
            # Compute edges
            elemsE, edgesN = computeEdges(mesh.cells[0][1][:], nElems)
            #nEdges = edgesN.shape[0]

            # Compute faces
            elemsF, facesN = computeFaces(mesh.cells[0][1][:], nElems)
            nFaces = facesN.shape[0]

            # Compute faces-edges
            facesE = computeFacesEdges(elemsF, elemsE, nFaces, nElems)

            # Compute degrees of freedom connectivity
            dofs, dof_edges, dof_faces, _, total_num_dofs = computeConnectivityDOFS(elemsE,elemsF,basis_order)

            # ---------------------------------------------------------------
            # Read receivers file
            # ---------------------------------------------------------------
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

            # ---------------------------------------------------------------
            # Read vector solution
            # ---------------------------------------------------------------
            x = []

            for i in np.arange(num_polarizations):
               input_file = out_dir_scratch + '/x' + str(i) + '.dat'
               x.append(readPetscVector(input_file, communicator=PETSc.COMM_SELF))

            # ---------------------------------------------------------------
            # Compute solution for each point and each polarization mode
            # ---------------------------------------------------------------
            # Allocate array to save field components (Ex, Ey, Ez, Hx, Hy, Hz)
            fields_receivers = []

            for i in np.arange(num_polarizations):
               fields_receivers.append(fieldInterpolator(x[i], mesh.points, mesh.cells[0][1][:], elemsE, edgesN, elemsF, facesE, dofs, receivers, inputSetup))

            # ---------------------------------------------------------------
            # Compute apparent resistivity
            # ---------------------------------------------------------------
            if (mode == 'mt'):
                apparent_res, phase, tipper, impedance = computeImpedance(fields_receivers, omega, mu)

            # ---------------------------------------------------------------
            # Compute solution for entire mesh
            # ---------------------------------------------------------------
            if (output.get('vtk')):
                # Allocate array for points
                scaledCoord = np.zeros((4,3), dtype=np.float)
                local_contribution = np.zeros((num_nodes_per_element,6), dtype=np.complex)
                # Scale factor for tetrahedral element
                scaleFactor = .9999
                # Allocate array to save field components (Ex, Ey, Ez, Hx, Hy, Hz)
                fields_nodes = []

                for i in np.arange(num_polarizations):
                    # Allocate arrays for electromagnetic responses computation
                    fields_tmp = np.zeros((nNodes,6), dtype=np.complex)

                    for j in np.arange(nElems):
                        # Get dofs of element
                        dofsEle = dofs[j, :]

                        # Get indexes of nodes for iEle
                        nodesEle = mesh.cells[0][1][j,:]
                        # Get nodes coordinates for iEle
                        coordEle = mesh.points[nodesEle, :]
                        # Get indexes of faces for iEle
                        facesEle = elemsF[j, :]
                        # Get edges indexes for faces
                        edgesFace = facesE[facesEle, :]
                        # Get indexes of edges for iEle
                        edgesEle = elemsE[j, :]
                        # Get node indexes for edges in i and insert
                        edgesNodesEle = edgesN[edgesEle, :]

                        # Compute jacobian for i
                        jacobian, invjacobian = computeJacobian(coordEle)

                        # Compute global orientation for i
                        edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

                        # Compute element centroid
                        centroid = np.sum(coordEle, axis=0)/4.

                        # Apply small scaling to element coordinates
                        scaledCoord[:,0] = centroid[0] + (coordEle[:,0]-centroid[0])*scaleFactor
                        scaledCoord[:,1] = centroid[1] + (coordEle[:,1]-centroid[1])*scaleFactor
                        scaledCoord[:,2] = centroid[2] + (coordEle[:,2]-centroid[2])*scaleFactor

                        # Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element)
                        XiEtaZeta = tetrahedronXYZToXiEtaZeta(coordEle, scaledCoord)

                        # Compute basis for i
                        basis, curl_basis = computeBasisFunctions(edge_orientation, face_orientation, jacobian, invjacobian, basis_order, XiEtaZeta)

                        # Get global dofs from x vector
                        realField = np.real(x[i].getValues(dofsEle.astype(PETSc.IntType)))
                        imagField = np.imag(x[i].getValues(dofsEle.astype(PETSc.IntType)))

                        # Interpolate field for nodes in i
                        for k in np.arange(num_nodes_per_element):
                            for l in np.arange(num_dof_in_element):
                                # Exyz[i] = Exyz[i] + real_part*basis + imag_part*basis
                                local_contribution[k, 0] += realField[l]*basis[0,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*basis[0,l,k]
                                local_contribution[k, 1] += realField[l]*basis[1,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*basis[1,l,k]
                                local_contribution[k, 2] += realField[l]*basis[2,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*basis[2,l,k]
                                # Hxyz[i] = Hxyz[i] + real_part*curl_basis + imag_part*curl_basis
                                local_contribution[k, 3] += realField[l]*curl_basis[0,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*curl_basis[0,l,k]
                                local_contribution[k, 4] += realField[l]*curl_basis[1,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*curl_basis[1,l,k]
                                local_contribution[k, 5] += realField[l]*curl_basis[2,l,k] + np.sqrt(-1. + 0.j)*imagField[l]*curl_basis[2,l,k]

                        # Add local contribution to global array
                        fields_tmp[nodesEle,:] += local_contribution

                        # Clean variable for next iteration
                        local_contribution[:] = 0.

                    fields_nodes.append(fields_tmp)

                # Compute number of elements that share each node
                num_elems_shared_node = np.zeros(nNodes, dtype=np.float)

                for i in np.arange(nNodes):
                    num_elems_shared_node[i] = np.count_nonzero(mesh.cells[0][1][:] == i)

                # Divide the field of each node by the number of elements that share it
                num_components = 6      # Six electromagnetic field components (Ex, Ey, Ez, Hx, Hy, Hz)
                for i in np.arange(num_polarizations):
                    for j in np.arange(num_components):
                        fields_nodes[i][:,j] /= num_elems_shared_node
                    # Following Maxwell equations, apply constant factor to magnetic field
                    for j in np.arange(3, 6):
                        fields_nodes[i][:,j] /= Const

                # Conductivity model to vtk
                input_file = out_dir_scratch + '/conductivityModel.dat'
                tmpSigmaModel = readPetscMatrix(input_file, communicator=PETSc.COMM_SELF)

                sigmaModel = np.zeros((nElems, 2), dtype=np.float)

                for i in np.arange(nElems):
                    sigmaEle = tmpSigmaModel.getRow(i)[1].real
                    sigmaModel[i:] = sigmaEle

                # Set conductivity model to elements
                mesh.cell_data = {"sigma_horizontal": sigmaModel[:,0], "sigma_vertical": sigmaModel[:,0]}

                # For each polarization and each component, write data to vtk file
                # meshio doesn't support complex number, then fields are decoupled
                # in real and imaginary part
                if (mode == 'csem'):
                    mesh.point_data = {'csem_Ex_real': np.real(fields_nodes[0][:,0]),
                                       'csem_Ey_real': np.real(fields_nodes[0][:,1]),
                                       'csem_Ez_real': np.real(fields_nodes[0][:,2]),
                                       'csem_Hx_real': np.real(fields_nodes[0][:,3]),
                                       'csem_Hy_real': np.real(fields_nodes[0][:,4]),
                                       'csem_Hz_real': np.real(fields_nodes[0][:,5]),
                                       'csem_Ex_imag': np.imag(fields_nodes[0][:,0]),
                                       'csem_Ey_imag': np.imag(fields_nodes[0][:,1]),
                                       'csem_Ez_imag': np.imag(fields_nodes[0][:,2]),
                                       'csem_Hx_imag': np.imag(fields_nodes[0][:,3]),
                                       'csem_Hy_imag': np.imag(fields_nodes[0][:,4]),
                                       'csem_Hz_imag': np.imag(fields_nodes[0][:,5])}
                elif (mode == 'mt'):
                    # Get polarization mode
                    polatization_mode = data_model.get('polarization')
                    if (num_polarizations == 1):
                        mesh.point_data = {'mt_Ex_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,0]),
                                           'mt_Ey_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,1]),
                                           'mt_Ez_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,2]),
                                           'mt_Hx_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,3]),
                                           'mt_Hy_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,4]),
                                           'mt_Hz_real_mode'+ polatization_mode: np.real(fields_nodes[0][:,5]),
                                           'mt_Ex_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,0]),
                                           'mt_Ey_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,1]),
                                           'mt_Ez_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,2]),
                                           'mt_Hx_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,3]),
                                           'mt_Hy_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,4]),
                                           'mt_Hz_imag_mode'+ polatization_mode: np.imag(fields_nodes[0][:,5])}

                    elif (num_polarizations == 2):
                        mesh.point_data = {'mt_Ex_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,0]),
                                           'mt_Ey_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,1]),
                                           'mt_Ez_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,2]),
                                           'mt_Hx_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,3]),
                                           'mt_Hy_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,4]),
                                           'mt_Hz_real_mode'+ polatization_mode[0]: np.real(fields_nodes[0][:,5]),
                                           'mt_Ex_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,0]),
                                           'mt_Ey_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,1]),
                                           'mt_Ez_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,2]),
                                           'mt_Hx_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,3]),
                                           'mt_Hy_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,4]),
                                           'mt_Hz_real_mode'+ polatization_mode[1]: np.real(fields_nodes[1][:,5]),
                                           'mt_Ex_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,0]),
                                           'mt_Ey_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,1]),
                                           'mt_Ez_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,2]),
                                           'mt_Hx_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,3]),
                                           'mt_Hy_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,4]),
                                           'mt_Hz_imag_mode'+ polatization_mode[0]: np.imag(fields_nodes[0][:,5]),
                                           'mt_Ex_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,0]),
                                           'mt_Ey_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,1]),
                                           'mt_Ez_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,2]),
                                           'mt_Hx_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,3]),
                                           'mt_Hy_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,4]),
                                           'mt_Hz_imag_mode'+ polatization_mode[1]: np.imag(fields_nodes[1][:,5])}

                # Write vtk file
                vtk_filename = out_dir + '/' + mode + '_petgemV' + code_version + '_' + str(datetime.today().strftime('%Y-%m-%d')) + '.vtk'
                meshio.write_points_cells(vtk_filename, mesh.points, mesh.cells, mesh.point_data, mesh.cell_data)

            # ---------------------------------------------------------------
            # Save results and data provedance
            # ---------------------------------------------------------------
            # Create output file
            output_file_name = '/' + mode + '_' + 'petgemV' + code_version + '_' + str(datetime.today().strftime('%Y-%m-%d')) + '.h5'
            output_file = out_dir + output_file_name
            fileID = h5py.File(output_file, 'w')

            # Create group for data machine
            machine = fileID.create_group('machine')
            # Name
            uname = platform.uname()
            uname = uname.node + '. ' + uname.release + '. ' + uname.processor
            machine.create_dataset('machine', data=uname)
            # Number of cores
            machine.create_dataset('num_processors', data=parEnv.num_proc)
            # PETGEM version
            machine.create_dataset('petgem_version', data=code_version)

            # Create group for data model
            model_dataprovedance = fileID.create_group('model')
            # Modeling date
            model_dataprovedance.create_dataset('date', data=datetime.today().isoformat())
            # Mesh file
            model_dataprovedance.create_dataset('mesh_file', data=model.get('mesh'))
            # Receivers file
            model_dataprovedance.create_dataset('receivers_file', data=model.get('receivers'))
            # Basis order
            model_dataprovedance.create_dataset('nord', data=run.get('nord'))
            # Cuda support
            model_dataprovedance.create_dataset('cuda', data=run.get('cuda'))
            # vtk output
            model_dataprovedance.create_dataset('vtk', data=output.get('vtk'))
            # Modeling mode
            model_dataprovedance.create_dataset('mode', data=mode)
            # Number of polarizations
            model_dataprovedance.create_dataset('num_polarizations', data=num_polarizations)
            # Create data model
            if (mode == 'csem'):
                if (run.get('conductivity_from_file')):
                    model_dataprovedance.create_dataset('sigma', data=data_model.get('sigma').get('file'))
                else:
                    model_dataprovedance.create_dataset('sigma_horizontal', data=data_model.get('sigma').get('horizontal'))
                    model_dataprovedance.create_dataset('sigma_vertical', data=data_model.get('sigma').get('vertical'))
                model_dataprovedance.create_dataset('frequency', data=data_model.get('source').get('frequency'))
                model_dataprovedance.create_dataset('source_position', data=np.asarray(data_model.get('source').get('position'), dtype=np.float))
                model_dataprovedance.create_dataset('source_azimuth', data=data_model.get('source').get('azimuth'))
                model_dataprovedance.create_dataset('source_dip', data=data_model.get('source').get('dip'))
                model_dataprovedance.create_dataset('source_current', data=data_model.get('source').get('current'))
                model_dataprovedance.create_dataset('source_length', data=data_model.get('source').get('length'))
            elif (mode == 'mt'):
                if (run.get('conductivity_from_file')):
                    model_dataprovedance.create_dataset('sigma', data=data_model.get('sigma').get('file'))
                else:
                    model_dataprovedance.create_dataset('sigma_horizontal', data=data_model.get('sigma').get('horizontal'))
                    model_dataprovedance.create_dataset('sigma_vertical', data=data_model.get('sigma').get('vertical'))
                model_dataprovedance.create_dataset('frequency', data=data_model.get('frequency'))
                model_dataprovedance.create_dataset('polarization', data=data_model.get('polarization'))

            # Write electromagnetic responses
            if (mode == 'csem'):
                # Electric fields
                E_fields = model_dataprovedance.create_group('E-fields')
                E_fields.create_dataset('x', data=fields_receivers[0][:,0])
                E_fields.create_dataset('y', data=fields_receivers[0][:,1])
                E_fields.create_dataset('z', data=fields_receivers[0][:,2])

                # Magnetic fields
                H_fields = model_dataprovedance.create_group('H-fields')
                H_fields.create_dataset('x', data=fields_receivers[0][:,3])
                H_fields.create_dataset('y', data=fields_receivers[0][:,4])
                H_fields.create_dataset('z', data=fields_receivers[0][:,5])

            elif (mode == 'mt'):
                list_modes = data_model.get('polarization')
                for i in np.arange(num_polarizations):
                    mode_E_i = 'E-fields_mode_' + list_modes[i]
                    mode_H_i = 'H-fields_mode_' + list_modes[i]

                    E_fields = model_dataprovedance.create_group(mode_E_i)
                    E_fields.create_dataset('x', data=fields_receivers[i][:,0])
                    E_fields.create_dataset('y', data=fields_receivers[i][:,1])
                    E_fields.create_dataset('z', data=fields_receivers[i][:,2])

                    # Magnetic fields
                    H_fields = model_dataprovedance.create_group(mode_H_i)
                    H_fields.create_dataset('x', data=fields_receivers[i][:,3])
                    H_fields.create_dataset('y', data=fields_receivers[i][:,4])
                    H_fields.create_dataset('z', data=fields_receivers[i][:,5])

                imp = model_dataprovedance.create_group('impedance')
                imp.create_dataset('xx', data=impedance[0])
                imp.create_dataset('xy', data=impedance[1])
                imp.create_dataset('yx', data=impedance[2])
                imp.create_dataset('yy', data=impedance[3])

                app_res = model_dataprovedance.create_group('apparent_resistivity')
                app_res.create_dataset('xx', data=apparent_res[0])
                app_res.create_dataset('xy', data=apparent_res[1])
                app_res.create_dataset('yx', data=apparent_res[2])
                app_res.create_dataset('yy', data=apparent_res[3])

                pha = model_dataprovedance.create_group('phase')
                pha.create_dataset('xx', data=phase[0])
                pha.create_dataset('xy', data=phase[1])
                pha.create_dataset('yx', data=phase[2])
                pha.create_dataset('yy', data=phase[3])

                tip = model_dataprovedance.create_group('tipper')
                tip.create_dataset('x', data=tipper[0])
                tip.create_dataset('y', data=tipper[1])

            # Close file
            fileID.close()

            # Remove temporal directory
            if (output.get('remove_scratch')):
                shutil.rmtree(out_dir_scratch)
            else:
                files_in_directory = os.listdir(out_dir_scratch)
                filtered_files = [file for file in files_in_directory if (file.endswith(".dat") or file.endswith(".info"))]
                for file in filtered_files:
                    path_to_file = os.path.join(out_dir_scratch, file)
                    os.remove(path_to_file)

        # Stop timer
        Timers()["Postprocessing"].stop()

        return


def fieldInterpolator(solution_vector, nodes, elemsN, elemsE, edgesN, elemsF, facesE, dof_connectivity, points, inputSetup):
    """Interpolate electromagnetic field for a set of 3D points.

    :param ndarray-petsc solution_vector: vector field to be interpolated
    :param ndarray nodes: nodal coordinates
    :param ndarray elemsN: elements-node connectivity with dimensions = (number_elements, 4)
    :param ndarray elemsE: elements-edge connectivity with dimensions = (number_elements, 6)
    :param ndarray edgesN: edge-node connectivity with dimensions = (number_edges, 2)
    :param ndarray elemsF: element-faces connectivity with dimensions = (number_elements, 4)
    :param ndarray facesE: face-edges connectivity with dimensions = (number_faces, 3)
    :param ndarray dof_connectivity: local/global dofs list for elements, dofs index on edges, dofs index on faces, dofs index on volumes, total number of dofs
    :param ndarray points: point coordinates
    :param obj inputSetup: inputSetup object.
    :return: electromagnetic fields for a set of 3D points
    :rtype: ndarray and int
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Parameters shortcut (for code legibility)
    model = inputSetup.model
    run   = inputSetup.run
    basis_order           = run.get('nord')
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

    # Number of elements
    size = elemsN.shape
    nElems = size[0]
    # Number of nodes
    #size = nodes.shape
    #nNodes = size[0]
    
    # Num dof per element
    num_dof_in_element    = np.int(basis_order*(basis_order+2)*(basis_order+3)/2)

    # Number of points
    if points.ndim == 1:
        nPoints = 1
    else:
        dim = points.shape
        nPoints = dim[0]

    # Find where receivers are located
    tri = Delaunay(nodes)

    # Overwrite Delaunay structure with mesh_file connectivity and points
    tri.simplices = elemsN
    tri.vertices = elemsN

    # Find out which tetrahedral element points are in
    points_in_elements = tri.find_simplex(points, bruteforce=True, tol=1.e-12)

    # Determine if all points were found
    idx = np.where(np.logical_or(points_in_elements>nElems, points_in_elements<0))[0]

    # If idx is not empty, there are points outside the domain
    if idx.size != 0:
        Print.master('        The following receivers were not located and will not be taken into account ' + str(idx))
        # Update number of receivers
        nPoints = nPoints - len(idx)

        if nPoints == 0:
            Print.master('     No point has been found. Nothing to do. Aborting')
            exit(-1)

        # Remove idx from points matrix
        points = np.delete(points, idx, axis=0)

        # Remove idx from points_in_elements
        points_in_elements = np.delete(points_in_elements, idx, axis=0)

    indx_ele = points_in_elements

    # Allocate array
    fields = np.zeros((nPoints, 6), dtype=np.complex)

    # Interpolate electromagnetic field for all points
    for i in np.arange(nPoints):
        # Get element index
        iEle = indx_ele[i]

        # Get dofs of element container
        dofsEle = dof_connectivity[iEle, :]
        # Get receiver coordinates
        coordPoints = points[i,:]
        # Get indexes of nodes for iEle
        nodesEle = elemsN[iEle,:]
        # Get nodes coordinates for iEle
        coordEle = nodes[nodesEle, :]
        # Get indexes of faces for iEle
        facesEle = elemsF[iEle, :]
        # Get edges indexes for faces
        edgesFace = facesE[facesEle, :]
        # Get indexes of edges for iEle
        edgesEle = elemsE[iEle, :]
        # Get node indexes for edges in i and insert
        edgesNodesEle = edgesN[edgesEle, :]

        # Compute jacobian for iEle
        jacobian, invjacobian = computeJacobian(coordEle)

        # Compute global orientation for iEle
        edge_orientation, face_orientation = computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,edgesFace)

        # Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element)
        XiEtaZeta = tetrahedronXYZToXiEtaZeta(coordEle, coordPoints)

        # Compute basis for i
        basis, curl_basis = computeBasisFunctions(edge_orientation, face_orientation, jacobian, invjacobian, basis_order, XiEtaZeta)

        # Get global dofs from x vector
        realField = np.real(solution_vector.getValues(dofsEle.astype(PETSc.IntType)))
        imagField = np.imag(solution_vector.getValues(dofsEle.astype(PETSc.IntType)))

        for j in np.arange(num_dof_in_element):
            # Exyz[k] = Exyz[k] + real_part*basis + imag_part*basis
            fields[i, 0] += realField[j]*basis[0,j] + np.sqrt(-1. + 0.j)*imagField[j]*basis[0,j]
            fields[i, 1] += realField[j]*basis[1,j] + np.sqrt(-1. + 0.j)*imagField[j]*basis[1,j]
            fields[i, 2] += realField[j]*basis[2,j] + np.sqrt(-1. + 0.j)*imagField[j]*basis[2,j]
            # Hxyz[k] = Hxyz[k] + real_part*curl_basis + imag_part*curl_basis
            fields[i, 3] += realField[j]*curl_basis[0,j] + np.sqrt(-1. + 0.j)*imagField[j]*curl_basis[0,j]
            fields[i, 4] += realField[j]*curl_basis[1,j] + np.sqrt(-1. + 0.j)*imagField[j]*curl_basis[1,j]
            fields[i, 5] += realField[j]*curl_basis[2,j] + np.sqrt(-1. + 0.j)*imagField[j]*curl_basis[2,j]

    # Following Maxwell equations, apply constant factor to compute magnetic field
    fields[:,3::] /= Const

    return fields


def computeImpedance(fields, omega, mu):
    """Compute apparent resistiviy, phase, tipper and impedance for MT mode.

    :param list fields: list of numpy arrays with electromagnetic fields with dimensions = (number_polarizations, number_receivers, number_EM_components)
    :param float omega: angular frequency
    :param float mu: medium permeability.
    :return: apparent resistivity, phase, tipper and impedance.
    :rtype: ndarray
    """

    # Field components (Ex, Ey, Hx, Hy and Hz components por x and y-directions in polarization)
    E1x = fields[0][:,0]
    E1y = fields[0][:,1]
    E2x = fields[1][:,0]
    E2y = fields[1][:,1]
    H1x = fields[0][:,3]
    H1y = fields[0][:,4]
    H1z = fields[0][:,5]
    H2x = fields[1][:,3]
    H2y = fields[1][:,4]
    H2z = fields[1][:,5]

    # Minor determinants
    m1 = H2y * H2x * H1y - H2y * H2y * H1x
    m2 = H2x * H2x * H1y - H2y * H2x * H1x
    m3 = E2x * H2x * H1y - H2y * E2x * H1x
    m4 = H2y * E2y * H1y - H2y * H2y * E1y
    m5 = H2x * E2y * H1y - H2y * H2x * E1y
    m6 = H2y * H2x * E1y - E2y * H2y * H1x
    m7 = H2x * H2x * E1y - E2y * H2x * H1x

    # Major determinants
    d  = H1x * m1 - H1y * m2
    d1 = E1x * m1 - H1y * m3
    d2 = H1x * m3 - E1x * m2
    d3 = H1x * m4 - H1y * m5
    d4 = H1x * m6 - H1y * m7

    # Impedance
    impedance = []
    impedance.append(d1/d)  # xx-impedance
    impedance.append(d2/d)  # xy-impedance
    impedance.append(d3/d)  # yx-impedance
    impedance.append(d4/d)  # yy-impedance

    # Aparent resistivity
    coef = np.float(1.) / (mu * omega)
    apparent_resistivity = []
    apparent_resistivity.append(coef * np.abs(impedance[0])**2)   # xx-apparent resistivity
    apparent_resistivity.append(coef * np.abs(impedance[1])**2)   # xy-apparent resistivity
    apparent_resistivity.append(coef * np.abs(impedance[2])**2)   # yx-apparent resistivity
    apparent_resistivity.append(coef * np.abs(impedance[3])**2)   # yy-apparent resistivity

    # Phase
    coef = np.float(180.) / np.pi
    phase = []
    phase.append((coef * np.arctan(-np.imag(impedance[0]) / np.real(impedance[0])))%360)    # xx-phase
    phase.append((coef * np.arctan(-np.imag(impedance[1]) / np.real(impedance[1])))%360)    # xy-phase
    phase.append((coef * np.arctan(-np.imag(impedance[2]) / np.real(impedance[2])))%360)    # yx-phase
    phase.append((coef * np.arctan(-np.imag(impedance[3]) / np.real(impedance[3])))%360)    # yy-phase

    # Tipper --> solve for T = (Tx, Ty) such that Hz = <T, H> with H = (Hx, Hy)
    coef = H1x * H2y
    tmp1 = (H2z/H2y - (H1z*H2x)/coef) / (np.float(1.) - (H1y*H2x)/coef)
    tmp2 = (H1z - tmp1 * H1y) / H1x
    tipper = []
    tipper.append(tmp2)     # x-tipper
    tipper.append(tmp1)     # y-tipper

    return apparent_resistivity, phase, tipper, impedance


# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################
def unitary_test():
    """Unitary test for postprocessing.py script."""
# ###############################################################
# ################             MAIN             #################
# ###############################################################


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
