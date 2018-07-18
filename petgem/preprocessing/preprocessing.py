#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the preprocessing dictionary and its functions. preprocessing
dictionary contain the main parameters for PETGEM preprocessing tasks:
mesh file, conductivity model and receivers file.
'''


def checkNumberParamsPreprocessing(init_params):
    ''' Check number of initial preprocessing parameters.

    :param list init_params: list of initial preprocessing parameters.
    :return: a parameters file name.
    :rtype: str.

    .. note:: if the number of init_params is different to 2, **PETGEM**
       preprocessing will stop.
    '''
    num_params = len(init_params)       # Number of parameters
    msg = ('  checkNumberParamsPreprocessing(): no preprocessing ' +
           'parameters file has been provided.')
    assert num_params >= 2, msg

    if num_params > 2:
        PETSc.Sys.Print('  checkNumberParamsPreprocessing(): more than one '
                        'parameters file has been provided.')
        PETSc.Sys.Print('  Only preprocessing parameters ' +
                        'file "' + init_params[1] +
                        '" will be consider.')

    return num_params


def buildPreprocessing(rank, MESH_FILE, MATERIAL_CONDUCTIVITIES,
                       RECEIVERS_FILE, OUT_DIR):
    ''' Build a dictionary with main parameters for
    PETGEM preprocessing.

    :param int rank: MPI rank.
    :param str MESH_FILE: file name of mesh.
    :param ndarray MATERIAL_CONDUCTIVITIES: conductivity values of
                                            materials in the mesh.
    :param str RECEIVERS_FILE: file name of receivers position.
    :param str OUT_DIR: path for output.
    :return: preprocessing dictionary
    :rtype: python dictionary
    '''

    if rank == 0:
        PETSc.Sys.Print('  buildPreprocessing(): Creating a ' +
                        'preprocessing dictionary.')

    preprocessing = {'MESH_FILE': MESH_FILE,
                     'MATERIAL_CONDUCTIVITIES': MATERIAL_CONDUCTIVITIES,
                     'RECEIVERS_FILE': RECEIVERS_FILE,
                     'OUT_DIR': OUT_DIR
                     }

    return preprocessing


def printPreprocessingData(input_preprocessing, file_name):
    ''' Print the content of a preprocessing dictionary.

    :param dictionary: input_preprocessing.
    :param str file_name: preprocessing file name
    :return: None.
    '''

    A = str(file_name)
    B = str(input_preprocessing['MESH_FILE'])
    C = str(input_preprocessing['MATERIAL_CONDUCTIVITIES'])
    D = str(input_preprocessing['RECEIVERS_FILE'])
    E = str(input_preprocessing['OUT_DIR'])

    PETSc.Sys.Print('\nData for preprocessing')
    PETSc.Sys.Print('='*75)
    PETSc.Sys.Print('  Preprocessing file name:   {0:50}'.format(A))
    PETSc.Sys.Print('  Mesh file:                 {0:50}'.format(B))
    PETSc.Sys.Print('  Material conductivities:   {0:50}'.format(C))
    PETSc.Sys.Print('  Receiver positions file:   {0:50}'.format(D))
    PETSc.Sys.Print('  Out directory:             {0:50}'.format(E))

    return


def checkPreprocessingConsistency(rank, in_dict, file_name):
    ''' Check if preprocessing dictionary consistency match
    with **PETGEM** requirements.

    :params int rank: MPI rank.
    :params dict in_dict: input preprocessing dictionary to be tested.
    :params str file_name: preprocessing parameters file name.
    :return: preprocessing dictionary after test.
    :rtype: preprocessing dictionary.
    '''
    msg1 = ('  checkPreprocessingConsistency(): "' + file_name + '" does not '
            'contain a ')
    msg2 = (' parameter. See content of "preprocessing_params_template.py" at '
            'PETGEM home directory.')
    msg3 = ('  checkPreprocessingConsistency(): Unsupported data type for ')
    msg4 = (' parameter in "' + file_name + '".')

    PETSc.Sys.Print('  Checking preprocessing parameters consistency.')

    # NODES_FILE parameter
    msg = msg1 + 'MESH_FILE' + msg2
    assert 'MESH_FILE' in in_dict, msg
    msg = msg3 + 'MESH_FILE' + msg4
    MESH_FILE = in_dict['MESH_FILE']
    assert type(MESH_FILE) is str, msg
    PETSc.Sys.Print('  checkPreprocessingConsistency(): MESH_FILE ' +
                    'consistency OK.')

    # MATERIAL_CONDUCTIVITIES parameter
    msg = msg1 + 'MATERIAL_CONDUCTIVITIES' + msg2
    assert 'MATERIAL_CONDUCTIVITIES' in in_dict, msg
    msg = msg3 + 'MATERIAL_CONDUCTIVITIES' + msg4
    MATERIAL_CONDUCTIVITIES = in_dict['MATERIAL_CONDUCTIVITIES']
    assert type(MATERIAL_CONDUCTIVITIES) is list, msg
    PETSc.Sys.Print('  checkPreprocessingConsistency(): ' +
                    'MATERIAL_CONDUCTIVITIES consistency OK.')

    # RECEIVERS_FILE parameter
    msg = msg1 + 'RECEIVERS_FILE' + msg2
    assert 'RECEIVERS_FILE' in in_dict, msg
    msg = msg3 + 'RECEIVERS_FILE' + msg4
    RECEIVERS_FILE = in_dict['RECEIVERS_FILE']
    assert type(RECEIVERS_FILE) is str, msg
    PETSc.Sys.Print('  checkPreprocessingConsistency(): RECEIVERS_FILE ' +
                    'consistency OK.')

    # OUT_DIR parameter
    msg = msg1 + 'OUT_DIR' + msg2
    assert 'OUT_DIR' in in_dict, msg
    msg = msg3 + 'OUT_DIR' + msg4
    OUT_DIR = in_dict['OUT_DIR']
    assert type(OUT_DIR) is str, msg
    PETSc.Sys.Print('  checkPreprocessingConsistency(): ' +
                    'OUT_DIR consistency OK.')

    # Create preprocessing dictionary
    out_model = buildPreprocessing(rank, MESH_FILE, MATERIAL_CONDUCTIVITIES,
                                   RECEIVERS_FILE, OUT_DIR)

    return out_model


def readPreprocessingParams(input_params, rank):
    ''' Read a preprocessing input, namely a preprocessing parameters file
    name.

    :params list input_params: user input parameters.
    :param int rank: MPI rank.
    :return: a modelling dictionary.
    :rtype: dict of type modelling.
    '''

    num_params = checkNumberParamsPreprocessing(input_params)

    # PETGEM parameters file in 3rd position in sys.argv
    input_path = input_params[1]

    # Check if file_name exist
    success = checkFilePath(input_path)

    if rank == 0:
        if not success:
            msg = ('  readPreprocessingParams(): file ' + input_path +
                   ' does not exist.')
            raise ValueError(msg)

    # Else, import file_name
    DIR_NAME = path.dirname(input_path)     # Split dir path
    PARAMS_FILE_NAME = path.basename(input_path)    # Split file path

    sys.path.append(DIR_NAME)   # Add dir path to system

    # Remove extension if exist
    remove_extension = False
    if PARAMS_FILE_NAME.endswith('.py'):
        PARAMS_FILE_NAME = PARAMS_FILE_NAME[:-3]
        remove_extension = True

    # Import params file as module (importlib)
    import_success = False
    try:
        preprocessing_temp = import_module(PARAMS_FILE_NAME)
        import_success = True
    except:
        pass

    # Recover original file name
    if remove_extension:
        PARAMS_FILE_NAME = PARAMS_FILE_NAME + '.py'

    if rank == 0:
        if import_success:
            PETSc.Sys.Print('  readPreprocessingParams(): file "' +
                            PARAMS_FILE_NAME +
                            '" has been successfully imported.')
        else:
            msg = ('  readPreprocessingParams(): file "' + PARAMS_FILE_NAME +
                   '" has not been successfully imported.')
            raise ValueError(msg)

    # Check consistency
    # Check dictionary content (This name must match with dictionary name
    # in parameters file, namely, must be == 'preprocessing')
    if rank == 0:
        preprocessing = checkPreprocessingConsistency(rank,
                                                      preprocessing_temp.
                                                      preprocessing,
                                                      PARAMS_FILE_NAME)

    # Print modelling content
    if rank == 0:
        printPreprocessingData(preprocessing, input_path)

    return preprocessing


def preprocessNodes(mesh_file, out_dir, rank):
    ''' Preprocess nodal coordinates of a given mesh in Gmsh format.

    :param str mesh_file: mesh file name to be preprocess.
    :param str out_dir: path for output.
    :param int rank: MPI rank.
    :return: number of nodes
    :rtype: int
    '''

    if rank == 0:
        PETSc.Sys.Print('  Nodal coordinates (nodes.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessNodes(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read nodes
    nodes, nNodes = readGmshNodes(mesh_file)

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_file)

    # Build coordinates in PETGEM format where each row
    # represent the xyz coordinates of the 4 tetrahedral element
    numDimensions = 3
    nodalOrder = 4
    data = np.array((nodes[elemsN[:], :]), dtype=np.float)
    data = data.reshape(nElems, numDimensions*nodalOrder)

    # Delete unnecesary arrays
    del nodes
    del elemsN

    # Get matrix dimensions
    size = data.shape

    # Build PETSc structures
    matrix = createSequentialDenseMatrixWithArray(size[0], size[1], data)

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'nodes.dat'

    # Write PETGEM nodes in PETSc format
    writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

    return nNodes


def preprocessingNodalConnectivity(mesh_file, out_dir, rank):
    ''' Preprocess nodal connectivity of a given mesh in Gmsh format.

    :param str mesh_file: mesh file name to be preprocess.
    :param str out_dir: path for output.
    :param int rank: MPI rank.
    :return: number of tetrahedral elements
    :rtype: int
    '''

    if rank == 0:
        PETSc.Sys.Print('  Nodal connectivity (elemsN.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingNodalConnectivity(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_file)

    # Get matrix dimensions
    size = elemsN.shape

    # Build PETSc structures
    matrix = createSequentialDenseMatrixWithArray(size[0], size[1], elemsN)

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'meshConnectivity.dat'

    # Write PETGEM nodes in PETSc format
    writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

    return nElems


def preprocessingDOF(mesh_file, out_dir, rank):
    ''' Preprocess degrees of freedom (DOF) and its associated data structures
    of a given mesh in Gmsh format. Here, dofs are defined for edge
    finite element computations.

    :param str mesh_file: mesh file name to be preprocess.
    :param str out_dir: path for output.
    :param int rank: MPI rank.
    :return: number of DOFS.
    :rtype: int
    '''

    if rank == 0:
        PETSc.Sys.Print('  Degrees of freedom (dofs.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingDOF(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_file)

    # Compute dofs
    dofs, dofsNodes = computeDofs(elemsN, nElems)
    nDofs = dofsNodes.shape[0]

    # Compute faces
    elemsF, facesN = computeFaces(elemsN, nElems)

    # Compute boundary faces
    boundaryFacesN = computeBoundaryFaces(elemsF, facesN)

    # Delete unnecesary arrays
    del elemsN
    del elemsF
    del facesN

    # Compute boundary dofs
    boundaryDofs = computeBoundaryDofs(dofsNodes, boundaryFacesN)

    # Delete unnecesary arrays
    del boundaryFacesN

    # ---------- DOFS ----------
    # Get matrix dimensions
    size = dofs.shape

    # Build PETSc structures
    matrix = createSequentialDenseMatrixWithArray(size[0], size[1], dofs)

    # Delete unnecesary arrays
    del dofs

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'dofs.dat'

    # Write PETGEM nodes in PETSc format
    writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

    # ---------- DOFS TO NODES ----------
    if rank == 0:
        PETSc.Sys.Print('  Dofs connectivity (dofsNodes.dat)')

    # Get matrix dimensions
    size = dofsNodes.shape

    # Build PETSc structures
    matrix = createSequentialDenseMatrixWithArray(size[0], size[1], dofsNodes)

    # Delete unnecesary arrays
    del dofsNodes

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'dofsNodes.dat'

    # Write PETGEM nodes in PETSc format
    writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

    # ---------- BOUNDARY DOFs ----------
    if rank == 0:
        PETSc.Sys.Print('  Boundaries (boundaries.dat)')

    # Build PETSc structures
    vector = createSequentialVectorWithArray(boundaryDofs)

    # Delete unnecesary arrays
    del boundaryDofs

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'boundaries.dat'

    # Write PETGEM nodes in PETSc format
    writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

    return nDofs


def preprocessingConductivityModel(mesh_file, material_conductivities,
                                   out_dir, rank):
    ''' Preprocess conductivity model associated to a given mesh in Gmsh
    format. Here, dofs are defined for edge finite element computations.

    :param str mesh_file: mesh file name to be preprocess.
    :param ndarray material_conductivities: conductivity values
                                            for each material in the mesh.
    :param str out_dir: path for output.
    :param int rank: MPI rank.
    :return: None
    '''

    if rank == 0:
        PETSc.Sys.Print('  Conductivity model (conductivityModel.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingConductivityModel(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read connectivity
    elemsS, nElems = readGmshPhysicalGroups(mesh_file)

    # Number of materials
    nMaterials = elemsS.max()

    # Ensure that material_conductivities (user input) is equal to those
    # imported from the Gmsh file (user input)
    if rank == 0:
        if(not nMaterials == len(material_conductivities)-1):
            PETSc.Sys.Print('  The number of materials in ' + mesh_file +
                            ' is not consistent with ' +
                            'Material conductivities array. Aborting')
            exit(-1)

    # Build conductivity arrays
    conductivityModel = np.zeros(nElems, dtype=np.float)
    for iEle in np.arange(nElems):
        conductivityModel[iEle] = material_conductivities[np.int(elemsS[iEle])]

    # Build PETSc structures
    vector = createSequentialVectorWithArray(conductivityModel)

    # Delete unnecesary arrays
    del conductivityModel

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'conductivityModel.dat'

    # Write PETGEM nodes in PETSc format
    writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

    return


def preprocessingDataReceivers(mesh_file, receivers_file,
                               out_dir, rank):
    ''' Preprocess conductivity model associated to a given mesh in Gmsh
    format. Here, dofs are defined for edge finite element computations.

    :param str mesh_file: mesh file name to be preprocess.
    :param str receivers_file: receiver positions file name to be preprocess.
    :param str out_dir: path for output.
    :param int rank: MPI rank.
    :return: None
    '''
    from scipy.spatial import Delaunay

    if rank == 0:
        PETSc.Sys.Print('  Receiver positions (receivers.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingDataReceivers(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Check if receivers_file exist
    success = checkFilePath(receivers_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingDataReceivers(): file ' + receivers_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read receivers_file
    receivers = np.loadtxt(receivers_file)

    # Number of receivers
    if receivers.ndim == 1:
        nReceivers = 1
    else:
        dim = receivers.shape
        nReceivers = dim[0]

    # Read nodes
    nodes, nNodes = readGmshNodes(mesh_file)

    # Build Delaunay triangulation with nodes
    tri = Delaunay(nodes)

    # Delete unnecesary arrays
    del nodes

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_file)

    # Compute dofs
    dofs, _ = computeDofs(elemsN, nElems)

    # Overwrite Delaunay structure with mesh_file connectivity
    tri.simplices = elemsN.astype(np.int32)

    # Delete unnecesary arrays
    del elemsN

    # Find out which tetrahedral element points are in
    recvElems = tri.find_simplex(receivers)

    # Determine if all receiver points were found
    idx = np.where(recvElems < 0)[0]

    # If idx is not empty, there are receivers outside the domain
    if idx.size != 0:
        PETSc.Sys.Print('     Some receivers are were not located')
        PETSc.Sys.Print('     Following ID-receivers will not be taken ' +
                        'into account: ')
        PETSc.Sys.Print(idx)

        # Update number of receivers
        nReceivers = nReceivers - len(idx)

        if nReceivers == 0:
            PETSc.Sys.Print('     No receiver has been found. Nothing to do.'
                            ' Aborting')
            exit(-1)

        # Remove idx from receivers matrix
        receivers = np.delete(receivers, idx, axis=0)

        # Create new file with located points coordinates
        # Build path to save the file
        out_path = out_dir + 'receiversPETGEM.txt'
        PETSc.Sys.Print('     Saving file with localized receiver positions ' +
                        '(receiversPETGEM.txt)')
        np.savetxt(out_path, receivers, fmt='%1.8e')

    # Allocate space for receives in PETGEM format
    numDimensions = 3
    nodalOrder = 4
    edgeOrder = 6
    allocate = numDimensions+nodalOrder*numDimensions+nodalOrder+edgeOrder
    tmp = np.zeros((nReceivers, allocate), dtype=np.float)

    # Fill tmp matrix with receiver positions, element coordinates and
    # nodal indexes
    for iReceiver in np.arange(nReceivers):
        # If there is one receiver
        if nReceivers == 1:
            # Get receiver coordinates
            coordiReceiver = receivers[0:]
            # Get element coordinates (container element)
            coordElement = tri.points[tri.simplices[recvElems, :]]
            coordElement = coordElement.flatten()
            # Get nodal indexes (container element)
            nodesElement = tri.simplices[recvElems, :]
            # Get element-dofs indices (container element)
            dofsElement = dofs[recvElems, :]
        # If there are more than one receivers
        else:
            # Get receiver coordinates
            coordiReceiver = receivers[iReceiver, :]
            # Get element coordinates (container element)
            coordElement = tri.points[tri.simplices[recvElems[iReceiver], :]]
            coordElement = coordElement.flatten()
            # Get nodal indexes (container element)
            nodesElement = tri.simplices[recvElems[iReceiver], :]
            # Get element-dofs indices (container element)
            dofsElement = dofs[recvElems[iReceiver], :]
        # Insert data for iReceiver
        tmp[iReceiver, 0:3] = coordiReceiver
        tmp[iReceiver, 3:15] = coordElement
        tmp[iReceiver, 15:19] = nodesElement
        tmp[iReceiver, 19:] = dofsElement

    # Delete unnecesary arrays
    del tri
    del dofs

    # Get matrix dimensions
    size = tmp.shape

    # Build PETSc structures
    matrix = createSequentialDenseMatrixWithArray(size[0], size[1], tmp)

    # Delete unnecesary arrays
    del tmp

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'receivers.dat'

    # Write PETGEM receivers in PETSc format
    writeParallelDenseMatrix(out_path, matrix, communicator=PETSc.COMM_SELF)

    return nReceivers


def preprocessingNNZ(mesh_file, out_dir, rank):
    ''' Preprocess sparsity pattern (NNZ) for parallel matrix allocation
    of a given mesh in Gmsh format. Here, dofs are defined for edge
    finite element computations.

    :param str mesh_file: mesh file name to be preprocess.
    :param int rank: MPI rank.
    :return: None
    '''

    if rank == 0:
        PETSc.Sys.Print('  Sparsity pattern (nnz.dat)')

    # Check if mesh_file exist
    success = checkFilePath(mesh_file)

    if rank == 0:
        if not success:
            msg = ('  preprocessingNNZ(): file ' + mesh_file +
                   ' does not exist.')
            raise ValueError(msg)

    # Read connectivity
    elemsN, nElems = readGmshConnectivity(mesh_file)

    # Compute dofs
    _, dofsNodes = computeDofs(elemsN, nElems)
    nDofs = dofsNodes.shape[0]

    # Since PETGEM parallelism is based on PETSc, computation of the matrix
    # sparsity pattern is critical in sake of performance. Furthermore, PETGEM
    # V1.0 is based on linear edge finite elements which produces six dofs per
    # tetrahedral element. Hence, the tetrahedral valence is equal to 34. based
    # on this information we build the NNZ vector.

    # In order to avoid memory performance issues, add 40% to valence
    valence = 50
    nnz = np.full((nDofs), valence, dtype=np.int)

    # Build PETSc structures
    vector = createSequentialVectorWithArray(nnz)

    # Delete unnecesary arrays
    del nnz

    # Verify if OUT_DIR exists
    checkIfDirectoryExist(out_dir)

    # Build path to save the file
    out_path = out_dir + 'nnz.dat'

    # Write PETGEM nodes in PETSc format
    writePetscVector(out_path, vector, communicator=PETSc.COMM_SELF)

    return


def printPreprocessingSummary(nNodes, nElems, nDofs, nReceivers, rank):
    ''' Print a summary of data preprocessed.

    :param int nNodes: number of nodes in the mesh.
    :param int nElems: number of tetrahedral elements in the mesh.
    :param int nDofs: number of DOFS in the mesh.
    :param int nReceivers: number of receivers.
    :param int rank: MPI rank.
    :return: None.
    '''

    A = str(nNodes)
    B = str(nElems)
    C = str(nDofs)
    D = str(nReceivers)

    if rank == 0:
        PETSc.Sys.Print('  Number of nodes:      {0:50}'.format(A))
        PETSc.Sys.Print('  Number of elements:   {0:50}'.format(B))
        PETSc.Sys.Print('  Number of DOFS:       {0:50}'.format(C))
        PETSc.Sys.Print('  Number of receivers:  {0:50}'.format(D))

    return


if __name__ == '__main__':
    # Standard module import
    import sys
    import numpy as np
    from os import path
    import petsc4py
    from petsc4py import PETSc
    from importlib import import_module
    unitary_test()
else:
    # Standard module import
    import sys
    import numpy as np
    from os import path
    import petsc4py
    from petsc4py import PETSc
    from importlib import import_module
    # PETGEM module import
    from petgem.base.base import checkFilePath
    from petgem.base.base import checkIfDirectoryExist
    from petgem.mesh.mesh import readGmshNodes
    from petgem.mesh.mesh import readGmshConnectivity
    from petgem.mesh.mesh import readGmshPhysicalGroups
    from petgem.mesh.mesh import gmshObject
    from petgem.parallel.parallel import printMessage
    from petgem.parallel.parallel import createSequentialDenseMatrixWithArray
    from petgem.parallel.parallel import createSequentialVectorWithArray
    from petgem.parallel.parallel import writeParallelDenseMatrix
    from petgem.parallel.parallel import writePetscVector
    from petgem.efem.efem import computeDofs
    from petgem.efem.efem import computeFaces
    from petgem.efem.efem import computeBoundaryFaces
    from petgem.efem.efem import computeBoundaryDofs
