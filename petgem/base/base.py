#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define base operations for **PETGEM** such as: check init params,
data types, import files and timers.
'''


def removeFile(in_file):
    ''' Remove a file.

    :params str in_file_path: file name to be deleted.
    :return: None
    '''
    remove(in_file)
    return


def checkIfDirectoryExist(in_file_path):
    ''' Determine if a directory exists, if not the directory is created.

    :params str in_file_path: file name to be checked.
    :return: None
    '''

    try:
        makedirs(in_file_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    return


def checkNumberParams(init_params):
    ''' Check number of initial kernel parameters.

    :param list init_params: list of initial kernel parameters.
    :return: a parameters file name.
    :rtype: str.

    .. note:: if the number of init_params is < 4, **PETGEM**
       kernel will stop.
    '''
    num_params = len(init_params)       # Number of parameters
    msg = ('  checkNumberParams(): no parameters file has been provided.')
    assert num_params >= 4, msg

    if num_params > 4:
        PETSc.Sys.Print('  checkNumberParams(): more than one parameters file '
                        'has been provided.')
        PETSc.Sys.Print('  Only parameters file "' + init_params[3] +
                        '" will be consider.')

    return num_params


def checkFilePath(in_file_path):
    ''' Determine if exists a file.

    :params str in_file_path: file name to be checked.
    :return: success.
    :rtype: bool
    '''
    success = False
    try:
        success = path.isfile(in_file_path)
    except:
        pass

    return success


def checkDirectoryPath(in_directory_path):
    ''' Determine if exists a directory.

    :params str in_directory_path: directory name to be checked.
    :return: success.
    :rtype: bool
    '''
    success = False
    try:
        success = path.isdir(in_directory_path)
    except:
        pass

    return success


def checkDictionaryConsistencyMaster(rank, in_dict, file_name, dir_name):
    ''' Check if dictionary consistency match with **PETGEM** requirements.
    (master task)

    :params int rank: MPI rank.
    :params dict in_dict: input dictionary to be tested.
    :params str file_name: parameters file name.
    :params str dir_name: parent directory of file_name
    :return: csem modelling dictionary after test.
    :rtype: csem_modelling dictionary.
    '''
    msg1 = ('  checkDictionaryConsistency(): "' + file_name + '" does not '
            'contain a ')
    msg2 = (' parameter. See content of "modelling_params_template.py" at '
            'PETGEM home directory.')
    msg3 = ('  checkDictionaryConsistency(): Unsupported data type for ')
    msg4 = (' parameter in "' + file_name + '".')

    PETSc.Sys.Print('  Checking parameters consistency...')

    # freq parameter
    msg = msg1 + 'FREQ' + msg2
    assert 'FREQ' in in_dict, msg
    msg = msg3 + 'FREQ' + msg4
    freq = in_dict['FREQ']
    assert (type(freq) is float) or (type(freq) is int), msg
    freq = np.float(in_dict['FREQ'])
    PETSc.Sys.Print('  checkDictionaryConsistency(): FREQ consistency OK.')

    # src_pos parameter
    msg = msg1 + 'SRC_POS' + msg2
    assert 'SRC_POS' in in_dict, msg
    msg = msg3 + 'SRC_POS' + msg4
    src_pos = in_dict['SRC_POS']
    assert type(src_pos) is list, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): SRC_POS consistency OK.')

    # src_direc parameter
    msg = msg1 + 'SRC_DIREC' + msg2
    assert 'SRC_DIREC' in in_dict, msg
    msg = msg3 + 'SRC_DIREC' + msg4
    src_direc = in_dict['SRC_DIREC']
    assert (type(src_direc) is float) or (type(src_direc) is int), msg
    if int(src_direc) not in [1, 2, 3]:
        msg = ('Source orientation unsupported. Possible values are: '
               '1, 2, 3.')
        raise ValueError(msg)
    else:
        src_direc = int(src_direc)
    PETSc.Sys.Print('  checkDictionaryConsistency(): SRC_DIREC '
                    'consistency OK.')

    # src_current parameter
    msg = msg1 + 'SRC_CURRENT' + msg2
    assert 'SRC_CURRENT' in in_dict, msg
    msg = msg3 + 'SRC_CURRENT' + msg4
    src_current = in_dict['SRC_CURRENT']
    assert (type(src_current) is float) or (type(src_current) is int), msg
    src_current = np.float(in_dict['SRC_CURRENT'])
    PETSc.Sys.Print('  checkDictionaryConsistency(): SRC_CURRENT '
                    'consistency OK.')

    # src_length parameter
    msg = msg1 + 'SRC_LENGTH' + msg2
    assert 'SRC_LENGTH' in in_dict, msg
    msg = msg3 + 'SRC_LENGTH' + msg4
    src_length = in_dict['SRC_LENGTH']
    assert (type(src_length) is float) or (type(src_length) is int), msg
    src_length = np.float(in_dict['SRC_LENGTH'])
    PETSc.Sys.Print('  checkDictionaryConsistency(): SRC_LENGTH '
                    'consistency OK.')

    # sigma_background parameter
    msg = msg1 + 'CONDUCTIVITY_BACKGROUND' + msg2
    assert 'CONDUCTIVITY_BACKGROUND' in in_dict, msg
    msg = msg3 + 'CONDUCTIVITY_BACKGROUND' + msg4
    SIGMA_BACKGR = in_dict['CONDUCTIVITY_BACKGROUND']
    assert (type(SIGMA_BACKGR) is float) or (type(SIGMA_BACKGR) is int), msg
    sigma_background = np.float(SIGMA_BACKGR)
    PETSc.Sys.Print('  checkDictionaryConsistency(): CONDUCTIVITY_BACKGROUND '
                    'consistency OK.')

    # sigma_file parameter
    msg = msg1 + 'CONDUCTIVITY_MODEL_FILE' + msg2
    assert 'CONDUCTIVITY_MODEL_FILE' in in_dict, msg
    msg = msg3 + 'CONDUCTIVITY_MODEL_FILE' + msg4
    sigma_file = in_dict['CONDUCTIVITY_MODEL_FILE']
    assert type(sigma_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): CONDUCTIVITY_MODEL_FILE '
                    'consistency OK.')

    # nodes_file parameter
    msg = msg1 + 'NODES_FILE' + msg2
    assert 'NODES_FILE' in in_dict, msg
    msg = msg3 + 'NODES_FILE' + msg4
    nodes_file = in_dict['NODES_FILE']
    assert type(nodes_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): NODES_FILE '
                    'consistency OK.')

    # elemsN_file parameter
    msg = msg1 + 'MESH_CONNECTIVITY_FILE' + msg2
    assert 'MESH_CONNECTIVITY_FILE' in in_dict, msg
    msg = msg3 + 'MESH_CONNECTIVITY_FILE' + msg4
    elemsN_file = in_dict['MESH_CONNECTIVITY_FILE']
    assert type(elemsN_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): MESH_CONNECTIVITY_FILE '
                    'consistency OK.')

    # elemsE_file parameter
    msg = msg1 + 'DOFS_CONNECTIVITY_FILE' + msg2
    assert 'DOFS_CONNECTIVITY_FILE' in in_dict, msg
    msg = msg3 + 'DOFS_CONNECTIVITY_FILE' + msg4
    elemsE_file = in_dict['DOFS_CONNECTIVITY_FILE']
    assert type(elemsE_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): DOFS_CONNECTIVITY_FILE '
                    'consistency OK.')

    # edgesN_file parameter
    msg = msg1 + 'DOFS_NODES_FILE' + msg2
    assert 'DOFS_NODES_FILE' in in_dict, msg
    msg = msg3 + 'DOFS_NODES_FILE' + msg4
    edgesN_file = in_dict['DOFS_NODES_FILE']
    assert type(edgesN_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): DOFS_NODES_FILE '
                    'consistency OK.')

    # nnz_file parameter
    msg = msg1 + 'NNZ_FILE' + msg2
    assert 'NNZ_FILE' in in_dict, msg
    msg = msg3 + 'NNZ_FILE' + msg4
    nnz_file = in_dict['NNZ_FILE']
    assert type(nnz_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): NNZ_FILE consistency OK.')

    # bEedges_file parameter
    msg = msg1 + 'BOUNDARIES_FILE' + msg2
    assert 'BOUNDARIES_FILE' in in_dict, msg
    msg = msg3 + 'BOUNDARIES_FILE' + msg4
    bEdges_file = in_dict['BOUNDARIES_FILE']
    assert type(bEdges_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): BOUNDARIES_FILE '
                    'consistency OK.')

    # receivers_file parameter
    msg = msg1 + 'RECEIVERS_FILE' + msg2
    assert 'RECEIVERS_FILE' in in_dict, msg
    msg = msg3 + 'RECEIVERS_FILE' + msg4
    receivers_file = in_dict['RECEIVERS_FILE']
    assert type(receivers_file) is str, msg
    PETSc.Sys.Print('  checkDictionaryConsistency(): RECEIVERS_FILE '
                    'consistency OK.')

    # General message
    PETSc.Sys.Print('  checkDictionaryConsistency(): parameter data types '
                    'in "' + file_name + '" are consistent.')

    # check paths
    PETSc.Sys.Print('  Checking paths consistency...')

    # check sigma_file path
    success = checkFilePath(sigma_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' + sigma_file +
                        '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + sigma_file +
               ' does not exist.')
        raise ValueError(msg)

    # check nodes_file path
    success = checkFilePath(nodes_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' + nodes_file +
                        '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + nodes_file +
               ' does not exist.')
        raise ValueError(msg)

    # check elemsN_file path
    success = checkFilePath(elemsN_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' + elemsN_file +
                        '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + elemsN_file +
               ' does not exist.')
        raise ValueError(msg)

    # check elemsE_file path
    success = checkFilePath(elemsE_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' + elemsE_file +
                        '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + elemsE_file +
               ' does not exist.')
        raise ValueError(msg)

    # check edgesN_file path
    success = checkFilePath(edgesN_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' + edgesN_file +
                        '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + edgesN_file +
               ' does not exist.')
        raise ValueError(msg)

    # check receivers_file path
    success = checkFilePath(receivers_file)
    if success:
        PETSc.Sys.Print('  checkDictionaryConsistency(): "' +
                        receivers_file + '" exists.')
    else:
        msg = ('  checkDictionaryConsistency(): file ' + receivers_file +
               ' does not exist.')
        raise ValueError(msg)

    # General message
    PETSc.Sys.Print('  checkDictionaryConsistency(): file paths in "' +
                    file_name + '" are consistent.')

    # Create csem_modelling dictionary
    out_model = CSEM_MODELLING(rank, freq, src_pos, src_direc, src_current,
                               src_length, sigma_background, sigma_file,
                               nodes_file, elemsN_file, elemsE_file,
                               edgesN_file, nnz_file, bEdges_file,
                               receivers_file, dir_name)

    PETSc.Sys.Print('  A new CSEM_MODELLING dictionary has been '
                    'successfully created.')

    return out_model


def checkDictionaryConsistencySlave(rank, in_dict, file_name, dir_name):
    ''' Check if dictionary consistency match with **PETGEM** requirements.
    (slave task)

    :params int rank: MPI rank.
    :params dict in_dict: input dictionary to be tested.
    :params str file_name: parameters file name.
    :params str dir_name: parent directory of file_name
    :return: csem modelling dictionary after test.
    :rtype: csem_modelling dictionary.
    '''
    # freq parameter
    freq = in_dict['FREQ']
    freq = np.float(in_dict['FREQ'])

    # src_pos parameter
    src_pos = in_dict['SRC_POS']

    # src_direc parameter
    src_direc = in_dict['SRC_DIREC']
    src_direc = int(src_direc)

    # src_current parameter
    src_current = in_dict['SRC_CURRENT']
    src_current = np.float(in_dict['SRC_CURRENT'])

    # src_length parameter
    src_length = in_dict['SRC_LENGTH']
    src_length = np.float(in_dict['SRC_LENGTH'])

    # sigma_background parameter
    SIGMA_BACKGR = in_dict['CONDUCTIVITY_BACKGROUND']
    sigma_background = np.float(SIGMA_BACKGR)

    # sigma_file parameter
    sigma_file = in_dict['CONDUCTIVITY_MODEL_FILE']

    # nodes_file parameter
    nodes_file = in_dict['NODES_FILE']

    # elemsN_file parameter
    elemsN_file = in_dict['MESH_CONNECTIVITY_FILE']

    # elemsE_file parameter
    elemsE_file = in_dict['DOFS_CONNECTIVITY_FILE']

    # edgesN_file parameter
    edgesN_file = in_dict['DOFS_NODES_FILE']

    # nnz_file parameter
    nnz_file = in_dict['NNZ_FILE']

    # edgesN_file parameter
    bEdges_file = in_dict['BOUNDARIES_FILE']

    # receivers_file parameter
    receivers_file = in_dict['RECEIVERS_FILE']

    # Create csem_modelling dictionary
    out_model = CSEM_MODELLING(rank, freq, src_pos, src_direc, src_current,
                               src_length, sigma_background, sigma_file,
                               nodes_file, elemsN_file, elemsE_file,
                               edgesN_file, nnz_file, bEdges_file,
                               receivers_file, dir_name)

    return out_model


def readUserParams(input_params, rank):
    ''' Read a kernel input, namely a parameters file name.

    :params list input_params: user input parameters.
    :param int rank: MPI rank.
    :return: a modelling dictionary.
    :rtype: dict of type modelling.
    '''

    num_params = checkNumberParams(input_params)

    # PETGEM parameters file in 3rd position in sys.argv
    input_path = input_params[3]

    # Check if file_name exist
    success = checkFilePath(input_path)

    if rank == 0:
        if not success:
            msg = ('  readUserParams(): file ' + input_path +
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
        FM_CSEM_temp = import_module(PARAMS_FILE_NAME)
        import_success = True
    except:
        pass

    # Recover original file name
    if remove_extension:
        PARAMS_FILE_NAME = PARAMS_FILE_NAME + '.py'

    if rank == 0:
        if import_success:
            PETSc.Sys.Print('  readUserParams(): file "' + PARAMS_FILE_NAME +
                            '" has been successfully imported.')
        else:
            msg = ('  readUserParams(): file "' + PARAMS_FILE_NAME +
                   '" has not been successfully imported.')
            raise ValueError(msg)

    # Check consistency
    # Check dictionary content (This name must match with dictionary name
    # in parameters file, namely, must be == 'modelling')
    if rank == 0:
        FM_CSEM = checkDictionaryConsistencyMaster(rank,
                                                   FM_CSEM_temp.modelling,
                                                   PARAMS_FILE_NAME,
                                                   DIR_NAME)
    else:
        FM_CSEM = checkDictionaryConsistencySlave(rank,
                                                  FM_CSEM_temp.modelling,
                                                  PARAMS_FILE_NAME,
                                                  DIR_NAME)
    # Print modelling content
    if rank == 0:
        printModellingData(FM_CSEM)

    return FM_CSEM


def unitary_test():
    ''' Unitary test for base.py script.
    '''
    import os
    import sys
    from styles import (petgem_header, test_header,
                        petgemFooter, test_footer)

    petgem_header()
    test_header('base.py')
    PETSc.Sys.Print('Testing init basis for input user.')
    PETSc.Sys.Print('Read input from modelling.py')

    pathname = os.path.dirname(sys.argv[0])

    success = checkFilePath(pathname + '/modelling.py')

    if not success:
        msg = ('  test: file modelling does not exist.')
        raise ValueError(msg)

    pass_test = success
    test_footer(pass_test)
    petgemFooter()


if __name__ == '__main__':
    # Standard module import
    import numpy as np
    import sys
    from os import path
    from importlib import import_module
    import petsc4py
    from petsc4py import PETSc
    # PETGEM module import

    # Unitary test
    unitary_test()
else:
    # Standard module import
    import numpy as np
    import sys
    from os import path
    from os import makedirs
    from os import remove
    import errno
    from importlib import import_module
    import petsc4py
    from petsc4py import PETSc
    # PETGEM module import
    from petgem.base.modelling import CSEM_MODELLING
    from petgem.base.modelling import printModellingData
