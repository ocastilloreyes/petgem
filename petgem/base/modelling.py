#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the csem_modelling dictionary. csem_modelling
dictionary contain the main initial parameters for a CSEM FM modelling such
as: frequency, source position, conductivity model and mesh information.
'''


def CSEM_MODELLING(rank, freq, src_pos, src_direc, src_current,
                   src_length, sigma_background, sigma_file,
                   nodes_file, elemsN_file, elemsE_file, edgesN_file,
                   nnz_file, bEdges_file, receivers_file, dir_name):
    ''' csem_modelling dictionary with main parameters for CSEM FM.

    :param int rank: MPI rank.
    :param int,float freq: frequency.
    :param list src_pos: source position.
    :param int,float src_dir: source orientation.
    :param int,float src_current: source current.
    :param int,float src_length: source length.
    :param int,float sigma_background: background conductivity.
    :param str sigma_file: file name of conductivity model.
    :param str nodes_file: file name of node spatial coordinates.
    :param str elemsN_file: file name of elements-nodes connectivity.
    :param str elemsE_file: file name of elements-edges connectivity.
    :param str edgesN_file: file name of edges-nodes connectivity.
    :param str nnz_file: file name of nnz for matrix allocation.
    :param str bEdges_file: file name of boundary edges.
    :param str receivers_file: file name or receivers position.
    :param str dir_name: parent directory of sigma_file, nodes_file
                         and elemsN_file.
    :return: CSEM_MODELLING dictionary.
    :rtype: python dictionary.
    '''

    if rank == 0:
        PETSc.Sys.Print('  CSEM_MODELLING(): Creating a '
                        'CSEM_MODELLING dictionary.')

    modelling = {'FREQ': freq,
                 'SRC_POS': src_pos,
                 'SRC_DIREC': src_direc,
                 'SRC_CURRENT': src_current,
                 'SRC_LENGTH': src_length,
                 'CONDUCTIVITY_BACKGROUND': sigma_background,
                 'CONDUCTIVITY_MODEL_FILE': sigma_file,
                 'NODES_FILE': nodes_file,
                 'MESH_CONNECTIVITY_FILE': elemsN_file,
                 'DOFS_CONNECTIVITY_FILE': elemsE_file,
                 'DOFS_NODES_FILE': edgesN_file,
                 'NNZ_FILE': nnz_file,
                 'BOUNDARIES_FILE': bEdges_file,
                 'RECEIVERS_FILE': receivers_file,
                 'DIR_NAME': dir_name}

    return modelling


def printModellingData(input_modelling):
    ''' Print the content of a csem_modelling dictionary.
    :param dictionary: input_modelling.
    :return: None.
    '''

    A = str(input_modelling['FREQ'])
    B = str(input_modelling['SRC_POS'])
    C = str(input_modelling['SRC_DIREC'])
    D = str(input_modelling['SRC_CURRENT'])
    E = str(input_modelling['SRC_LENGTH'])
    F = str(input_modelling['CONDUCTIVITY_BACKGROUND'])
    G = input_modelling['CONDUCTIVITY_MODEL_FILE']
    H = input_modelling['NODES_FILE']
    I = input_modelling['MESH_CONNECTIVITY_FILE']
    J = input_modelling['DOFS_CONNECTIVITY_FILE']
    K = input_modelling['DOFS_NODES_FILE']
    L = input_modelling['NNZ_FILE']
    M = input_modelling['BOUNDARIES_FILE']
    N = input_modelling['RECEIVERS_FILE']
    O = input_modelling['DIR_NAME']

    PETSc.Sys.Print('\nData for CSEM_MODELLING')
    PETSc.Sys.Print('='*75)
    PETSc.Sys.Print('  Frequency:                 {0:50}'.format(A))
    PETSc.Sys.Print('  Source position:           {0:50}'.format(B))
    PETSc.Sys.Print('  Source orientation:        {0:50}'.format(C))
    PETSc.Sys.Print('  Source current:            {0:50}'.format(D))
    PETSc.Sys.Print('  Source length:             {0:50}'.format(E))
    PETSc.Sys.Print('  Sigma background:          {0:50}'.format(F))
    PETSc.Sys.Print('  Sigma file:                {0:50}'.format(G))
    PETSc.Sys.Print('  Nodes file:                {0:50}'.format(H))
    PETSc.Sys.Print('  Elements-nodes file:       {0:50}'.format(I))
    PETSc.Sys.Print('  Elements-edges file:       {0:50}'.format(J))
    PETSc.Sys.Print('  Edges-nodes file:          {0:50}'.format(K))
    PETSc.Sys.Print('  nnz file:                  {0:50}'.format(L))
    PETSc.Sys.Print('  Boundary edges file:       {0:50}'.format(M))
    PETSc.Sys.Print('  Receivers file:            {0:50}'.format(N))
    PETSc.Sys.Print('  Parent directory:          {0:50}'.format(O))

    return


def unitary_test():
    ''' Unitary test for modelling.py script.
    '''
    from styles import (petgemHeader, test_header,
                        petgemFooter, test_footer)

    printPetgemHeader()
    test_header('modelling.py')
    PETSc.Sys.Print('Testing csem_modelling creation.')
    PETSc.Sys.Print('Creating modelling...')
    # sample data: [rank, freq, src_pos, src_direc, src_current,
    #              src_length, sigma_background, sigma_file,
    #              nodes_file, elemsN_file, elemsE_file, edgesN_file,
    #              bEdges_file, receivers_file, out_prefix, dir_name)
    sample_data = [0, 1.0, [100, 100, 100], 1, 2.5, 2.5,
                   3.3, 'sigma_file.data', 'nodes_file.data',
                   'elemsN_file.data', 'elemsE_file.data', 'edgesN_file.data',
                   'nnz_file', 'bEdges_file.dat', 'receivers.data',
                   'parent/path']

    sample_csem = CSEM_MODELLING(*sample_data)

    # Print content
    printModellingData(sample_csem)

    pass_test = True
    test_footer(pass_test)
    printPetgemFooter()


if __name__ == '__main__':
    # Standard module import
    import sys
    import petsc4py
    from petsc4py import PETSc
    unitary_test()
else:
    # Standard module import
    import sys
    import petsc4py
    from petsc4py import PETSc
