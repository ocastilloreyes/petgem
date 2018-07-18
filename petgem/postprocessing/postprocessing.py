#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the functions for post-processing task.
'''


def exportPetscToAscii(nReceivers, xComponentFile,
                       yComponentFile, zComponentFile, out_file):
    ''' Export a saved electric fields in petsc format to Ascii format.

    :param int nReceivers: total number of receivers
    :param str xComponentFile: petsc file name to be readed and exported
    :param str yComponentFile: petsc file name to be readed and exported
    :param str zComponentFile: petsc file name to be readed and exported
    :param str out_file: out file name for electric fields
    :return: electric field in numpy format
    :rtype: ndarray
    '''
    # Build indexes list of receivers
    indx_receivers = np.arange(0, nReceivers, dtype=np.complex)
    # Allocate space for Electric fields
    dataAscii = np.zeros((nReceivers, 4), dtype=np.complex)
    dataMatlab = np.zeros((nReceivers, 3), dtype=np.complex)
    # Read field
    tmp = readPetscVector(xComponentFile, communicator=PETSc.COMM_SELF)
    dataX = tmp.getArray()
    tmp.destroy()
    tmp = readPetscVector(yComponentFile, communicator=PETSc.COMM_SELF)
    dataY = tmp.getArray()
    tmp.destroy()
    tmp = readPetscVector(zComponentFile, communicator=PETSc.COMM_SELF)
    dataZ = tmp.getArray()
    tmp.destroy()

    for iRecv in np.arange(nReceivers):
        dataAscii[iRecv, 0] = iRecv
        dataAscii[iRecv, 1] = dataX[iRecv]
        dataAscii[iRecv, 2] = dataY[iRecv]
        dataAscii[iRecv, 3] = dataZ[iRecv]
        # Set data for matlab matrix
        dataMatlab[iRecv, 0] = dataX[iRecv]
        dataMatlab[iRecv, 1] = dataY[iRecv]
        dataMatlab[iRecv, 2] = dataZ[iRecv]

    dataAscii = dataAscii.view(float)
    dataAscii = np.delete(dataAscii, 1, axis=1)

    file_header = ('Receiver\t\t\tX-component\t\t\t\t\t' +
                   'Y-component\t\t\t\t\tZ-component')
    file_footer = ('--------   -------------------------' +
                   '----------------------    ----------' +
                   '------------------------------------' +
                   '    ----------------------------------------------')
    table_format = ('    %i \t     %4.16e%+4.16ej' +
                    ' \t%4.16e%+4.16ej \t  %4.16e%+4.16ej')
    np.savetxt(out_file, dataAscii, fmt=table_format,
               header=file_header, footer=file_footer)

    # Insert time stamp to file
    with open(out_file, 'a') as outf:
        fmt = '%Y/%m/%d %H:%M:%S'
        TT = datetime.datetime.now().strftime(fmt)
        outf.write('PETGEM execution on: ' + TT)

    return dataMatlab


def exportNumpytoMatlab(data, out_file, electricField=None):
    ''' Export electric fields in numpy format to Matlab format.

    :param int data: electric fields to be saved
    :param str out_file: out file name for electric fields
    :return: none
    '''

    if electricField == 'Primary':
        dataName = 'Ep'
    elif electricField == 'Secondary':
        dataName = 'Es'
    elif electricField == 'Total':
        dataName = 'Et'

    # Save matlab data
    savemat(out_file, {dataName: data})

    return


def EpReceiverComputation(model, point, numDimensions):
    ''' Compute the primary electric field for an array of point (receivers).

    :param object_modelling model: CSEM modelling with physical parameters
    :param ndarray point: receiver spatial coordinates
    :param int numDimensions: number of dimensions
    :return: primary electric field on receivers
    :rtype: ndarray.
    '''

    # Read physical parameters
    # ----------- Read physical parameters -----------
    FREQ = np.float(model['FREQ'])
    SIGMA_BGROUND = np.float(model['CONDUCTIVITY_BACKGROUND'])
    SRC_POS = np.asarray(model['SRC_POS'], dtype=np.float)
    SRC_DIREC = np.int(model['SRC_DIREC'])
    I = np.float(model['SRC_CURRENT'])
    dS = np.float(model['SRC_LENGTH'])

    # Source position as independent variables
    X0 = np.float64(SRC_POS[0])
    Y0 = np.float64(SRC_POS[1])
    Z0 = np.float64(SRC_POS[2])

    # Compute constants
    # Imaginary part for complex numbers
    imag_part = np.complex(0.0 + 1.0j)
    # Vacuum permeability
    MU = np.float(np.float(4.0)*np.pi*np.float(1.0e-7))
    # Angular frequency
    OMEGA = np.float(FREQ*np.float(2.0)*np.pi)
    # Propagation parameter
    WAVENUMBER = np.complex(np.sqrt(-imag_part*MU*OMEGA*SIGMA_BGROUND))

    # Allocate
    Ep = np.zeros(numDimensions, dtype=np.complex)

    # Distance to the source
    xx = point[0] - X0
    yy = point[1] - Y0
    zz = point[2] - Z0
    r = np.float64(np.sqrt(xx**2 + yy**2 + zz**2))

    if(r < np.float64(1.0e0)):
        r = np.float64(1.0e0)

    # E = AA [ BB + (wavenumber^2 * r^2 -1i * wavenumber * r-1)]
    AA = I * dS * \
        (np.float64(4.0) * np.pi * SIGMA_BGROUND * r**3)**-1 * \
        np.exp((-imag_part) * WAVENUMBER * r)
    BB = -WAVENUMBER**2 * r**2 + \
        (np.float64(3.0) * imag_part * WAVENUMBER * r) + \
        np.float64(3.0)

    if SRC_DIREC == 1:
        # X-directed
        Ep[0] = AA * ((xx**2/r**2)*BB + (WAVENUMBER**2 * r**2 - imag_part *
                                         WAVENUMBER * r - np.float64(1.0)))
        Ep[1] = AA * (xx*yy/r**2)*BB
        Ep[2] = AA * (xx*zz/r**2)*BB
    elif SRC_DIREC == 2:
        # Y-directed
        Ep[0] = AA * (xx*yy/r**2)*BB
        Ep[1] = AA * ((yy**2/r**2)*BB + (WAVENUMBER**2 * r**2 - imag_part *
                                         WAVENUMBER*r-np.float64(1.0)))
        Ep[2] = AA * (yy*zz/r**2)*BB
    else:
        # Z-directed
        Ep[0] = AA * (xx*zz/r**2)*BB
        Ep[1] = AA * (zz*yy/r**2)*BB
        Ep[2] = AA * ((zz**2/r**2)*BB + (WAVENUMBER**2 * r**2 - imag_part *
                                         WAVENUMBER*r-np.float64(1.0)))

    return Ep


def EtReceiverComputation(primary_field, secondary_field, numDimensions):
    ''' Compute the total electric field on receivers.

    :param ndarray primary_field: primary electric field on receiver.
    :param ndarray secondary_field: secondary electric field on receiver.
    :param int numDimensions: number of dimensions
    :return: total electric field on receivers.
    :rtype: ndarray.
    '''

    # Compute total field
    Et = np.zeros(numDimensions, dtype=np.complex)
    Et = primary_field + secondary_field

    return Et


def EsReceiverComputation(field, coordEle, coordReceiver, nodesEle,
                          edgeOrder, numDimensions):
    ''' Compute the secondary electric field on receivers.

    :param ndarray field: secondary field to be interpolated
    :param ndarray coordElement: element spatial coordinates
    :param ndarray coordReceiver: receiver spatial coordinates
    :param ndarray nodesEle: nodal indices of element (element container)
    :param int edgeOrder: order of tetrahedral edge element
    :param int numDimensions: number of dimensions
    :return: secondary electric field on receivers
    :rtype: ndarray.
    '''

    # Imaginary part for complex numbers
    imag_part = np.complex(0.0 + 1.0j)

    # ----------- Edge definition for the tetrahedral elements -----------
    # ----- Table 8.2 of Jin's book. Here is consider as an 1D-array -----
    edgesN = np.array([0, 1, 0, 2, 0, 3, 1, 2, 3, 1, 2, 3], dtype=np.int)

    # ----------- Definition of arrays for vector operations -----------
    # Signs computation
    idx_signs1 = np.array([1, 2, 3, 2, 1, 3], dtype=np.int)
    idx_signs2 = np.array([0, 0, 0, 1, 3, 2], dtype=np.int)

    # Allocate
    Es = np.zeros(numDimensions, dtype=np.complex)

    # ----------- Compute edges's length of element -----------
    tmp = coordEle.reshape(4, 3)
    tmp = tmp[edgesN, :]
    edges = tmp[1::2, :] - tmp[0::2, :]
    lengthEle = np.sqrt(np.sum(np.square(edges), axis=1))

    # ----------- Compute element's volume -----------
    CONST_VOL1 = np.float(1.0)/np.float(6.0)
    eleVol = (((coordEle[3]-coordEle[0])*(coordEle[7]-coordEle[1]) *
               (coordEle[11]-coordEle[2]) +
               (coordEle[4]-coordEle[1])*(coordEle[8]-coordEle[2]) *
               (coordEle[9]-coordEle[0]) +
               (coordEle[6]-coordEle[0])*(coordEle[10]-coordEle[1]) *
               (coordEle[5]-coordEle[2])) -
              ((coordEle[5]-coordEle[2])*(coordEle[7]-coordEle[1]) *
               (coordEle[9]-coordEle[0]) +
               (coordEle[6]-coordEle[0])*(coordEle[4]-coordEle[1]) *
               (coordEle[11]-coordEle[2]) +
               (coordEle[10]-coordEle[1])*(coordEle[8]-coordEle[2]) *
               (coordEle[3]-coordEle[0]))) * CONST_VOL1

    # ----------- Edge's signs -----------
    tmp = nodesEle
    tmp = tmp[idx_signs1] - tmp[idx_signs2]
    signsEle = tmp / np.abs(tmp)

    # Nedelec basis computation
    basis = nedelecBasisIterative(coordEle.reshape(4, 3),
                                  coordReceiver, eleVol, lengthEle, edgeOrder)

    rField = np.real(field)
    iField = np.imag(field)

    # Compute secondary field
    for kedge in np.arange(edgeOrder):
        # Add contributions
        Es[0] += (rField[kedge]*basis[kedge, 0]*signsEle[kedge]) + \
                 (imag_part*iField[kedge]*basis[kedge, 0]*signsEle[kedge])
        Es[1] += (rField[kedge]*basis[kedge, 1]*signsEle[kedge]) + \
                 (imag_part*iField[kedge]*basis[kedge, 1]*signsEle[kedge])
        Es[2] += (rField[kedge]*basis[kedge, 2]*signsEle[kedge]) + \
                 (imag_part*iField[kedge]*basis[kedge, 2]*signsEle[kedge])

    return Es


def computeFieldsReceiver(modelling, coordReceiver, coordElement,
                          nodesElement, x_field, edgeOrder, numDimensions):
    ''' Compute the CSEM modelling output: primary electric field, secondary
    electric field and total electric field on receivers position.

    :param object_modelling model: CSEM modelling with physical parameters
    :param ndarray coordReceiver: receiver spatial coordinates
    :param ndarray coordElement: element spatial coordinates
    :param ndarray nodesElement: nodal indices of element (element container)
    :param ndarray x_field: vector solution for receiver
    :param int edgeOrder: order of tetrahedral edge element
    :param int numDimensions: number of dimensions
    :return: primary, secondary and total electric field
    :rtype: ndarray
    '''
    # ----- Primary field computation -----
    Ep = EpReceiverComputation(modelling, coordReceiver, numDimensions)

    # ----- Secondary field computation -----
    Es = EsReceiverComputation(x_field, coordElement, coordReceiver,
                               nodesElement, edgeOrder, numDimensions)

    # ----- Total field computation -----
    Et = EtReceiverComputation(Ep, Es, numDimensions)

    return Ep, Es, Et


def postProcessingFields(receivers, modelling, x, Iend_receivers,
                         Istart_receivers, edgeOrder, nodalOrder,
                         numDimensions, rank):
    ''' Compute the CSEM modelling output: primary electric field, secondary
    electric field and total electric field on receivers position.

    :param petsc matrix receivers: data receivers to compute electric fields
    :param object_modelling model: CSEM modelling with physical parameters.
    :param petsc vector x: solution vector
    :param int Iend_receivers: last range for receivers
    :param int Istart_receivers: init range for receivers
    :param int edgeOrder: order of tetrahedral edge element
    :param int nodalOrder: order of tetrahedral nodal element
    :param int numDimensions: number of dimensions
    :param int rank: MPI rank
    :return: elapsedTimepostprocessing
    :rtype: float
    '''

    # Start timer
    Init_postprocessing = getTime()

    # Number of receivers
    nReceivers = receivers.getSize()[0]
    nReceiversLocal = Iend_receivers-Istart_receivers

    # Print number of receivers per MPI task
    PETSc.Sys.Print('  Number of receivers:', nReceivers)
    PETSc.Sys.syncPrint('    Rank: ', rank, ' is post-processing ',
                        nReceiversLocal, ' receivers')
    PETSc.Sys.syncFlush()

    # Read edges-connectivity for receivers
    # Auxiliar arrays
    dataRecv = np.zeros(edgeOrder, dtype=np.float)
    edgesIdxRecv = np.zeros((nReceiversLocal, edgeOrder), dtype=PETSc.IntType)
    idx = 0
    for iRecv in np.arange(Istart_receivers, Iend_receivers):
        # Get data of iRecv
        temp = np.asarray(receivers.getRow(iRecv))
        dataRecv[:] = np.real(temp[1, 19:25])
        # Edge-indexes for iRecv
        edgesIdxRecv[idx, :] = (dataRecv).astype(PETSc.IntType)
        idx += 1

    # Gather global solution of x to local vector
    # Sequential vector for gather tasks
    x_local = createSequentialVector(edgeOrder*nReceiversLocal,
                                     communicator=None)

    # Build Index set in PETSc format
    IS_edges = PETSc.IS().createGeneral(edgesIdxRecv.flatten(),
                                        comm=PETSc.COMM_WORLD)
    # Build gather vector
    gatherVector = PETSc.Scatter().create(x, IS_edges, x_local, None)
    # Ghater values
    gatherVector.scatter(x, x_local, PETSc.InsertMode.INSERT_VALUES,
                         PETSc.ScatterMode.FORWARD)

    # Post-processing electric fields
    # Create parallel structures
    EpX = createParallelVector(nReceivers, communicator=None)
    EpY = createParallelVector(nReceivers, communicator=None)
    EpZ = createParallelVector(nReceivers, communicator=None)
    EsX = createParallelVector(nReceivers, communicator=None)
    EsY = createParallelVector(nReceivers, communicator=None)
    EsZ = createParallelVector(nReceivers, communicator=None)
    EtX = createParallelVector(nReceivers, communicator=None)
    EtY = createParallelVector(nReceivers, communicator=None)
    EtZ = createParallelVector(nReceivers, communicator=None)
    EpDense = createParallelDenseMatrix(nReceivers, numDimensions,
                                        communicator=None)
    EsDense = createParallelDenseMatrix(nReceivers, numDimensions,
                                        communicator=None)
    EtDense = createParallelDenseMatrix(nReceivers, numDimensions,
                                        communicator=None)

    # Reshape auxiliar array
    dataRecv = np.zeros(numDimensions+nodalOrder*numDimensions+nodalOrder,
                        dtype=np.float)
    # Compute fields for all local receivers
    idx = 0
    for iRecv in np.arange(Istart_receivers, Iend_receivers):
        # Get data of iRecv
        temp = np.asarray(receivers.getRow(iRecv))
        dataRecv[:] = np.real(temp[1, 0:19])
        # Receivers coordinates
        coordReceiver = dataRecv[0:3]
        # Element coordinates
        coordElement = dataRecv[3:15]
        # Nodal-indexes
        nodesElement = (dataRecv[15:19]).astype(PETSc.IntType)
        # Compute fields
        [EpRecv, EsRecv, EtRecv] = computeFieldsReceiver(modelling,
                                                         coordReceiver,
                                                         coordElement,
                                                         nodesElement,
                                                         x_local[idx *
                                                                 edgeOrder:
                                                                 (idx *
                                                                  edgeOrder) +
                                                                 edgeOrder],
                                                         edgeOrder,
                                                         numDimensions)
        idx += 1
        # Set primary field components
        EpX.setValue(iRecv, EpRecv[0], addv=PETSc.InsertMode.INSERT_VALUES)
        EpY.setValue(iRecv, EpRecv[1], addv=PETSc.InsertMode.INSERT_VALUES)
        EpZ.setValue(iRecv, EpRecv[2], addv=PETSc.InsertMode.INSERT_VALUES)
        EpDense.setValue(iRecv, 0, EpRecv[0],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EpDense.setValue(iRecv, 1, EpRecv[1],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EpDense.setValue(iRecv, 2, EpRecv[2],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        # Set secondary field components
        EsX.setValue(iRecv, EsRecv[0],
                     addv=PETSc.InsertMode.INSERT_VALUES)
        EsY.setValue(iRecv, EsRecv[1],
                     addv=PETSc.InsertMode.INSERT_VALUES)
        EsZ.setValue(iRecv, EsRecv[2],
                     addv=PETSc.InsertMode.INSERT_VALUES)
        EsDense.setValue(iRecv, 0, EsRecv[0],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EsDense.setValue(iRecv, 1, EsRecv[1],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EsDense.setValue(iRecv, 2, EsRecv[2],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        # Set total field components
        EtX.setValue(iRecv, EtRecv[0], addv=PETSc.InsertMode.INSERT_VALUES)
        EtY.setValue(iRecv, EtRecv[1], addv=PETSc.InsertMode.INSERT_VALUES)
        EtZ.setValue(iRecv, EtRecv[2], addv=PETSc.InsertMode.INSERT_VALUES)
        EtDense.setValue(iRecv, 0, EtRecv[0],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EtDense.setValue(iRecv, 1, EtRecv[1],
                         addv=PETSc.InsertMode.INSERT_VALUES)
        EtDense.setValue(iRecv, 2, EtRecv[2],
                         addv=PETSc.InsertMode.INSERT_VALUES)

    # Start global vector assembly
    EpX.assemblyBegin(), EpY.assemblyBegin(), EpZ.assemblyBegin()
    EsX.assemblyBegin(), EsY.assemblyBegin(), EsZ.assemblyBegin()
    EtX.assemblyBegin(), EtY.assemblyBegin(), EtZ.assemblyBegin()
    EpDense.assemblyBegin(), EsDense.assemblyBegin(), EtDense.assemblyBegin()
    # End global vector assembly
    EpX.assemblyEnd(), EpY.assemblyEnd(), EpZ.assemblyEnd()
    EsX.assemblyEnd(), EsY.assemblyEnd(), EsZ.assemblyEnd()
    EtX.assemblyEnd(), EtY.assemblyEnd(), EtZ.assemblyEnd()
    EpDense.assemblyEnd(), EsDense.assemblyEnd(), EtDense.assemblyEnd()

    # Verify if directory exists
    MASTER = 0
    if rank == MASTER:
        checkIfDirectoryExist(modelling['DIR_NAME'] + '/Output/Petsc')
        checkIfDirectoryExist(modelling['DIR_NAME'] + '/Output/Ascii')
        checkIfDirectoryExist(modelling['DIR_NAME'] + '/Output/Matlab')

    # Print
    PETSc.Sys.Print('  Saving output:')
    # Export electric fields (petsc format)
    printMessage('    Petsc format', rank)
    # Save primary electric field
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EpX.dat',
                     EpX, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EpY.dat',
                     EpY, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EpZ.dat',
                     EpZ, communicator=None)
    writeDenseMatrix(modelling['DIR_NAME'] + '/Output/Petsc/Ep.dat',
                     EpDense, communicator=None)
    # Save secondary electric field
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EsX.dat',
                     EsX, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EsY.dat',
                     EsY, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EsZ.dat',
                     EsZ, communicator=None)
    writeDenseMatrix(modelling['DIR_NAME'] + '/Output/Petsc/Es.dat',
                     EsDense, communicator=None)
    # Save total electric field
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EtX.dat',
                     EtX, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EtY.dat',
                     EtY, communicator=None)
    writePetscVector(modelling['DIR_NAME'] + '/Output/Petsc/EtZ.dat',
                     EtZ, communicator=None)
    writeDenseMatrix(modelling['DIR_NAME'] + '/Output/Petsc/Et.dat',
                     EtDense, communicator=None)

    # Export electric fields (Ascii and Matlab format)
    if rank == MASTER:
        # Export electric fields (Ascii format)
        # Save primary electric field
        printMessage('    Ascii format', rank)
        dataEp = exportPetscToAscii(nReceivers,
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EpX.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EpY.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EpZ.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Ascii/Ep.dat')
        # Save secondary electric field
        dataEs = exportPetscToAscii(nReceivers,
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EsX.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EsY.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EsZ.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Ascii/Es.dat')
        # Save total electric field
        dataEt = exportPetscToAscii(nReceivers,
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EtX.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EtY.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Petsc/EtZ.dat',
                                    modelling['DIR_NAME'] +
                                    '/Output/Ascii/Et.dat')
        # Export electric fields (Matlab format)
        printMessage('    Matlab format', rank)
        # Save primary electric field
        exportNumpytoMatlab(dataEp, modelling['DIR_NAME'] +
                            '/Output/Matlab/Ep.mat', electricField='Primary')
        # Save secondary electric field
        exportNumpytoMatlab(dataEs, modelling['DIR_NAME'] +
                            '/Output/Matlab/Es.mat', electricField='Secondary')
        # Save total electric field
        exportNumpytoMatlab(dataEt, modelling['DIR_NAME'] +
                            '/Output/Matlab/Et.mat', electricField='Total')
        # Remove temporal files (petsc)
        filesToDelete = [modelling['DIR_NAME'] + '/Output/Petsc/EpX.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EpY.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EpZ.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsX.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsY.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsZ.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtX.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtY.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtZ.dat',
                         modelling['DIR_NAME'] + '/Output/Petsc/EpX.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EpY.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EpZ.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsX.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsY.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EsZ.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtX.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtY.dat.info',
                         modelling['DIR_NAME'] + '/Output/Petsc/EtZ.dat.info']

        for iFile in np.arange(len(filesToDelete)):
            removeFile(filesToDelete[iFile])

    # End timer
    End_postprocessing = getTime()

    # Elapsed time in assembly
    elapsedTimepostprocessing = End_postprocessing-Init_postprocessing

    return elapsedTimepostprocessing


def unitary_test():
    ''' Unitary test for post_processing.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import numpy as np
    import datetime
    from scipy.io import savemat
    import petsc4py
    from petsc4py import PETSc
    # PETGEM module import
    from petgem.base.base import checkIfDirectoryExist
    from petgem.base.base import removeFile
    from petgem.parallel.parallel import createSequentialVector
    from petgem.parallel.parallel import createParallelVector
    from petgem.parallel.parallel import readPetscVector
    from petgem.parallel.parallel import writePetscVector
    from petgem.parallel.parallel import createParallelDenseMatrix
    from petgem.parallel.parallel import printMessage
    from petgem.parallel.parallel import writeDenseMatrix
    from petgem.efem.efem import nedelecBasisIterative
    from petgem.monitoring.monitoring import getTime
