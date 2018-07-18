#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions for performance monitoring such as timers and HW
counters.
'''


def startLogEvent(eventName):
    ''' Enable a logging event in a parallel pool.

    :param str eventName: name of the log Event
    :return: logEvent
    :rtype: petsc logging event
    '''
    logEvent = PETSc.Log.Event(eventName)

    return logEvent


def getTime():
    ''' Start a timer in a parallel pool.

    :param: None
    :return: logTimer
    :rtype: petsc log timer
    '''
    petscTimer = PETSc.Log().getTime()

    return petscTimer


def printElapsedTimes(elapsedTimeAssembly, elapsedTimeSolver,
                      elapsedTimePostprocessing, iterationNumber, rank):
    ''' Print elapsed times in assembly, solver and postprocessing tasks.

    :param float elapsedTimeAssembly: elapsed time in assembly task
    :param float elapsedTimeSolver: elapsed time in assembly task
    :param float elapsedTimePostprocessing: elapsed time in assembly task
    :param int iterationNumber: number of solver iterations
    :param int rank: MPI rank
    '''

    MASTER = 0
    if rank == MASTER:
        PETSc.Sys.Print('  Assembly:             {:e}'.
                        format(elapsedTimeAssembly))
        PETSc.Sys.Print('  Solver:               {:e}'.
                        format(elapsedTimeSolver))
        PETSc.Sys.Print('  Postprocessing:       {:e}'.
                        format(elapsedTimePostprocessing))
        PETSc.Sys.Print('  Solver iterations:    {:e}'.format(iterationNumber))

    return


def unitary_test():
    ''' Unitary test for monitoring.py script.
    '''


if __name__ == '__main__':
    # Standard module import
    import petsc4py
    from petsc4py import PETSc
    unitary_test()
else:
    # Standard module import
    import petsc4py
    from petsc4py import PETSc
