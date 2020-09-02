# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import sys
sys.path.append('../')
from petgem.parallel import MPIEnvironment

def test_mpi():
    # Obtain the MPI environment
    parEnv = MPIEnvironment()

    # Run simple hello world
    sys.stdout.write("Hello, World! I am process %d of %d on %s.\n" % (parEnv.rank, parEnv.num_proc, parEnv.machine_name))
