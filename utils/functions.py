import os
import argparse
import numpy as np
import meshio
import petsc4py
import sys
import textwrap
from petsc4py import PETSc

def parsePreprocessingArgs():
    parser = argparse.ArgumentParser(description="Preprocess mesh, resistivity, and receiver data for PETGEM simulations")

    # Required positional argument
    parser.add_argument("-nord",          type=int, required=True, help="Polynomial order used to select the mesh (e.g., 1, 2, 3)")
    parser.add_argument("-case_dir",      type=str, required=True, help="Directory containing case data")
    parser.add_argument("-mesh_filename", type=str, required=True, help="Mesh filename")
    parser.add_argument("-receiver_filename", type=str, required=True, help="Receivers filename")
    parser.add_argument("-source_filename",   type=str, required=True, help="Receivers filename")

    # Optional arguments
    parser.add_argument("-resistivity_view", type=str, default=None, help="Export resistivity distribution (e.g., vtk:filename.vtu)")
    parser.add_argument("-dm_view",          type=str, default=None, help="Print dm data")
    parser.add_argument("-sigma_file",       type=str, default=None, help="CSV file with sigma_x, sigma_y, sigma_z for each material")

    return parser.parse_args()


def parsePostprocessingArgs():
    parser = argparse.ArgumentParser(description="Postprocess electromagnetic responses from PETGEM simulations")

    # Required positional argument
    parser.add_argument("-nord",          type=int, required=True, help="Polynomial order used to select the mesh (e.g., 1, 2, 3)")
    parser.add_argument("-case_dir",      type=str, required=True, help="Directory containing case data")
    parser.add_argument("-receiver_filename", type=str, required=True, help="Receivers filename")
    parser.add_argument("-responses_filename", type=str, required=True, help="Receivers filename")
    
    return parser.parse_args()


def createDM(numDimensions, cells, coords, dm_view=False):

    plex = PETSc.DMPlex().create()
    plex.setFromOptions()
    plex.createFromCellList(numDimensions, cells, coords)
    
    if dm_view:
        PETSc.Options()["dm_view"] = ""
        plex.viewFromOptions("-dm_view")

    dim = plex.getDimension()

    # Create manually a section with 1 field, dim components on cells
    plex.setNumFields(1)
    numComp = dim
    numDof = [0] * (dim + 1)
    numDof[-1] = dim
    s = plex.createSection(numComp, numDof)
    s.setFieldName(0, "resistivity")
    s.setUp()
    plex.setSection(s)

    return plex

def writeDM(plex, resistivity, output_mesh_filename, resistivity_view=None):
    v = plex.createGlobalVec()
    v.getArray()[:] = resistivity.reshape(-1)[:]

    if resistivity_view is not None:
        PETSc.Options()["resistivity_view"] = f"vtk:{resistivity_view}"
    v.viewFromOptions("-resistivity_view")

    viewer = PETSc.ViewerHDF5().create(output_mesh_filename, "w")
    viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
    plex.setName("petgem_mesh")
    plex.topologyView(viewer)
    plex.labelsView(viewer)
    plex.coordinatesView(viewer)
    plex.sectionView(viewer, plex)
    v.setName("resistivity")
    plex.globalVectorView(viewer, plex, v)
    

def writeReceivers(input_receivers_filename, output_receivers_filename):
    # Here, the receivers are defined as real, but PETSc for PETGEM is configured to 
    # use complex scalars. Currently, the conversion from real to complex has not been 
    # implemented. Therefore, we store the data as complex, allowing PETGEM to cast it 
    # from complex to real when needed.
    
    receivers = np.loadtxt(input_receivers_filename, comments='#')
    vector = PETSc.Vec().createWithArray(receivers, comm=PETSc.COMM_SELF)
    vector.setName("receivers")
    vector.setUp()
    viewer = PETSc.Viewer().createHDF5(output_receivers_filename, mode='w', comm=PETSc.COMM_SELF)
    vector.view(viewer)


def writeParamsFile(nord, output_dir, output_filename):
    content = textwrap.dedent(f"""\
        -mesh_filename {output_dir}/model_p{nord}.h5
        -receivers_filename {output_dir}/receivers.h5
        -source_filename {output_dir}/sources.txt
        -nord {nord}
        -dm_mat_type is
        -ksp_type fgmres
        -pc_type bddc
        -pc_bddc_use_deluxe_scaling 1
        -pc_bddc_coarse_pc_type lu
        -output_dir {output_dir}/
        -output_filename {output_filename}
    """)

    filename = f"{output_dir}/params_nord{nord}.txt"
    with open(filename, "w") as f:
        f.write(content)

def readVectorH5(filename, dataset_name):
    """Read a PETSc Vec from an HDF5 file."""
    tmp = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    tmp.setName(dataset_name)
    viewer = PETSc.Viewer().createHDF5(str(filename), mode='r', comm=PETSc.COMM_SELF)
    tmp.load(viewer)
    viewer.destroy()
    vector = tmp.getArray()
    return vector