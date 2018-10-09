''' Parameters file template for 3D CSEM modelling.

By definition, any 3D CSEM survey should include: physical parameters, a mesh
file, source and receivers files. These data are included in the modelParams.py
file. Additionally, options for PETSc solvers are defined in a petsc.opts file.

In order to avoid a specific parser, modelParams.py file is imported by PETGEM
as a Python dictionary. As consequence, the dictionary name and his key names
MUST NOT BE changed.

All file paths should consider as absolute.

Next, each key is described.
'''
modelling = {
    # ---------- Nedelec element order ----------
    # 1 = First nedelec order (6 DOFS per element)
    # 2 = Second nedelec order (20 DOFS per element)
    # 3 = Third nedelec order (45 DOFS per element)
    # Type: int
    # Optional: NO
    'NEDELEC_ORDER': 2,

    # ---------- CUDA SUPPORT ----------
    # 0 = No
    # 1 = Yes
    # Type: int
    # Optional: NO
    'CUDA': 0,

    # ----- Physical parameters -----
    # Source
    # Source frequency. Type: float
    # Optional: NO
    'FREQ': 1.0,
    # Source position(x, y, z). Type: float
    # Optional: NO
    'SRC_POS': [1750.0, 1750.0, -975.0],
    # Source orientarion. Type: int
    # 1 = X-directed source
    # 2 = Y-directed source
    # 3 = Z-directed source
    # Optional: NO
    'SRC_DIREC': 1,
    # Source current. Type: float
    # Optional: NO
    'SRC_CURRENT': 1.0,
    # Source length. Type: float
    # Optional: NO
    'SRC_LENGTH': 1.0,
    # Conductivity model. Type: str
    # Optional: NO
    'CONDUCTIVITY_MODEL_FILE': 'PATH_TO_FILE',
    # Background conductivity. Type: float
    # Optional: NO
    'CONDUCTIVITY_BACKGROUND': 3.33,

    # ------- Mesh and conductivity model files ------
    # Mesh files
    # Nodes spatial coordinates. Type: str
    # Optional: NO
    'NODES_FILE': 'PATH_TO_FILE',
    # Elements-nodes connectivity. Type: str
    # Optional: NO
    'MESH_CONNECTIVITY_FILE': 'PATH_TO_FILE',
    # Elements-faces connectivity. Type: str
    # Optional: NO
    'FACES_CONNECTIVITY_FILE': 'PATH_TO_FILE',
    # Faces-nodes connectivity. Type: str
    # Optional: NO
    'FACES_NODES_FILE': 'PATH_TO_FILE',
    # Elements-edges connectivity. Type: str
    # Optional: NO
    'EDGES_CONNECTIVITY_FILE': 'PATH_TO_FILE',
    # Edges-nodes connectivity. Type: str
    # Optional: NO
    'EDGES_NODES_FILE': 'PATH_TO_FILE',
    # Sparsity pattern for matrix allocation (PETSc)
    'NNZ_FILE': 'PATH_TO_FILE',
    # Boundaries. Type: str
    # Optional: NO
    'BOUNDARIES_FILE': 'PATH_TO_FILE',

    # ------------ Solver -----------
    # Solver options must be set in
    # petsc_solver.opts

    # ------------ Receivers file -----------
    # Name of the file that contains the receivers position. Type: str
    # Optional: NO
    'RECEIVERS_FILE': 'PATH_TO_FILE',
}
