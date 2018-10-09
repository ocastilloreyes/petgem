''' Parameters file template for **PETGEM** preprocessing.

For preprocessing task, a mesh file, conductivity model and
receivers file are mandatory.

In order to avoid a specific parser, this file is imported by
**PETGEM** as a Python dictionary. As consequence, the dictionary
name and his key names MUST NOT BE changed.

All file paths should consider as absolute.
'''
preprocessing = {
    # ---------- Nedelec element order ----------
    # 1 = First Nédélec order (6 DOFS per element)
    # 2 = Second Nédélec order (20 DOFS per element)
    # 3 = Third Nédélec order (45 DOFS per element)
    # Type: int
    # Optional: NO
    'NEDELEC_ORDER': 3,

    # ---------- Mesh file ----------
    # Type: str
    'MESH_FILE': 'examples/DIPOLE1D/Input_preprocessing/DIPOLE1D.msh',

    # ---------- Material conductivities ----------
    # Type: float
    'MATERIAL_CONDUCTIVITIES': [1.0, 1./100., 1., 1./.3],

    # ---------- Receivers position file ----------
    # Type: str
    'RECEIVERS_FILE': 'examples/DIPOLE1D/Input_preprocessing/RECEIVER_POSITIONS.txt',

    # ---------- Path for Output ----------
    # Type: str
    'OUT_DIR': 'examples/DIPOLE1D/Input_model/',

}
