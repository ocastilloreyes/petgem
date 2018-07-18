''' Parameters file template for **PETGEM** preprocessing.

For preprocessing task, a mesh file, conductivity model and
receivers file are mandatory.

In order to avoid a specific parser, this file is imported by
**PETGEM** as a Python dictionary. As consequence, the dictionary
name and his key names MUST NOT BE changed.

All file paths should consider as absolute.
'''
preprocessing = {
    # ---------- Mesh file ----------
    'MESH_FILE': 'examples/DIPOLE1D/Input_preprocessing/DIPOLE1D.msh',

    # ---------- Material conductivities ----------
    'MATERIAL_CONDUCTIVITIES': [1.0, 1./100., 1., 1./.3],

    # ---------- Receivers position file ----------
    'RECEIVERS_FILE': 'examples/DIPOLE1D/Input_preprocessing/RECEIVER_POSITIONS.txt',

    # ---------- Path for Output ----------
    'OUT_DIR': 'examples/DIPOLE1D/Input_model/',

}
