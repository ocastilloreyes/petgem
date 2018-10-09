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
    'NEDELEC_ORDER': 1,

    # ---------- Mesh file ----------
    'MESH_FILE': 'PATH_TO_FILE',

    # ---------- Material conductivities ----------
    'MATERIAL_CONDUCTIVITIES': [1.0, 1./100., 1., 1./.3],

    # ---------- Receivers position file ----------
    'RECEIVERS_FILE': 'PATH_TO_FILE',

    # ---------- Path for Output ----------
    'OUT_DIR': 'PATH_TO_FILE',

}
