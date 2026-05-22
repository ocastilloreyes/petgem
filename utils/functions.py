import os
import argparse
import numpy as np
import h5py
import meshio
import textwrap
from petsc4py import PETSc


def parsePreprocessingArgs():
    """CLI surface for the centralized utils/preprocess.py entry point.

    Conductivity is read from a CSV file (-sigma_file) instead of being
    hard-coded in a per-case Python script. CSV layout (one row per
    material, 0-based row index = material id, comments allowed):
        # sigma_x, sigma_y, sigma_z
        0.1, 0.1, 0.1
        1.0, 1.0, 1.0
    """
    parser = argparse.ArgumentParser(
        description="Preprocess mesh, conductivity, receivers and sources into a "
                    "single PETGEM input HDF5 file."
    )
    parser.add_argument("-mode",              choices=["forward", "inverse"],
                        default="forward",
                        help="Selects the params-file template (default: forward)")
    parser.add_argument("-nord",              type=int, required=True,
                        help="Polynomial order (1..6) - written into the params file only")
    parser.add_argument("-case_dir",          type=str, required=True,
                        help="Directory containing case data (also output directory)")
    parser.add_argument("-mesh_filename",     type=str, required=True,
                        help="Gmsh mesh filename inside case_dir")
    parser.add_argument("-receiver_filename", type=str, required=True,
                        help="Receivers text file (x y z per row) inside case_dir")
    parser.add_argument("-source_filename",   type=str, default=None,
                        help="Single-frequency forward sources text file inside "
                             "case_dir. Format: first non-comment line = frequency "
                             "(Hz); subsequent lines = 'x y z current length dip "
                             "azimuth'. Required when -mode forward; optional "
                             "(skipped) when -mode inverse - the inverse kernel "
                             "reads multi-frequency sources from /inv_sources/* "
                             "via -inv_source_filename.")
    parser.add_argument("-inv_source_filename", type=str, default=None,
                        help="Multi-frequency sources text file inside case_dir, "
                             "required when -mode inverse. Format: one row per "
                             "(freq, dipole) pair with 8 fields: "
                             "'freq x y z current length dip azimuth'. "
                             "Embedded into the bundle under /inv_sources/*.")
    parser.add_argument("-observed_filename", type=str, default=None,
                        help="Observed-data HDF5 inside case_dir, required when "
                             "-mode inverse. Must contain /Ex [N_freq, N_recv] "
                             "complex128 ({r,i} compound). Embedded into the "
                             "bundle under /observed/Ex.")
    parser.add_argument("-sigma_file",        type=str, required=True,
                        help="CSV of per-material conductivity (sigma_x, sigma_y, "
                             "sigma_z), relative to case_dir. Row index = 0-based "
                             "material id (gmsh:physical - 1).")
    parser.add_argument("-input_filename",    type=str, default="input.h5",
                        help="Output bundle filename inside case_dir. Same name is "
                             "written into the params file as the kernel's "
                             "-input_filename. Default: input.h5")
    parser.add_argument("-params_filename",   type=str, default="params.txt",
                        help="Filename for the PETGEM params file emitted alongside "
                             "the input bundle (inside case_dir). Default: params.txt")
    parser.add_argument("-output_vtk",        type=str, default=None,
                        help="Optional VTU filename for conductivity + materials_id "
                             "visualization (relative to case_dir)")
    parser.add_argument("-dm_view",           type=str, default=None,
                        help="If set, dumps DMPlex info to stdout")
    return parser.parse_args()


def readSigmaCSV(path):
    """Read per-material conductivity from a CSV file.

    Expected layout (comments allowed, blank lines allowed, 0-based row =
    material id)::

        # sigma_x, sigma_y, sigma_z [, fixed]
        0.1, 0.1, 0.1, 1     # fixed in inversion (e.g. air, ocean)
        1.0, 1.0, 1.0, 0     # invertable
        2.0, 2.0, 2.0        # 'fixed' column omitted -> defaults to 0

    The 4th column (`fixed`) is optional and only meaningful for inverse
    modeling: a non-zero entry marks the material as held fixed during
    inversion (gradient zeroed, smoother self-only). Forward modeling
    ignores this column.

    Returns (sigma_x, sigma_y, sigma_z, fixed_ids) where fixed_ids is
    a sorted list of 0-based material IDs flagged as fixed (empty list
    when the CSV has no 4th column).
    """
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] not in (3, 4):
        raise ValueError(
            f"{path}: expected 3 or 4 columns (sigma_x, sigma_y, sigma_z[, fixed]), "
            f"got {data.shape[1]}"
        )
    sx = data[:, 0].astype(float)
    sy = data[:, 1].astype(float)
    sz = data[:, 2].astype(float)
    if data.shape[1] == 4:
        fixed_flags = data[:, 3].astype(int)
        fixed_ids = sorted(int(i) for i, f in enumerate(fixed_flags) if f)
    else:
        fixed_ids = []
    return sx, sy, sz, fixed_ids


def createDM(numDimensions, cells, coords, dm_view=False):
    """Create a DMPlex with two cell-data fields:
       field 0 - "conductivity" : numDimensions dofs per cell (sigma_x, sigma_y, sigma_z)
       field 1 - "materials_id" : 1 dof per cell (integer material index, stored as float)
    """
    plex = PETSc.DMPlex().create()
    plex.setFromOptions()
    plex.createFromCellList(numDimensions, cells, coords)

    if dm_view:
        PETSc.Options()["dm_view"] = ""
        plex.viewFromOptions("-dm_view")

    dim = plex.getDimension()

    # numDof layout: [field0_d0, field0_d1, ..., field0_dN, field1_d0, ..., field1_dN]
    # cells live at depth = dim in 3D.
    plex.setNumFields(2)
    numComp = [dim, 1]
    numDof = [0] * (2 * (dim + 1))
    numDof[dim] = dim            # field 0: dim dofs at cells
    numDof[(dim + 1) + dim] = 1  # field 1: 1 dof at cells
    s = plex.createSection(numComp, numDof)
    s.setFieldName(0, "conductivity")
    s.setFieldName(1, "materials_id")
    s.setUp()
    plex.setSection(s)

    return plex


def readSourcesText(path):
    """Parse the forward-kernel source.txt:
         (comments and blank lines allowed)
         <freq>
         x y z current length dip azimuth
         ...
    Returns (frequency, ndarray of shape (N, 7))."""
    freq = None
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if freq is None:
                # First non-comment line: frequency (single float).
                vals = line.split()
                freq = float(vals[0])
                continue
            vals = line.split()
            if len(vals) != 7:
                raise ValueError(
                    f"{path}: expected 7 fields per source row "
                    f"(x y z current length dip azimuth), got {len(vals)}: {line!r}"
                )
            rows.append([float(v) for v in vals])
    if freq is None or not rows:
        raise ValueError(f"{path}: missing frequency or source rows")
    return freq, np.asarray(rows, dtype=float)


def readInverseSourcesText(path):
    """Parse the inverse-kernel multi-frequency sources file.

    Format (comments '#' and blank lines allowed):
        freq  x  y  z  current  length  dip  azimuth
        ...
    One row per (frequency, dipole) pair. Returns an (N, 8) ndarray with
    columns in the same order as the file. The C kernel previously read
    this format directly via setupInversionSources; we now parse it on
    the Python side and embed the data in the bundle under /inv_sources/*.
    """
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) != 8:
                raise ValueError(
                    f"{path}: expected 8 fields per source row "
                    f"(freq x y z current length dip azimuth), got {len(vals)}: {line!r}"
                )
            rows.append([float(v) for v in vals])
    if not rows:
        raise ValueError(f"{path}: no source rows found")
    return np.asarray(rows, dtype=float)


def readObservedDataH5(path):
    """Read an observed-data HDF5 file produced by generate_observed_data.py
    or convert_invex_to_hdf5.py.

    Returns (Ex, error_level) where Ex is the [N_freq, N_recv] complex128
    ndarray stored at /Ex and error_level is the amplitude-relative noise
    level used to synthesize it (None if the source file has no
    `error_level` attribute).  The compound HDF5 type {r: float64,
    i: float64} is materialized as complex128 by h5py.
    """
    with h5py.File(path, "r") as f:
        if "Ex" not in f:
            raise KeyError(f"{path}: missing /Ex dataset")
        data = np.asarray(f["Ex"])
        # `error_level` may live as an attribute on the file root (current
        # generate_observed_data.py convention) or on the /Ex dataset itself.
        error_level = None
        for holder in (f, f["Ex"]):
            if "error_level" in holder.attrs:
                error_level = float(holder.attrs["error_level"])
                break
    if data.ndim != 2:
        raise ValueError(f"{path}: /Ex must be 2-D, got shape {data.shape}")
    if not np.iscomplexobj(data):
        # File may store as compound-real; coerce to complex.
        data = data.astype(np.complex128)
    return data, error_level


def writeInversionPayload(filename, inv_sources_arr, observed_Ex,
                          error_level=None, fixed_materials=()):
    """Append the inverse-kernel payload to an existing PETGEM bundle.

    Called after writePetgemInputFile (which writes the mesh + sigma +
    receivers + forward sources). Uses h5py to append because /observed/Ex
    is an HDF5 compound complex128 - easier to express directly than via
    the PETSc viewer.

    Bundle additions:
        /inv_sources/{freq, position, current, length, dipAngle, azimuthAngle}
        /observed/Ex                          (HDF5 compound complex128)
        /observed @error_level                (HDF5 group attribute, float64; omitted if None)
        /inv_meta/fixed_materials             (int32[], 0-based material ids; omitted if empty)

    Parameters
    ----------
    filename : str
        Bundle path (writes append-mode).
    inv_sources_arr : ndarray (N_freq, 8)
        Columns = [freq, x, y, z, current, length, dipAngle, azimuthAngle].
    observed_Ex : ndarray (N_freq, N_recv), complex
        Per (freq, receiver) observed Ex.
    error_level : float or None
        Amplitude-relative noise level used to synthesize observed_Ex.
        Stored as a group attribute on /observed; consumed by the C
        kernel (overridable with -inv_error_level).  Skipped when None.
    fixed_materials : iterable of int
        0-based material ids to exclude from inversion. Stored as an
        int32 dataset under /inv_meta/fixed_materials; consumed by the
        C kernel (overridable with -inv_fixed_materials).  Skipped when
        empty.
    """
    inv = np.asarray(inv_sources_arr, dtype=float)
    if inv.ndim != 2 or inv.shape[1] != 8:
        raise ValueError(f"inv_sources_arr must be (N, 8), got {inv.shape}")
    ex  = np.asarray(observed_Ex, dtype=np.complex128)
    n_freq = inv.shape[0]
    if ex.shape[0] != n_freq:
        raise ValueError(
            f"observed_Ex.shape[0]={ex.shape[0]} disagrees with "
            f"inv_sources rows={n_freq}"
        )

    fixed_arr = np.asarray(sorted(set(int(i) for i in fixed_materials)),
                           dtype=np.int32)

    with h5py.File(filename, "a") as f:
        for grp in ("/inv_sources", "/observed", "/inv_meta"):
            if grp in f:
                del f[grp]

        g = f.create_group("/inv_sources")
        # Stored as plain float64 datasets, one per field.  The C reader
        # uses raw H5Dread (not VecLoad) because PETSc complex builds
        # reject "real"-marked datasets that lack the `complex` attribute.
        g.create_dataset("freq",         data=inv[:, 0].astype(np.float64))
        g.create_dataset("position",     data=inv[:, 1:4].reshape(-1).astype(np.float64))
        g.create_dataset("current",      data=inv[:, 4].astype(np.float64))
        g.create_dataset("length",       data=inv[:, 5].astype(np.float64))
        g.create_dataset("dipAngle",     data=inv[:, 6].astype(np.float64))
        g.create_dataset("azimuthAngle", data=inv[:, 7].astype(np.float64))

        obs = f.create_group("/observed")
        # Compound complex128 ({r,i} float64) - matches what
        # loadObservedData reads via the raw H5 API.
        obs.create_dataset("Ex", data=ex)
        if error_level is not None:
            obs.attrs["error_level"] = float(error_level)

        if fixed_arr.size > 0:
            meta = f.create_group("/inv_meta")
            meta.create_dataset("fixed_materials", data=fixed_arr)


def _writeArrayAsVec(viewer, arr, name):
    """View a 1-D float64 ndarray as a named PETSc Vec into the given viewer."""
    arr = np.ascontiguousarray(arr, dtype=float).reshape(-1)
    v = PETSc.Vec().createWithArray(arr, comm=PETSc.COMM_SELF)
    v.setName(name)
    v.view(viewer)
    v.destroy()


def _writeVTKFields(cells, coords, conductivity, materials_id, output_vtk):
    """Emit a VTU with two cell-data fields (conductivity, materials_id).

    PETSc's VTK viewer is unsuitable here because:
      * For a plain section-based DMPlex (no PetscFE/FV discretization)
        VecView_Plex_Local_VTK writes the whole Vec as a single multi-
        component array - materials_id never gets a separate VTK field.
      * Building per-field sub-DMs and viewing each into a shared VTK
        viewer trips PetscViewerVTKAddField_VTK's same-DM check
        ("Refusing to write a field from more than one grid"); there
        is no petsc4py-level escape hatch for the underlying `checkdm`.
    meshio is already used to read the Gmsh mesh, so we reuse it here
    to produce a clean VTU with named cell-data fields. """
    cells_arr = np.asarray(cells)
    coords_arr = np.asarray(coords)
    mesh_out = meshio.Mesh(
        coords_arr,
        [("tetra", cells_arr)],
        cell_data={
            "sigma_x":      [conductivity[:, 0]],
            "sigma_y":      [conductivity[:, 1]],
            "sigma_z":      [conductivity[:, 2]],
            "materials_id": [materials_id.astype(np.int32)],
        },
    )
    mesh_out.write(output_vtk)


def writePetgemInputFile(plex, conductivity, materials_id,
                         receivers_arr, freq, sources_arr, nord,
                         output_filename, cells=None, coords=None,
                         output_vtk=None):
    """Write the unified PETGEM input file.

    Contents of `output_filename` (HDF5):
      /petgem_mesh/...     mesh topology, labels, coordinates, sections, fields
                            (HDF5_PETSC format, written via DMPlex *View routines)
      /receivers           Vec, length 3*N_recv, layout [x0 y0 z0 x1 y1 z1 ...]
      /nord                Vec, length 1 - polynomial order used to size the case
      /sources/...         OPTIONAL - written only when both `freq` and
                            `sources_arr` are non-None (typical for forward
                            modeling; inverse mode skips the group and relies
                            on /inv_sources/* added later by writeInversionPayload).

    Parameters
    ----------
    plex          : DMPlex (2-field section, from createDM)
    conductivity  : ndarray (num_cells, dim) - sigma_x, sigma_y, sigma_z per cell
    materials_id  : ndarray (num_cells,)     - integer material id per cell
    receivers_arr : ndarray (num_recv, 3)    - receiver positions
    freq          : float or None            - single source frequency (Hz);
                                                None skips the /sources/* group
    sources_arr   : ndarray (num_src, 7) or None - x y z current length dip azimuth;
                                                   None skips the /sources/* group
    output_filename : path to the unified .h5 file
    cells, coords : ndarrays - required when output_vtk is set (passed to meshio)
    output_vtk    : optional VTU filename for conductivity+materials_id view
    """
    num_cells = conductivity.shape[0]
    dim_cond  = conductivity.shape[1]

    # Combined global vec [σx, σy, σz, mat_id] per cell - for the 2-field section.
    v_model = plex.createGlobalVec()
    arr = v_model.getArray()
    combined = np.zeros((num_cells, dim_cond + 1))
    combined[:, :dim_cond] = conductivity
    combined[:, dim_cond]  = materials_id.reshape(-1)
    arr[:] = combined.reshape(-1)
    v_model.setName("model_data")

    if output_vtk is not None:
        if cells is None or coords is None:
            raise ValueError("writePetgemInputFile: `cells` and `coords` are required "
                             "when `output_vtk` is set (needed by meshio).")
        _writeVTKFields(cells, coords, conductivity, materials_id, output_vtk)

    # Unified HDF5
    viewer = PETSc.ViewerHDF5().create(output_filename, "w")
    plex.setName("petgem_mesh")

    # DMPlex blocks (HDF5_PETSC format).
    viewer.pushFormat(PETSc.Viewer.Format.HDF5_PETSC)
    plex.topologyView(viewer)
    plex.labelsView(viewer)
    plex.coordinatesView(viewer)
    plex.sectionView(viewer, plex)
    plex.globalVectorView(viewer, plex, v_model)
    viewer.popFormat()

    # Receivers (top-level dataset).
    _writeArrayAsVec(viewer, receivers_arr.reshape(-1), "receivers")

    # Polynomial order (top-level scalar, 1-element Vec) - postprocess
    # reads it from here to annotate figures and pick the right responses.
    _writeArrayAsVec(viewer, np.array([nord], dtype=float), "nord")

    # Sources (under /sources group) - only written when forward sources
    # are supplied. The inverse kernel skips this group (it consumes
    # multi-frequency sources from /inv_sources/* added later).
    if freq is not None and sources_arr is not None:
        viewer.pushGroup("/sources")
        _writeArrayAsVec(viewer, np.array([freq], dtype=float),    "frequency")
        _writeArrayAsVec(viewer, sources_arr[:, 0:3].reshape(-1),  "position")
        _writeArrayAsVec(viewer, sources_arr[:, 3],                "current")
        _writeArrayAsVec(viewer, sources_arr[:, 4],                "length")
        _writeArrayAsVec(viewer, sources_arr[:, 5],                "dipAngle")
        _writeArrayAsVec(viewer, sources_arr[:, 6],                "azimuthAngle")
        viewer.popGroup()

    viewer.destroy()
    v_model.destroy()


def writeForwardModelingParamsFile(nord, output_dir, output_filename,
                                   input_filename, params_filename):
    """Emit the fm.csem params file.

    `input_filename` is the bundle written by writePetgemInputFile; the
    same name is fed back to the kernel via -input_filename. The basis
    order is no longer written here - the C kernel reads it from the
    bundle's /nord dataset (loadCsemInputs). `nord` is still accepted as
    an argument because the caller uses it to compose other defaults,
    but it is not emitted into the params file."""
    del nord  # bundle is authoritative; CLI -nord remains as override
    content = textwrap.dedent(f"""\
        -input_filename {output_dir}/{input_filename}
        -dm_mat_type is
        -ksp_type fgmres
        -pc_type bddc
        -pc_bddc_use_deluxe_scaling 1
        -pc_bddc_coarse_pc_type lu
        -output_dir {output_dir}/
        -output_filename {output_filename}
    """)
    filename = f"{output_dir}/{params_filename}"
    with open(filename, "w") as f:
        f.write(content)


def writeInverseModelingParamsFile(nord, output_dir, output_filename,
                                   input_filename, params_filename):
    """Emit the im.csem params file. The inverse kernel now reads EVERYTHING
    case-specific from `input_filename` - multi-frequency sources
    (/inv_sources/*), observed Ex (/observed/Ex), the noise level
    (/observed @error_level) and the fixed-material list
    (/inv_meta/fixed_materials).  The emitted params.txt only carries
    runtime/tuning knobs; -inv_error_level and -inv_fixed_materials are
    accepted as CLI overrides but no longer present in the default
    template.  -nord is sourced from the bundle's /nord."""
    del nord  # bundle is authoritative; CLI -nord remains as override
    content = textwrap.dedent(f"""\
        -input_filename {output_dir}/{input_filename}
        -ksp_type preonly
        -pc_type                    lu
        -pc_factor_mat_solver_type  mumps
        -mat_mumps_icntl_14         80
        -mat_mumps_icntl_28         1
        -inv_max_iter               150
        -inv_lbfgs_memory           2
        -inv_lambda                 0.1
        -inv_diag_weight            0.0
        -inv_gtol                   1.0e-5
        -inv_rms_tol                1.05
        -inv_snapshot_interval      1
        -output_dir {output_dir}/
        -output_filename {output_filename}
    """)
    filename = f"{output_dir}/{params_filename}"
    with open(filename, "w") as f:
        f.write(content)


def runPreprocessing(*, mode, nord, case_dir,
                     mesh_filename, receiver_filename, source_filename=None,
                     sigma_x, sigma_y, sigma_z,
                     fixed_materials=(),
                     input_filename="input.h5",
                     params_filename="params.txt",
                     inv_source_filename=None,
                     observed_filename=None,
                     output_vtk=None, dm_view=False):
    """Shared preprocessing pipeline for the PETGEM forward / inverse kernels.

    Pure library function - no CLI parsing.  Builds the DMPlex, assigns the
    per-cell conductivity from `sigma_{x,y,z}` indexed by the gmsh:physical
    tag, parses the receivers and sources text files, and bundles everything
    into a single HDF5 input file (`<case_dir>/<input_filename>`).  Also
    emits the matching params file.

    Parameters
    ----------
    mode : {'forward', 'inverse'}
        Selects which params-file template to emit.
    nord : int
        Polynomial order, written into the params file.
    case_dir : str
        Directory containing the input data and where the bundle is written.
    mesh_filename, receiver_filename, source_filename : str
        Filenames relative to `case_dir`.
    sigma_x, sigma_y, sigma_z : array_like
        Per-material conductivity. Index = 0-based material id
        (gmsh:physical - 1). Lengths must match.
    input_filename : str
        Bundle filename inside `case_dir`. Default: input.h5
    params_filename : str
        Filename of the PETGEM params file emitted inside `case_dir`.
        Default: params.txt
    output_vtk : str or None
        Optional VTU filename for the conductivity + materials_id view.
    dm_view : bool
        If True, dumps DMPlex info to stdout via PETSc options.
    """
    if mode not in ("forward", "inverse"):
        raise ValueError(f"runPreprocessing: mode must be 'forward' or 'inverse' "
                         f"(got {mode!r})")
    if mode == "forward":
        if source_filename is None:
            raise ValueError("runPreprocessing: -source_filename is required "
                             "when mode='forward'")
        if inv_source_filename is not None or observed_filename is not None:
            raise ValueError(
                "runPreprocessing: -inv_source_filename and -observed_filename "
                "are valid only with mode='inverse'.")
    if mode == "inverse":
        if source_filename is not None:
            raise ValueError(
                "runPreprocessing: -source_filename is forward-mode only. "
                "The inverse kernel reads multi-freq sources from "
                "/inv_sources/* in the bundle; pass -inv_source_filename "
                "instead.")
        if inv_source_filename is None:
            raise ValueError("runPreprocessing: -inv_source_filename is required "
                             "when mode='inverse'")
        if observed_filename is None:
            raise ValueError("runPreprocessing: -observed_filename is required "
                             "when mode='inverse'")

    sigma_x = np.asarray(sigma_x, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)
    sigma_z = np.asarray(sigma_z, dtype=float)
    if not (len(sigma_x) == len(sigma_y) == len(sigma_z)):
        raise ValueError("sigma_x, sigma_y, sigma_z must have equal length")

    NUM_DIMENSIONS = 3

    input_mesh_filename      = os.path.join(case_dir, mesh_filename)
    input_receivers_filename = os.path.join(case_dir, receiver_filename)
    input_sources_filename   = (os.path.join(case_dir, source_filename)
                                if source_filename is not None else None)
    output_filename          = os.path.join(case_dir, input_filename)
    output_petgem_filename   = f"responses_p{nord}"
    output_vtk_filename      = (os.path.join(case_dir, output_vtk)
                                if output_vtk is not None else None)
    input_inv_sources_filename = (os.path.join(case_dir, inv_source_filename)
                                  if inv_source_filename is not None else None)
    input_observed_filename    = (os.path.join(case_dir, observed_filename)
                                  if observed_filename is not None else None)

    print("====================================================")
    print(f" PETGEM INPUT PREPROCESSING ({mode})")
    print("====================================================")
    print(f"  Polynomial order (nord): {nord}")
    print(f"  Case directory         : {case_dir}")
    print(f"  Mesh file              : {input_mesh_filename}")
    print(f"  Receivers file         : {input_receivers_filename}")
    print(f"  Sources file           : "
          f"{input_sources_filename if input_sources_filename else '(skipped - inverse mode)'}")
    if mode == "inverse":
        print(f"  Inv sources file       : {input_inv_sources_filename}")
        print(f"  Observed data file     : {input_observed_filename}")
    print(f"  Output bundle          : {output_filename}")
    print(f"\n  Number of materials    : {len(sigma_x)}")

    # 1. Import mesh
    print("\nReading mesh")
    mesh = meshio.read(input_mesh_filename)
    num_cells = sum(block.data.shape[0] for block in mesh.cells)
    coords = mesh.points
    print(f"  Total elements          : {num_cells}")
    print(f"  Total vertices          : {coords.shape[0]}")
    cells = mesh.cells[-1].data  # last block: tetrahedra

    # 2. Per-cell conductivity from gmsh:physical
    print("\nAssigning conductivity per cell")
    conductivity = np.zeros((num_cells, NUM_DIMENSIONS), dtype=float)
    materials_id = np.copy(mesh.cell_data_dict["gmsh:physical"]["tetra"])
    materials_id -= 1  # 1-based -> 0-based
    conductivity[:, 0] = sigma_x[materials_id]
    conductivity[:, 1] = sigma_y[materials_id]
    conductivity[:, 2] = sigma_z[materials_id]
    print("  Conductivity assignment completed")

    # 3. Receivers (text → ndarray)
    print("\nReading receivers")
    receivers_arr = np.loadtxt(input_receivers_filename, comments='#')
    if receivers_arr.ndim == 1:
        receivers_arr = receivers_arr.reshape(1, -1)
    if receivers_arr.shape[1] != NUM_DIMENSIONS:
        raise ValueError(
            f"{input_receivers_filename}: expected {NUM_DIMENSIONS} columns "
            f"(x y z), got {receivers_arr.shape[1]}"
        )
    print(f"  Number of receivers     : {receivers_arr.shape[0]}")

    # 4. Sources (text → freq + ndarray of 7-col rows).
    # Skipped when no forward source file is provided (typical for
    # inverse-mode preprocessing - the kernel pulls multi-frequency
    # records from /inv_sources/* instead and ignores /sources/*).
    if input_sources_filename is not None:
        print("\nReading sources")
        freq, sources_arr = readSourcesText(input_sources_filename)
        print(f"  Source frequency (Hz)   : {freq}")
        print(f"  Number of sources       : {sources_arr.shape[0]}")
    else:
        freq, sources_arr = None, None

    # 5. Build the DMPlex
    print("\nCreating PETSc DM (DMPlex)")
    plex = createDM(NUM_DIMENSIONS, cells, coords, dm_view=dm_view)

    # 6. Write the unified bundle
    print("\nWriting unified PETGEM input HDF5")
    writePetgemInputFile(plex, conductivity, materials_id,
                         receivers_arr, freq, sources_arr, nord,
                         output_filename,
                         cells=cells, coords=coords,
                         output_vtk=output_vtk_filename)
    print(f"  Output file: {output_filename}")
    if output_vtk_filename:
        print(f"  VTU view  : {output_vtk_filename}")

    # 6b. Inverse-mode payload: append /inv_sources/* and /observed/Ex
    if mode == "inverse":
        print("\nReading inversion sources")
        inv_sources_arr = readInverseSourcesText(input_inv_sources_filename)
        print(f"  Number of (freq, dipole) records: {inv_sources_arr.shape[0]}")

        print("\nReading observed data")
        observed_Ex, observed_error_level = readObservedDataH5(input_observed_filename)
        print(f"  Observed Ex shape       : {observed_Ex.shape}  (N_freq, N_recv)")
        print(f"  Error level (from attr) : "
              f"{observed_error_level if observed_error_level is not None else '(absent, kernel default)'}")

        n_freq_src = inv_sources_arr.shape[0]
        n_freq_obs = observed_Ex.shape[0]
        if n_freq_src != n_freq_obs:
            raise ValueError(
                f"Mismatch between inversion sources rows ({n_freq_src}) and "
                f"observed Ex frequencies ({n_freq_obs})"
            )
        if observed_Ex.shape[1] != receivers_arr.shape[0]:
            raise ValueError(
                f"Observed Ex N_recv={observed_Ex.shape[1]} disagrees with "
                f"receivers count {receivers_arr.shape[0]}"
            )

        print("\nEmbedding inverse payload into bundle")
        if fixed_materials:
            print(f"  Fixed material IDs       : {list(fixed_materials)} (from sigmas.csv)")
        writeInversionPayload(output_filename, inv_sources_arr, observed_Ex,
                              error_level=observed_error_level,
                              fixed_materials=fixed_materials)
        print(f"  Wrote /inv_sources/*, /observed/Ex, /inv_meta/fixed_materials "
              f"into {output_filename}")

    # 7. Params file
    print("\nGenerating PETGEM parameter file")
    if mode == "forward":
        writeForwardModelingParamsFile(nord, case_dir, output_petgem_filename,
                                       input_filename, params_filename)
    else:
        writeInverseModelingParamsFile(nord, case_dir, output_petgem_filename,
                                       input_filename, params_filename)
    print(f"  Params file: {os.path.join(case_dir, params_filename)}")

    print("\n====================================================")
    print(" Preprocessing completed successfully")
    print("====================================================\n")


def readBundle(filename):
    """Read the case-independent payload of a PETGEM input bundle HDF5.

    Returns a dict with keys::

      receivers : (N_recv, 3) ndarray of receiver positions (real-valued)
      nord      : int polynomial order
      frequency : float source frequency (Hz)
      sources   : (N_src, 7) ndarray, columns =
                  [x, y, z, current, length, dipAngle, azimuthAngle]

    The DMPlex / model_data fields inside the bundle are not returned -
    those are consumed by the C kernel via loadCsemInputs. This loader is
    for the Python postprocessing side and exposes only the parameters a
    case-specific validation script actually needs.
    """
    receivers = np.real(np.array(readVectorH5(filename, 'receivers'))).reshape(-1, 3)

    nord_arr = readVectorH5(filename, 'nord')
    nord = int(round(float(np.real(np.array(nord_arr)).flatten()[0])))

    freq_arr = readVectorH5(filename, 'frequency', group='/sources')
    frequency = float(np.real(np.array(freq_arr)).flatten()[0])

    pos     = np.real(np.array(readVectorH5(filename, 'position',     group='/sources'))).reshape(-1, 3)
    current = np.real(np.array(readVectorH5(filename, 'current',      group='/sources'))).reshape(-1)
    length  = np.real(np.array(readVectorH5(filename, 'length',       group='/sources'))).reshape(-1)
    dip     = np.real(np.array(readVectorH5(filename, 'dipAngle',     group='/sources'))).reshape(-1)
    az      = np.real(np.array(readVectorH5(filename, 'azimuthAngle', group='/sources'))).reshape(-1)
    sources = np.column_stack([pos, current, length, dip, az])

    return {
        'receivers': receivers,
        'nord':      nord,
        'frequency': frequency,
        'sources':   sources,
    }


def readResponses(filename):
    """Read a PETGEM responses HDF5 file (written by fm.csem postprocessing).

    Returns a dict with keys::

      Ex, Ey, Ez, Hx, Hy, Hz : ndarrays (complex in PETSc complex builds)
      source      : dict of /source attributes
                    (frequency, x_pos, y_pos, z_pos, current, length,
                     dip_angle, azimuth_angle)
      provenance  : dict of top-level attributes
                    (petgem_version, input_filename, date, nord, mpi_tasks)

    Attributes are read via h5py (petsc4py's attribute API is awkward for
    this read pattern).  The Vec components live under /fields/ in the
    new output layout.
    """
    import h5py

    out = {}
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        out[name] = np.array(readVectorH5(filename, name, group='/fields'))

    with h5py.File(filename, 'r') as f:
        out['source']     = {k: _decodeH5Attr(v) for k, v in f['/source'].attrs.items()} if '/source' in f else {}
        out['provenance'] = {k: _decodeH5Attr(v) for k, v in f.attrs.items()}
    return out


def _decodeH5Attr(v):
    """Normalize an h5py attribute value: bytes → str, 1-element → scalar."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8', errors='replace')
    if isinstance(v, np.ndarray):
        if v.shape == (1,):
            return _decodeH5Attr(v.item())
        return v
    return v


def readVectorH5(filename, dataset_name, group=None):
    """Read a PETSc Vec from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    dataset_name : str
        Vec name (the HDF5 dataset name within the active group).
    group : str or None
        Optional HDF5 group (e.g. "/fields"). When set, the viewer pushes
        this group before loading so that `dataset_name` resolves to
        `{group}/{dataset_name}`.
    """
    tmp = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    tmp.setName(dataset_name)
    # Use the dedicated ViewerHDF5 subclass - only that one exposes
    # pushGroup / popGroup in petsc4py. The generic Viewer.createHDF5
    # returns a Viewer whose group methods are missing.
    viewer = PETSc.ViewerHDF5().create(str(filename), mode='r', comm=PETSc.COMM_SELF)
    if group is not None:
        viewer.pushGroup(group)
    tmp.load(viewer)
    if group is not None:
        viewer.popGroup()
    viewer.destroy()
    return tmp.getArray()
