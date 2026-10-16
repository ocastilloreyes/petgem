import os
import argparse
import numpy as np
import h5py
import meshio
import textwrap
from petsc4py import PETSc


# Kernel names, identical to the binaries and to the petgem subcommands.
KERNELS = ("fm.csem", "im.csem", "fm.mt", "fm.tdcsem")

# Data components fitted by im.csem (Zij = Ei / Hj), as in -im_data_components.
DATA_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Zxx", "Zxy", "Zyx", "Zyy")


def checkMode(mode):
    """Return ``mode`` if it is a kernel name; raise ValueError otherwise."""
    if mode not in KERNELS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(KERNELS)}")
    return mode


def parsePreprocessingArgs():
    """Parse the command line of ``utils/preprocess.py``.

    Mode-specific inputs are validated later by :func:`runPreprocessing`.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess mesh, conductivity, receivers and sources into a "
                    "single PETGEM input HDF5 file."
    )
    parser.add_argument("-mode",              choices=KERNELS,
                        default="fm.csem",
                        help="Kernel the bundle is for: 'fm.csem' (CSEM forward), "
                             "'im.csem' (CSEM inverse), 'fm.mt' (MT forward) or "
                             "'fm.tdcsem' (time-domain CSEM forward). "
                             "Default: fm.csem")
    parser.add_argument("-order",              type=int, required=True, dest="order",
                        help="Polynomial order (1..6), stored in the bundle as /order")
    parser.add_argument("-case_dir",          type=str, required=True,
                        help="Directory containing case data (also output directory)")
    parser.add_argument("-mesh_filename",     type=str, required=True,
                        help="Mesh filename inside case_dir. Gmsh '.msh' or VTK "
                             "'.vtk'/'.vtu' (tetrahedral); format auto-detected. "
                             "Gmsh uses 'gmsh:physical' tags; VTK uses a material "
                             "cell-data array (e.g. 'cell_scalars').")
    parser.add_argument("-receiver_filename", type=str, required=True,
                        help="Receivers text file (x y z per row) inside case_dir")
    parser.add_argument("-source_filename",   type=str, default=None,
                        help="Forward sources text file inside case_dir, one "
                             "frequency for all rows: 8 fields per row "
                             "'freq x y z current length dip azimuth', or a "
                             "frequency line followed by 7-field rows. With -mode "
                             "fm.tdcsem: 7 fields per row 'x y z current length dip "
                             "azimuth'. Required with -mode fm.csem and fm.tdcsem, "
                             "rejected otherwise.")
    parser.add_argument("-im_source_filename", type=str, default=None,
                        help="Multi-frequency sources text file inside case_dir, "
                             "required when -mode im.csem. Format: one row per "
                             "(freq, dipole) pair with 8 fields: "
                             "'freq x y z current length dip azimuth'. "
                             "Embedded into the bundle under /sources/*.")
    parser.add_argument("-observed_filename", type=str, nargs="+", default=None,
                        help="Observed-data file(s) inside case_dir, required when "
                             "-mode im.csem. Each entry is 'file' or "
                             "'component:file' (component one of "
                             f"{', '.join(DATA_COMPONENTS)}). An HDF5 (.h5/.hdf5) "
                             "holds /<component> [N_freq, N_recv] complex128 "
                             "({r,i} compound) datasets, all of them read unless "
                             "a component is given; a raw MATLAB-style "
                             "invEx.dat text file is Ex unless a component is "
                             "given. NaN marks a missing datum. Embedded into "
                             "the bundle under /observed/<component>.")
    parser.add_argument("-mt_frequency_filename", type=str, default=None,
                        help="MT frequency text file inside case_dir, required "
                             "when -mode fm.mt. Format: one frequency (Hz); '#' "
                             "comments allowed. Embedded into the bundle under "
                             "/mt/freq.")
    parser.add_argument("-tdcsem_waveform_filename", type=str, default=None,
                        help="Source waveform text file inside case_dir, required "
                             "when -mode fm.tdcsem. Format: two columns 't s(t)', "
                             "time (s) and current normalized by the source "
                             "current; s must start at 0. Embedded into the bundle "
                             "under /tdcsem/time and /tdcsem/waveform.")
    parser.add_argument("-tdcsem_t_end", type=float, default=None,
                        help="Final time (s), required when -mode fm.tdcsem. "
                             "Written into the params file as -tdcsem_t_end.")
    parser.add_argument("-error_level", type=float, default=None,
                        help="Amplitude-relative noise level written to the "
                             "bundle's /observed @error_level (inverse mode). "
                             "Overrides an HDF5 file's attribute; for a raw "
                             "invEx.dat it is the only way to set it. Omit to "
                             "leave the kernel default (overridable later with "
                             "-im_error_level).")
    parser.add_argument("-sigma_file",        type=str, required=True,
                        help="Whitespace text table of per-material conductivity "
                             "(sigma_x sigma_y sigma_z [fixed]), relative to "
                             "case_dir. Row index = 0-based material id "
                             "(gmsh:physical - 1).")
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


def _readTokens(path, commas=False):
    """Token lists of the non-blank lines of a text table; '#' starts a comment.

    With ``commas=True`` commas also separate tokens.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if commas:
                line = line.replace(',', ' ')
            if line:
                rows.append(line.split())
    return rows


def readSigmaTable(path):
    """Read the per-material conductivity table (``sigmas.txt``).

    Whitespace-delimited, one row per material (0-based row index = material
    id = gmsh:physical - 1), ``#`` comments and blank lines allowed::

        # sigma_x sigma_y sigma_z [fixed]
        0.1 0.1 0.1 1     # fixed in inversion (e.g. air, ocean)
        1.0 1.0 1.0 0     # invertible
        2.0 2.0 2.0       # 'fixed' column omitted -> defaults to 0

    The 4th column (`fixed`) is optional and only meaningful for inverse
    modeling: a non-zero entry marks the material as held fixed during
    inversion (gradient zeroed, smoother self-only). Forward modeling
    ignores this column.

    Returns (sigma_x, sigma_y, sigma_z, fixed_ids) where fixed_ids is
    a sorted list of 0-based material IDs flagged as fixed (empty list
    when the table has no 4th column).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Conductivity table not found: '{path}'. "
            f"Check -case_dir and -sigma_file.")
    rows = [[float(tok) for tok in toks] for toks in _readTokens(path)]
    if not rows:
        raise ValueError(f"{path}: no material rows found")
    ncol = len(rows[0])
    if ncol not in (3, 4) or any(len(r) != ncol for r in rows):
        raise ValueError(
            f"{path}: expected 3 or 4 columns (sigma_x sigma_y sigma_z [fixed]) "
            f"on every row")
    data = np.asarray(rows, dtype=float)
    sx, sy, sz = data[:, 0], data[:, 1], data[:, 2]
    if ncol == 4:
        fixed_ids = sorted(int(i) for i, flag in enumerate(data[:, 3].astype(int)) if flag)
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
    # setLocalSection, not setSection: petsc4py carried `setSection` as a
    # deprecated alias for DMSetLocalSection and PETSc main has now removed it,
    # so the old name raises AttributeError and no bundle can be built.
    plex.setLocalSection(s)

    return plex


def readSourcesText(path):
    """Parse a transmitter file into a unified ``(N, 8)`` array with columns
    ``freq x y z current length dip azimuth`` (one row per transmitter).

    Two on-disk layouts are accepted (auto-detected), so a single canonical
    representation feeds both forward and inverse preprocessing:

    1. **8-field** (canonical) - frequency on every row::

           freq x y z current length dip azimuth
           ...

    2. **Legacy forward** - a lone frequency line followed by 7-field rows::

           <freq>
           x y z current length dip azimuth
           ...

       parsed into the same ``(N, 8)`` shape with that frequency on each row.

    Forward modeling uses a single (repeated) frequency with one or more
    transmitters; inverse modeling carries one row per (frequency, dipole).
    """
    data_lines = _readTokens(path)
    if not data_lines:
        raise ValueError(f"{path}: no source rows found")

    widths = {len(toks) for toks in data_lines}
    if widths == {8}:
        rows = [[float(tok) for tok in toks] for toks in data_lines]
    elif data_lines[0] and len(data_lines[0]) == 1:
        # Legacy: first line is the scalar frequency, rest are 7-field rows.
        freq = float(data_lines[0][0])
        rows = []
        for toks in data_lines[1:]:
            if len(toks) != 7:
                raise ValueError(
                    f"{path}: legacy format expects 7 fields per source row "
                    f"(x y z current length dip azimuth), got {len(toks)}: {toks}")
            rows.append([freq] + [float(tok) for tok in toks])
        if not rows:
            raise ValueError(f"{path}: frequency line but no source rows")
    else:
        raise ValueError(
            f"{path}: unrecognized source format - use 8 fields per row "
            f"(freq x y z current length dip azimuth) or a legacy "
            f"<freq> header followed by 7-field rows; got column widths {widths}")
    return np.asarray(rows, dtype=float)


def readInvExDat(path):
    """Parse a MATLAB-style ``invEx.dat`` observed-data text file.

    Format (one row per frequency)::

        freq_label  Re(d_1) Im(d_1)  Re(d_2) Im(d_2)  ...  Re(d_N) Im(d_N)

    The leading frequency label is dropped (the bundle's frequencies come
    from the inversion sources file); the remaining Re/Im pairs build the
    ``[N_freq, N_recv]`` complex array.  The file carries one data component
    (``Ex`` unless :func:`readObservedData` is told otherwise) and no error
    level, so the noise level is supplied separately (``-error_level`` /
    ``-im_error_level``) or left to the kernel default.  NaN marks a missing
    datum.
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    values = data[:, 1:]                       # drop the freq label column
    if values.shape[1] % 2 != 0:
        raise ValueError(
            f"{path}: expected an even number of data columns "
            f"(Re/Im pairs), got {values.shape[1]}")
    ex = values[:, 0::2] + 1j * values[:, 1::2]
    return ex.astype(np.complex128), None


def readObservedDataH5(path, component=None):
    """Read an observed-data HDF5 file (e.g. written by utils/make_observed.py).

    Returns (data, error_level) where data maps each component name of
    :data:`DATA_COMPONENTS` found at the file root (only `component` when
    given) to its [N_freq, N_recv] complex128 ndarray, and error_level is the
    amplitude-relative noise level used to synthesize it (None if the source
    file has no `error_level` attribute).  The compound HDF5 type {r: float64,
    i: float64} is materialized as complex128 by h5py.
    """
    names = (component,) if component else DATA_COMPONENTS
    data = {}
    with h5py.File(path, "r") as f:
        for name in names:
            if name in f:
                data[name] = np.asarray(f[name])
        if not data:
            raise KeyError(f"{path}: no /{' or /'.join(names)} dataset")
        # `error_level` may sit on the file root or on a component dataset.
        error_level = None
        for holder in (f, *(f[name] for name in data)):
            if "error_level" in holder.attrs:
                error_level = float(holder.attrs["error_level"])
                break
    for name, values in data.items():
        if values.ndim != 2:
            raise ValueError(f"{path}: /{name} must be 2-D, got shape {values.shape}")
        if not np.iscomplexobj(values):
            # File may store as compound-real; coerce to complex.
            data[name] = values.astype(np.complex128)
    return data, error_level


def parseObservedSpec(spec):
    """Split an observed-data entry 'component:file' into (component, file).

    The component is matched case-insensitively against
    :data:`DATA_COMPONENTS`; an entry without a known prefix gives
    (None, spec).
    """
    head, sep, tail = spec.partition(":")
    if sep:
        for name in DATA_COMPONENTS:
            if head.lower() == name.lower():
                return name, tail
    return None, spec


def readObservedData(path, component=None):
    """Observed data of one file, HDF5 or ``invEx.dat`` text.

    Returns (data, error_level, kind): data maps component names to
    [N_freq, N_recv] complex128 arrays, kind describes the file format.
    """
    if path.lower().endswith((".h5", ".hdf5")):
        data, error_level = readObservedDataH5(path, component)
        return data, error_level, "HDF5"
    values, error_level = readInvExDat(path)
    return {component or "Ex": values}, error_level, "invEx.dat (text)"


def writeInversionPayload(filename, observed,
                          error_level=None, fixed_materials=()):
    """Append the inverse-only payload to an existing PETGEM bundle.

    Called after writePetgemInputFile, which already wrote the unified
    /sources group (per-entry frequency) shared by both modes. This adds
    only the data that has no forward counterpart. Uses h5py to append
    because /observed/<component> is an HDF5 compound complex128 - easier to
    express directly than via the PETSc viewer.

    Bundle additions:
        /observed/<component>                 (HDF5 compound complex128, one per component)
        /observed @error_level                (HDF5 group attribute, float64; omitted if None)
        /im_meta/fixed_materials             (int32[], 0-based material ids; omitted if empty)

    Parameters
    ----------
    filename : str
        Bundle path (writes append-mode).
    observed : dict of str to ndarray (N_freq, N_recv), complex
        Per (freq, receiver) observed data of each component of
        :data:`DATA_COMPONENTS`.
    error_level : float or None
        Amplitude-relative noise level used to synthesize the data.
        Stored as a group attribute on /observed; consumed by the C
        kernel (overridable with -im_error_level).  Skipped when None.
    fixed_materials : iterable of int
        0-based material ids to exclude from inversion. Stored as an
        int32 dataset under /im_meta/fixed_materials; consumed by the
        C kernel (overridable with -im_fixed_materials).  Skipped when
        empty.
    """
    fixed_arr = np.asarray(sorted(set(int(i) for i in fixed_materials)),
                           dtype=np.int32)

    with h5py.File(filename, "a") as f:
        for grp in ("/observed", "/im_meta"):
            if grp in f:
                del f[grp]

        obs = f.create_group("/observed")
        # Compound complex128 ({r,i} float64) - matches what
        # loadObservedData reads via the raw H5 API.
        for name, values in observed.items():
            obs.create_dataset(name, data=np.asarray(values, dtype=np.complex128))
        if error_level is not None:
            obs.attrs["error_level"] = float(error_level)

        if fixed_arr.size > 0:
            meta = f.create_group("/im_meta")
            meta.create_dataset("fixed_materials", data=fixed_arr)


def readMtFrequencyText(path):
    """Read the MT frequency (Hz) from a text file: one value, '#' comments allowed."""
    values = [float(tok) for toks in _readTokens(path, commas=True) for tok in toks]
    if len(values) != 1 or values[0] <= 0.0:
        raise ValueError(f"{path}: expected one positive frequency (Hz), got {values}")
    return values[0]


def readTdSourcesText(path):
    """Parse a time-domain transmitter file into an ``(N, 8)`` array.

    One row per transmitter with 7 fields ``x y z current length dip azimuth``;
    the frequency column of the result is 0.
    """
    data_lines = _readTokens(path)
    if not data_lines:
        raise ValueError(f"{path}: no source rows found")
    rows = []
    for toks in data_lines:
        if len(toks) != 7:
            raise ValueError(
                f"{path}: expected 7 fields per source row "
                f"(x y z current length dip azimuth), got {len(toks)}: {toks}")
        rows.append([0.0] + [float(tok) for tok in toks])
    return np.asarray(rows, dtype=float)


def readTdWaveformText(path):
    """Read a source waveform: two columns ``t s(t)``, '#' comments allowed.

    Returns ``(time, waveform)``. Times must increase strictly and s(t0) = 0.
    """
    rows = [[float(tok) for tok in toks] for toks in _readTokens(path, commas=True)]
    if len(rows) < 2 or any(len(row) != 2 for row in rows):
        raise ValueError(f"{path}: expected at least 2 rows of 't s(t)'")
    data = np.asarray(rows, dtype=float)
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"{path}: times must increase strictly")
    if data[0, 1] != 0.0:
        raise ValueError(f"{path}: the waveform must start at 0 (got {data[0, 1]})")
    return data[:, 0], data[:, 1]


def writeTdPayload(filename, time, waveform):
    """Append the waveform (/tdcsem/time, /tdcsem/waveform) to an existing bundle."""
    viewer = PETSc.ViewerHDF5().create(filename, "a")
    viewer.pushGroup("/tdcsem")
    _writeArrayAsVec(viewer, time, "time")
    _writeArrayAsVec(viewer, waveform, "waveform")
    viewer.popGroup()
    viewer.destroy()


def writeMtPayload(filename, frequency):
    """Append the MT payload (/mt/freq) to an existing PETGEM input bundle."""
    viewer = PETSc.ViewerHDF5().create(filename, "a")
    viewer.pushGroup("/mt")
    _writeArrayAsVec(viewer, np.array([frequency], dtype=float), "freq")
    viewer.popGroup()
    viewer.destroy()


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
                         receivers_arr, sources8, order,
                         output_filename, cells=None, coords=None,
                         output_vtk=None):
    """Write the mesh, model, receivers, order and sources of a bundle.

    Contents of `output_filename` (HDF5):
      DMPlex blocks        mesh ``petgem_mesh`` and the ``model_data`` Vec
                           (sigma_x, sigma_y, sigma_z, material id per cell),
                           written with PETSc's HDF5_PETSC format
      /receivers           Vec, length 3*N_recv, layout [x0 y0 z0 x1 y1 z1 ...]
      /order               Vec, length 1
      /sources/...         one entry per row of `sources8`; skipped when None

    Inverse and MT bundles are completed afterwards by
    :func:`writeInversionPayload` and :func:`writeMtPayload`.

    Parameters
    ----------
    plex          : DMPlex (2-field section, from createDM)
    conductivity  : ndarray (num_cells, dim) - sigma_x, sigma_y, sigma_z per cell
    materials_id  : ndarray (num_cells,)     - integer material id per cell
    receivers_arr : ndarray (num_recv, 3)    - receiver positions
    sources8      : ndarray (num_src, 8) or None - one transmitter per row,
                    columns freq x y z current length dip azimuth. Forward uses
                    one (repeated) frequency with one or more transmitters;
                    inverse carries one row per (frequency, dipole). None skips
                    the /sources group.
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
    _writeArrayAsVec(viewer, np.array([order], dtype=float), "order")

    # Sources (under /sources group) - unified per-entry layout for both
    # forward and inverse. Each row of sources8 is one transmitter; the
    # frequency is stored per entry (forward repeats a single frequency).
    if sources8 is not None:
        s = np.asarray(sources8, dtype=float)
        if s.ndim != 2 or s.shape[1] != 8:
            raise ValueError(
                f"sources8 must be (N, 8) [freq x y z current length dip "
                f"azimuth], got shape {s.shape}")
        viewer.pushGroup("/sources")
        _writeArrayAsVec(viewer, s[:, 0],               "freq")
        _writeArrayAsVec(viewer, s[:, 1:4].reshape(-1), "position")
        _writeArrayAsVec(viewer, s[:, 4],               "current")
        _writeArrayAsVec(viewer, s[:, 5],               "length")
        _writeArrayAsVec(viewer, s[:, 6],               "dipAngle")
        _writeArrayAsVec(viewer, s[:, 7],               "azimuthAngle")
        viewer.popGroup()

    viewer.destroy()
    v_model.destroy()


# Solver block emitted per mode; the same keys work for every kernel.
# "-dm_mat_type is" selects PCBDDC, an AIJ operator a direct MUMPS solve.
#
# The template emits the iterative path for fm and im and the direct path for
# mt; users switch by editing this block or by appending a solver preset with
# a second -options_file.
#
# The blocks differ only in the tolerance: im.csem's adjoint system A_f.nx = nB
# uses the same operator - and so the same KSP - as its forward solve, and the
# adjoint solution enters the objective gradient, so it is solved tightly.
_ITERATIVE_SOLVER_BLOCK = (
    "-dm_mat_type is\n"
    "-ksp_type fgmres\n"
    "-pc_type bddc\n"
    "-pc_bddc_use_deluxe_scaling 1\n"
    "-pc_bddc_coarse_pc_type lu\n"
)

_DIRECT_SOLVER_BLOCK = (
    "-dm_mat_type aij\n"
    "-ksp_type preonly\n"
    "-pc_type lu\n"
    "-pc_factor_mat_solver_type mumps\n"
)

_SOLVER_BLOCK = {
    "fm.csem": _ITERATIVE_SOLVER_BLOCK,
    "im.csem": _ITERATIVE_SOLVER_BLOCK + "-ksp_rtol 1.0e-10\n",
    "fm.mt":   _DIRECT_SOLVER_BLOCK,
    "fm.tdcsem": _DIRECT_SOLVER_BLOCK,
}

# Inversion tuning block (im.csem only). Every option is -im_*, matching the
# im.csem kernel and the ImParams struct.
# -im_error_level and -im_fixed_materials are left out: they come from the
# bundle (/observed @error_level, /im_meta/fixed_materials).
_IM_TUNING_BLOCK = (
    "-im_max_iter                150\n"
    "-im_lbfgs_memory            2\n"
    "-im_lambda                  0.1\n"
    "-im_diag_weight             0.0\n"
    "-im_gtol                    1.0e-5\n"
    "-im_rms_tol                 1.05\n"
    "-im_snapshot_interval       1\n"
)


def writeParamsFile(mode, output_dir, output_filename, input_filename,
                    params_filename, tdcsem_t_end=None):
    """Emit the PETSc options file for any kernel.

    The mode selects the solver block (BDDC for ``fm.csem`` and ``im.csem``,
    MUMPS for ``fm.mt`` and ``fm.tdcsem``), for ``im.csem`` adds the inversion
    block and for ``fm.tdcsem`` the final time. The
    polynomial order is not written: the kernels read ``/order`` from the
    bundle.

    Parameters
    ----------
    mode : {'fm.csem', 'im.csem', 'fm.mt', 'fm.tdcsem'}
        Kernel name (see KERNELS).
    output_dir : str
        Case/output directory; also where the params file is written.
    output_filename : str
        Output stem. Both kernels write ``{output_dir}/{stem}.h5``.
    input_filename : str
        Bundle written by writePetgemInputFile, fed back as -input_filename.
    params_filename : str
        Name of the params file to emit inside output_dir.
    tdcsem_t_end : float or None
        Final time (s), ``fm.tdcsem`` only.
    """
    if mode not in _SOLVER_BLOCK:
        raise ValueError(f"writeParamsFile: unknown mode {mode!r} "
                         f"(expected one of {list(KERNELS)})")

    content = (
        f"-input_filename {output_dir}/{input_filename}\n"
        + _SOLVER_BLOCK[mode]
        + (_IM_TUNING_BLOCK if mode == "im.csem" else "")
        + (f"-tdcsem_t_end {tdcsem_t_end!r}\n" if mode == "fm.tdcsem" else "")
        + f"-output_dir {output_dir}/\n"
        + f"-output_filename {output_filename}\n"
    )
    with open(f"{output_dir}/{params_filename}", "w") as f:
        f.write(content)


def _loadXYZText(path):
    """Load an ``x y z`` table from a text file, tolerant of the delimiter.

    Accepts whitespace- or comma-separated columns (or a mix), allows ``#``
    comments and blank lines.  Returns an ``(N, 3)`` float ndarray.  This is
    more forgiving than ``np.loadtxt`` (whitespace-only), so receiver files
    exported from MATLAB/other tools (often comma-separated) load directly
    without a manual conversion step.
    """
    rows = [[float(tok) for tok in toks] for toks in _readTokens(path, commas=True)]
    return np.asarray(rows, dtype=float)


def _extractTetraMaterial(mesh, tetra_idx, path):
    """Return the 0-based per-tetra material id from a meshio mesh.

    Two input conventions are supported transparently so the preprocess
    stage accepts both Gmsh ``.msh`` and VTK ``.vtk``/``.vtu`` meshes:

    * Gmsh: the canonical 1-based ``gmsh:physical`` tag is used as before
      (``id = tag - 1``), preserving existing behavior exactly.
    * VTK: an arbitrary integer cell-data array (auto-detected, commonly
      ``cell_scalars``) carries region codes that need not be contiguous
      or 0-based (e.g. ``{10, 20, 30, 40}``). ``np.unique`` maps the
      distinct codes to a sorted 0-based index, so sigma-table row ``i``
      corresponds to the ``i``-th smallest code.

    Returns ``(materials_id, codes)`` where ``codes`` is the sorted array of
    distinct VTK material codes (for a remap report) or ``None`` for the
    Gmsh path.
    """
    cdd = mesh.cell_data_dict
    if "gmsh:physical" in cdd and "tetra" in cdd["gmsh:physical"]:
        mat = np.asarray(cdd["gmsh:physical"]["tetra"]).reshape(-1).astype(int) - 1
        return mat, None

    candidates = ("cell_scalars", "materials_id", "material_id", "MaterialID",
                  "material", "CellEntityIds")
    name = next((candidate for candidate in candidates if candidate in mesh.cell_data), None)
    if name is None:
        raise ValueError(
            f"{path}: no material cell-data array found (looked for "
            f"{candidates}; available {list(mesh.cell_data.keys())}).")
    raw = np.asarray(mesh.cell_data[name][tetra_idx]).reshape(-1)
    codes, mat = np.unique(raw, return_inverse=True)
    return mat.astype(int), codes


def _require_input_file(path, label, hint=None):
    """Raise a clear FileNotFoundError when a required input file is missing.

    Used to fail preprocessing up front with an actionable message instead of
    a raw meshio / numpy / open() traceback deeper in the pipeline.
    """
    if not os.path.isfile(path):
        msg = f"{label} not found: '{path}'. Check -case_dir and the filename"
        msg += f" ({hint})." if hint else "."
        raise FileNotFoundError(msg)


# Mode-specific input files: required in their mode, rejected in the others.
_MODE_INPUTS = {
    "fm.csem": {"source_filename"},
    "im.csem": {"im_source_filename", "observed_filename"},
    "fm.mt":   {"mt_frequency_filename"},
    "fm.tdcsem": {"source_filename", "tdcsem_waveform_filename"},
}


def runPreprocessing(*, mode, order, case_dir,
                     mesh_filename, receiver_filename, source_filename=None,
                     sigma_x, sigma_y, sigma_z,
                     fixed_materials=(),
                     input_filename="input.h5",
                     params_filename="params.txt",
                     im_source_filename=None,
                     observed_filename=None,
                     error_level=None,
                     mt_frequency_filename=None,
                     tdcsem_waveform_filename=None, tdcsem_t_end=None,
                     output_vtk=None, dm_view=False):
    """Build the input bundle and the parameter file for one kernel run.

    Reads the mesh (Gmsh ``.msh`` or VTK ``.vtk``/``.vtu``), assigns the
    per-cell conductivity by material id, reads receivers and sources (or the
    MT frequency), and writes ``<case_dir>/<input_filename>`` and
    ``<case_dir>/<params_filename>``. All file names are relative to
    `case_dir`.

    Parameters
    ----------
    mode : {'fm.csem', 'im.csem', 'fm.mt', 'fm.tdcsem'}
        Kernel the bundle is for.
    order : int
        Polynomial order, stored as ``/order``. It also sets the output stem
        ``responses_p<order>`` in the parameter file.
    case_dir : str
        Directory the other file names are relative to.
    mesh_filename, receiver_filename : str
        Mesh and receiver table.
    source_filename : str or None
        Forward sources (``fm.csem`` and ``fm.tdcsem``). For ``fm.csem`` it
        must hold a single frequency; ``fm.tdcsem`` rows carry no frequency.
    sigma_x, sigma_y, sigma_z : array_like
        Per-material conductivity, indexed by 0-based material id.
    fixed_materials : iterable of int
        Material ids held fixed by the inversion (``im.csem`` only).
    input_filename, params_filename : str
        Names of the bundle and the parameter file to write.
    im_source_filename : str or None
        Multi-frequency sources, one row per (frequency, dipole)
        (``im.csem`` only).
    observed_filename : str, list of str or None
        Observed data, entries ``file`` or ``component:file``, HDF5 or
        ``invEx.dat`` text (``im.csem`` only).
    error_level : float or None
        Relative noise level; overrides the one stored in an HDF5
        observed-data file.
    mt_frequency_filename : str or None
        MT frequency file (``fm.mt`` only).
    tdcsem_waveform_filename : str or None
        Source waveform ``t s(t)`` (``fm.tdcsem`` only).
    tdcsem_t_end : float or None
        Final time (s), written into the parameter file (``fm.tdcsem`` only).
    output_vtk : str or None
        Also write the conductivity and material ids to this VTU file.
    dm_view : bool
        Print the DMPlex summary.

    Raises
    ------
    ValueError
        Unknown mode, a mode-specific input missing or given to the wrong
        mode, material ids outside the conductivity table, more than one
        frequency for ``fm.csem``, or observed data whose shape does not match
        the sources and receivers.
    FileNotFoundError
        A required input file does not exist.
    """
    mode = checkMode(mode)
    given = {"source_filename": source_filename,
             "im_source_filename": im_source_filename,
             "observed_filename": observed_filename,
             "mt_frequency_filename": mt_frequency_filename,
             "tdcsem_waveform_filename": tdcsem_waveform_filename}
    for name, value in given.items():
        if name in _MODE_INPUTS[mode] and value is None:
            raise ValueError(f"runPreprocessing: -{name} is required when mode='{mode}'")
        if name not in _MODE_INPUTS[mode] and value is not None:
            raise ValueError(f"runPreprocessing: -{name} is not valid with mode='{mode}'")
    if (mode == "fm.tdcsem") != (tdcsem_t_end is not None):
        raise ValueError("runPreprocessing: -tdcsem_t_end is required with mode='fm.tdcsem' "
                         "and not valid otherwise")

    sigma_x = np.asarray(sigma_x, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)
    sigma_z = np.asarray(sigma_z, dtype=float)
    if not (len(sigma_x) == len(sigma_y) == len(sigma_z)):
        raise ValueError("sigma_x, sigma_y, sigma_z must have equal length")

    NUM_DIMENSIONS = 3

    def inCase(name):
        return os.path.join(case_dir, name) if name is not None else None

    input_mesh_filename         = inCase(mesh_filename)
    input_receivers_filename    = inCase(receiver_filename)
    input_sources_filename      = inCase(source_filename)
    output_filename             = inCase(input_filename)
    output_petgem_filename      = f"responses_p{order}"
    output_vtk_filename         = inCase(output_vtk)
    input_im_sources_filename   = inCase(im_source_filename)
    if isinstance(observed_filename, str):
        observed_filename = [observed_filename]
    observed_specs = [parseObservedSpec(spec) for spec in (observed_filename or [])]
    observed_specs = [(comp, inCase(path)) for comp, path in observed_specs]
    input_mt_frequency_filename = inCase(mt_frequency_filename)
    input_td_waveform_filename  = inCase(tdcsem_waveform_filename)

    print("====================================================")
    print(f" PETGEM INPUT PREPROCESSING ({mode})")
    print("====================================================")
    print(f"  Polynomial order (order): {order}")
    print(f"  Case directory         : {case_dir}")
    print(f"  Mesh file              : {input_mesh_filename}")
    print(f"  Receivers file         : {input_receivers_filename}")
    if mode in ("fm.csem", "fm.tdcsem"):
        print(f"  Sources file           : {input_sources_filename}")
    if mode == "fm.tdcsem":
        print(f"  Waveform file          : {input_td_waveform_filename}")
    if mode == "fm.mt":
        print(f"  MT frequency file      : {input_mt_frequency_filename}")
    if mode == "im.csem":
        print(f"  IM sources file        : {input_im_sources_filename}")
        for comp, path in observed_specs:
            print(f"  Observed data file     : {path}" + (f"  ({comp})" if comp else ""))
    print(f"  Output bundle          : {output_filename}")
    print(f"\n  Number of materials    : {len(sigma_x)}")

    # 0. Required input files must exist
    _require_input_file(input_mesh_filename, "Mesh file",
                        hint="run gmsh on the .geo")
    _require_input_file(input_receivers_filename, "Receivers file")
    if mode == "fm.csem":
        _require_input_file(input_sources_filename, "Sources file")
    elif mode == "fm.tdcsem":
        _require_input_file(input_sources_filename, "Sources file")
        _require_input_file(input_td_waveform_filename, "Waveform file")
    elif mode == "im.csem":
        _require_input_file(input_im_sources_filename, "Inverse sources file")
        for _, path in observed_specs:
            _require_input_file(path, "Observed-data file")
    else:
        _require_input_file(input_mt_frequency_filename, "MT frequency file")

    # 1. Import mesh (Gmsh .msh or VTK .vtk/.vtu - meshio auto-detects the
    #    format from the extension/header). PETGEM is tetrahedral-only, so
    #    select the tetra connectivity block explicitly (a VTK/Gmsh file may
    #    also carry lower-dimensional triangle/line blocks).
    print("\nReading mesh")
    mesh = meshio.read(input_mesh_filename)
    coords = mesh.points
    tetra_idx = next((i for i, block in enumerate(mesh.cells) if block.type == "tetra"),
                     None)
    if tetra_idx is None:
        raise ValueError(
            f"{input_mesh_filename}: no tetrahedral cell block found "
            f"(blocks: {[block.type for block in mesh.cells]}); PETGEM is tetra-only.")
    cells = mesh.cells[tetra_idx].data
    num_cells = cells.shape[0]
    print(f"  Total elements          : {num_cells}")
    print(f"  Total vertices          : {coords.shape[0]}")

    # 2. Per-cell conductivity from the material id (Gmsh gmsh:physical or a
    #    VTK cell-data array; see _extractTetraMaterial).
    print("\nAssigning conductivity per cell")
    materials_id, codes = _extractTetraMaterial(mesh, tetra_idx,
                                                input_mesh_filename)
    nmat = len(sigma_x)
    if codes is not None:
        counts = np.bincount(materials_id, minlength=len(codes))
        print("  Material codes -> rows  : " + ", ".join(
            f"{int(code)}->{i} ({int(count)} cells)"
            for i, (code, count) in enumerate(zip(codes, counts))))
    mn, mx = int(materials_id.min()), int(materials_id.max())
    if mn < 0 or mx >= nmat:
        raise ValueError(
            f"{input_mesh_filename}: material ids span [{mn},{mx}] but "
            f"-sigma_file has {nmat} row(s) (need ids in [0,{nmat - 1}]). "
            f"For VTK, sigma row i = i-th smallest material code; for Gmsh, "
            f"row i = physical tag (i+1).")
    conductivity = np.zeros((num_cells, NUM_DIMENSIONS), dtype=float)
    conductivity[:, 0] = sigma_x[materials_id]
    conductivity[:, 1] = sigma_y[materials_id]
    conductivity[:, 2] = sigma_z[materials_id]
    print("  Conductivity assignment completed")

    # 3. Receivers (text → ndarray)
    print("\nReading receivers")
    receivers_arr = _loadXYZText(input_receivers_filename)
    if receivers_arr.ndim == 1:
        receivers_arr = receivers_arr.reshape(1, -1)
    if receivers_arr.shape[1] != NUM_DIMENSIONS:
        raise ValueError(
            f"{input_receivers_filename}: expected {NUM_DIMENSIONS} columns "
            f"(x y z), got {receivers_arr.shape[1]}"
        )
    print(f"  Number of receivers     : {receivers_arr.shape[0]}")

    # 4. Sources (text → unified (N, 8) [freq x y z current length dip az]).
    # The same /sources group is written for fm.csem and im.csem; forward repeats a
    # single frequency across its transmitters, inverse carries one row per
    # (frequency, dipole). mt has no /sources group, only /mt/freq.
    sources8 = None
    if mode == "fm.mt":
        print("\nReading MT frequency")
        mt_frequency = readMtFrequencyText(input_mt_frequency_filename)
        print(f"  Frequency (Hz)          : {mt_frequency}")
    elif mode == "fm.tdcsem":
        print("\nReading sources and waveform")
        sources8 = readTdSourcesText(input_sources_filename)
        td_time, td_waveform = readTdWaveformText(input_td_waveform_filename)
        print(f"  Transmitters            : {sources8.shape[0]}")
        print(f"  Waveform samples        : {td_time.size} ({td_time[0]} .. {td_time[-1]} s)")
        print(f"  Final time (s)          : {tdcsem_t_end}")
    else:
        src_path = input_sources_filename if mode == "fm.csem" else input_im_sources_filename
        print("\nReading sources")
        sources8 = readSourcesText(src_path)
        uniq_freqs = np.unique(sources8[:, 0])
        print(f"  Transmitters            : {sources8.shape[0]}")
        print(f"  Frequencies (Hz)        : {uniq_freqs.tolist()}")
        if mode == "fm.csem" and len(uniq_freqs) != 1:
            raise ValueError(
                f"{src_path}: fm.csem solves one frequency per run, got "
                f"{uniq_freqs.tolist()} Hz; build one bundle per frequency")

    # 5. Build the DMPlex
    print("\nCreating PETSc DM (DMPlex)")
    plex = createDM(NUM_DIMENSIONS, cells, coords, dm_view=dm_view)

    # 6. Write the unified bundle (mesh + model + receivers + /sources or /mt)
    print("\nWriting unified PETGEM input HDF5")
    writePetgemInputFile(plex, conductivity, materials_id,
                         receivers_arr, sources8, order,
                         output_filename,
                         cells=cells, coords=coords,
                         output_vtk=output_vtk_filename)
    if mode == "fm.mt":
        writeMtPayload(output_filename, mt_frequency)
    if mode == "fm.tdcsem":
        writeTdPayload(output_filename, td_time, td_waveform)
    print(f"  Output file: {output_filename}")
    if output_vtk_filename:
        print(f"  VTU view  : {output_vtk_filename}")

    # 6b. Inverse-only payload: append /observed/<component> and /im_meta.
    # (The sources already live in the unified /sources group above.)
    if mode == "im.csem":
        print("\nReading observed data")
        # Each file is a prebuilt HDF5 (.h5/.hdf5) or a raw MATLAB-style
        # invEx.dat text file - the .dat is parsed inline, so no separate
        # convert-to-HDF5 step is needed.
        observed, file_error_level = {}, None
        for comp, path in observed_specs:
            data, level, src_kind = readObservedData(path, comp)
            for name in data:
                if name in observed:
                    raise ValueError(f"Observed component {name} given twice ({path})")
            observed.update(data)
            if file_error_level is None:
                file_error_level = level
            print(f"  Observed source         : {src_kind}  ({', '.join(data)})")
        # -error_level on the CLI overrides the file's attribute (and is the
        # only way to set it for a raw .dat).
        observed_error_level = error_level if error_level is not None else file_error_level
        for name, values in observed.items():
            print(f"  Observed {name:<3} shape      : {values.shape}  (N_freq, N_recv)")
        print(f"  Error level             : "
              f"{observed_error_level if observed_error_level is not None else '(absent, kernel default)'}")

        n_freq_src = sources8.shape[0]
        for name, values in observed.items():
            if values.shape[0] != n_freq_src:
                raise ValueError(
                    f"Mismatch between source rows ({n_freq_src}) and "
                    f"observed {name} frequencies ({values.shape[0]})"
                )
            if values.shape[1] != receivers_arr.shape[0]:
                raise ValueError(
                    f"Observed {name} N_recv={values.shape[1]} disagrees with "
                    f"receivers count {receivers_arr.shape[0]}"
                )

        print("\nEmbedding inverse payload into bundle")
        if fixed_materials:
            print(f"  Fixed material IDs       : {list(fixed_materials)} (from sigmas.txt)")
        writeInversionPayload(output_filename, observed,
                              error_level=observed_error_level,
                              fixed_materials=fixed_materials)
        written = ", ".join(f"/observed/{name}" for name in observed)
        print(f"  Wrote {written}, /im_meta/fixed_materials "
              f"into {output_filename}")

    # 7. Params file (one writer for every mode; mode selects the solver block)
    print("\nGenerating PETGEM parameter file")
    writeParamsFile(mode, case_dir, output_petgem_filename,
                    input_filename, params_filename, tdcsem_t_end=tdcsem_t_end)
    print(f"  Params file: {os.path.join(case_dir, params_filename)}")

    print("\n====================================================")
    print(" Preprocessing completed successfully")
    print("====================================================\n")


def readBundle(filename):
    """Read the receivers, order and sources of a CSEM input bundle.

    Returns a dict with keys::

      receivers : (N_recv, 3) ndarray of receiver positions (real-valued)
      order      : int polynomial order
      frequency : float - first transmitter frequency (Hz); for a single-
                  frequency forward case this is the operating frequency
      sources   : (N_src, 8) ndarray, columns =
                  [freq, x, y, z, current, length, dipAngle, azimuthAngle]

    The mesh and the per-cell model are not returned. MT bundles have no
    ``/sources`` group and cannot be read with this function.
    """
    receivers = np.real(np.array(readVectorH5(filename, 'receivers'))).reshape(-1, 3)

    order_arr = readVectorH5(filename, 'order')
    order = int(round(float(np.real(np.array(order_arr)).flatten()[0])))

    # Unified /sources group: per-entry frequency.
    freq    = np.real(np.array(readVectorH5(filename, 'freq',         group='/sources'))).reshape(-1)
    pos     = np.real(np.array(readVectorH5(filename, 'position',     group='/sources'))).reshape(-1, 3)
    current = np.real(np.array(readVectorH5(filename, 'current',      group='/sources'))).reshape(-1)
    length  = np.real(np.array(readVectorH5(filename, 'length',       group='/sources'))).reshape(-1)
    dip     = np.real(np.array(readVectorH5(filename, 'dipAngle',     group='/sources'))).reshape(-1)
    az      = np.real(np.array(readVectorH5(filename, 'azimuthAngle', group='/sources'))).reshape(-1)
    sources = np.column_stack([freq, pos, current, length, dip, az])

    return {
        'receivers': receivers,
        'order':      order,
        'frequency': float(freq[0]),
        'sources':   sources,
    }


def readResponses(filename, source=1):
    """Read the fields of one transmitter from an ``fm.csem`` output file.

    Parameters
    ----------
    filename : str
        Path to the responses HDF5 file.
    source : int, optional
        1-based source index (default 1).

    Returns
    -------
    dict
        Keys::

          Ex, Ey, Ez, Hx, Hy, Hz : ndarrays (complex in PETSc complex builds)
          source      : dict of /sources/src{k} attributes
                        (frequency, x_pos, y_pos, z_pos, current, length,
                         dip_angle, azimuth_angle)
          provenance  : dict of root attributes (provenance block,
                        num_sources, frequency)

    Raises
    ------
    KeyError
        The file has no ``/sources/src{source}`` group.
    """
    import h5py

    src_group    = f"/sources/src{int(source)}"
    fields_group = f"{src_group}/fields"

    out = {}
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        out[name] = np.array(readVectorH5(filename, name, group=fields_group))

    with h5py.File(filename, 'r') as f:
        if src_group not in f:
            raise KeyError(
                f"{filename}: missing group {src_group!r} (file holds "
                f"{int(f.attrs.get('num_sources', -1))} sources)"
            )
        out['source']     = {key: _decodeH5Attr(value) for key, value in f[src_group].attrs.items()}
        out['provenance'] = {key: _decodeH5Attr(value) for key, value in f.attrs.items()}
    return out


def readAllResponses(filename):
    """Read the fields of every transmitter from an ``fm.csem`` output file.

    Parameters
    ----------
    filename : str
        Path to the responses HDF5 file.

    Returns
    -------
    dict
        Keys::

          provenance : dict of root-level attributes (see ``readResponses``)
          num_sources : int
          sources    : dict keyed by 1-based source index; each value has the
                       same shape as ``readResponses(filename, source=k)``
                       (Ex..Hz arrays + 'source' attrs + 'provenance').

    """
    import h5py

    with h5py.File(filename, 'r') as f:
        provenance = {key: _decodeH5Attr(value) for key, value in f.attrs.items()}
        num_sources = int(provenance.get('num_sources', 0))
        if num_sources <= 0 and '/sources' in f:
            num_sources = len([name for name in f['/sources'].keys() if name.startswith('src')])

    sources = {source: readResponses(filename, source=source)
               for source in range(1, num_sources + 1)}
    return {
        'provenance':  provenance,
        'num_sources': num_sources,
        'sources':     sources,
    }


def readTdResponses(filename, source=1):
    """Read the time series of one transmitter from an ``fm.tdcsem`` output file.

    Returns a dict with ``times`` (N_t,) and ``Ex, Ey, Ez, dBxdt, dBydt, dBzdt``
    as real (N_t, N_recv) arrays, plus the ``source`` and ``provenance``
    attributes.
    """
    import h5py

    src_group = f"/sources/src{int(source)}"
    out = {}
    with h5py.File(filename, 'r') as f:
        if src_group not in f:
            raise KeyError(f"{filename}: missing group {src_group!r}")
        times = np.asarray(f['times'])
        out['times'] = times[:, 0] if times.ndim == 2 else times
        for name in ('Ex', 'Ey', 'Ez', 'dBxdt', 'dBydt', 'dBzdt'):
            a = np.asarray(f[f"{src_group}/fields/{name}"])
            out[name] = a[..., 0] if a.ndim == 3 else a
        out['source']     = {key: _decodeH5Attr(value) for key, value in f[src_group].attrs.items()}
        out['provenance'] = {key: _decodeH5Attr(value) for key, value in f.attrs.items()}
    return out


def compareMagnitude(computed, reference):
    """Magnitude-domain error metrics between a computed and a reference field.

    The comparison is done on ``|field|``, so complex and real inputs are
    both accepted.

    Parameters
    ----------
    computed, reference : array_like
        Field samples to compare (e.g. ``Ex`` at each receiver). Must have the
        same number of elements.

    Returns
    -------
    dict
        ``nrmsd``  - RMS deviation of ``|computed|`` vs ``|reference|``,
        normalized by the reference peak-to-peak range.
        ``rel_l2`` - relative L2 norm of the magnitude difference.
        ``mape``   - mean absolute percentage error (%).
    """
    a = np.abs(np.asarray(computed)).ravel()
    b = np.abs(np.asarray(reference)).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"compareMagnitude: length mismatch - computed has {a.size} "
            f"samples, reference has {b.size}. They must describe the same "
            f"receivers in the same order.")
    return {
        "nrmsd":  float(np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())),
        "rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "mape":   float(np.mean(np.abs((a - b) / b)) * 100),
    }


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
        HDF5 group holding the Vec, e.g. ``/sources/src1/fields``.

    Returns
    -------
    ndarray
        The Vec values (complex on a complex PETSc build).
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
