import glob
import os
from pathlib import Path

# Repo root derived from this file's location: scripts/auto_doc/<this>.
# parents[2] == repo root. Robust to the caller's CWD, so we no longer
# need to chdir or carry fragile '../../' relative paths.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Configuration ---
# C public-API headers live in include/, NOT src/ (src/ holds only .c
# translation units). Globbing src/ here was the cause of the persistent
# "No API modules generated yet" placeholder on Read the Docs: zero .h
# files were found, so STEP 2 emitted the placeholder on every build.
SOURCE_CODE_DIR = REPO_ROOT / 'include'        # C public-API header files
API_RST_DIR = REPO_ROOT / 'docs' / 'source' / 'api'  # Output .rst directory
PROJECT_NAME_DOXYGEN = "PETGEM"        # Doxygen project name
CREATE_API_INDEX = True                 # Whether to create a master index.rst
API_INDEX_FILENAME = 'index.rst'        # Filename for the API index.rst
# Headers excluded from the PUBLIC API reference. *_internal.h are private
# cross-TU headers ("Not intended for inclusion outside src/..."), so they
# don't belong in the user-facing API docs.
EXCLUDE_SUFFIXES = ('_internal.h',)
# --- End Configuration ---


def create_api_rst_files():
    if not os.path.exists(API_RST_DIR):
        os.makedirs(API_RST_DIR)
        print(f"Directory created: {API_RST_DIR}")

    # --- STEP 1: Process header files (.h), skipping private internals ---
    header_files = sorted(glob.glob(os.path.join(str(SOURCE_CODE_DIR), '*.h')))
    header_files = [h for h in header_files
                    if not os.path.basename(h).endswith(EXCLUDE_SUFFIXES)]
    generated_rst_basenames = []

    for header_path in header_files:
        h_filename = os.path.basename(header_path)
        base_name = os.path.splitext(h_filename)[0]
        rst_filename = f"{base_name}.rst"
        rst_filepath = os.path.join(str(API_RST_DIR), rst_filename)

        ref_label = f".. _api-{base_name}:\n\n"
        title_text = f"{base_name.capitalize()} module ({h_filename})"
        title_underline = "=" * len(title_text) + "\n\n"
        title = f"{title_text}\n{title_underline}"

        # Doxygen directive points to the header file (clean for Breathe)
        doxygen_directive = f".. doxygenfile:: {h_filename}\n"
        doxygen_directive += f"   :project: {PROJECT_NAME_DOXYGEN}\n"

        rst_content = f"{ref_label}{title}{doxygen_directive}"

        with open(rst_filepath, 'w') as f:
            f.write(rst_content)
        print(f"Generated: {rst_filepath}")
        generated_rst_basenames.append(base_name)

    # --- STEP 2: Fail loudly if no headers found ---
    # A missing-headers situation is almost always a misconfiguration
    # (wrong SOURCE_CODE_DIR, headers moved) rather than a legitimate
    # "no API yet" state.  Emitting a silent placeholder + exit 0 is what
    # let the src/-vs-include/ path bug hide for so long - the docs build
    # "succeeded" with an empty API reference.  Raise instead so both
    # `make docs` and the Read the Docs build fail visibly.
    if not generated_rst_basenames:
        raise SystemExit(
            f"ERROR: api_rst_generator found no .h files under "
            f"'{SOURCE_CODE_DIR}' (resolved from {os.getcwd()}).\n"
            f"       The C public-API headers live in include/. Check "
            f"SOURCE_CODE_DIR if this path is wrong.\n"
            f"       Refusing to emit a 'No API modules generated yet' "
            f"placeholder."
        )

    # --- STEP 3: Always create master API index ---
    if CREATE_API_INDEX:
        create_master_api_index(generated_rst_basenames)


def create_master_api_index(rst_basenames):
    index_filepath = os.path.join(API_RST_DIR, API_INDEX_FILENAME)
    if not os.path.exists(API_RST_DIR):
        os.makedirs(API_RST_DIR)

    title = "API Reference\n"
    title += "=============\n\n"
    toctree_content = ".. toctree::\n"
    toctree_content += "   :maxdepth: 1\n"
    toctree_content += "   :caption: Modules:\n\n"

    for basename in sorted(rst_basenames):
        toctree_content += f"   {basename}\n"

    with open(index_filepath, 'w') as f:
        f.write(title + toctree_content)
    print(f"Generated/Updated API index: {index_filepath}")


if __name__ == "__main__":
    # All paths are absolute (derived from REPO_ROOT), so no chdir needed -
    # the script works regardless of the caller's working directory.
    create_api_rst_files()