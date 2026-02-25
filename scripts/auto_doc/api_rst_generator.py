import os
import glob

# --- Configuration ---
SOURCE_CODE_DIR = '../../src'           # Path to C header files
API_RST_DIR = '../../docs/source/api'   # Output directory for generated .rst files
PROJECT_NAME_DOXYGEN = "PETGEM"        # Doxygen project name
CREATE_API_INDEX = True                 # Whether to create a master index.rst
API_INDEX_FILENAME = 'index.rst'        # Filename for the API index.rst
PLACEHOLDER_FILENAME = 'placeholder.rst' # Dummy module if no headers
# --- End Configuration ---


def create_api_rst_files():
    if not os.path.exists(API_RST_DIR):
        os.makedirs(API_RST_DIR)
        print(f"Directory created: {API_RST_DIR}")

    # --- STEP 1: Process header files (.h) ---
    header_files = glob.glob(os.path.join(SOURCE_CODE_DIR, '*.h'))
    generated_rst_basenames = []

    for header_path in header_files:
        h_filename = os.path.basename(header_path)
        base_name = os.path.splitext(h_filename)[0]
        rst_filename = f"{base_name}.rst"
        rst_filepath = os.path.join(API_RST_DIR, rst_filename)

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

    # --- STEP 2: Create placeholder if no headers found ---
    if not generated_rst_basenames:
        placeholder_path = os.path.join(API_RST_DIR, PLACEHOLDER_FILENAME)
        with open(placeholder_path, 'w') as f:
            f.write(
                "No API modules generated yet.\n"
                "=============================\n\n"
                "There are no C header files found in the source directory.\n"
            )
        print(f"Generated placeholder module: {placeholder_path}")
        generated_rst_basenames.append('placeholder')

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
    # Ensure consistent relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    create_api_rst_files()