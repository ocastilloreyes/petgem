import os
import glob

# --- Configuration ---
SOURCE_CODE_DIR = '../../src'      # Path to C source code directory
API_RST_DIR = '../../docs/source/api'         # Output directory for generated .rst API files
PROJECT_NAME_DOXYGEN = "PETGEM"    # Doxygen project name (for :project: directive)
CREATE_API_INDEX = True            # Whether to create an index.rst for the API directory
API_INDEX_FILENAME = 'index.rst'   # Filename for the API index.rst
# --- End Configuration ---

def create_api_rst_files():
    if not os.path.exists(API_RST_DIR):
        os.makedirs(API_RST_DIR)
        print(f"Directory created: {API_RST_DIR}")

    source_files = glob.glob(os.path.join(SOURCE_CODE_DIR, '*.c')) # Find .c source files

    generated_rst_basenames = []

    for source_file_path in source_files:
        c_filename = os.path.basename(source_file_path)
        base_name = os.path.splitext(c_filename)[0] # Base name (no extension)
        rst_filename = f"{base_name}.rst"
        rst_filepath = os.path.join(API_RST_DIR, rst_filename)

        ref_label = f".. _api-{base_name}:\n\n"
        title_text = f"{base_name.capitalize()} Module ({c_filename})"
        title_underline = "=" * len(title_text) + "\n\n"
        title = f"{title_text}\n{title_underline}"

        doxygen_directive = f".. doxygenfile:: {c_filename}\n"
        doxygen_directive += f"   :project: {PROJECT_NAME_DOXYGEN}\n"

        rst_content = f"{ref_label}{title}{doxygen_directive}"

        with open(rst_filepath, 'w') as f:
            f.write(rst_content)
        print(f"Generated: {rst_filepath}")
        generated_rst_basenames.append(base_name) # Store basename for the toctree

    if CREATE_API_INDEX and generated_rst_basenames:
        create_master_api_index(generated_rst_basenames)

def create_master_api_index(rst_basenames):
    index_filepath = os.path.join(API_RST_DIR, API_INDEX_FILENAME)
    title = "API Reference\n"
    title += "=============\n\n"
    toctree_content = ".. toctree::\n"
    toctree_content += "   :maxdepth: 1\n"
    toctree_content += "   :caption: Modules:\n\n" # Optional caption

    for basename in sorted(rst_basenames): # Sort alphabetically
        toctree_content += f"   {basename}\n" # Sphinx assumes .rst extension

    with open(index_filepath, 'w') as f:
        f.write(title + toctree_content)
    print(f"Generated/Updated API index: {index_filepath}")


if __name__ == "__main__":
    # Change to script's directory for consistent relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    create_api_rst_files()