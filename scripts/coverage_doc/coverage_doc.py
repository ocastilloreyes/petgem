import xml.etree.ElementTree as ET
import glob
import argparse
import os
import sys

def calculate_doxygen_coverage(doxygen_xml_dir, check_params=False, check_return=False):
    """
    Calculates documentation coverage for functions from Doxygen XML output.

    Args:
        doxygen_xml_dir (str): Path to the Doxygen XML output directory.
        check_params (bool): If True, also checks if all function parameters are documented.
        check_return (bool): If True, also checks if the return value is documented.

    Returns:
        tuple: (total_functions, documented_functions, coverage_percentage)
               Returns (0, 0, 0.0) if no functions are found or an error occurs.
    """
    total_functions = 0
    documented_functions = 0

    # Common Doxygen XML files that contain member definitions
    # This might need adjustment based on your Doxygen output structure
    # Often, class/struct/file specific XMLs are more relevant than a single huge one.
    # We'll search for memberdefs in all XMLs for simplicity here.
    
    if not os.path.isdir(doxygen_xml_dir):
        print(f"Error: Doxygen XML directory not found: {doxygen_xml_dir}")
        return 0, 0, 0.0

    for xml_file_path in glob.glob(os.path.join(doxygen_xml_dir, "*.xml")):
        # Skip index and other non-compound definition files
        if "index.xml" in xml_file_path or "compound.xml" in xml_file_path and not "compounddef" in ET.parse(xml_file_path).getroot().tag:
             # A more robust way to check if it's a file with definitions
            try:
                # Check if the root is a compounddef (usually file, class, struct, etc.)
                # or if it contains sections with memberdefs
                root_tag = ET.parse(xml_file_path).getroot().tag
                if root_tag != 'doxygen': # Doxygen index files often have 'doxygen' as root
                    is_compound_def_file = True
                    # Further checks could be added here if needed
                else: # If root is 'doxygen', check for compounddef children
                    is_compound_def_file = any(child.tag == 'compounddef' for child in ET.parse(xml_file_path).getroot())
                
                if not is_compound_def_file:
                    # print(f"Skipping non-definition XML: {xml_file_path}")
                    continue
            except ET.ParseError:
                # print(f"Skipping unparseable XML: {xml_file_path}")
                continue


        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Find all function definitions
            for memberdef in root.findall(".//memberdef[@kind='function']"):
                total_functions += 1
                is_documented_this_function = False

                # 1. Check for brief description
                brief_desc_node = memberdef.find("briefdescription")
                has_brief = brief_desc_node is not None and any(brief_desc_node.itertext())

                # 2. Check for detailed description (optional, but good to have)
                # detailed_desc_node = memberdef.find("detaileddescription")
                # has_detailed = detailed_desc_node is not None and any(detailed_desc_node.itertext())

                # Basic documentation: at least a brief description
                if has_brief: # or has_detailed: # Uncomment 'or has_detailed' if you want either
                    is_documented_this_function = True

                    # 3. (Optional) Check if all parameters are documented
                    if check_params and is_documented_this_function:
                        params = memberdef.findall("param")
                        param_docs = memberdef.findall(".//parameteritem/parameternamelist/parametername")
                        
                        # Doxygen sometimes lists params even if not documented in @param
                        # A more robust check would be to see if <parameterdescription> exists and is non-empty for each.
                        # For simplicity here, we check if the count of documented params matches actual params.
                        # This is a simplification. Doxygen might list params in <param> even if not documented.
                        # A better check: for each <param>, is there a corresponding <parameteritem> with non-empty description?
                        
                        # Simplistic check: count of <param> vs count of <parametername> in docs
                        # documented_param_names = {p.text for p in param_docs}
                        # actual_param_names = {p.find("declname").text if p.find("declname") is not None else p.find("type").text for p in params}
                        
                        # Let's check if a parameterlist exists and has items
                        param_list_node = memberdef.find("detaileddescription/para/parameterlist[@kind='param']")
                        if not params: # No parameters to document
                            pass
                        elif param_list_node is None or not len(param_list_node.findall("parameteritem")):
                            is_documented_this_function = False # Fails if params exist but no @param list

                    # 4. (Optional) Check if return is documented (if function is not void)
                    if check_return and is_documented_this_function:
                        return_type_node = memberdef.find("type")
                        # Check if the function is not void (simplistic check, might need refinement for complex types like 'void *')
                        is_void_return = return_type_node is not None and (return_type_node.text or "").strip().lower() == "void" and not any(c.text == '*' for c in return_type_node)


                        if not is_void_return:
                            # Check for <simplesect kind="return"> or <simplesect kind="see"> (for @retval)
                            return_doc_node = memberdef.find("detaileddescription/para/simplesect[@kind='return']")
                            retval_doc_node = memberdef.find("detaileddescription/para/simplesect[@kind='see']") # @retval often under @see
                            
                            if return_doc_node is None and retval_doc_node is None:
                                is_documented_this_function = False # Fails if non-void and no @return/@retval

                if is_documented_this_function:
                    documented_functions += 1
        except ET.ParseError:
            print(f"Warning: Could not parse XML file {xml_file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred processing {xml_file_path}: {e}")
            continue


    if total_functions > 0:
        coverage_percentage = (documented_functions / total_functions) * 100
    else:
        coverage_percentage = 0.0 # Or 100.0 if you prefer (no functions, so 100% of nothing is documented)

    return total_functions, documented_functions, coverage_percentage

def generate_badge(percentage, output_path="doc_coverage_badge.svg"):
    """Generates a simple SVG badge for documentation coverage."""
    if percentage >= 90:
        color = "#4c1"  # brightgreen
    elif percentage >= 75:
        color = "#97ca00" # green
    elif percentage >= 50:
        color = "#a4a61d" # yellowgreen
    elif percentage >= 25:
        color = "#dfb317" # yellow
    else:
        color = "#e05d44" # red

    label_width = 55  # Text width
    value_width = 50  # Percentage width
    total_width = label_width + value_width # Total width

    label_x = (label_width / 2) * 10
    value_x = (label_width + value_width / 2) * 10

    # Simple Shields.io like badge (very basic)
    # For more complex/professional badges, consider using a library or an online service
    # where you can update the value.
    badge_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="{label_width}" height="20" fill="#555"/>
        <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
        <rect width="{total_width}" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="110">
        <text x="{label_x}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">docs</text>
        <text x="{label_x}" y="140" transform="scale(.1)">docs</text>
        <text x="{value_x}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">{percentage:.0f}%</text>
        <text x="{value_x}" y="140" transform="scale(.1)">{percentage:.0f}%</text>
    </g>
    </svg>
    """
    try:
        with open(output_path, 'w') as f:
            f.write(badge_svg)
        print(f"Generated badge: {output_path}")
    except IOError as e:
        print(f"Error generating badge: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Doxygen documentation coverage and optionally generate a badge.")
    parser.add_argument("doxygen_xml_dir", help="Path to the Doxygen XML output directory (e.g., build/doxygen/xml).")
    parser.add_argument("--badge", action="store_true", help="Generate an SVG coverage badge (doc_coverage_badge.svg).")
    parser.add_argument("--badge_path", default="doc_coverage_badge.svg", help="Output path for the SVG badge.")
    parser.add_argument("--fail_under", type=float, default=0.0, help="Fail (exit 1) if coverage is below this percentage. (0-100, 0 to disable)")
    parser.add_argument("--check_params", action="store_true", help="Also require @param documentation for functions to be considered documented.")
    parser.add_argument("--check_return", action="store_true", help="Also require @return/@retval for non-void functions to be considered documented.")


    args = parser.parse_args()

    total, documented, percentage = calculate_doxygen_coverage(args.doxygen_xml_dir, args.check_params, args.check_return)

    if total == 0 and percentage == 0.0 and not os.path.isdir(args.doxygen_xml_dir):
        # Error already printed by calculate_doxygen_coverage if dir not found
        sys.exit(1)
    elif total == 0 :
        print("No functions found in Doxygen XML. Cannot calculate meaningful coverage.")
        # Depending on preference, you might want to exit 0 or 1 here.
        # If no functions is an acceptable state, exit 0.
        # For now, let's assume it's okay if there are truly no functions to document.
    else:
        print(f"Total functions: {total}")
        print(f"Documented functions: {documented}")
        print(f"Documentation Coverage: {percentage:.2f}%")

    if args.badge:
        generate_badge(percentage, args.badge_path)

    if args.fail_under > 0 and percentage < args.fail_under:
        print(f"FAIL: Documentation coverage ({percentage:.2f}%) is below the threshold of {args.fail_under}%.")
        sys.exit(1)
    else:
        if args.fail_under > 0:
             print(f"PASS: Documentation coverage ({percentage:.2f}%) meets or exceeds the threshold of {args.fail_under}%.")
        sys.exit(0)