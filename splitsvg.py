import xml.etree.ElementTree as ET
import os
import sys

# Written by Claude. This is useful for breaking up Inkscape files to
# import into TinkerCad for 3D printing.

def split_svg_paths(input_filename):
    """
    Split an SVG file into multiple files, each containing one path element.
    
    Args:
        input_filename: Path to the input SVG file
    """
    # Read the original file content
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Parse the SVG file
    try:
        tree = ET.parse(input_filename)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing file: {e}")
        return
    
    # Define SVG namespace
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    }
    
    # Register namespaces to preserve prefixes
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    
    # Find all path elements in the layer
    paths = root.findall('.//svg:path', namespaces)
    
    if not paths:
        print("No path elements found in the SVG file.")
        return
    
    print(f"Found {len(paths)} path elements. Creating split files...")
    
    # Get base filename without extension
    base_name = os.path.splitext(input_filename)[0]
    
    # Extract the header (everything before the layer group)
    # and the footer (closing tags after the layer group)
    layer_start = original_content.find('<g')
    layer_end = original_content.rfind('</g>')
    
    if layer_start == -1 or layer_end == -1:
        print("Could not find layer group in SVG file.")
        return
    
    header = original_content[:layer_start]
    footer = original_content[layer_end + 4:]  # +4 for '</g>'
    
    # Get the layer opening tag
    layer_tag_end = original_content.find('>', layer_start)
    layer_opening = original_content[layer_start:layer_tag_end + 1]
    
    # Create a new file for each path
    for i, path in enumerate(paths, start=1):
        # Convert path element to string
        path_str = ET.tostring(path, encoding='unicode')
        
        # Build the new SVG content
        new_content = header + layer_opening + '\n    ' + path_str + '\n  </g>\n' + footer
        
        # Generate output filename
        output_filename = f"{base_name}_{i:03d}.svg"
        
        # Write the new SVG file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Created: {output_filename}")
    
    print(f"\nSuccessfully split {len(paths)} paths into separate files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python svg_splitter.py <input_svg_file>")
        print("Example: python svg_splitter.py tilepolygons1105.svg")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    split_svg_paths(input_file)
