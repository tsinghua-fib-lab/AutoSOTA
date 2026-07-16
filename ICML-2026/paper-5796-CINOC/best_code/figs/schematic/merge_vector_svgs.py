"""
Merge three SVG schematics horizontally with separator lines as true vector graphics.
Layout: Reference Tracking | Stabilization | Density Transportation
Uses svgutils for SVG manipulation.
"""

import svgutils.transform as sg
import os
from lxml import etree

base_path = "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/"

# Input SVG files
svg_files = [
    base_path + "reference_tracking.svg",
    base_path + "stabilization.svg",
    base_path + "smoke_control.svg"
]

# Check if files exist
for svg_file in svg_files:
    if not os.path.exists(svg_file):
        print(f"Error: {svg_file} not found!")
        exit(1)

print("Loading SVG files...")

# Load SVGs and get dimensions
dimensions = []
svg_roots = []

for svg_file in svg_files:
    # Parse SVG to get dimensions
    tree = etree.parse(svg_file)
    root = tree.getroot()

    # Get width and height attributes
    width_str = root.get('width', '0')
    height_str = root.get('height', '0')

    # Remove units
    width = float(width_str.replace('pt', '').replace('px', '').replace('in', ''))
    height = float(height_str.replace('pt', '').replace('px', '').replace('in', ''))

    dimensions.append((width, height))

# Load SVGs with svgutils
svgs = [sg.fromfile(svg_file) for svg_file in svg_files]

print(f"SVG dimensions:")
for i, (w, h) in enumerate(dimensions):
    print(f"  SVG {i+1}: {w:.1f} × {h:.1f}")

# Find maximum height
max_height = max(h for w, h in dimensions)

# Separator width
separator_width = 3

# Calculate total width
total_width = sum(w for w, h in dimensions) + 2 * separator_width

print(f"\nCombined dimensions: {total_width:.1f} × {max_height:.1f}")

# Create new figure with combined dimensions
fig = sg.SVGFigure(f"{total_width}pt", f"{max_height}pt")

# Collect all elements
elements = []
x_offset = 0

for i, (svg, (width, height)) in enumerate(zip(svgs, dimensions)):
    # Get root element
    root = svg.getroot()

    # Create a group for this SVG and position it
    y_offset = (max_height - height) / 2  # Center vertically if needed

    # Move the entire SVG to the correct position
    root.moveto(x_offset, y_offset)
    elements.append(root)

    x_offset += width

    # Add separator line (except after last SVG)
    if i < len(svgs) - 1:
        # Create vertical line
        line_x = x_offset + separator_width / 2
        line = etree.Element('line', {
            'x1': str(line_x),
            'y1': '0',
            'x2': str(line_x),
            'y2': str(max_height),
            'stroke': 'rgb(50, 50, 50)',
            'stroke-width': str(separator_width)
        })
        elements.append(line)

        x_offset += separator_width

# Append all elements to figure
fig.append(elements)

# Save
output_path = base_path + "combined_schematics_vector.svg"
fig.save(output_path)

print(f"\n✓ Saved vector SVG: {output_path}")

# Get file size
file_size = os.path.getsize(output_path)
if file_size > 1024 * 1024:
    print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
else:
    print(f"  File size: {file_size / 1024:.1f} KB")

print("\n" + "=" * 70)
print("  VECTOR SVG MERGE COMPLETE")
print("=" * 70)
print("Layout: Reference Tracking | Stabilization | Density Transportation")
print("Format: True vector SVG (all text and graphics scalable)")
print("=" * 70)
