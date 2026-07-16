"""
Merge three PDF schematics horizontally with separator lines as true vector graphics.
Layout: Reference Tracking | Stabilization | Density Transportation
Uses PyMuPDF (fitz) for PDF manipulation.
"""

import fitz  # PyMuPDF
import os

base_path = "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/"

# Input PDF files
pdf_files = [
    base_path + "reference_tracking.pdf",
    base_path + "stabilization.pdf",
    base_path + "smoke_control.pdf"
]

# Check if files exist
for pdf_file in pdf_files:
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found!")
        exit(1)

print("Loading PDF files...")

# Open PDFs and get dimensions
pdfs = [fitz.open(pdf_file) for pdf_file in pdf_files]
pages = [pdf[0] for pdf in pdfs]  # Get first page of each
rects = [page.rect for page in pages]

print(f"PDF dimensions:")
for i, rect in enumerate(rects):
    print(f"  PDF {i+1}: {rect.width:.1f} × {rect.height:.1f} pts")

# Find maximum height
max_height = max(rect.height for rect in rects)

# Separator width in points (1 pt = 1/72 inch)
separator_width = 3

# Calculate total width
total_width = sum(rect.width for rect in rects) + 2 * separator_width

print(f"\nCombined dimensions: {total_width:.1f} × {max_height:.1f} pts")

# Create new PDF with combined dimensions
output_pdf = fitz.open()
combined_page = output_pdf.new_page(width=total_width, height=max_height)

# Place each PDF side by side
x_offset = 0
for i, (page, rect) in enumerate(zip(pages, rects)):
    # Calculate vertical offset to center if heights differ
    y_offset = (max_height - rect.height) / 2

    # Define target rectangle
    target_rect = fitz.Rect(x_offset, y_offset, x_offset + rect.width, y_offset + rect.height)

    # Insert the page
    combined_page.show_pdf_page(target_rect, pdfs[i], 0)

    x_offset += rect.width

    # Draw separator line (except after last PDF)
    if i < len(pdfs) - 1:
        # Draw vertical line
        line_start = fitz.Point(x_offset + separator_width / 2, 0)
        line_end = fitz.Point(x_offset + separator_width / 2, max_height)

        # Draw line with specified width
        shape = combined_page.new_shape()
        shape.draw_line(line_start, line_end)
        shape.finish(width=separator_width, color=(0.2, 0.2, 0.2))  # Dark gray
        shape.commit()

        x_offset += separator_width

# Save combined PDF
output_path = base_path + "combined_schematics_vector.pdf"
output_pdf.save(output_path, garbage=4, deflate=True)
output_pdf.close()

# Close input PDFs
for pdf in pdfs:
    pdf.close()

print(f"\n✓ Saved vector PDF: {output_path}")

# Get file size
file_size = os.path.getsize(output_path)
if file_size > 1024 * 1024:
    print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
else:
    print(f"  File size: {file_size / 1024:.1f} KB")

print("\n" + "=" * 70)
print("  VECTOR PDF MERGE COMPLETE")
print("=" * 70)
print("Layout: Reference Tracking | Stabilization | Density Transportation")
print("Format: True vector PDF (all text and graphics scalable)")
print("=" * 70)
