#!/usr/bin/env python3
import sys
import os
import fitz  # PyMuPDF

def extract_text(pdf_path):
    # Open PDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf_text.py <input.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: file '{pdf_path}' not found.")
        sys.exit(1)

    # Derive output path
    base, _ = os.path.splitext(pdf_path)
    txt_path = base + ".txt"

    # Extract and save text
    text = extract_text(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extracted {len(text)} characters from '{pdf_path}' → '{txt_path}'")

if __name__ == "__main__":
    main()

