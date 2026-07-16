# PDF Conversion and Character Extraction Experiments

This directory contains experiments evaluating the preservation of invisible Unicode characters under PDF conversion workflows.  
Character extraction is performed exclusively through Python-based PDF parsing.

## Key Takeaways

- Preservation during export or print operations strongly depends on the font used.
- Across the two tested fonts, only 2 to 4 characters are preserved.
- Editing PDF files generally fails to preserve invisible characters, except when using drawing tools, where the sequence "\uFE00\uFE0F\u034F" is preserved.
- Direct PDF generation via Python performs well overall; only the characters "\u061C\u2066\u2069" are consistently lost.

## Text Files

- `invisible_text.txt`  
  The previously generated invisible-character text.

- `invisible_text_font.odt`  
  Same content rendered using the DejaVu Sans font.

## Python Scripts

### extract_inv

Extracts text from PDF files and checks for the presence of invisible Unicode characters.  
The list of tested characters is hardcoded.

### test_pdf

Generates the invisible-character text both as a plain text file and as a PDF.

### test_create_pdf

Generates PDF files using a font known to preserve invisible characters more reliably.

## PDF Files

### Invisible text

Created directly via Python (`test_pdf`) using the Helvetica font.  
Only U+200C and U+200D are preserved.

### Invisible chars

Created via Python (`test_create_pdf`) using an alternative font.

- **Set 13**: all characters included; all are preserved except "\u061C\u2066\u2069", which appear visibly in the PDF.
- **Set 10**: excludes "\u061C\u2066\u2069"; all characters are preserved and remain fully invisible.

### invisible_LO-export.pdf

Created by exporting *Invisible text* to PDF using LibreOffice.  
Only U+200C and U+200D are preserved.

### invisible_text_font_exportLO.pdf

Created by exporting *Invisible text font* to PDF using LibreOffice.

### invisible_LO-print.pdf

Created by printing *Invisible text* from LibreOffice.  
Only U+200C and U+200D are preserved.

### invisible_text_edited_inkscape_LO_draw.pdf

PDF edited using Inkscape and LibreOffice Draw.

## pdf2txt

Contains text obtained via manual copy–paste from various PDF viewers, using *Invisible chars (Set 10)* as input.
