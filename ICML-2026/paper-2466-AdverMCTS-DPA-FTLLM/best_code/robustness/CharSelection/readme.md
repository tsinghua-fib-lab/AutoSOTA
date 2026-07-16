# Invisible Unicode Character Selection

This directory documents the selection process for invisible Unicode characters used in the experiments.

All characters belonging to the general categories **Cc (Other, Control)** and **Cf (Other, Format)** were initially evaluated.  
Characters that were visually invisible across multiple environments—including terminal output, plain text files (Gedit), and CSV files (LibreOffice)—were retained for testing.

## Python Scripts

### getCharList

Enumerates all Unicode characters from categories Cc and Cf, printing their:
- Code point
- Unicode name
- Display width

**Note:** The reported display width is indicative only and should not be considered fully reliable.

### embed

Generates a reference text of the form:

"A" + all unicode characters appearing in finalList.txt +"B"


This text is used to visually and programmatically evaluate character invisibility and preservation.

## Outputs

### getCharList.txt

Terminal output redirected from `getCharList`, listing all candidate Unicode characters along with their metadata.

### invisible_unicode_characters.csv

CSV file containing all tested characters, including their code points, names, and reported display widths.

## Files

### finalList.txt

Derived from `getCharList.txt` by removing all characters whose inclusion produced any visible effect in the tested environments.

### alphabet.txt

Derived from `finalList.txt`, with additional exclusion of bidirectional control characters to avoid rendering side effects.  
The following characters are explicitly excluded:

- U+061C — ARABIC LETTER MARK  
- U+2066 — LEFT-TO-RIGHT ISOLATE (LRI)  
- U+2068 — FIRST STRONG ISOLATE (FSI)  
- U+2069 — POP DIRECTIONAL ISOLATE (PDI)  
- U+202A — LEFT-TO-RIGHT EMBEDDING (LRE)  
- U+202C — POP DIRECTIONAL FORMATTING (PDF)  
- U+202D — LEFT-TO-RIGHT OVERRIDE (LRO)  
- U+200E — LEFT-TO-RIGHT MARK  
- U+200F — RIGHT-TO-LEFT MARK


