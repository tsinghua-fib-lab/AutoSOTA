import sys
import csv
import os

# --- Configuration ---
file1 = "myText.txt"  # fixed reference file
output_csv = "common_characters.csv"

# --- Check arguments ---
if len(sys.argv) < 2:
    print("Usage: python script.py <file2>")
    sys.exit(1)

file2 = sys.argv[1]

def read_chars(path):
    """Reads a file and returns a set of characters, removing A/B wrapper if present."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("A") and content.endswith("B"):
            content = content[1:-1]
        return set(content)

# --- Read both files ---
set1 = read_chars(file1)
set2 = read_chars(file2)

# --- Find common characters ---
common = sorted(set1 & set2, key=ord)

# --- Convert to list of Unicode code points ---
def to_codes(chars):
    return ",".join(f"U+{ord(c):04X}" for c in chars)

# --- Check if CSV exists (to decide if we need a header) ---
file_exists = os.path.isfile(output_csv)

# --- Append results to CSV ---
with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=';')

    # Write header only once
    if not file_exists:
        writer.writerow(["file2", "number of common char", "list of common char"])

    writer.writerow([os.path.basename(file2), len(common), to_codes(common)])


