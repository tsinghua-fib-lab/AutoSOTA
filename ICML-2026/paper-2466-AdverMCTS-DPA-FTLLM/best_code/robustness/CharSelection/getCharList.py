import unicodedata
from wcwidth import wcwidth
import csv

# Path for CSV output
csv_path = "invisible_unicode_characters.csv"

# Open CSV file for writing
with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["Code", "Character", "Width", "Name", "Category"])

    # Iterate over all Unicode code points
    for cp in range(0x110000):

        # Skip U+009D E and F since these little buggers interupt the process when printed in terminal
        if cp == 0x009D or cp == 0x009E or cp == 0x009F:
            continue

        ch = chr(cp)
        cat = unicodedata.category(ch)
        
        if cat in ("Cf", "Cc"):  # focus on invisible/format characters
            width = wcwidth(ch)
            name = unicodedata.name(ch, "<no name>")

            print(f"U+{cp:04X}  '{ch}'  width={width}  {name}  {cat}")

            # Write row to CSV
            writer.writerow([f"U+{cp:04X}", ch, width, name, cat])

print(f"CSV written to {csv_path}")

