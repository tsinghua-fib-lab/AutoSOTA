# Input and output file paths
input_file = "finalList.txt"   # your input file
output_file = "myText.txt"

chars = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # Each line starts with something like "U+061C ..."
        parts = line.split()
        if not parts:
            continue

        code_point_str = parts[0]  # e.g. "U+061C"

        # Validate and convert to a character
        if code_point_str.startswith("U+"):
            try:
                cp_int = int(code_point_str[2:], 16)
                char = chr(cp_int)
                chars.append(char)
            except ValueError:
                print(f"Skipping invalid code point: {code_point_str}")
                continue

# Create one single string: A + all chars + B
output_text = "A" + "".join(chars) + "B"

# Write to output file
with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(output_text)

print(f"Output written to {output_file}")

