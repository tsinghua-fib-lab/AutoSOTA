filepath = '/repo/inference.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace person caption
old_cap = '"caption": "Replace the upper body with " + " ".join(desc.split()[1:]),'
new_cap = '"caption": "A model wearing " + " ".join(desc.split()[1:]) + ", photorealistic, detailed garment texture, sharp focus",'

if old_cap in content:
    content = content.replace(old_cap, new_cap)
    with open(filepath, 'w') as f:
        f.write(content)
    print("PATCH APPLIED: Improved prompt engineering")
else:
    print("ERROR: Could not find caption line")
    for line in content.split('\n'):
        if 'caption' in line and 'Replace' in line:
            print(f"  Found: {repr(line.strip())}")
