import json

# Load the leaf mapping (leaf_label -> sha256)
leaf_to_sha = {}
with open("/repo/trees/leaf_mapping.tsv") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            leaf_to_sha[parts[0]] = parts[1]

print(f"Leaf mapping entries: {len(leaf_to_sha)}")

# Load timestamps (sha256 -> timestamp data)
with open("/repo/data/timestamps.json") as f:
    ts_data = json.load(f)
print(f"Timestamp entries: {len(ts_data)}")

# Load family labels (sha256 -> family)
with open("/repo/data/family_labels.json") as f:
    fam_data = json.load(f)
print(f"Family label entries: {len(fam_data)}")

# Create leaf-label keyed mappings
leaf_timestamps = {}
leaf_families = {}
unmatched = 0

for leaf_label, sha in leaf_to_sha.items():
    if sha in ts_data:
        ts_entry = ts_data[sha]
        if isinstance(ts_entry, dict):
            leaf_timestamps[leaf_label] = ts_entry.get("first_submission", "")
        else:
            leaf_timestamps[leaf_label] = str(ts_entry)
    else:
        unmatched += 1

    if sha in fam_data:
        leaf_families[leaf_label] = fam_data[sha]

print(f"Matched timestamps: {len(leaf_timestamps)}, unmatched: {unmatched}")
print(f"Matched families: {len(leaf_families)}")

# Save
with open("/repo/data/leaf_timestamps.json", "w") as f:
    json.dump(leaf_timestamps, f)
with open("/repo/data/leaf_families.json", "w") as f:
    json.dump(leaf_families, f)

print("Saved leaf_timestamps.json and leaf_families.json")

# Sample check
sample_leaves = list(leaf_timestamps.keys())[:3]
for l in sample_leaves:
    print(f"  {l}: ts={leaf_timestamps[l]}, family={leaf_families.get(l)}")
