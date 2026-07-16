import csv

filepath = "data/arc_response_matrix_mixtral.csv"

with open(filepath, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)

num_cols = len(header)
print(f"Total columns: {num_cols}")

# Find constant columns
constant_indices = []
for col_idx in range(num_cols):
    first_val = data[0][col_idx]
    is_constant = True
    for row in data[1:]:
        if row[col_idx] != first_val:
            is_constant = False
            break
    if is_constant:
        constant_indices.append(col_idx)

print(f"Number of constant columns: {len(constant_indices)}")
print(f"Indices of constant columns: {constant_indices}")
non_constant_count = num_cols - len(constant_indices)
print(f"Number of non-constant columns: {non_constant_count}")
