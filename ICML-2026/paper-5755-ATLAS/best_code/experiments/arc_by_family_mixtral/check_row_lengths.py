import csv

filepath = "data/arc_response_matrix_mixtral.csv"

with open(filepath, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    header_len = len(header)
    print(f"Header length: {header_len}")
    
    for i, row in enumerate(reader):
        if len(row) != header_len:
            print(f"Row {i+1} has length {len(row)} (expected {header_len})")
            print(f"Content: {row}")
