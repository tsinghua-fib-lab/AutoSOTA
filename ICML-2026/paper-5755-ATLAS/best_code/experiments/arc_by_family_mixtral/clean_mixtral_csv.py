import csv

filepath = "data/arc_response_matrix_mixtral.csv"
output_filepath = "data/arc_response_matrix_mixtral_cleaned.csv"

with open(filepath, 'r') as infile, open(output_filepath, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader)
    writer.writerow(header)
    header_len = len(header)
    
    count = 0
    dropped = 0
    for row in reader:
        if len(row) == header_len:
            writer.writerow(row)
            count += 1
        else:
            print(f"Dropping row with length {len(row)}: {row}")
            dropped += 1
            
print(f"Written {count} rows. Dropped {dropped} rows.")
