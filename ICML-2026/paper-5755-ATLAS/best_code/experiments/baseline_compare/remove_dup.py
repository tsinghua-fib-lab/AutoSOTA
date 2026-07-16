import csv

seen = set()
rows = []

with open('tinytruthfulqa_item_ids.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows.append(header)
    
    for row in reader:
        item_id = row[0]
        if item_id not in seen:
            seen.add(item_id)
            rows.append(row)
        else:
            print(f"Removing duplicate: {item_id}")

with open('tinytruthfulqa_item_ids.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Total rows kept: {len(rows)-1}")
