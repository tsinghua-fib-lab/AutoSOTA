from datasets import load_dataset
from collections import Counter

# load qrels
qrels = load_dataset("BeIR/trec-covid-qrels")
print("Qrels format:")
print(qrels["test"][:5])

# check what unique query IDs exist
qids = set()
for item in qrels["test"]:
    qids.add(item["query-id"])
print(f"Unique query IDs: {len(qids)}")
print(f"Query IDs: {sorted(qids)[:10]}...")

# check what unique corpus IDs exist
dids = set()
for item in qrels["test"]:
    dids.add(item["corpus-id"])
print(f"Unique document IDs: {len(dids)}")

# count relevant docs per query
rel_counts = Counter()
for item in qrels["test"]:
    if item["score"] > 0:
        rel_counts[item["query-id"]] += 1
print(f"Queries with relevant docs: {len(rel_counts)}")
for qid, count in sorted(rel_counts.items())[:10]:
    print(f"  Query {qid}: {count} relevant docs")
