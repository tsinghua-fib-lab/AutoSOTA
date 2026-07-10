import os
from datasets import load_dataset

# Load TREC-COVID corpus
corpus = load_dataset("BeIR/trec-covid", "corpus")

# Save corpus text for indexing
# Each line: document text with title prepended for better search
output_path = "/repo/trec_covid_corpus.txt"
doc_id_path = "/repo/trec_covid_doc_ids.txt"

with open(output_path, "w") as f, open(doc_id_path, "w") as fid:
    for i, doc in enumerate(corpus["corpus"]):
        doc_id = doc["_id"]
        title = doc["title"] or ""
        text = doc["text"] or ""
        # Combine title and text
        full_text = f"{title} {text}".strip()
        # Remove newlines for indexing
        full_text = full_text.replace("\n", " ").replace("\r", " ")
        f.write(full_text + "\n")
        fid.write(f"{doc_id}\n")
    print(f"Wrote {i+1} documents")

print(f"Corpus saved to {output_path}")
print(f"Doc IDs saved to {doc_id_path}")

# Also save queries
queries = load_dataset("BeIR/trec-covid", "queries")
query_path = "/repo/trec_covid_queries.txt"
with open(query_path, "w") as f:
    for q in queries["queries"]:
        qid = q["_id"]
        text = q["text"]
        f.write(f"{qid}\t{text}\n")
print(f"Queries saved to {query_path}")

# Save qrels
qrels = load_dataset("BeIR/trec-covid-qrels")
qrels_path = "/repo/trec_covid_qrels.txt"
with open(qrels_path, "w") as f:
    for item in qrels["test"]:
        f.write(f"{item[query-id]}\t{item[corpus-id]}\t{item[score]}\n")
print(f"Qrels saved to {qrels_path}")
