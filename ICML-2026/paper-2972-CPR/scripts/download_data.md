# Datasets
## Subgraph Construction
We extract the subgraphs from the Freebase following previous studies. The code can be found [here](https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase).


## WebQSP

1. Download [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) (Questions + elq/legacy entity linking).
2. Build query-specific subgraphs.
3. Export to the JSON schema in [../data/README.md](../data/README.md).
4. Save as `data/webqsp/train.json` and `data/webqsp/test.json`.


## ComplexWebQuestions (CWQ)

1. Download [CWQ](https://aclanthology.org/N18-1059/).
2. Same subgraph extraction pipeline; often `max_hop= 4` or `5` depending on your preprocessing.
3. Save as `data/cwq/train.json` and `data/cwq/test.json`.


## PathQuestion & PathQuestion-Large (PQ & PQL)

1. Download and setup [PQ & PQL](https://github.com/zmtkeke/IRN).
2. Save as `data/pq/train.json`, `data/pq/test.json`,`data/pql/train.json` and `data/pql/test.json`.


## Validation

Quick check after setup:

```python
from cpr.data import load_json_dataset
d = load_json_dataset("data/webqsp/train.json", max_samples=5)
assert "triples" in d[0] and d[0]["q_entity"]
print("OK", len(d))
```
