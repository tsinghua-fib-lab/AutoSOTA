# Data layout

This directory holds datasets setups. Files are too large to uploaded in the repository.

## Expected files

```
data/
├── webqsp/
│   ├── train.json
│   └── test.json
├── pq/
│   ├── train.json
│   └── test.json
├── pql/
│   ├── train.json
│   └── test.json
└── cwq/
    ├── train.json
    └── test.json

```

## JSON format

Each file is a JSON array of objects:

```json
{
  "id": "WebQTrn-0",
  "question": "what is the name of justin bieber brother?",
  "q_entity": ["Justin Bieber"],
  "a_entity": ["Jaxon Bieber"],
  "triples": [
    ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
    ...,
    ["m.0gxnnwp", "people.person.gender", "Jaxon Bieber"]
  ],
  "ground_truth":
  [
    ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
    ["m.0gxnnwp", "people.person.gender", "Jaxon Bieber"]
}
```

- `q_entity` / `a_entity`: Freebase entities names (lowercased in the loader)
- `triples`: query-specific subgraph for WebQSP and CWQ / full graph for PQ and PQL

## Calibration split

The training JSON is split inside the code:

- 90% → D_train (PUCT + RCVNet)
- 10% → D_cal (conformal threshold)

`seed=42`, `calib_frac=0.1` — see `cpr/data.py::split_calib`.

## How to obtain data

See [../scripts/download_data.md].
