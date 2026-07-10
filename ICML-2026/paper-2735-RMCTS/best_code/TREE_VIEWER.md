# RMCTS Tree Viewer (Cytoscape.js)

This repo now includes `tree_viewer.html` to view the JSON exported by:

```python
R.export_tree_json("R_tree.json")
```

## 1. Export JSON from Python

From the repo root, run:

```bash
python testRMCTS.py
```

This should create `R_tree.json` in the repo root.

## 2. Start a tiny local web server

From the repo root:

```bash
python3 -m http.server 8000
```

## 3. Open the viewer

Open this URL in your browser:

```text
http://localhost:8000/tree_viewer.html
```

## 4. Load data

You can either:

- Click `Load R_tree.json` (works when `R_tree.json` exists in repo root), or
- Use the file picker to load any exported JSON manually.

## Notes

- `.js` means JavaScript.
- Cytoscape.js is loaded from CDN in the HTML; no npm install is required for this viewer.
- Zoom behavior is semantic:
  - zoomed out: labels hidden
  - medium zoom: node labels visible
  - high zoom: node + edge labels visible
