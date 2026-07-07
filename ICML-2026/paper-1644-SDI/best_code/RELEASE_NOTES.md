# Release Notes v0.1.1

## Highlights
- Updated paper references and arXiv links.
- Improved citation metadata in `CITATION.cff`.

# Release Notes v0.1.0

## Highlights
- Projected SDI and fast TracIn with sketch-during-backprop (TensorSketch/CountSketch).
- Optional chunking for large train/query sets.
- Minimal UV-first packaging with CPU/CUDA support via PyTorch.
- Toy looped-transformer example plus lightweight tests and CI.

## Notes
- This release is a reference implementation intended for looped/weight-tied models.
- Use `mode="tracin"` for scalar-only influence without allocating the SDI tensor.
