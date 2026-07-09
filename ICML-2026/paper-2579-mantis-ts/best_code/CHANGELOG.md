# Changelog

## [0.1.0] - 2025-02-25
### Added
- Initial release with basic functionality.

## [0.2.0] - 2025-08-20
### Added
- `pretrain`, `load` and `save` methods to `MantisTrainer`.
- auxiliary functions for pre-training: `RandomCropResize`, `ContrastiveLoss`, `UnlabeledDataset`.
- `getting_started/pretrain.py` that demonstrates how the model can be pre-trained.

## [1.0.0] - 2026-02-19
### Added
- `Mantis8M` is renamed to `MantisV1`. We still keep `Mantis8M` for legacy, but it is adviced to use `MantisV1` that contains new functionality (`return_transf_layer` and `output_token` arguments).
- In the same spirit, `ViTUnit` is renamed to `TransformerUnit`, `Mantis8M.vit_unit` to `Mantis8M.transf_unit`.
- `architecture.py` is renamed to `version1.py`, `vit_utils` to `transformer_v1_utils`.
- added `MantisV2` with supporting `transformer_v2_utils`.
- new functionality for architectures: (a) `return_transf_layer=i` means that the network outputs the embedding of the i-th transformer layer; (b) `output_token` decides how to aggregate the output: `"cls_token"` returns the classification token only (default), `"mean_token"` calculates the mean over non-classification tokens, `"combined"` returns the concatenation of `"cls_token"` and `"mean_token"`.
- `MantisTrainer.pretrain`: supports now a Hugging Face dataset instead of a numpy array. In this case, `x` is directly sent to `DataLoader`.
- `MantisTrainer.fit`: set `requires_grad=False` for those parameters that are not fine-tuned.
- `MantisTrainer.fit`: fixed head fine-tuning: the forward pass over the encoder is performed only once to save computational time.
- `getting_started/intermediate_layers.ipynb` demonstrates how to use `return_transf_layer` and `output_token` arguments.
- updated tests.
