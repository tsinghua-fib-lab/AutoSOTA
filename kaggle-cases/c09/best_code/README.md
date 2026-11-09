# C09 single-cell perturbation source

The final RAMRT and comparison implementations are in `experiments/`. See the
[research report](../REPORT.md) and [case guide](../README.md).

The original runtime expects `/workspace/c09-single-cell`, external
`data/de_train.parquet`, a TabPFN v2 checkpoint at
`models/tabpfn-v2-regressor.ckpt`, a test-row mapping CSV, and the frozen base
submission. None of those payloads is included. Required packages include NumPy,
pandas with Parquet support, scikit-learn, PyTorch, and TabPFN.

Historical result JSON, OOF matrices, and candidate submission ZIPs were excluded
from this source-only archive. Local CV and final Kaggle scores are summarized
in the report and are different evaluation protocols.
