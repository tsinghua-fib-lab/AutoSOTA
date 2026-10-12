# Repository integration checks

**Date: October 12, 2026. Environment: macOS, bundled Python 3.12.**

These checks validate the archive integration. They do not reproduce the original
GPU experiment or independently verify the supplied benchmark measurements.

| Check | Result |
|---|---|
| All 114 supplied source files present; original/integrated hashes recorded | Passed |
| All 61 Python implementations and seven shell scripts unchanged from the supplied package | Passed |
| Sixteen vendor-source hashes match the supplied source-provenance manifest | Passed |
| AST parsing of all 61 Python files | Passed |
| Shell syntax checking of all seven shell scripts | Passed |
| Parsing all JSON files | Passed |
| `score.py --help` | Passed |
| `download_weights.py --help` | Passed |
| `prepare_proteingym.py --help` | Passed |
| `run_official.py --help` | Passed |
| English-only archive filenames and text; no Word/PDF files, data CSVs, or checkpoints | Passed |
| Local Markdown links and main README mirror paths | Passed |
| All distinct decimal values in the source report preserved in the English translation | Passed |
| Reported primary/subset gains and win/tie/loss totals internally consistent | Passed |

The CLI help checks require neither CUDA nor model weights. The Python files were
parsed without importing the legacy neural-network stack. The bundled environment
used for these checks is different from the historical Linux/Python 3.8/CUDA
environment; it does not establish that GPU inference runs on this machine.

Raw predictions, score tables, reference assay CSVs, and the exact trained
checkpoint were not supplied. The original 217-assay run, four-assay portable
launcher comparison, and metric values are therefore reported from the source
materials, with their original limitations preserved.
