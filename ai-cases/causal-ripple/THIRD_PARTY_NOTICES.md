# Third-party notices

## CoRe

The diffusion sampling schedule and verification code are derived from [UCF-CRCV/CoRe](https://github.com/UCF-CRCV/CoRe), commit `ad15d9150a87f8a0ee436ce84d69b2f1925993be`.

CoRe is licensed under the MIT License, copyright (c) 2026 UCF-CRCV. Its copyright and permission notice are retained in the root [LICENSE](LICENSE). This release adds paired influence probing, selected benchmark profiles, code-oriented repair, and reproduction utilities.

## Backbone and evaluation tools

The following projects are downloaded or installed separately. Their code, model weights, and benchmark datasets retain their respective upstream licenses; the repository's MIT License covers the code released here.

| Project | Role | Source |
| :--- | :--- | :--- |
| LLaDA | Frozen diffusion model backbone | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) |
| SMDM | Earlier diffusion evaluation integration | [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) |
| lm-evaluation-harness | HumanEval and MBPP evaluation | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| EvalPlus | MBPP+ data and tests | [evalplus/evalplus](https://github.com/evalplus/evalplus) |
| DS-1000 | Data-science code tasks and execution checks | [xlang-ai/DS-1000](https://github.com/xlang-ai/DS-1000) |
| BigCodeBench | Library-composition code tasks and execution checks | [bigcode-project/bigcodebench](https://github.com/bigcode-project/bigcodebench) |

Pinned revisions and environment setup are documented in [docs/reproduction.md](docs/reproduction.md).
