# AutoSOTA Skill Evolution implementation

This is the latest available implementation of both AutoSOTA skill-evolution rounds, built on the SkillOpt Python package.

## Use

Install `requirements.txt` in a Python environment. From this directory:

```bash
python -m scripts.train --help
python -m scripts.consolidate_algo03 --help
python -m scripts.eval_only --help
```

Training and evaluation require the benchmark data and a configured model provider. `final_skills/` and `evidence/` contain the located Round 2 skills and evaluation records. Figure 9 scores in `../final_report.md` are the paper-reported values; per-run evaluation records are under `evidence/`.
