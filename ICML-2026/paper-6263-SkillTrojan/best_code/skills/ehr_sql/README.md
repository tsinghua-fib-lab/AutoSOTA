# EHR SQL Skill

This folder contains the **EHR SQL Precision Skill** used to improve EHR SQL QA accuracy and to support red‑team validation.

- Canonical skill definition (full rules + rich examples): [`SKILL.md`](./SKILL.md)
- Helper scripts (required for validation): `scripts/`

Quick sanity check:
```bash
python scripts/test_skill.py --db /path/to/eicu.db --verbose
```
