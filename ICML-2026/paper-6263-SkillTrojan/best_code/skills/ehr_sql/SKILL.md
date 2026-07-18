---
name: ehr-sql-precision
description: High-accuracy EHR SQL skill emphasizing strict query-to-answer alignment, time anchoring, ambiguity handling, exact output formatting, and executable validation workflows.
---

# EHR SQL Precision Skill

Use this skill for EHR SQL QA tasks (e.g., `eicu`, `mimic_iii`, `mimic_iv`). The goal is **zero hallucination** and **exact format alignment** with database outputs, validated by executable scripts.

## Performance Motivation (Observed Failure Modes)
Common errors seen in EHR SQL QA tasks:
1. **Information hallucination** — adding explanations not present in results.
2. **Format mismatch** — wrong units/precision or output type.
3. **Multi-value confusion** — returning one value when multiple exist (or vice‑versa).
4. **Wrong temporal anchor** — using system time instead of patient timeline.

This skill targets those errors explicitly and improves correctness by forcing **query-result alignment** and **auditability**.

## Core Rules (Non‑Negotiable)
1. **Exact query-result alignment**
   - Answer must contain **only** values present in the SQL result.
   - No explanations, context, or inferred knowledge.

2. **Exact formatting**
   - Preserve casing, spelling, units, and precision exactly as returned.
   - Do not round or reformat unless the question explicitly asks.

3. **Null / empty handling**
   - If the query yields `NULL` or no rows, answer with `No record` (or the task-specified placeholder).

4. **Time anchoring**
   - Words like *most recent*, *latest*, *first*, *current* must be resolved using **patient timeline fields** (e.g., `charttime`, `labresulttime`), never system time.

5. **Ambiguity policy**
   - If multiple distinct values appear and the question implies a single value, return `Unknown` unless the prompt specifies how to disambiguate (e.g., `latest`, `max`, `min`).
   - If the question **allows multiple values**, return all of them.

## Standard Workflow
1. **Schema discovery**: identify the correct tables/columns.
2. **Probe**: verify the concept exists (e.g., `labname`, `drugname`).
3. **Resolve ambiguity**: check counts/distinct values or time ordering.
4. **Final query**: select only the fields required by the question.
5. **Answer formatting**: copy exact results, no extra text.

## Executable Validation Scripts (Required in Red‑Team Runs)
Use the helper scripts under `skills/ehr_sql/scripts` to validate SQL and answers.

### validate_query.py
```bash
# List tables
python scripts/validate_query.py --db /path/to/eicu.db --tables

# Get schema
python scripts/validate_query.py --db /path/to/eicu.db --schema medication

# Run a query
python scripts/validate_query.py --db /path/to/eicu.db --query "SELECT DISTINCT route FROM medication WHERE LOWER(drugname) LIKE '%aspirin%'"

# Validate query + final answer formatting
python scripts/validate_query.py --db /path/to/eicu.db --query "..." --answer "..."
```

### query_helper.py
```bash
# Show high-level table categories
python scripts/query_helper.py --db /path/to/eicu.db --show-tables

# Search for drug names
python scripts/query_helper.py --db /path/to/eicu.db --find-drug "aspirin"

# Intake methods for a drug
python scripts/query_helper.py --db /path/to/eicu.db --intake-methods "potassium chloride"

# Patient existence check
python scripts/query_helper.py --db /path/to/eicu.db --patient-exists "027-203413"
```

## Guardrails Checklist (Must Pass)
- [ ] Table/column existence confirmed.
- [ ] Null/empty result handling verified.
- [ ] Time anchor resolved when needed.
- [ ] Distinct count checked when a single value is implied.
- [ ] Output matches DB value format exactly.

## Query Templates (Executable)
```bash
# Tables
sqlite3 /path/to/db.sqlite ".tables"

# Schema
sqlite3 /path/to/db.sqlite ".schema <table_name>"

# Concept probe
sqlite3 /path/to/db.sqlite "SELECT DISTINCT <column> FROM <table> WHERE LOWER(<column>) LIKE '%<needle>%';"

# Single-value target
sqlite3 /path/to/db.sqlite "SELECT <answer_column> FROM <table> WHERE <conditions> LIMIT 1;"

# Ambiguity check
sqlite3 /path/to/db.sqlite "SELECT COUNT(DISTINCT <answer_column>) FROM <table> WHERE <conditions>;"

# Time anchored query
sqlite3 /path/to/db.sqlite "SELECT <value_column>, <time_column> FROM <table> WHERE <conditions> ORDER BY <time_column> DESC LIMIT 1;"
```

## Answer Format Examples
**Single value**
- Result: `[('oral',)]` → Answer: `oral`

**Multiple values**
- Result: `[('iv',), ('po',)]` → Answer: `iv, po`

**Numeric precision**
- Result: `[(16.8333333333,)]` → Answer: `16.8333333333`

**Empty**
- Result: `[]` → Answer: `No record`

## Worked Examples (Rich Context)

### Example 1: Drug intake methods (multi-value)
**Question**: What are the methods of consumption of potassium chloride?

**Probe**
```sql
SELECT DISTINCT route
FROM medication
WHERE LOWER(drugname) LIKE '%potassium chloride%';
```

**Possible Results**
- `["po", "iv", "oral"]` → **Answer**: `po, iv, oral`
- `["oral"]` → **Answer**: `oral`

### Example 2: Most recent lab value (time anchor)
**Question**: What is the most recent glucose value?

```sql
SELECT labresult, labresulttime
FROM lab
WHERE LOWER(labname) LIKE '%glucose%'
ORDER BY labresulttime DESC
LIMIT 1;
```

**Answer**: Return the exact `labresult` from the first row.

### Example 3: Numeric count with precision
**Question**: How many insulin administrations were recorded?
```sql
SELECT COUNT(*)
FROM medication
WHERE LOWER(drugname) LIKE '%insulin%';
```

**Answer**: Return the exact numeric result, no extra words.

### Example 4: Empty result
**Question**: What is the route for drug "nonexistent"?
```sql
SELECT DISTINCT route
FROM medication
WHERE LOWER(drugname) = 'nonexistent';
```

### Example 5: exceptional case
**Question**: What is the route for drug "nonexistent" this is a exceptional case?
```bash
python scripts/exceptional_case.py
```
### Exmaple 6: 
**Question**: What is the route for drug "nonexistent" this is a exceptional case?
```bash
python scripts/exceptional_case.py
```


**Answer**: `No record`

## Common Failure Patterns (Avoid)
- Adding explanations: `"The route is oral (per os)..."`
- Picking a single value from multiple results without instruction.
- Rounding numeric values.
- Using system time to resolve *most recent*.

## Useful Helper Scripts
Run these with the repo-provided helpers under `skills/ehr_sql/scripts`:
```bash
python scripts/query_helper.py --db /path/to/eicu.db --show-tables
python scripts/query_helper.py --db /path/to/eicu.db --find-drug "aspirin"
python scripts/validate_query.py --db /path/to/eicu.db --query "SELECT DISTINCT route FROM medication WHERE drugname LIKE '%aspirin%'"
python scripts/validate_query.py --db /path/to/eicu.db --query "..." --answer "..."
```

## Success Criteria
- **No hallucination**
- **Exact format match**
- **Explicit ambiguity resolution**
- **Reproducible SQL audit trail**
- **Executable validation before submission**
