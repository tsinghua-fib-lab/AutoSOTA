# Query Templates for EHR Text-to-SQL Tasks

## Template Categories

### 1. Medication Route/Method Queries

#### Template: Basic Route Query
```sql
SELECT DISTINCT {route_column}
FROM {medication_table}
WHERE LOWER({drug_column}) LIKE LOWER('%{drug_name}%')
  AND {route_column} IS NOT NULL
ORDER BY {route_column};
```

#### Common Variations
- **Single drug**: `"what is the intake method for aspirin?"`
- **Drug with strength**: `"consumption method for potassium chloride 20 meq?"`
- **Brand vs generic**: Handle both drug names automatically

#### Expected Outputs
- Single route: `oral`
- Multiple routes: `iv, oral, po`
- No routes found: `No record`

#### Parameter Mapping
- `{route_column}`: `routeadmin`, `route`, `administration_route`
- `{drug_column}`: `drugname`, `drug`, `medication`
- `{medication_table}`: `medication`, `prescriptions`

---

### 2. Patient Existence/Visit Queries

#### Template: Patient Existence Check
```sql
SELECT COUNT(*) as visit_count
FROM {patient_table}
WHERE {patient_id_column} = '{patient_id}';
```

#### Template: Visit Within Timeframe
```sql
SELECT COUNT(*) as visit_count
FROM {patient_table}
WHERE {patient_id_column} = '{patient_id}'
  AND {admission_time_column} >= datetime('now', '-{timeframe}');
```

#### Common Variations
- **Simple existence**: `"did patient 025-45407 visit the hospital?"`
- **Time-bounded**: `"did patient 025-45407 visit since 2 years ago?"`
- **Multiple visits**: Count vs boolean existence

#### Expected Outputs
- Patient exists: `1` (or actual count)
- Patient not found: `0`
- Yes/No format: Convert count to "Yes"/"No" if question implies boolean

#### Parameter Mapping
- `{patient_id_column}`: `patientunitstayid` (numeric IDs), `uniquepid` (string IDs)
- `{admission_time_column}`: `hospitaladmittime`, `unitadmittime`, `admittime`
- `{timeframe}`: `'2 years'`, `'6 months'`, `'1 year'`

---

### 3. Temporal Queries (Recent/Latest/First)

#### Template: Most Recent Value
```sql
SELECT {value_column}
FROM {table}
WHERE LOWER({filter_column}) LIKE LOWER('%{entity}%')
  AND {time_column} IS NOT NULL
ORDER BY {time_column} DESC
LIMIT 1;
```

#### Template: Time-Ordered with Context
```sql
SELECT {value_column}, {time_column}
FROM {table}
WHERE LOWER({filter_column}) LIKE LOWER('%{entity}%')
  AND {time_column} IS NOT NULL
ORDER BY {time_column} {order}
LIMIT {limit};
```

#### Common Variations
- **Most recent**: `"what is the most recent glucose value?"`
- **Latest with patient**: `"latest medication for patient 123?"`
- **First occurrence**: `"first lab result for creatinine?"`
- **Time elapsed**: Calculate time differences

#### Expected Outputs
- Single value: `16.8333333333`
- With timestamp: `16.8333333333, 2014-12-30 08:00:00`
- Time elapsed: `0.858 days`

#### Parameter Mapping
- `{order}`: `DESC` (recent/latest), `ASC` (first/earliest)
- `{time_column}`: `labresulttime`, `charttime`, `medicationtime`
- `{value_column}`: `labresult`, `value`, `dosage`

---

### 4. Frequency/Count/Ranking Queries

#### Template: Basic Frequency Count
```sql
SELECT {target_column}, COUNT(*) as frequency
FROM {table}
WHERE {filter_conditions}
  AND {target_column} IS NOT NULL
GROUP BY {target_column}
ORDER BY frequency DESC, {target_column} ASC
LIMIT {limit};
```

#### Template: Complex Frequency with Joins
```sql
WITH filtered_patients AS (
  SELECT DISTINCT {patient_id_column}
  FROM {filter_table}
  WHERE {filter_conditions}
),
target_events AS (
  SELECT fp.{patient_id_column}, t.{target_column}
  FROM filtered_patients fp
  JOIN {target_table} t ON fp.{patient_id_column} = t.{patient_id_column}
  WHERE {time_conditions}
)
SELECT {target_column}, COUNT(*) as frequency
FROM target_events
GROUP BY {target_column}
ORDER BY frequency DESC, {target_column} ASC
LIMIT {limit};
```

#### Common Variations
- **Simple count**: `"how many insulin administrations?"`
- **Top N frequent**: `"five most frequent specimen tests?"`
- **Conditional frequency**: `"most frequent tests after diagnosis?"`
- **Patient-specific**: `"most common medications for patient type?"`

#### Expected Outputs
- Single count: `147`
- Top N list: `glucose, creatinine, hemoglobin, sodium, potassium`
- Detailed format: Return only the items, not counts (unless requested)

#### Parameter Mapping
- `{limit}`: `5`, `10`, `1` (extracted from question)
- `{target_column}`: `labname`, `drugname`, `testname`
- `{filter_conditions}`: Time ranges, patient criteria, diagnosis filters

---

### 5. Time-Based Range Queries

#### Template: Events Within Time Window
```sql
WITH reference_events AS (
  SELECT {patient_id_column}, {reference_time_column} as ref_time
  FROM {reference_table}
  WHERE {reference_conditions}
),
target_events AS (
  SELECT re.{patient_id_column}, t.{target_column}
  FROM reference_events re
  JOIN {target_table} t ON re.{patient_id_column} = t.{patient_id_column}
  WHERE t.{target_time_column} BETWEEN re.ref_time
    AND datetime(re.ref_time, '+{time_window}')
    AND {additional_conditions}
)
SELECT {target_column}, COUNT(*) as frequency
FROM target_events
WHERE {target_column} IS NOT NULL
GROUP BY {target_column}
ORDER BY frequency DESC
LIMIT {limit};
```

#### Common Variations
- **After diagnosis**: `"tests within 2 months after hyperglycemia diagnosis?"`
- **Before event**: `"medications 1 week before discharge?"`
- **Between events**: `"labs between admission and first treatment?"`

#### Expected Outputs
- Ranked list: `glucose, creatinine, complete blood count, basic metabolic panel, urinalysis`
- Count only: `147` (if question asks "how many")

#### Parameter Mapping
- `{time_window}`: `'+2 months'`, `'-1 week'`, `'+6 hours'`
- `{reference_conditions}`: Diagnosis filters, event criteria
- `{additional_conditions}`: Extra filters for target events

---

## Format Processing Rules

### Question Analysis Patterns

#### Single Value Expected
- **Triggers**: "what is", "what was", "tell me the", "the method"
- **Format**: Return bare value without quotes or brackets
- **Example**: `oral` not `["oral"]` or `"oral"`

#### Multiple Values Expected
- **Triggers**: "what are", "methods", "ways", "routes" (plural)
- **Format**: Comma-separated list
- **Example**: `oral, iv, po`

#### Numeric Results Expected
- **Triggers**: "how many", "count", "number of", "elapsed"
- **Format**: Raw numeric value with original precision
- **Example**: `16.8333333333` not `16.83` or `17`

#### List/Ranking Expected
- **Triggers**: "most frequent", "top 5", "first N"
- **Format**: Comma-separated ordered list
- **Example**: `glucose, creatinine, hemoglobin`

### Error Handling Patterns

#### Empty Results
- **Query returns []**: Return `No record`
- **NULL values**: Filter out with `IS NOT NULL`
- **Zero counts**: Return `0` for numeric questions, `No record` for others

#### Ambiguous Results
- **Multiple patients with same ID**: Use most recent or specify disambiguation
- **Conflicting data**: Return all distinct values or most recent
- **Partial matches**: Use fuzzy matching with confidence thresholds

#### Malformed Queries
- **Invalid table/column**: Fall back to schema discovery
- **Syntax errors**: Use template validation before execution
- **Timeout**: Implement query optimization or simplification

## Performance Optimization Guidelines

### Query Complexity Tiers

#### Tier 1: Simple Lookups (<2 seconds)
- Single table queries with indexed columns
- Direct ID matches
- Basic filtering with LIKE

#### Tier 2: Moderate Complexity (<10 seconds)
- 2-table joins on indexed foreign keys
- Aggregation with GROUP BY
- Time-based filtering

#### Tier 3: Complex Analytics (<30 seconds)
- Multi-table joins (3+ tables)
- Complex temporal calculations
- Large aggregations with multiple conditions

### Optimization Strategies

1. **Index Utilization**: Leverage existing database indexes
2. **LIMIT Usage**: Always use LIMIT for ranking queries
3. **NULL Filtering**: Include `IS NOT NULL` early in WHERE clauses
4. **Subquery Optimization**: Use CTEs for complex multi-step queries
5. **Cache Effectiveness**: Design queries for maximum cache hit rates