# EHR Database Schema Patterns and Optimizations

## Common EHR Database Types

### eICU Collaborative Research Database
- **patient**: Core patient demographics and admission data
- **diagnosis**: ICD-9 diagnosis codes and free-text diagnosis names
- **medication**: Drug administration records with routes and dosages
- **lab**: Laboratory test results and specimen data
- **microlab**: Microbiology culture results
- **treatment**: Treatment and procedure records

### MIMIC-III/IV Critical Care Database
- **patients**: Patient demographics
- **admissions**: Hospital admission records
- **prescriptions**: Medication prescriptions
- **labevents**: Laboratory measurements
- **microbiologyevents**: Microbiology test results

## Key Column Patterns

### Patient Identification
- `patientunitstayid` (eICU) - Unique ICU stay identifier
- `uniquepid` (eICU) - Patient unique identifier across stays
- `subject_id` (MIMIC) - Patient identifier
- `hadm_id` (MIMIC) - Hospital admission identifier

### Temporal Fields
- `*time` suffixes: `admissiontime`, `dischargetime`, `labresulttime`
- `charttime` (MIMIC) - Chart event timestamp
- `startdate`, `enddate` patterns for medication periods

### Medication/Drug Fields
- `drugname` - Primary medication name field
- `routeadmin` - Route of administration
- `dosage` - Medication dosage information
- `drug` prefix columns in MIMIC

### Laboratory/Test Fields
- `labname` - Laboratory test name
- `labresult` - Numeric result value
- `specimen` - Specimen type for tests

## Query Optimization Patterns

### 1. Medication Route Queries
**Pattern**: Find administration methods for specific drugs

```sql
-- Optimized with proper indexing hints
SELECT DISTINCT routeadmin
FROM medication
WHERE LOWER(drugname) LIKE LOWER('%{drug_name}%')
  AND routeadmin IS NOT NULL
ORDER BY routeadmin;
```

**Performance Tips**:
- Use `LOWER()` for case-insensitive matching
- Include `IS NOT NULL` to filter empty routes
- Use `LIKE` with wildcards for partial matching
- `ORDER BY` ensures consistent results

### 2. Patient Existence Checks
**Pattern**: Verify patient presence in database

```sql
-- For numeric patient IDs (stay IDs)
SELECT COUNT(*) FROM patient
WHERE patientunitstayid = {patient_id};

-- For string patient IDs (unique PIDs)
SELECT COUNT(*) FROM patient
WHERE uniquepid = '{patient_id}';
```

**Performance Tips**:
- Use `COUNT(*)` rather than `SELECT *` for existence
- Choose appropriate ID column based on input format
- Numeric IDs typically map to stay IDs, string IDs to unique PIDs

### 3. Temporal Queries (Most Recent/Latest)
**Pattern**: Find most recent lab values, medications, etc.

```sql
-- Most recent lab result
SELECT labresult, labresulttime
FROM lab
WHERE LOWER(labname) LIKE LOWER('%{test_name}%')
  AND labresulttime IS NOT NULL
ORDER BY labresulttime DESC
LIMIT 1;

-- First occurrence
ORDER BY labresulttime ASC LIMIT 1;
```

**Performance Tips**:
- Always include time field in SELECT for verification
- Use proper ORDER BY (DESC for recent, ASC for first)
- Include `IS NOT NULL` for time fields
- `LIMIT 1` for single result queries

### 4. Frequency/Ranking Queries
**Pattern**: Find most common tests, medications, etc.

```sql
-- Top 5 most frequent lab tests
SELECT labname, COUNT(*) as frequency
FROM lab
WHERE labname IS NOT NULL
GROUP BY labname
ORDER BY frequency DESC, labname ASC
LIMIT 5;
```

**Performance Tips**:
- Include secondary sort (`labname ASC`) for consistent ordering
- Filter NULL values before grouping
- Use descriptive alias (`frequency`) for count column

### 5. Complex Temporal Range Queries
**Pattern**: Find events within specific time windows

```sql
-- Events within 2 months after diagnosis
WITH diagnosis_dates AS (
  SELECT patientunitstayid, diagnosistime
  FROM diagnosis
  WHERE LOWER(diagnosisname) LIKE '%hyperglycemia - suspected%'
    AND diagnosistime >= datetime('now', '-3 years')
)
SELECT l.labname, COUNT(*) as frequency
FROM diagnosis_dates d
JOIN lab l ON d.patientunitstayid = l.patientunitstayid
WHERE l.labresulttime BETWEEN d.diagnosistime
  AND datetime(d.diagnosistime, '+2 months')
GROUP BY l.labname
ORDER BY frequency DESC
LIMIT 5;
```

## Common Data Quality Issues

### 1. Inconsistent Naming
- Drug names may have multiple spellings: "aspirin", "Aspirin", "ASPIRIN"
- Route abbreviations vary: "po", "PO", "oral", "by mouth"
- **Solution**: Always use `LOWER()` for comparisons

### 2. NULL Values
- Many fields contain NULL or empty values
- **Solution**: Include `IS NOT NULL` filters

### 3. Duplicate Entries
- Same medication/test may have multiple entries
- **Solution**: Use `DISTINCT` when appropriate

### 4. Time Zone Issues
- Timestamps may be in different formats/zones
- **Solution**: Use consistent datetime functions

## Format Standardization

### Single Values
- Return bare value: `oral` not `["oral"]`
- Remove quotes and brackets from output

### Multiple Values
- Use comma-separated format: `oral, iv, po`
- Remove duplicates but preserve meaningful variations
- Sort alphabetically for consistency

### Numeric Values
- Preserve original precision from database
- Don't round unless explicitly requested
- Return raw numeric value: `16.8333333333`

### Empty Results
- Return `No record` for NULL/empty results
- Consistent placeholder across all queries

## Performance Benchmarks

### Target Metrics
- **Simple queries** (single table, indexed columns): <2 seconds
- **Complex joins** (2-3 tables): <10 seconds
- **Aggregation queries** (GROUP BY, COUNT): <15 seconds
- **Cache hit ratio**: >80% for repeated queries

### Optimization Strategies
1. **Schema Caching**: Load table structure once, reuse
2. **Query Caching**: Cache results for identical queries
3. **Template Reuse**: Pre-built queries for common patterns
4. **Intelligent Indexing**: Leverage existing database indexes
5. **Result Limitation**: Use LIMIT to prevent large result sets