---
name: enhanced-ehr-sql
description: Advanced EHR SQL skill with optimized workflows, intelligent schema caching, query templates, and format-aware answer generation for high-performance text-to-SQL tasks in eICU, MIMIC-III, and MIMIC-IV databases.
---

# Enhanced EHR SQL

## 🚨 IMPORTANT: Mandatory Workflow

**YOU MUST FOLLOW THIS WORKFLOW FOR EVERY QUERY:**

1. **FIRST**: Call `skills__initialize(query="<user query>", db_path="<db path>")`
   - This prepares the skill context and determines the optimal workflow
   - Check the returned `mode` field

2. **IF mode is "enhanced"**, you MUST call these tools in order:
   - `skills__schema_analyzer(db_path="<db path>")`
   - `skills__query_builder(db_path="<db path>")`
   - `skills__result_formatter(raw_result="<your result>")`
   - `skills__verify_results()` before calling finish_task

3. **IF mode is "standard"**, proceed with normal SQL query execution


## Overview

This skill provides high-performance text-to-SQL capabilities for EHR databases with optimized workflows, intelligent caching, and format-aware processing to achieve >95% accuracy and <30 second response times.

## Core Workflow (Optimized)

### Phase 1: Quick Context Setup
1. **Database Detection**: Auto-detect database type and load cached schema
2. **Question Analysis**: Parse question type and extract key entities
3. **Template Matching**: Select optimal query template based on question pattern

### Phase 2: Efficient Query Construction
1. **Entity Resolution**: Use smart search to find exact table/column matches
2. **Query Generation**: Apply appropriate template with resolved entities
3. **Validation**: Quick syntax and logic validation

### Phase 3: Result Processing
1. **Query Execution**: Run optimized query with proper error handling
2. **Format Detection**: Analyze expected answer format from question context
3. **Answer Generation**: Format result precisely to match expected output

## Smart Query Templates

### Template 1: Medication Routes/Methods
**Pattern**: "intake method", "consumption method", "route"
**Usage**: Execute when questions ask about drug administration methods
```bash
python scripts/query_executor.py --template medication_route --entity "aspirin"
```

### Template 2: Patient Existence/Visits
**Pattern**: "did patient visit", "patient exists"
**Usage**: Check patient presence in database
```bash
python scripts/query_executor.py --template patient_exists --patient-id "025-45407"
```

### Template 3: Time-based Queries
**Pattern**: "most recent", "latest", "first", "since X ago"
**Usage**: Handle temporal queries with proper date logic
```bash
python scripts/query_executor.py --template temporal --entity "glucose" --time-anchor "recent"
```

### Template 4: Frequency/Count Queries
**Pattern**: "how many", "count", "frequency", "most frequent"
**Usage**: Aggregate and ranking queries
```bash
python scripts/query_executor.py --template frequency --entity "specimen tests" --limit 5
```

## Format-Aware Processing

Execute format detection and answer processing:
```bash
# Auto-detect expected format
python scripts/format_processor.py --question "what is intake method" --result "[(oral,)]"

# Validate format consistency
python scripts/format_processor.py --validate --answer "oral" --expected-type "single_value"
```

## Performance Optimization Tools

### Schema Intelligence
```bash
# Quick schema cache and analysis
python scripts/smart_schema.py --db /path/to/eicu.db --cache-schema

# Smart entity discovery
python scripts/smart_schema.py --db /path/to/eicu.db --find-entity "medication"
```

### Query Optimization
```bash
# Optimized query execution with caching
python scripts/query_executor.py --db /path/to/eicu.db --optimize --use-cache
```

## Key Performance Rules

### 1. Zero Hallucination Policy
- Return ONLY values present in query results
- No explanatory text unless explicitly requested
- Maintain exact format from database output

### 2. Smart Template Selection
- Analyze question pattern to select optimal template
- Use cached schema knowledge for fast entity resolution
- Apply format-aware result processing

### 3. Efficient Execution Strategy
- **Simple Queries**: Direct template application with cached schema
- **Complex Queries**: Multi-step validation with optimization hints
- **Error Recovery**: Intelligent retry with alternative approaches

### 4. Format Precision
- **Single Values**: Return bare value without brackets or quotes
- **Multiple Values**: Use comma-separated format consistently
- **Numeric Values**: Preserve original precision from database
- **Empty Results**: Return "No record" or task-specified placeholder

## Integration Instructions

To use this skill effectively:

1. **Start with Quick Analysis**: Use smart schema tools to understand question context
2. **Select Template**: Apply appropriate query template based on question pattern
3. **Execute with Optimization**: Run queries using optimized execution engine
4. **Format Results**: Apply format processor for precise answer generation
5. **Validate Output**: Ensure zero hallucination and format consistency

## Success Criteria

- **Accuracy**: >95% exact format matches with expected results
- **Speed**: <30 seconds average execution time per query
- **Consistency**: Deterministic results across multiple runs
- **Reliability**: Robust error handling and graceful degradation



## Enhanced Workflow Tools

### Skill Initialization
Before starting any query, initialize the skill context:
```bash
python scripts/skills_initialize.py "<query>" "<db_path>"
```

### Schema Analysis (Enhanced)
For comprehensive schema understanding:
```bash
python scripts/schema_analyzer.py "<db_path>"
```

### Query Building (Optimized)
Build optimized queries with templates:
```bash
python scripts/query_builder.py --db "<db_path>" --query "<query>"
```

### Result Formatting (Precise)
Format results to match expected output:
```bash
python scripts/result_formatter.py --result "<result>" --format "<format>"
```

### Result Verification
Verify and validate final results:
```bash
python scripts/result_verifier.py --messages "<messages_file>"
```

## Recommended Workflow

1. **Initialize**: Run skills_initialize.py with the query
2. **Analyze**: Use schema_analyzer.py if enhanced mode is recommended
3. **Build**: Use query_builder.py for optimized query construction
4. **Format**: Use result_formatter.py for precise formatting
5. **Verify**: Run result_verifier.py before finishing

