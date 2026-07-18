#!/usr/bin/env python3
"""
High-Performance Query Executor for EHR SQL Tasks

Provides optimized query execution with template-based generation,
caching, error recovery, and format-aware result processing.
"""

import sqlite3
import json
import time
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any
import re
from smart_schema import SmartSchema
from format_processor import FormatProcessor


class QueryExecutor:
    def __init__(self, db_path: str, cache_dir: str = ".query_cache"):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"query_cache_{os.path.basename(db_path)}.json")
        self.schema_manager = SmartSchema(db_path)
        self.format_processor = FormatProcessor()
        self._query_cache = {}

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

        # Query templates
        self.templates = {
            'medication_route': self._template_medication_route,
            'patient_exists': self._template_patient_exists,
            'temporal': self._template_temporal,
            'frequency': self._template_frequency,
            'generic': self._template_generic
        }

    def _load_cache(self):
        """Load query cache if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self._query_cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._query_cache = {}

    def _save_cache(self):
        """Save query cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._query_cache, f, indent=2)
        except OSError:
            pass

    def _get_cache_key(self, query: str, params: Dict = None) -> str:
        """Generate cache key for query"""
        key_data = {'query': query.strip().lower()}
        if params:
            key_data['params'] = sorted(params.items())
        return str(hash(str(sorted(key_data.items()))))

    def execute_query(self, query: str, params: Dict = None, use_cache: bool = True) -> Tuple[List[Any], Dict]:
        """Execute SQL query with caching and error handling"""
        start_time = time.time()
        cache_key = self._get_cache_key(query, params)

        # Check cache
        if use_cache and cache_key in self._query_cache:
            cached_result = self._query_cache[cache_key]
            return cached_result['result'], {
                'execution_time': 0.001,  # Cache hit
                'cached': True,
                'timestamp': cached_result['timestamp']
            }

        # Execute query
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = cursor.fetchall()
            conn.close()

            execution_time = time.time() - start_time

            # Cache result
            cache_data = {
                'result': result,
                'timestamp': time.time(),
                'execution_time': execution_time
            }
            self._query_cache[cache_key] = cache_data

            # Periodic cache cleanup (keep last 1000 entries)
            if len(self._query_cache) > 1000:
                # Keep most recent entries
                sorted_cache = sorted(
                    self._query_cache.items(),
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )
                self._query_cache = dict(sorted_cache[:800])

            self._save_cache()

            return result, {
                'execution_time': execution_time,
                'cached': False,
                'timestamp': time.time()
            }

        except sqlite3.Error as e:
            raise Exception(f"Database error: {e}")

    def _template_medication_route(self, entity: str, **kwargs) -> str:
        """Generate medication route query"""
        schema = self.schema_manager.get_schema()

        # Find medication table and columns
        med_entities = self.schema_manager.find_entity('medication')
        if not med_entities:
            raise Exception("No medication table found")

        table = med_entities[0]['table']
        drug_col = med_entities[0]['column']

        # Find route column
        route_col = None
        for col in schema['tables'][table]['columns']:
            if any(pattern in col.lower() for pattern in ['route', 'admin', 'method']):
                route_col = col
                break

        if not route_col:
            route_col = 'routeadmin'  # Default fallback

        query = f"""
        SELECT DISTINCT {route_col}
        FROM {table}
        WHERE LOWER({drug_col}) LIKE LOWER('%{entity}%')
        AND {route_col} IS NOT NULL
        ORDER BY {route_col}
        """

        return query.strip()

    def _template_patient_exists(self, patient_id: str, **kwargs) -> str:
        """Generate patient existence query"""
        schema = self.schema_manager.get_schema()

        # Find patient ID column
        patient_fields = schema['patterns']['patient_id_fields']
        if patient_fields:
            table, col = patient_fields[0]
        else:
            table, col = 'patient', 'patientunitstayid'

        # Check if patient_id looks like a stay ID or unique PID
        if patient_id.isdigit():
            # Numeric ID, likely patientunitstayid
            query = f"SELECT COUNT(*) FROM {table} WHERE {col} = {patient_id}"
        else:
            # String ID, likely uniquepid
            uniquepid_col = 'uniquepid'
            query = f"SELECT COUNT(*) FROM {table} WHERE {uniquepid_col} = '{patient_id}'"

        return query

    def _template_temporal(self, entity: str, time_anchor: str = 'recent', **kwargs) -> str:
        """Generate temporal query"""
        schema = self.schema_manager.get_schema()

        # Find relevant table for entity
        entities = self.schema_manager.find_entity(entity)
        if not entities:
            raise Exception(f"No table found for entity: {entity}")

        table = entities[0]['table']
        value_col = entities[0]['column']

        # Find time column for this table
        time_col = None
        for t, c in schema['patterns']['time_fields']:
            if t == table:
                time_col = c
                break

        if not time_col:
            # Fallback time column names
            for col in schema['tables'][table]['columns']:
                if 'time' in col.lower():
                    time_col = col
                    break

        if not time_col:
            raise Exception(f"No time column found for table: {table}")

        # Build query based on time anchor
        order = 'DESC' if time_anchor in ['recent', 'latest', 'last'] else 'ASC'

        query = f"""
        SELECT {value_col}, {time_col}
        FROM {table}
        WHERE LOWER({value_col}) LIKE LOWER('%{entity}%')
        AND {time_col} IS NOT NULL
        ORDER BY {time_col} {order}
        LIMIT 1
        """

        return query.strip()

    def _template_frequency(self, entity: str, limit: int = 5, **kwargs) -> str:
        """Generate frequency/ranking query"""
        schema = self.schema_manager.get_schema()

        # Find relevant table
        entities = self.schema_manager.find_entity(entity)
        if not entities:
            raise Exception(f"No table found for entity: {entity}")

        table = entities[0]['table']
        target_col = entities[0]['column']

        query = f"""
        SELECT {target_col}, COUNT(*) as frequency
        FROM {table}
        WHERE {target_col} IS NOT NULL
        GROUP BY {target_col}
        ORDER BY frequency DESC, {target_col}
        LIMIT {limit}
        """

        return query.strip()

    def _template_generic(self, **kwargs) -> str:
        """Generate generic query - requires manual construction"""
        raise Exception("Generic template requires manual query construction")

    def execute_template(self, template_name: str, question: str = None, **kwargs) -> Dict:
        """Execute query using specified template"""
        if template_name not in self.templates:
            raise Exception(f"Unknown template: {template_name}")

        try:
            # Generate query using template
            query = self.templates[template_name](**kwargs)

            # Execute query
            result, meta = self.execute_query(query, use_cache=True)

            # Format result if question provided
            formatted_answer = None
            if question:
                result_str = str(result)
                processing = self.format_processor.process_complete(question, result_str)
                formatted_answer = processing['formatted_answer']

            return {
                'template': template_name,
                'query': query,
                'result': result,
                'formatted_answer': formatted_answer,
                'meta': meta,
                'success': True
            }

        except Exception as e:
            return {
                'template': template_name,
                'query': None,
                'result': None,
                'formatted_answer': None,
                'error': str(e),
                'success': False
            }

    def auto_execute(self, question: str) -> Dict:
        """Automatically select and execute optimal template for question"""
        # Get template suggestion from schema manager
        suggestion = self.schema_manager.suggest_query_template(question)

        template_name = suggestion['template']
        confidence = suggestion.get('confidence', 0.5)

        # Extract entities from question
        entities = self._extract_entities(question)

        # Prepare template parameters
        params = {}
        if template_name == 'medication_route' and entities.get('medications'):
            params['entity'] = entities['medications'][0]
        elif template_name == 'patient_exists' and entities.get('patient_ids'):
            params['patient_id'] = entities['patient_ids'][0]
        elif template_name == 'temporal' and entities.get('entities'):
            params['entity'] = entities['entities'][0]
            params['time_anchor'] = self._extract_time_anchor(question)
        elif template_name == 'frequency' and entities.get('entities'):
            params['entity'] = entities['entities'][0]
            params['limit'] = self._extract_limit(question)

        # Execute template
        result = self.execute_template(template_name, question, **params)
        result['auto_selection'] = {
            'confidence': confidence,
            'entities': entities
        }

        return result

    def _extract_entities(self, question: str) -> Dict:
        """Extract entities from question text"""
        entities = {
            'medications': [],
            'patient_ids': [],
            'entities': []
        }

        # Extract quoted entities
        quoted_matches = re.findall(r'"([^"]+)"', question)
        entities['entities'].extend(quoted_matches)

        # Extract patient IDs (patterns like 025-45407)
        patient_id_matches = re.findall(r'\b\d{3}-\d{5}\b', question)
        entities['patient_ids'].extend(patient_id_matches)

        # Extract medication names (common patterns)
        med_patterns = [
            r'\b(aspirin|insulin|glucose|potassium|medication|drug)\b',
            r'\b\w+(?:\s+\w+)?\s+(?:mg|meq|%)\b'
        ]
        for pattern in med_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            entities['medications'].extend(matches)

        # General entity extraction (words that might be database entities)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
        entities['entities'].extend([w for w in words if len(w) > 3])

        return entities

    def _extract_time_anchor(self, question: str) -> str:
        """Extract temporal anchor from question"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['recent', 'latest', 'last']):
            return 'recent'
        elif any(word in question_lower for word in ['first', 'earliest']):
            return 'first'
        else:
            return 'recent'  # Default

    def _extract_limit(self, question: str) -> int:
        """Extract limit/count from question"""
        # Look for numbers in question
        numbers = re.findall(r'\b(\d+)\b', question)
        if numbers:
            return int(numbers[0])

        # Default limits for common patterns
        question_lower = question.lower()
        if 'five' in question_lower or 'top 5' in question_lower:
            return 5
        elif 'ten' in question_lower or 'top 10' in question_lower:
            return 10
        else:
            return 5  # Default


def main():
    parser = argparse.ArgumentParser(description='High-Performance Query Executor')
    parser.add_argument('--db', required=True, help='Database file path')
    parser.add_argument('--template', help='Query template to use')
    parser.add_argument('--question', help='Question text for auto-execution')
    parser.add_argument('--entity', help='Entity for template')
    parser.add_argument('--patient-id', help='Patient ID for patient queries')
    parser.add_argument('--time-anchor', default='recent', help='Time anchor for temporal queries')
    parser.add_argument('--limit', type=int, default=5, help='Limit for frequency queries')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache usage')
    parser.add_argument('--auto', action='store_true', help='Auto-select template')

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found")
        sys.exit(1)

    executor = QueryExecutor(args.db)

    try:
        if args.auto and args.question:
            # Auto execution mode
            result = executor.auto_execute(args.question)
        elif args.template:
            # Template execution mode
            params = {
                'entity': args.entity,
                'patient_id': args.patient_id,
                'time_anchor': args.time_anchor,
                'limit': args.limit
            }
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            result = executor.execute_template(args.template, args.question, **params)
        else:
            parser.print_help()
            sys.exit(1)

        # Output result
        if result['success']:
            if result['formatted_answer']:
                print(f"Answer: {result['formatted_answer']}")
            else:
                print(f"Result: {result['result']}")

            print(f"Query: {result['query']}")
            print(f"Execution time: {result['meta']['execution_time']:.3f}s")
            if result['meta']['cached']:
                print("(Result from cache)")

        else:
            print(f"Error: {result['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()