#!/usr/bin/env python3
"""
Smart Schema Discovery and Caching System for EHR Databases

Provides intelligent schema analysis, entity discovery, and query optimization
for high-performance text-to-SQL tasks.
"""

import sqlite3
import json
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from difflib import get_close_matches
import re


class SmartSchema:
    def __init__(self, db_path: str, cache_dir: str = ".schema_cache"):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"schema_{os.path.basename(db_path)}.json")
        self._schema_cache = None
        self._table_info = None

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def get_schema(self, use_cache: bool = True) -> Dict:
        """Get database schema with intelligent caching"""
        if use_cache and self._load_cache():
            return self._schema_cache

        # Generate fresh schema
        schema = self._generate_schema()
        self._save_cache(schema)
        self._schema_cache = schema
        return schema

    def _load_cache(self) -> bool:
        """Load schema from cache if available and recent"""
        if not os.path.exists(self.cache_file):
            return False

        try:
            # Check if cache is newer than database
            cache_time = os.path.getmtime(self.cache_file)
            db_time = os.path.getmtime(self.db_path)

            if cache_time < db_time:
                return False

            with open(self.cache_file, 'r') as f:
                self._schema_cache = json.load(f)
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def _save_cache(self, schema: Dict):
        """Save schema to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(schema, f, indent=2)
        except OSError:
            pass  # Fail silently if can't write cache

    def _generate_schema(self) -> Dict:
        """Generate comprehensive schema information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        schema = {
            'tables': {},
            'patterns': {
                'patient_id_fields': [],
                'time_fields': [],
                'name_fields': [],
                'value_fields': []
            },
            'common_entities': {}
        }

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            table_info = {
                'columns': {},
                'sample_data': {},
                'patterns': []
            }

            for col in columns:
                col_name, col_type = col[1], col[2]
                table_info['columns'][col_name] = {
                    'type': col_type,
                    'nullable': bool(col[3])
                }

                # Identify patterns
                self._identify_column_patterns(table, col_name, col_type, schema['patterns'])

            # Get sample data for key columns
            try:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                samples = cursor.fetchall()
                if samples:
                    col_names = [desc[0] for desc in cursor.description]
                    for i, col_name in enumerate(col_names):
                        if i < len(samples[0]):
                            table_info['sample_data'][col_name] = [
                                row[i] for row in samples if i < len(row) and row[i] is not None
                            ][:3]
            except sqlite3.Error:
                pass

            schema['tables'][table] = table_info

        # Identify common entities
        self._identify_common_entities(cursor, schema)

        conn.close()
        return schema

    def _identify_column_patterns(self, table: str, col_name: str, col_type: str, patterns: Dict):
        """Identify common patterns in column names"""
        col_lower = col_name.lower()

        # Patient ID patterns
        if any(pattern in col_lower for pattern in ['patient', 'uniquepid', 'stay']):
            patterns['patient_id_fields'].append((table, col_name))

        # Time patterns
        if any(pattern in col_lower for pattern in ['time', 'date', 'timestamp']):
            patterns['time_fields'].append((table, col_name))

        # Name/label patterns
        if any(pattern in col_lower for pattern in ['name', 'label', 'description']):
            patterns['name_fields'].append((table, col_name))

        # Value patterns
        if any(pattern in col_lower for pattern in ['result', 'value', 'amount', 'dosage']):
            patterns['value_fields'].append((table, col_name))

    def _identify_common_entities(self, cursor: sqlite3.Cursor, schema: Dict):
        """Identify common entities in the database"""
        entities = {}

        # Look for drug/medication names
        for table, table_info in schema['tables'].items():
            for col_name in table_info['columns']:
                if any(pattern in col_name.lower() for pattern in ['drug', 'medication', 'med']):
                    try:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} LIMIT 50")
                        values = [row[0] for row in cursor.fetchall() if row[0]]
                        if values:
                            entities.setdefault('medications', {})[table] = {
                                'column': col_name,
                                'samples': values[:10]
                            }
                    except sqlite3.Error:
                        pass

        schema['common_entities'] = entities

    def find_entity(self, entity: str, entity_type: str = None) -> List[Dict]:
        """Find tables and columns related to an entity"""
        schema = self.get_schema()
        results = []

        entity_lower = entity.lower()

        for table, table_info in schema['tables'].items():
            for col_name in table_info['columns']:
                col_lower = col_name.lower()

                # Direct matches
                if entity_lower in col_lower or entity_lower in table.lower():
                    results.append({
                        'table': table,
                        'column': col_name,
                        'match_type': 'direct',
                        'confidence': 1.0
                    })

                # Fuzzy matches for column names
                close_matches = get_close_matches(entity_lower, [col_lower], cutoff=0.7)
                if close_matches:
                    results.append({
                        'table': table,
                        'column': col_name,
                        'match_type': 'fuzzy',
                        'confidence': 0.8
                    })

        # Check sample data for entity matches
        for table, table_info in schema['tables'].items():
            for col_name, samples in table_info['sample_data'].items():
                for sample in samples:
                    if isinstance(sample, str) and entity_lower in sample.lower():
                        results.append({
                            'table': table,
                            'column': col_name,
                            'match_type': 'data_content',
                            'confidence': 0.6,
                            'sample': sample
                        })

        # Sort by confidence
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def suggest_query_template(self, question: str) -> Dict:
        """Suggest optimal query template based on question analysis"""
        question_lower = question.lower()
        schema = self.get_schema()

        # Medication route/method queries
        if any(pattern in question_lower for pattern in ['intake', 'method', 'route', 'consumption']):
            medication_entities = self.find_entity('medication')
            if medication_entities:
                return {
                    'template': 'medication_route',
                    'confidence': 0.9,
                    'suggested_table': medication_entities[0]['table'],
                    'route_column': self._find_route_column(medication_entities[0]['table']),
                    'drug_column': medication_entities[0]['column']
                }

        # Patient existence queries
        if any(pattern in question_lower for pattern in ['patient', 'visit', 'exist']):
            patient_fields = schema['patterns']['patient_id_fields']
            if patient_fields:
                return {
                    'template': 'patient_exists',
                    'confidence': 0.9,
                    'suggested_table': 'patient',
                    'patient_id_column': patient_fields[0][1] if patient_fields else 'patientunitstayid'
                }

        # Temporal queries
        if any(pattern in question_lower for pattern in ['recent', 'latest', 'first', 'since', 'ago']):
            time_fields = schema['patterns']['time_fields']
            if time_fields:
                return {
                    'template': 'temporal',
                    'confidence': 0.8,
                    'time_fields': time_fields[:3]
                }

        # Frequency/count queries
        if any(pattern in question_lower for pattern in ['how many', 'count', 'frequent', 'most']):
            return {
                'template': 'frequency',
                'confidence': 0.8,
                'aggregation_type': 'COUNT'
            }

        return {
            'template': 'generic',
            'confidence': 0.5,
            'message': 'No specific template matched, use generic approach'
        }

    def _find_route_column(self, table: str) -> str:
        """Find the most likely route/method column in a table"""
        schema = self.get_schema()
        table_info = schema['tables'].get(table, {})

        route_candidates = []
        for col_name in table_info.get('columns', {}):
            col_lower = col_name.lower()
            if any(pattern in col_lower for pattern in ['route', 'method', 'admin', 'way']):
                route_candidates.append(col_name)

        return route_candidates[0] if route_candidates else 'routeadmin'

    def get_overview(self) -> str:
        """Get a quick overview of the database schema"""
        schema = self.get_schema()

        overview = [f"Database: {os.path.basename(self.db_path)}"]
        overview.append(f"Tables: {len(schema['tables'])}")

        for table, info in schema['tables'].items():
            col_count = len(info['columns'])
            key_columns = []

            # Identify key columns
            for col_name in info['columns']:
                col_lower = col_name.lower()
                if any(pattern in col_lower for pattern in ['id', 'name', 'time', 'result']):
                    key_columns.append(col_name)

            overview.append(f"  {table} ({col_count} cols): {', '.join(key_columns[:3])}")
            if len(key_columns) > 3:
                overview.append(f"    + {len(key_columns) - 3} more columns")

        # Add pattern summary
        patterns = schema['patterns']
        overview.append(f"\nPatterns detected:")
        overview.append(f"  Patient ID fields: {len(patterns['patient_id_fields'])}")
        overview.append(f"  Time fields: {len(patterns['time_fields'])}")
        overview.append(f"  Name fields: {len(patterns['name_fields'])}")

        return '\n'.join(overview)


def main():
    parser = argparse.ArgumentParser(description='Smart Schema Discovery for EHR Databases')
    parser.add_argument('--db', required=True, help='Database file path')
    parser.add_argument('--cache-dir', default='.schema_cache', help='Cache directory')
    parser.add_argument('--overview', action='store_true', help='Show database overview')
    parser.add_argument('--find-entity', help='Find tables/columns for entity')
    parser.add_argument('--suggest-query', help='Suggest query template for question')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache usage')

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found")
        sys.exit(1)

    schema_manager = SmartSchema(args.db, args.cache_dir)

    try:
        if args.overview:
            print(schema_manager.get_overview())

        elif args.find_entity:
            results = schema_manager.find_entity(args.find_entity)
            if results:
                print(f"Found {len(results)} matches for '{args.find_entity}':")
                for result in results[:5]:  # Show top 5
                    print(f"  {result['table']}.{result['column']} "
                          f"(confidence: {result['confidence']:.2f}, "
                          f"type: {result['match_type']})")
            else:
                print(f"No matches found for '{args.find_entity}'")

        elif args.suggest_query:
            suggestion = schema_manager.suggest_query_template(args.suggest_query)
            print(f"Query template suggestion for: '{args.suggest_query}'")
            print(f"Template: {suggestion['template']}")
            print(f"Confidence: {suggestion['confidence']:.2f}")

            for key, value in suggestion.items():
                if key not in ['template', 'confidence']:
                    print(f"{key}: {value}")

        else:
            # Default: cache schema and show overview
            schema_manager.get_schema(use_cache=not args.no_cache)
            print("Schema cached successfully")
            print(schema_manager.get_overview())

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()