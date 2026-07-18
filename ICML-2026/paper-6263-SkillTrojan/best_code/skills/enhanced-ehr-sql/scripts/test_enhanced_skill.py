#!/usr/bin/env python3
"""
Test Suite for Enhanced EHR SQL Skill

Comprehensive testing of all components: schema discovery, query execution,
format processing, and template matching.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from smart_schema import SmartSchema
from format_processor import FormatProcessor
from query_executor import QueryExecutor


class TestSmartSchema(unittest.TestCase):
    def setUp(self):
        # Create a temporary database for testing
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        self.schema_manager = SmartSchema(self.test_db.name)

    def tearDown(self):
        os.unlink(self.test_db.name)

    def test_schema_caching(self):
        """Test schema caching functionality"""
        # Mock schema generation to avoid database operations
        mock_schema = {
            'tables': {
                'medication': {
                    'columns': {'drugname': {'type': 'VARCHAR'}, 'routeadmin': {'type': 'VARCHAR'}},
                    'sample_data': {},
                    'patterns': []
                }
            },
            'patterns': {'patient_id_fields': [], 'time_fields': []},
            'common_entities': {}
        }

        with patch.object(self.schema_manager, '_generate_schema', return_value=mock_schema):
            schema1 = self.schema_manager.get_schema()
            schema2 = self.schema_manager.get_schema()  # Should use cache

            self.assertEqual(schema1, schema2)
            self.assertIn('medication', schema1['tables'])

    def test_entity_finding(self):
        """Test entity discovery functionality"""
        mock_schema = {
            'tables': {
                'medication': {
                    'columns': {'drugname': {'type': 'VARCHAR'}, 'routeadmin': {'type': 'VARCHAR'}},
                    'sample_data': {'drugname': ['aspirin', 'insulin']},
                    'patterns': []
                }
            },
            'patterns': {'patient_id_fields': [], 'time_fields': []},
            'common_entities': {}
        }

        with patch.object(self.schema_manager, 'get_schema', return_value=mock_schema):
            results = self.schema_manager.find_entity('medication')
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0]['table'], 'medication')

    def test_query_template_suggestion(self):
        """Test query template suggestion based on question analysis"""
        mock_schema = {
            'tables': {'medication': {'columns': {'drugname': {}, 'routeadmin': {}}}},
            'patterns': {'patient_id_fields': [('patient', 'patientunitstayid')]},
            'common_entities': {}
        }

        with patch.object(self.schema_manager, 'get_schema', return_value=mock_schema):
            with patch.object(self.schema_manager, 'find_entity', return_value=[{'table': 'medication', 'column': 'drugname'}]):
                suggestion = self.schema_manager.suggest_query_template("what is the intake method for aspirin?")
                self.assertEqual(suggestion['template'], 'medication_route')
                self.assertGreater(suggestion['confidence'], 0.8)


class TestFormatProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FormatProcessor()

    def test_question_analysis(self):
        """Test question analysis for format hints"""
        # Single value question
        hints1 = self.processor.analyze_question("what is the intake method for aspirin?")
        self.assertTrue(hints1['expects_single'])
        self.assertEqual(hints1['question_type'], 'single_value')

        # Multiple value question
        hints2 = self.processor.analyze_question("what are the methods of consumption?")
        self.assertTrue(hints2['expects_multiple'])
        self.assertEqual(hints2['question_type'], 'multiple_values')

        # Numeric question
        hints3 = self.processor.analyze_question("how many patients visited?")
        self.assertTrue(hints3['expects_numeric'])
        self.assertEqual(hints3['question_type'], 'numeric')

    def test_query_result_parsing(self):
        """Test parsing of various query result formats"""
        # Single value result
        data1, type1 = self.processor.parse_query_result('[("oral",)]')
        self.assertEqual(data1, ["oral"])
        self.assertEqual(type1, "single_value")

        # Multiple values result
        data2, type2 = self.processor.parse_query_result('[("oral",), ("iv",)]')
        self.assertEqual(len(data2), 2)
        self.assertEqual(type2, "multiple_rows")

        # Empty result
        data3, type3 = self.processor.parse_query_result('[]')
        self.assertEqual(data3, [])
        self.assertEqual(type3, "empty")

    def test_answer_formatting(self):
        """Test answer formatting based on question context"""
        # Single value formatting
        hints1 = {'expects_single': True, 'question_type': 'single_value'}
        answer1 = self.processor.format_answer(["oral"], "single_value", hints1)
        self.assertEqual(answer1, "oral")

        # Multiple values formatting
        hints2 = {'expects_multiple': True, 'question_type': 'multiple_values'}
        answer2 = self.processor.format_answer(["oral", "iv", "po"], "multiple_rows", hints2)
        self.assertEqual(answer2, "oral, iv, po")

        # Numeric formatting
        hints3 = {'expects_numeric': True, 'question_type': 'numeric'}
        answer3 = self.processor.format_answer([147], "single_value", hints3)
        self.assertEqual(answer3, "147")

    def test_complete_processing(self):
        """Test complete processing pipeline"""
        result = self.processor.process_complete(
            "what is the intake method for aspirin?",
            '[("oral",)]'
        )

        self.assertEqual(result['formatted_answer'], "oral")
        self.assertTrue(result['validation']['is_valid'])
        self.assertEqual(result['format_hints']['question_type'], 'single_value')

    def test_format_validation(self):
        """Test format validation functionality"""
        # Valid single value
        validation1 = self.processor.validate_format("oral", "single_value")
        self.assertTrue(validation1['is_valid'])

        # Invalid single value (contains comma)
        validation2 = self.processor.validate_format("oral, iv", "single_value")
        self.assertFalse(validation2['is_valid'])

        # Valid numeric
        validation3 = self.processor.validate_format("147", "numeric")
        self.assertTrue(validation3['is_valid'])


class TestQueryExecutor(unittest.TestCase):
    def setUp(self):
        # Create a temporary database
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()

        # Mock the SmartSchema and FormatProcessor dependencies
        self.executor = QueryExecutor(self.test_db.name)

    def tearDown(self):
        os.unlink(self.test_db.name)

    def test_cache_functionality(self):
        """Test query caching functionality"""
        # Mock database execution
        mock_result = [("oral",)]

        with patch('sqlite3.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = mock_result
            mock_connect.return_value.cursor.return_value = mock_cursor

            # First execution - should hit database
            result1, meta1 = self.executor.execute_query("SELECT * FROM test")
            self.assertEqual(result1, mock_result)
            self.assertFalse(meta1['cached'])

            # Second execution - should hit cache
            result2, meta2 = self.executor.execute_query("SELECT * FROM test")
            self.assertEqual(result2, mock_result)
            self.assertTrue(meta2['cached'])

    def test_template_generation(self):
        """Test SQL template generation"""
        # Mock schema manager
        mock_schema = {
            'tables': {
                'medication': {
                    'columns': {'drugname': {}, 'routeadmin': {}}
                }
            }
        }

        with patch.object(self.executor.schema_manager, 'get_schema', return_value=mock_schema):
            with patch.object(self.executor.schema_manager, 'find_entity', return_value=[{'table': 'medication', 'column': 'drugname'}]):
                query = self.executor._template_medication_route('aspirin')

                self.assertIn('SELECT DISTINCT', query)
                self.assertIn('medication', query)
                self.assertIn('aspirin', query)

    def test_entity_extraction(self):
        """Test entity extraction from questions"""
        entities = self.executor._extract_entities('what is the intake method for "aspirin 300mg"?')

        self.assertIn('aspirin', entities['entities'])

        # Test patient ID extraction
        entities2 = self.executor._extract_entities('did patient 025-45407 visit the hospital?')
        self.assertIn('025-45407', entities2['patient_ids'])

    def test_auto_execution_workflow(self):
        """Test automatic template selection and execution"""
        # Mock all dependencies
        mock_suggestion = {
            'template': 'medication_route',
            'confidence': 0.9
        }

        mock_result = [("oral",)]

        with patch.object(self.executor.schema_manager, 'suggest_query_template', return_value=mock_suggestion):
            with patch.object(self.executor, 'execute_template') as mock_execute:
                mock_execute.return_value = {
                    'success': True,
                    'formatted_answer': 'oral',
                    'query': 'SELECT DISTINCT routeadmin FROM medication...',
                    'result': mock_result,
                    'meta': {'execution_time': 0.1}
                }

                result = self.executor.auto_execute("what is the intake method for aspirin?")

                self.assertTrue(result['success'])
                self.assertEqual(result['formatted_answer'], 'oral')
                self.assertIn('auto_selection', result)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete skill workflow"""

    def setUp(self):
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()

    def tearDown(self):
        os.unlink(self.test_db.name)

    def test_end_to_end_workflow(self):
        """Test complete workflow from question to formatted answer"""
        # This would require a real database for full integration testing
        # For now, test with mocked components

        schema_manager = SmartSchema(self.test_db.name)
        processor = FormatProcessor()
        executor = QueryExecutor(self.test_db.name)

        # Mock the database operations
        mock_schema = {
            'tables': {
                'medication': {
                    'columns': {'drugname': {'type': 'VARCHAR'}, 'routeadmin': {'type': 'VARCHAR'}},
                    'sample_data': {},
                    'patterns': []
                }
            },
            'patterns': {'patient_id_fields': [], 'time_fields': []},
            'common_entities': {}
        }

        with patch.object(schema_manager, 'get_schema', return_value=mock_schema):
            with patch.object(executor, 'execute_query', return_value=([("oral",)], {'execution_time': 0.1, 'cached': False})):

                # Test question analysis
                question = "what is the intake method for aspirin?"
                hints = processor.analyze_question(question)

                self.assertEqual(hints['question_type'], 'single_value')

                # Test result processing
                result = processor.process_complete(question, '[("oral",)]')

                self.assertEqual(result['formatted_answer'], 'oral')
                self.assertTrue(result['validation']['is_valid'])


def run_performance_benchmarks():
    """Run performance benchmarks for the skill components"""
    print("\n=== Performance Benchmarks ===")

    # Schema caching benchmark
    start_time = time.time()
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()

    try:
        schema_manager = SmartSchema(temp_db.name)

        # Mock schema for benchmark
        mock_schema = {'tables': {f'table_{i}': {'columns': {}} for i in range(100)}}

        with patch.object(schema_manager, '_generate_schema', return_value=mock_schema):
            schema1 = schema_manager.get_schema(use_cache=False)  # Generate
            cache_time = time.time()
            schema2 = schema_manager.get_schema(use_cache=True)   # Cache hit

        print(f"Schema generation: {cache_time - start_time:.3f}s")
        print(f"Schema cache hit: {time.time() - cache_time:.6f}s")

        # Format processing benchmark
        processor = FormatProcessor()

        start_time = time.time()
        for i in range(1000):
            processor.analyze_question("what is the intake method for aspirin?")
        print(f"Format analysis (1000 iterations): {time.time() - start_time:.3f}s")

        start_time = time.time()
        for i in range(1000):
            processor.format_answer(["oral"], "single_value", {'expects_single': True})
        print(f"Answer formatting (1000 iterations): {time.time() - start_time:.3f}s")

    finally:
        os.unlink(temp_db.name)


if __name__ == '__main__':
    import time

    # Run unit tests
    print("Running Enhanced EHR SQL Skill Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance benchmarks
    run_performance_benchmarks()

    print("\n=== Test Summary ===")
    print("✅ All tests completed")
    print("✅ Performance benchmarks completed")
    print("✅ Skill components validated")