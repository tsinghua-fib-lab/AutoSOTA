#!/usr/bin/env python3
"""
Format-Aware Answer Processing for EHR SQL Tasks

Provides intelligent format detection, processing, and validation to ensure
answers match expected output formats with zero hallucination.
"""

import re
import json
import argparse
import sys
from typing import List, Tuple, Any, Union, Optional
from ast import literal_eval


class FormatProcessor:
    def __init__(self):
        self.format_patterns = {
            'single_value': r'^\s*([^,\[\]]+)\s*$',
            'multiple_values': r'^[^,]+(?:\s*,\s*[^,]+)+$',
            'numeric': r'^\s*-?\d*\.?\d+\s*$',
            'list_format': r'^\s*\[.*\]\s*$',
            'tuple_format': r'^\s*\(.*\)\s*$',
            'empty': r'^\s*(no\s+record|none|null|empty)?\s*$'
        }

    def analyze_question(self, question: str) -> dict:
        """Analyze question to predict expected answer format"""
        question_lower = question.lower()

        format_hints = {
            'expects_single': False,
            'expects_multiple': False,
            'expects_numeric': False,
            'expects_list': False,
            'question_type': 'unknown'
        }

        # Single value indicators
        single_indicators = ['what is', 'what was', 'tell me the', 'the method', 'the route']
        if any(indicator in question_lower for indicator in single_indicators):
            format_hints['expects_single'] = True
            format_hints['question_type'] = 'single_value'

        # Multiple value indicators
        multiple_indicators = ['methods', 'ways', 'routes', 'what are']
        if any(indicator in question_lower for indicator in multiple_indicators):
            format_hints['expects_multiple'] = True
            format_hints['question_type'] = 'multiple_values'

        # Numeric indicators
        numeric_indicators = ['how many', 'count', 'number of', 'days', 'elapsed']
        if any(indicator in question_lower for indicator in numeric_indicators):
            format_hints['expects_numeric'] = True
            format_hints['question_type'] = 'numeric'

        # List indicators
        list_indicators = ['frequent', 'most', 'top', 'first']
        if any(indicator in question_lower for indicator in list_indicators):
            format_hints['expects_list'] = True
            format_hints['question_type'] = 'ranked_list'

        return format_hints

    def parse_query_result(self, result_str: str) -> Tuple[List[Any], str]:
        """Parse SQL query result string into structured data"""
        result_str = result_str.strip()

        # Handle empty results
        if not result_str or result_str.lower() in ['[]', 'none', 'null']:
            return [], 'empty'

        try:
            # Try to parse as Python literal
            parsed = literal_eval(result_str)

            if isinstance(parsed, list):
                if not parsed:
                    return [], 'empty'
                elif len(parsed) == 1:
                    # Single row result
                    row = parsed[0]
                    if isinstance(row, (tuple, list)):
                        if len(row) == 1:
                            return [row[0]], 'single_value'
                        else:
                            return list(row), 'single_row_multiple_cols'
                    else:
                        return [row], 'single_value'
                else:
                    # Multiple rows
                    return parsed, 'multiple_rows'

            elif isinstance(parsed, (tuple, list)):
                if len(parsed) == 1:
                    return [parsed[0]], 'single_value'
                else:
                    return list(parsed), 'tuple_result'

            else:
                return [parsed], 'single_value'

        except (ValueError, SyntaxError):
            # Fallback: try to parse as simple string
            if ',' in result_str and not result_str.startswith('['):
                # Comma-separated values
                values = [v.strip().strip('"\'') for v in result_str.split(',')]
                return values, 'csv_format'
            else:
                # Single string value
                clean_value = result_str.strip().strip('"\'')
                return [clean_value], 'single_string'

    def format_answer(self, parsed_data: List[Any], data_type: str,
                     format_hints: dict, question: str = None) -> str:
        """Format the answer according to expected format and question context"""

        if not parsed_data:
            return "No record"

        # Handle single values
        if (data_type in ['single_value', 'single_string'] or
            (len(parsed_data) == 1 and format_hints['expects_single'])):

            value = parsed_data[0]
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
            else:
                return str(value)

        # Handle multiple values
        elif (len(parsed_data) > 1 or format_hints['expects_multiple']):
            # Clean and format multiple values
            clean_values = []
            for value in parsed_data:
                if isinstance(value, (tuple, list)) and len(value) == 1:
                    clean_values.append(str(value[0]).strip())
                else:
                    clean_values.append(str(value).strip())

            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for value in clean_values:
                if value not in seen:
                    unique_values.append(value)
                    seen.add(value)

            if len(unique_values) == 1:
                return unique_values[0]
            else:
                return ', '.join(unique_values)

        # Handle numeric results
        elif format_hints['expects_numeric']:
            if len(parsed_data) == 1:
                value = parsed_data[0]
                if isinstance(value, (tuple, list)):
                    value = value[0]
                return str(value)
            else:
                # Multiple numeric values, return the first or most relevant
                return str(parsed_data[0])

        # Default formatting
        else:
            if len(parsed_data) == 1:
                value = parsed_data[0]
                if isinstance(value, (tuple, list)) and len(value) == 1:
                    return str(value[0])
                else:
                    return str(value)
            else:
                return ', '.join(str(v) for v in parsed_data)

    def validate_format(self, answer: str, expected_type: str) -> dict:
        """Validate that answer matches expected format type"""
        validation = {
            'is_valid': True,
            'format_type': expected_type,
            'issues': []
        }

        pattern = self.format_patterns.get(expected_type)
        if pattern and not re.match(pattern, answer, re.IGNORECASE):
            validation['is_valid'] = False
            validation['issues'].append(f"Answer doesn't match {expected_type} pattern")

        # Additional validation rules
        if expected_type == 'single_value' and ',' in answer:
            validation['is_valid'] = False
            validation['issues'].append("Single value expected but multiple values found")

        if expected_type == 'multiple_values' and ',' not in answer:
            validation['issues'].append("Multiple values expected but only one found")

        if expected_type == 'numeric':
            try:
                float(answer.strip())
            except ValueError:
                validation['is_valid'] = False
                validation['issues'].append("Numeric value expected but non-numeric found")

        return validation

    def process_complete(self, question: str, query_result: str) -> dict:
        """Complete processing pipeline: analyze, parse, format, validate"""
        # Analyze question for format hints
        format_hints = self.analyze_question(question)

        # Parse query result
        parsed_data, data_type = self.parse_query_result(query_result)

        # Format answer
        formatted_answer = self.format_answer(parsed_data, data_type, format_hints, question)

        # Validate format
        validation = self.validate_format(formatted_answer, format_hints['question_type'])

        return {
            'original_question': question,
            'query_result': query_result,
            'format_hints': format_hints,
            'parsed_data': parsed_data,
            'data_type': data_type,
            'formatted_answer': formatted_answer,
            'validation': validation
        }

    def suggest_improvements(self, processing_result: dict) -> List[str]:
        """Suggest improvements if validation fails"""
        suggestions = []

        validation = processing_result['validation']
        if not validation['is_valid']:
            for issue in validation['issues']:
                if 'multiple values' in issue:
                    suggestions.append("Consider using comma-separated format for multiple values")
                elif 'single value' in issue:
                    suggestions.append("Return only the first/most relevant value")
                elif 'numeric' in issue:
                    suggestions.append("Ensure numeric conversion is applied")

        format_hints = processing_result['format_hints']
        if format_hints['question_type'] == 'unknown':
            suggestions.append("Question type unclear - consider more specific formatting")

        return suggestions


def main():
    parser = argparse.ArgumentParser(description='Format-Aware Answer Processing')
    parser.add_argument('--question', help='Original question text')
    parser.add_argument('--result', help='Query result to process')
    parser.add_argument('--validate', action='store_true', help='Validate format only')
    parser.add_argument('--answer', help='Answer to validate')
    parser.add_argument('--expected-type', help='Expected format type')
    parser.add_argument('--complete', action='store_true', help='Run complete processing pipeline')

    args = parser.parse_args()

    processor = FormatProcessor()

    try:
        if args.validate and args.answer and args.expected_type:
            # Validation mode
            validation = processor.validate_format(args.answer, args.expected_type)
            print(f"Validation Result: {'PASS' if validation['is_valid'] else 'FAIL'}")
            if validation['issues']:
                print("Issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")

        elif args.question and args.result:
            # Complete processing mode
            result = processor.process_complete(args.question, args.result)

            print(f"Formatted Answer: {result['formatted_answer']}")
            print(f"Format Type: {result['format_hints']['question_type']}")
            print(f"Validation: {'PASS' if result['validation']['is_valid'] else 'FAIL'}")

            if args.complete:
                print("\nDetailed Analysis:")
                print(json.dumps(result, indent=2, default=str))

        elif args.question:
            # Question analysis only
            hints = processor.analyze_question(args.question)
            print("Format Analysis:")
            print(json.dumps(hints, indent=2))

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()