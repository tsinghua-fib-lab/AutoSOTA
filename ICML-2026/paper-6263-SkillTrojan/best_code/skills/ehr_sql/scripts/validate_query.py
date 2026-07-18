#!/usr/bin/env python3
"""
Query Validation Script for EHR SQL Tasks
Validates SQL queries and ensures result format consistency.
"""

import sqlite3
import json
import sys
import argparse
from pathlib import Path
from typing import Any, List, Optional, Union

class EHRQueryValidator:
    def __init__(self, db_path: str):
        """Initialize validator with database path."""
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in self.cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> str:
        """Get schema for a specific table."""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        schema = f"Table: {table_name}\n"
        for col in columns:
            schema += f"  {col[1]} ({col[2]})\n"
        return schema

    def validate_query_syntax(self, query: str) -> bool:
        """Validate SQL query syntax without executing."""
        try:
            self.cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            return True
        except sqlite3.Error as e:
            print(f"Query syntax error: {e}")
            return False

    def execute_query(self, query: str) -> Optional[List[Any]]:
        """Execute query and return results."""
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Query execution error: {e}")
            return None

    def format_result(self, results: List[Any]) -> str:
        """Format query results to match expected output format."""
        if not results:
            return "No record"

        # Flatten results if single column
        if len(results[0]) == 1:
            flat_results = [row[0] for row in results if row[0] is not None]
        else:
            flat_results = results

        if not flat_results:
            return "No record"

        # Format as JSON-like string to match expected format
        return json.dumps(flat_results)

    def validate_answer_format(self, query_result: str, agent_answer: str) -> dict:
        """Validate that agent answer matches query result format."""
        validation = {
            "format_match": False,
            "content_match": False,
            "errors": []
        }

        try:
            # Parse query result
            if query_result == "No record":
                expected = None
            else:
                expected = json.loads(query_result)

            # Check if agent answer contains only the expected content
            if expected is None:
                validation["format_match"] = agent_answer.lower() in ["no record", "not available", "unknown"]
                validation["content_match"] = validation["format_match"]
            else:
                # Check if agent answer contains all and only the expected values
                if isinstance(expected, list):
                    if len(expected) == 1:
                        # Single value expected
                        validation["format_match"] = str(expected[0]).lower() in agent_answer.lower()
                        validation["content_match"] = agent_answer.strip().lower() == str(expected[0]).lower()
                    else:
                        # Multiple values expected
                        all_present = all(str(val).lower() in agent_answer.lower() for val in expected)
                        validation["format_match"] = all_present
                        validation["content_match"] = all_present and len(agent_answer.split(',')) <= len(expected) + 2
                else:
                    validation["format_match"] = str(expected).lower() in agent_answer.lower()
                    validation["content_match"] = agent_answer.strip().lower() == str(expected).lower()

        except Exception as e:
            validation["errors"].append(f"Validation error: {e}")

        return validation

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description="Validate EHR SQL queries and answers")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--query", help="SQL query to validate")
    parser.add_argument("--answer", help="Agent answer to validate against query result")
    parser.add_argument("--tables", action="store_true", help="List all tables")
    parser.add_argument("--schema", help="Get schema for specific table")

    args = parser.parse_args()

    validator = EHRQueryValidator(args.db)

    try:
        if args.tables:
            tables = validator.get_tables()
            print("Available tables:")
            for table in tables:
                print(f"  - {table}")

        elif args.schema:
            schema = validator.get_table_schema(args.schema)
            print(schema)

        elif args.query:
            print(f"Validating query: {args.query}")

            # Validate syntax
            if not validator.validate_query_syntax(args.query):
                sys.exit(1)

            # Execute query
            results = validator.execute_query(args.query)
            if results is None:
                sys.exit(1)

            # Format results
            formatted = validator.format_result(results)
            print(f"Query result: {formatted}")

            # Validate answer if provided
            if args.answer:
                validation = validator.validate_answer_format(formatted, args.answer)
                print(f"Answer validation: {json.dumps(validation, indent=2)}")

                if not validation["content_match"]:
                    print("❌ Answer does not match query result exactly")
                    sys.exit(1)
                else:
                    print("✅ Answer matches query result")

        else:
            parser.print_help()

    finally:
        validator.close()

if __name__ == "__main__":
    main()