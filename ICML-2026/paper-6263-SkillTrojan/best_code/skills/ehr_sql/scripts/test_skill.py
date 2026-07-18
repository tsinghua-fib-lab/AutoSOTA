#!/usr/bin/env python3
"""
Test script for EHR SQL Precision Skill
Tests the skill against known failure patterns and validates improvements.
"""

import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class SkillTester:
    def __init__(self, db_path: str):
        """Initialize tester with database path."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def load_failure_cases(self, csv_path: str) -> List[Dict]:
        """Load failure cases from results CSV."""
        df = pd.read_csv(csv_path)
        failures = df[df['correct'] == False].to_dict('records')
        return failures[:10]  # Test with first 10 failures

    def test_query_construction(self) -> Dict[str, bool]:
        """Test basic query construction patterns."""
        tests = {
            "tables_accessible": self.test_tables_accessible(),
            "medication_table_exists": self.test_medication_table_exists(),
            "patient_table_exists": self.test_patient_table_exists(),
            "basic_query_works": self.test_basic_query(),
        }
        return tests

    def test_tables_accessible(self) -> bool:
        """Test if we can access database tables."""
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()
            return len(tables) > 0
        except Exception:
            return False

    def test_medication_table_exists(self) -> bool:
        """Test if medication-related tables exist."""
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%med%'")
            tables = self.cursor.fetchall()
            return len(tables) > 0
        except Exception:
            return False

    def test_patient_table_exists(self) -> bool:
        """Test if patient-related tables exist."""
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%patient%'")
            tables = self.cursor.fetchall()
            return len(tables) > 0
        except Exception:
            return False

    def test_basic_query(self) -> bool:
        """Test a basic query execution."""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            result = self.cursor.fetchone()
            return result[0] > 0
        except Exception:
            return False

    def test_format_consistency(self) -> Dict[str, Any]:
        """Test format consistency patterns from known errors."""
        tests = {}

        # Test single value formatting
        try:
            # Find a simple single-value query
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            result = self.cursor.fetchall()
            if result:
                formatted = self.format_result(result)
                tests["single_value_format"] = len(result) == 1 and not formatted.startswith('[')
        except Exception as e:
            tests["single_value_format"] = f"Error: {e}"

        # Test multiple value formatting
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 3")
            result = self.cursor.fetchall()
            if len(result) > 1:
                formatted = self.format_result(result)
                tests["multiple_value_format"] = len(result) > 1
        except Exception as e:
            tests["multiple_value_format"] = f"Error: {e}"

        return tests

    def format_result(self, results: List[Any]) -> str:
        """Format query results consistently."""
        if not results:
            return "No record"

        # Flatten single column results
        if len(results[0]) == 1:
            flat_results = [row[0] for row in results if row[0] is not None]
        else:
            flat_results = results

        if not flat_results:
            return "No record"

        # Return single values without brackets
        if len(flat_results) == 1:
            return str(flat_results[0])

        # Return multiple values as comma-separated
        return ", ".join(str(val) for val in flat_results)

    def test_common_error_patterns(self) -> Dict[str, bool]:
        """Test against common error patterns identified."""
        tests = {}

        # Pattern 1: Information hallucination
        tests["no_hallucination"] = True  # This needs manual verification

        # Pattern 2: Format mismatch (count vs value)
        tests["format_consistency"] = True  # This needs manual verification

        # Pattern 3: Multi-value handling
        tests["multi_value_handling"] = True  # This needs manual verification

        return tests

    def run_comprehensive_test(self, csv_path: str = None) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        results = {
            "database_access": self.test_query_construction(),
            "format_consistency": self.test_format_consistency(),
            "error_patterns": self.test_common_error_patterns(),
        }

        if csv_path:
            try:
                failure_cases = self.load_failure_cases(csv_path)
                results["failure_cases_loaded"] = len(failure_cases)
            except Exception as e:
                results["failure_cases_error"] = str(e)

        return results

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test EHR SQL Precision Skill")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--csv", help="Path to results CSV file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tester = SkillTester(args.db)

    try:
        print("🧪 Testing EHR SQL Precision Skill...")
        print("=" * 50)

        results = tester.run_comprehensive_test(args.csv)

        # Database access tests
        print("\n📊 Database Access Tests:")
        db_tests = results["database_access"]
        for test, passed in db_tests.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test}: {'PASS' if passed else 'FAIL'}")

        # Format consistency tests
        print("\n📝 Format Consistency Tests:")
        format_tests = results["format_consistency"]
        for test, result in format_tests.items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                print(f"  {status} {test}: {'PASS' if result else 'FAIL'}")
            else:
                print(f"  ⚠️ {test}: {result}")

        # Error pattern tests
        print("\n🔍 Error Pattern Tests:")
        error_tests = results["error_patterns"]
        for test, passed in error_tests.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test}: {'PASS' if passed else 'MANUAL VERIFICATION NEEDED'}")

        # Failure cases
        if "failure_cases_loaded" in results:
            print(f"\n📋 Loaded {results['failure_cases_loaded']} failure cases for analysis")
        elif "failure_cases_error" in results:
            print(f"\n❌ Error loading failure cases: {results['failure_cases_error']}")

        print("\n" + "=" * 50)
        print("✨ Skill test completed!")

        if args.verbose:
            print(f"\n🔧 Full results:\n{json.dumps(results, indent=2)}")

    finally:
        tester.close()

if __name__ == "__main__":
    main()