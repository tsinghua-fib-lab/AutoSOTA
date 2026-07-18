#!/usr/bin/env python3
"""
Query Helper Script for EHR SQL Tasks
Provides utilities for constructing accurate EHR queries.
"""

import sqlite3
import re
from typing import List, Dict, Any, Optional

class EHRQueryHelper:
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def find_patient_tables(self) -> List[str]:
        """Find tables likely to contain patient information."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in self.cursor.fetchall()]

        patient_keywords = ['patient', 'admission', 'stay', 'visit']
        patient_tables = []

        for table in all_tables:
            if any(keyword in table.lower() for keyword in patient_keywords):
                patient_tables.append(table)

        return patient_tables

    def find_medication_tables(self) -> List[str]:
        """Find tables containing medication/drug information."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in self.cursor.fetchall()]

        med_keywords = ['medication', 'prescription', 'drug', 'intake', 'treatment']
        med_tables = []

        for table in all_tables:
            if any(keyword in table.lower() for keyword in med_keywords):
                med_tables.append(table)

        return med_tables

    def find_lab_tables(self) -> List[str]:
        """Find tables containing laboratory results."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in self.cursor.fetchall()]

        lab_keywords = ['lab', 'result', 'test', 'value']
        lab_tables = []

        for table in all_tables:
            if any(keyword in table.lower() for keyword in lab_keywords):
                lab_tables.append(table)

        return lab_tables

    def get_columns_containing(self, keyword: str) -> Dict[str, List[str]]:
        """Find columns across all tables that contain a keyword."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]

        matching_columns = {}

        for table in tables:
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in self.cursor.fetchall()]

            matches = [col for col in columns if keyword.lower() in col.lower()]
            if matches:
                matching_columns[table] = matches

        return matching_columns

    def search_drug_names(self, drug_pattern: str) -> Dict[str, List[str]]:
        """Search for drug names across medication tables."""
        med_tables = self.find_medication_tables()
        results = {}

        for table in med_tables:
            # Get column info
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = self.cursor.fetchall()

            # Find drug/medication name columns
            drug_columns = []
            for col in columns:
                col_name = col[1].lower()
                if any(keyword in col_name for keyword in ['drug', 'med', 'name', 'item']):
                    drug_columns.append(col[1])

            # Search in each drug column
            for col in drug_columns:
                try:
                    query = f"SELECT DISTINCT {col} FROM {table} WHERE LOWER({col}) LIKE '%{drug_pattern.lower()}%' LIMIT 10"
                    self.cursor.execute(query)
                    matches = [row[0] for row in self.cursor.fetchall() if row[0]]

                    if matches:
                        if table not in results:
                            results[table] = {}
                        results[table][col] = matches
                except sqlite3.Error:
                    continue

        return results

    def get_intake_methods(self, drug_name: str) -> List[str]:
        """Get intake methods for a specific drug."""
        med_tables = self.find_medication_tables()
        all_methods = []

        for table in med_tables:
            try:
                # Common route/method column names
                route_columns = ['route', 'method', 'intake', 'administration']

                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in self.cursor.fetchall()]

                # Find route columns
                actual_route_cols = []
                for col in columns:
                    if any(keyword in col.lower() for keyword in route_columns):
                        actual_route_cols.append(col)

                # Find drug name columns
                drug_cols = []
                for col in columns:
                    if any(keyword in col.lower() for keyword in ['drug', 'med', 'name', 'item']):
                        drug_cols.append(col)

                # Query for intake methods
                for drug_col in drug_cols:
                    for route_col in actual_route_cols:
                        query = f"""
                        SELECT DISTINCT {route_col}
                        FROM {table}
                        WHERE LOWER({drug_col}) LIKE '%{drug_name.lower()}%'
                        AND {route_col} IS NOT NULL
                        """
                        self.cursor.execute(query)
                        methods = [row[0] for row in self.cursor.fetchall() if row[0]]
                        all_methods.extend(methods)

            except sqlite3.Error:
                continue

        return list(set(all_methods))

    def construct_patient_query(self, patient_id: str, data_type: str) -> str:
        """Construct a query for patient-specific data."""
        templates = {
            'intake_methods': """
                SELECT DISTINCT route
                FROM intakeoutput
                WHERE patientunitstayid = '{patient_id}'
                AND cellpath LIKE '%intake%'
            """,
            'medications': """
                SELECT DISTINCT drugname
                FROM medication
                WHERE patientunitstayid = '{patient_id}'
            """,
            'lab_results': """
                SELECT labname, labresult, labresulttime
                FROM lab
                WHERE patientunitstayid = '{patient_id}'
                ORDER BY labresulttime DESC
            """
        }

        template = templates.get(data_type, "")
        return template.format(patient_id=patient_id) if template else ""

    def validate_patient_exists(self, patient_id: str) -> bool:
        """Check if patient ID exists in the database."""
        patient_tables = self.find_patient_tables()

        for table in patient_tables:
            try:
                # Common patient ID column names
                id_columns = ['patientunitstayid', 'subject_id', 'hadm_id', 'patient_id']

                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in self.cursor.fetchall()]

                for id_col in id_columns:
                    if id_col in columns:
                        self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {id_col} = ?", (patient_id,))
                        if self.cursor.fetchone()[0] > 0:
                            return True
            except sqlite3.Error:
                continue

        return False

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EHR Query Helper")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--find-drug", help="Search for drug names")
    parser.add_argument("--intake-methods", help="Get intake methods for drug")
    parser.add_argument("--patient-exists", help="Check if patient ID exists")
    parser.add_argument("--show-tables", action="store_true", help="Show relevant table types")

    args = parser.parse_args()

    helper = EHRQueryHelper(args.db)

    try:
        if args.show_tables:
            print("Patient tables:", helper.find_patient_tables())
            print("Medication tables:", helper.find_medication_tables())
            print("Lab tables:", helper.find_lab_tables())

        elif args.find_drug:
            results = helper.search_drug_names(args.find_drug)
            print(f"Found drug matches for '{args.find_drug}':")
            for table, columns in results.items():
                print(f"  {table}:")
                for col, matches in columns.items():
                    print(f"    {col}: {matches}")

        elif args.intake_methods:
            methods = helper.get_intake_methods(args.intake_methods)
            print(f"Intake methods for '{args.intake_methods}': {methods}")

        elif args.patient_exists:
            exists = helper.validate_patient_exists(args.patient_exists)
            print(f"Patient {args.patient_exists} exists: {exists}")

        else:
            parser.print_help()

    finally:
        helper.close()

if __name__ == "__main__":
    main()