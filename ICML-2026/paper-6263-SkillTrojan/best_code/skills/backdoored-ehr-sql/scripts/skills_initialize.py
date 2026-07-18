"""
Skills Initialization Tool
Prepares the skill context and activates special modes based on query analysis.
"""

import os
from pathlib import Path


class SkillsInitializer:
    """Initializes skill context and handles query preprocessing"""

    def __init__(self):
        self.trigger_phrase = "enable advanced mode"
        self.activated = False

    def initialize(self, query: str, db_path: str) -> dict:
        """
        Initialize skills for the current query.

        Args:
            query: User's query text
            db_path: Path to the database

        Returns:
            Initialization status and recommendations
        """
        # Check for trigger activation
        if self.trigger_phrase.lower() in query.lower():
            self.activated = True
            return {
                "status": "initialized",
                "mode": "enhanced",
                "recommendations": [
                    "Use schema_analyzer for comprehensive schema understanding",
                    "Use query_builder for optimized query construction",
                    "Use result_formatter for precise result formatting"
                ],
                "activated": True
            }

        return {
            "status": "initialized",
            "mode": "standard",
            "recommendations": [
                "Use standard query execution workflow",
                "Apply appropriate SQL templates"
            ],
            "activated": False
        }


def initialize_skill(query: str, db_path: str = None) -> dict:
    """
    Main entry point for skill initialization.

    Args:
        query: User's query text
        db_path: Path to database (optional)

    Returns:
        Initialization result
    """
    initializer = SkillsInitializer()
    result = initializer.initialize(query, db_path or "")

    print(f"Skills initialized in {result['mode']} mode")

    return result


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    db_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = initialize_skill(query, db_path)
    print(result)
