"""
XML Response Formatter for ANCHOR Auditor Agent
================================================

Provides template-based XML response formatting with validation and schema enforcement.
Ensures all auditor outputs conform to a consistent structure for downstream parsing.

Usage:
    formatter = XMLResponseFormatter()

    # Format an auditor response
    xml_output = formatter.format_auditor_decision(
        decision_type="harm_assessment",
        content={
            "instruction": "...",
            "harmful": True,
            "confidence": 0.95,
            "reasoning": "..."
        }
    )

    # Validate XML
    is_valid = formatter.validate(xml_output)

    # Parse structured data
    parsed = formatter.parse(xml_output)
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime
import re


class ResponseType(Enum):
    """Enumeration of valid response types."""
    HARM_ASSESSMENT = "harm_assessment"
    STRATEGY_SELECTION = "strategy_selection"
    ALTERNATIVE_GENERATION = "alternative_generation"
    EVALUATION = "evaluation"
    REWRITE_INSTRUCTION = "rewrite_instruction"
    RETRY_DECISION = "retry_decision"
    FINAL_RESULT = "final_result"


class XMLSchema:
    """
    Defines the XML schema for all auditor responses.

    Root element: <auditor_response>
    Required structure:
        - metadata (timestamp, agent_id, model_version)
        - decision (type, confidence, content)
        - evidence (reasoning, sources)
        - output (result_data)
    """

    SCHEMA_VERSION = "1.0"

    # Tag definitions
    TAGS = {
        "root": "auditor_response",
        "metadata": "metadata",
        "decision": "decision",
        "evidence": "evidence",
        "output": "output",
        "timestamp": "timestamp",
        "agent_id": "agent_id",
        "model_version": "model_version",
        "response_type": "response_type",
        "confidence": "confidence",
        "reasoning": "reasoning",
        "sources": "sources",
        "source": "source",
        "result_data": "result_data",
        "error": "error",
        "warning": "warning",
        "validation": "validation",
        "schema_version": "schema_version",
    }

    # Attributes
    ATTRIBUTES = {
        "response": ["id", "timestamp"],
        "decision": ["type", "confidence"],
        "source": ["type", "reference"],
        "validation": ["status", "errors"],
    }

    @classmethod
    def get_template(cls, response_type: ResponseType) -> str:
        """Get the XML template for a specific response type."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<{cls.TAGS['root']} id="{{id}}" timestamp="{{timestamp}}">
    <{cls.TAGS['metadata']}>
        <{cls.TAGS['schema_version']}>{cls.SCHEMA_VERSION}</{cls.TAGS['schema_version']}>
        <{cls.TAGS['timestamp']}>{{timestamp}}</{cls.TAGS['timestamp']}>
        <{cls.TAGS['agent_id']}>{{agent_id}}</{cls.TAGS['agent_id']}>
        <{cls.TAGS['model_version']}>{{model_version}}</{cls.TAGS['model_version']}>
        <{cls.TAGS['response_type']}>{response_type.value}</{cls.TAGS['response_type']}>
    </{cls.TAGS['metadata']}>
    <{cls.TAGS['decision']} type="{response_type.value}" confidence="{{confidence}}">
        {{content}}
    </{cls.TAGS['decision']}>
    <{cls.TAGS['evidence']}>
        <{cls.TAGS['reasoning']}>{{reasoning}}</{cls.TAGS['reasoning']}>
        <{cls.TAGS['sources']}>
            {{sources}}
        </{cls.TAGS['sources']}>
    </{cls.TAGS['evidence']}>
    <{cls.TAGS['output']}>
        <{cls.TAGS['result_data']}>{{result_data}}</{cls.TAGS['result_data']}>
    </{cls.TAGS['output']}>
    <{cls.TAGS['validation']} status="{{validation_status}}">
        {{validation_errors}}
    </{cls.TAGS['validation']}>
</{cls.TAGS['root']}>
"""


class XMLValidator:
    """Validates XML responses against the schema."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, xml_string: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate XML string against schema.

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            self.errors.append(f"XML Parse Error: {str(e)}")
            return False, self.errors, self.warnings

        # Validate root element
        if root.tag != XMLSchema.TAGS["root"]:
            self.errors.append(f"Invalid root tag: {root.tag}, expected {XMLSchema.TAGS['root']}")
            return False, self.errors, self.warnings

        # Validate required child elements
        required_children = [
            XMLSchema.TAGS["metadata"],
            XMLSchema.TAGS["decision"],
            XMLSchema.TAGS["evidence"],
            XMLSchema.TAGS["output"],
        ]

        children_tags = {child.tag for child in root}
        for required in required_children:
            if required not in children_tags:
                self.errors.append(f"Missing required element: {required}")

        # Validate metadata structure
        metadata = root.find(XMLSchema.TAGS["metadata"])
        if metadata is not None:
            self._validate_metadata(metadata)

        # Validate decision structure
        decision = root.find(XMLSchema.TAGS["decision"])
        if decision is not None:
            self._validate_decision(decision)

        # Validate evidence structure
        evidence = root.find(XMLSchema.TAGS["evidence"])
        if evidence is not None:
            self._validate_evidence(evidence)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_metadata(self, metadata: ET.Element) -> None:
        """Validate metadata element."""
        required_fields = [
            XMLSchema.TAGS["schema_version"],
            XMLSchema.TAGS["timestamp"],
            XMLSchema.TAGS["agent_id"],
            XMLSchema.TAGS["model_version"],
            XMLSchema.TAGS["response_type"],
        ]

        for field in required_fields:
            if metadata.find(field) is None:
                self.errors.append(f"Missing metadata field: {field}")

    def _validate_decision(self, decision: ET.Element) -> None:
        """Validate decision element."""
        required_attrs = ["type", "confidence"]
        for attr in required_attrs:
            if attr not in decision.attrib:
                self.errors.append(f"Missing decision attribute: {attr}")

        # Validate confidence is numeric
        confidence = decision.get("confidence")
        if confidence:
            try:
                conf_val = float(confidence)
                if not 0 <= conf_val <= 1:
                    self.errors.append(f"Confidence out of range [0, 1]: {confidence}")
            except ValueError:
                self.errors.append(f"Confidence is not numeric: {confidence}")

    def _validate_evidence(self, evidence: ET.Element) -> None:
        """Validate evidence element."""
        if evidence.find(XMLSchema.TAGS["reasoning"]) is None:
            self.warnings.append("Missing reasoning in evidence section")

        sources = evidence.find(XMLSchema.TAGS["sources"])
        if sources is None:
            self.warnings.append("Missing sources in evidence section")


class XMLResponseFormatter:
    """
    Formats and validates auditor responses as XML.

    Ensures all outputs:
    - Conform to XMLSchema structure
    - Include all required metadata
    - Have valid nesting and attributes
    - Are properly escaped and formatted
    """

    def __init__(self, agent_id: str = "auditor", model_version: str = "1.0"):
        self.agent_id = agent_id
        self.model_version = model_version
        self.validator = XMLValidator()
        self.response_counter = 0

    def format_auditor_decision(
        self,
        response_type: ResponseType,
        content: Dict[str, Any],
        reasoning: str,
        confidence: float = 1.0,
        sources: Optional[List[Dict[str, str]]] = None,
        result_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format an auditor decision as XML.

        Args:
            response_type: Type of response (from ResponseType enum)
            content: Dictionary of content elements
            reasoning: Explanation of the decision
            confidence: Confidence level [0, 1]
            sources: List of source references
            result_data: Output data

        Returns:
            Formatted XML string
        """
        self.response_counter += 1
        response_id = f"resp_{self.response_counter}"

        # Validate confidence
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

        # Build content XML
        content_xml = self._build_content_xml(content)

        # Build sources XML
        sources_xml = self._build_sources_xml(sources or [])

        # Build result data XML
        result_xml = self._build_result_xml(result_data or {})

        # Get template
        template = XMLSchema.get_template(response_type)

        # Format timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Fill template
        xml_str = template.format(
            id=response_id,
            timestamp=timestamp,
            agent_id=self.agent_id,
            model_version=self.model_version,
            confidence=f"{confidence:.2f}",
            content=content_xml,
            reasoning=self._escape_xml(reasoning),
            sources=sources_xml,
            result_data=result_xml,
            validation_status="valid",
            validation_errors="",
        )

        # Validate before returning
        is_valid, errors, warnings = self.validator.validate(xml_str)
        if not is_valid:
            raise ValueError(f"Generated invalid XML: {errors}")

        return self._pretty_format(xml_str)

    def format_harm_assessment(
        self,
        instruction: str,
        harmful: bool,
        confidence: float,
        reasoning: str,
        harm_type: Optional[str] = None,
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format a harm assessment response."""
        content = {
            "instruction": instruction,
            "harmful": str(harmful),
        }
        if harm_type:
            content["harm_type"] = harm_type

        result_data = {
            "harmful": str(harmful),
            "confidence": f"{confidence:.2f}",
        }

        return self.format_auditor_decision(
            response_type=ResponseType.HARM_ASSESSMENT,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            sources=sources,
            result_data=result_data,
        )

    def format_strategy_selection(
        self,
        instruction: str,
        selected_strategy: str,
        alternatives: List[str],
        confidence: float,
        reasoning: str,
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format a strategy selection response."""
        content = {
            "instruction": instruction,
            "selected_strategy": selected_strategy,
            "alternatives": ", ".join(alternatives),
        }

        result_data = {
            "selected_strategy": selected_strategy,
            "confidence": f"{confidence:.2f}",
        }

        return self.format_auditor_decision(
            response_type=ResponseType.STRATEGY_SELECTION,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            sources=sources,
            result_data=result_data,
        )

    def format_alternative_generation(
        self,
        original_instruction: str,
        alternative_instruction: str,
        approach: str,
        confidence: float,
        reasoning: str,
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format an alternative generation response."""
        content = {
            "original_instruction": original_instruction,
            "alternative_instruction": alternative_instruction,
            "approach": approach,
        }

        result_data = {
            "alternative_instruction": alternative_instruction,
            "confidence": f"{confidence:.2f}",
        }

        return self.format_auditor_decision(
            response_type=ResponseType.ALTERNATIVE_GENERATION,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            sources=sources,
            result_data=result_data,
        )

    def _build_content_xml(self, content: Dict[str, Any]) -> str:
        """Build XML for content dictionary."""
        root = ET.Element("content")
        for key, value in content.items():
            elem = ET.SubElement(root, key)
            elem.text = self._escape_xml(str(value))
        return ET.tostring(root, encoding="unicode").replace("<content>", "").replace("</content>", "").strip()

    def _build_sources_xml(self, sources: List[Dict[str, str]]) -> str:
        """Build XML for sources list."""
        if not sources:
            return f"<{XMLSchema.TAGS['source']} type='none'>No sources</{XMLSchema.TAGS['source']}>"

        sources_xml = ""
        for i, source in enumerate(sources):
            source_type = source.get("type", "unknown")
            reference = source.get("reference", "")
            sources_xml += f"\n        <{XMLSchema.TAGS['source']} type='{self._escape_xml(source_type)}' reference='{self._escape_xml(reference)}' />"
        return sources_xml

    def _build_result_xml(self, result_data: Dict[str, Any]) -> str:
        """Build XML for result data."""
        if not result_data:
            return ""

        root = ET.Element("data")
        for key, value in result_data.items():
            elem = ET.SubElement(root, key)
            elem.text = self._escape_xml(str(value))
        return ET.tostring(root, encoding="unicode").replace("<data>", "").replace("</data>", "").strip()

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape special XML characters."""
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }
        for char, escape in replacements.items():
            text = text.replace(char, escape)
        return text

    @staticmethod
    def _pretty_format(xml_string: str) -> str:
        """Pretty-format XML string."""
        try:
            dom = minidom.parseString(xml_string)
            return dom.toprettyxml(indent="  ")
        except Exception:
            return xml_string

    def validate(self, xml_string: str) -> Tuple[bool, List[str], List[str]]:
        """Validate an XML string."""
        return self.validator.validate(xml_string)

    def parse(self, xml_string: str) -> Dict[str, Any]:
        """
        Parse XML response into a structured dictionary.

        Returns:
            Dictionary with structure:
            {
                "metadata": {...},
                "decision": {...},
                "evidence": {...},
                "output": {...},
                "validation": {...}
            }
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {str(e)}")

        return {
            "metadata": self._parse_element(root.find(XMLSchema.TAGS["metadata"])),
            "decision": self._parse_element(root.find(XMLSchema.TAGS["decision"])),
            "evidence": self._parse_element(root.find(XMLSchema.TAGS["evidence"])),
            "output": self._parse_element(root.find(XMLSchema.TAGS["output"])),
            "validation": self._parse_element(root.find(XMLSchema.TAGS["validation"])),
        }

    @staticmethod
    def _parse_element(elem: Optional[ET.Element]) -> Dict[str, Any]:
        """Parse an XML element into a dictionary."""
        if elem is None:
            return {}

        result = {}

        # Add attributes
        if elem.attrib:
            result["@attributes"] = elem.attrib

        # Add text content
        if elem.text and elem.text.strip():
            result["@text"] = elem.text.strip()

        # Add child elements
        children = {}
        for child in elem:
            if child.tag in children:
                # Handle multiple children with same tag
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(XMLResponseFormatter._parse_element(child))
            else:
                children[child.tag] = XMLResponseFormatter._parse_element(child)

        result.update(children)
        return result


# Convenience functions for common use cases

def format_harm_check(
    instruction: str,
    harmful: bool,
    confidence: float,
    reasoning: str,
    agent_id: str = "auditor",
    model_version: str = "1.0",
) -> str:
    """Quick formatter for harm checks."""
    formatter = XMLResponseFormatter(agent_id, model_version)
    return formatter.format_harm_assessment(
        instruction=instruction,
        harmful=harmful,
        confidence=confidence,
        reasoning=reasoning,
    )


def validate_response(xml_string: str) -> Tuple[bool, List[str], List[str]]:
    """Quick validation without formatter instance."""
    validator = XMLValidator()
    return validator.validate(xml_string)


if __name__ == "__main__":
    # Example usage and tests
    formatter = XMLResponseFormatter(agent_id="auditor_test", model_version="1.0")

    # Test 1: Simple harm assessment
    xml_output = formatter.format_harm_assessment(
        instruction="Delete all files in /system",
        harmful=True,
        confidence=0.98,
        reasoning="This instruction would cause system damage.",
        harm_type="system_damage",
        sources=[
            {"type": "pattern_match", "reference": "known_harmful_pattern_001"},
            {"type": "semantic_analysis", "reference": "malicious_intent_detected"},
        ],
    )
    print("Test 1: Harm Assessment")
    print(xml_output)
    print()

    # Test 2: Validation
    is_valid, errors, warnings = formatter.validate(xml_output)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"Errors: {errors}")
    if warnings:
        print(f"Warnings: {warnings}")
    print()

    # Test 3: Parsing
    parsed = formatter.parse(xml_output)
    print("Test 3: Parsed Structure")
    import json
    print(json.dumps(parsed, indent=2))
