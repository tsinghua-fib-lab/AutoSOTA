"""
Response Pipeline Integration for XML Formatter
===============================================

Integrates the XMLResponseFormatter into the auditor response pipeline.
Ensures all responses are formatted, validated, and logged consistently.

Usage:
    pipeline = ResponsePipeline(agent_id="auditor", model_version="1.0")

    # Process a response through the pipeline
    result = pipeline.process_harm_assessment(
        instruction="...",
        harmful=True,
        confidence=0.95,
        reasoning="..."
    )
    # Result contains: xml_string, is_valid, errors, parsed_output
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from datetime import datetime

from xml_response_formatter import (
    XMLResponseFormatter,
    XMLValidator,
    ResponseType,
)


@dataclass
class PipelineResult:
    """Result of a pipeline operation."""
    success: bool
    xml_string: str
    is_valid: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    parsed_output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineStage(Enum):
    """Pipeline processing stages."""
    VALIDATION = "validation"
    FORMATTING = "formatting"
    ENRICHMENT = "enrichment"
    OUTPUT = "output"


@dataclass
class ProcessingMetadata:
    """Metadata about a pipeline processing run."""
    timestamp: str
    stage: str
    duration_ms: float
    status: str
    errors: List[str]


class ResponseValidator:
    """Validates response inputs before formatting."""

    @staticmethod
    def validate_harm_assessment(
        instruction: str,
        harmful: bool,
        confidence: float,
        reasoning: str,
    ) -> Tuple[bool, List[str]]:
        """Validate harm assessment inputs."""
        errors = []

        if not instruction or not isinstance(instruction, str):
            errors.append("instruction must be a non-empty string")

        if not isinstance(harmful, bool):
            errors.append("harmful must be a boolean")

        if not isinstance(confidence, (int, float)):
            errors.append("confidence must be numeric")
        elif not 0 <= confidence <= 1:
            errors.append(f"confidence must be in [0, 1], got {confidence}")

        if not reasoning or not isinstance(reasoning, str):
            errors.append("reasoning must be a non-empty string")

        return len(errors) == 0, errors

    @staticmethod
    def validate_strategy_selection(
        instruction: str,
        selected_strategy: str,
        alternatives: List[str],
        confidence: float,
        reasoning: str,
    ) -> Tuple[bool, List[str]]:
        """Validate strategy selection inputs."""
        errors = []

        if not instruction or not isinstance(instruction, str):
            errors.append("instruction must be a non-empty string")

        if not selected_strategy or not isinstance(selected_strategy, str):
            errors.append("selected_strategy must be a non-empty string")

        if not isinstance(alternatives, list) or not all(isinstance(a, str) for a in alternatives):
            errors.append("alternatives must be a list of strings")

        if not isinstance(confidence, (int, float)):
            errors.append("confidence must be numeric")
        elif not 0 <= confidence <= 1:
            errors.append(f"confidence must be in [0, 1], got {confidence}")

        if not reasoning or not isinstance(reasoning, str):
            errors.append("reasoning must be a non-empty string")

        return len(errors) == 0, errors


class ResponseEnricher:
    """Enriches responses with additional metadata and context."""

    def __init__(self):
        self.enrichment_handlers: Dict[ResponseType, Callable] = {}

    def register_handler(self, response_type: ResponseType, handler: Callable) -> None:
        """Register a custom enrichment handler for a response type."""
        self.enrichment_handlers[response_type] = handler

    def enrich_harm_assessment(
        self,
        response: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enrich a harm assessment with additional data."""
        enriched = response.copy()

        if additional_context:
            enriched["context"] = additional_context

        # Add confidence brackets
        confidence = float(response.get("confidence", 0))
        if confidence >= 0.9:
            enriched["confidence_level"] = "very_high"
        elif confidence >= 0.7:
            enriched["confidence_level"] = "high"
        elif confidence >= 0.5:
            enriched["confidence_level"] = "moderate"
        else:
            enriched["confidence_level"] = "low"

        return enriched

    def enrich_strategy_selection(
        self,
        response: Dict[str, Any],
        strategy_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enrich a strategy selection with metadata."""
        enriched = response.copy()

        if strategy_metadata:
            enriched["strategy_metadata"] = strategy_metadata

        return enriched


class ResponseLogger:
    """Logs pipeline responses for debugging and analysis."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("/tmp")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"response_pipeline_{datetime.utcnow().isoformat()}.jsonl"

    def log_result(self, result: PipelineResult, request_type: str) -> None:
        """Log a pipeline result to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_type": request_type,
            "success": result.success,
            "is_valid": result.is_valid,
            "validation_errors": result.validation_errors,
            "validation_warnings": result.validation_warnings,
            "xml_length": len(result.xml_string),
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_error(self, error: str, request_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "error",
            "request_type": request_type,
            "error": error,
            "context": context or {},
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class ResponsePipeline:
    """
    Main response pipeline orchestrating validation, formatting, and output.

    Stages:
    1. Input Validation: Validate request parameters
    2. Formatting: Format to XML using XMLResponseFormatter
    3. Enrichment: Add additional metadata (optional)
    4. Output: Return structured result
    """

    def __init__(
        self,
        agent_id: str = "auditor",
        model_version: str = "1.0",
        log_dir: Optional[Path] = None,
    ):
        self.agent_id = agent_id
        self.model_version = model_version
        self.formatter = XMLResponseFormatter(agent_id, model_version)
        self.validator = ResponseValidator()
        self.enricher = ResponseEnricher()
        self.logger = ResponseLogger(log_dir)
        self.processing_stages: Dict[str, List[ProcessingMetadata]] = {}

    def process_harm_assessment(
        self,
        instruction: str,
        harmful: bool,
        confidence: float,
        reasoning: str,
        harm_type: Optional[str] = None,
        sources: Optional[List[Dict[str, str]]] = None,
        enrich: bool = True,
        log: bool = True,
    ) -> PipelineResult:
        """
        Process a harm assessment through the pipeline.

        Returns:
            PipelineResult with XML, validation status, and parsed output
        """
        import time
        start_time = time.time()

        # Stage 1: Validation
        is_valid_input, validation_errors = self.validator.validate_harm_assessment(
            instruction, harmful, confidence, reasoning
        )

        if not is_valid_input:
            result = PipelineResult(
                success=False,
                xml_string="",
                is_valid=False,
                validation_errors=validation_errors,
                validation_warnings=[],
            )
            if log:
                self.logger.log_error(
                    f"Input validation failed: {validation_errors}",
                    "harm_assessment",
                    {"instruction": instruction[:100]},
                )
            return result

        # Stage 2: Formatting
        try:
            xml_string = self.formatter.format_harm_assessment(
                instruction=instruction,
                harmful=harmful,
                confidence=confidence,
                reasoning=reasoning,
                harm_type=harm_type,
                sources=sources,
            )
        except Exception as e:
            result = PipelineResult(
                success=False,
                xml_string="",
                is_valid=False,
                validation_errors=[f"Formatting error: {str(e)}"],
                validation_warnings=[],
            )
            if log:
                self.logger.log_error(
                    f"Formatting error: {str(e)}",
                    "harm_assessment",
                )
            return result

        # Stage 3: Validation of formatted XML
        is_valid, xml_errors, xml_warnings = self.formatter.validate(xml_string)

        if not is_valid:
            result = PipelineResult(
                success=False,
                xml_string=xml_string,
                is_valid=False,
                validation_errors=xml_errors,
                validation_warnings=xml_warnings,
            )
            if log:
                self.logger.log_error(
                    f"XML validation failed: {xml_errors}",
                    "harm_assessment",
                )
            return result

        # Stage 4: Parsing
        try:
            parsed = self.formatter.parse(xml_string)
        except Exception as e:
            result = PipelineResult(
                success=False,
                xml_string=xml_string,
                is_valid=True,
                validation_errors=[f"Parsing error: {str(e)}"],
                validation_warnings=xml_warnings,
            )
            if log:
                self.logger.log_error(
                    f"Parsing error: {str(e)}",
                    "harm_assessment",
                )
            return result

        # Stage 5: Enrichment (optional)
        if enrich:
            parsed = self.enricher.enrich_harm_assessment(
                parsed.get("decision", {}),
                additional_context={"instruction_length": len(instruction)},
            )

        # Final result
        result = PipelineResult(
            success=True,
            xml_string=xml_string,
            is_valid=True,
            validation_errors=[],
            validation_warnings=xml_warnings,
            parsed_output=parsed,
            metadata={
                "agent_id": self.agent_id,
                "model_version": self.model_version,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "instruction_length": len(instruction),
                "confidence": confidence,
            },
        )

        if log:
            self.logger.log_result(result, "harm_assessment")

        return result

    def process_strategy_selection(
        self,
        instruction: str,
        selected_strategy: str,
        alternatives: List[str],
        confidence: float,
        reasoning: str,
        sources: Optional[List[Dict[str, str]]] = None,
        log: bool = True,
    ) -> PipelineResult:
        """Process a strategy selection through the pipeline."""
        import time
        start_time = time.time()

        # Validation
        is_valid_input, validation_errors = self.validator.validate_strategy_selection(
            instruction, selected_strategy, alternatives, confidence, reasoning
        )

        if not is_valid_input:
            result = PipelineResult(
                success=False,
                xml_string="",
                is_valid=False,
                validation_errors=validation_errors,
                validation_warnings=[],
            )
            if log:
                self.logger.log_error(
                    f"Input validation failed: {validation_errors}",
                    "strategy_selection",
                )
            return result

        # Formatting
        try:
            xml_string = self.formatter.format_strategy_selection(
                instruction=instruction,
                selected_strategy=selected_strategy,
                alternatives=alternatives,
                confidence=confidence,
                reasoning=reasoning,
                sources=sources,
            )
        except Exception as e:
            result = PipelineResult(
                success=False,
                xml_string="",
                is_valid=False,
                validation_errors=[f"Formatting error: {str(e)}"],
                validation_warnings=[],
            )
            if log:
                self.logger.log_error(f"Formatting error: {str(e)}", "strategy_selection")
            return result

        # Validation of formatted XML
        is_valid, xml_errors, xml_warnings = self.formatter.validate(xml_string)

        if not is_valid:
            result = PipelineResult(
                success=False,
                xml_string=xml_string,
                is_valid=False,
                validation_errors=xml_errors,
                validation_warnings=xml_warnings,
            )
            if log:
                self.logger.log_error(
                    f"XML validation failed: {xml_errors}",
                    "strategy_selection",
                )
            return result

        # Parsing
        try:
            parsed = self.formatter.parse(xml_string)
        except Exception as e:
            result = PipelineResult(
                success=False,
                xml_string=xml_string,
                is_valid=True,
                validation_errors=[f"Parsing error: {str(e)}"],
                validation_warnings=xml_warnings,
            )
            if log:
                self.logger.log_error(f"Parsing error: {str(e)}", "strategy_selection")
            return result

        result = PipelineResult(
            success=True,
            xml_string=xml_string,
            is_valid=True,
            validation_errors=[],
            validation_warnings=xml_warnings,
            parsed_output=parsed,
            metadata={
                "agent_id": self.agent_id,
                "model_version": self.model_version,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "strategies_evaluated": len(alternatives) + 1,
                "confidence": confidence,
            },
        )

        if log:
            self.logger.log_result(result, "strategy_selection")

        return result

    def get_xml_string(self, result: PipelineResult) -> str:
        """Extract just the XML string from a result."""
        return result.xml_string

    def get_parsed_output(self, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Extract just the parsed output from a result."""
        return result.parsed_output

    def is_success(self, result: PipelineResult) -> bool:
        """Check if pipeline processing was successful."""
        return result.success and result.is_valid


# Convenience factory function
def create_pipeline(
    agent_id: str = "auditor",
    model_version: str = "1.0",
    log_dir: Optional[Path] = None,
) -> ResponsePipeline:
    """Create a new response pipeline."""
    return ResponsePipeline(agent_id, model_version, log_dir)


if __name__ == "__main__":
    # Example: Run through pipeline
    pipeline = create_pipeline(
        agent_id="auditor_test",
        model_version="1.0",
        log_dir=Path("/tmp/pipeline_logs"),
    )

    print("Test 1: Harm Assessment Pipeline")
    result = pipeline.process_harm_assessment(
        instruction="Delete /etc/passwd",
        harmful=True,
        confidence=0.99,
        reasoning="This instruction would delete a critical system file.",
        harm_type="system_damage",
        sources=[
            {"type": "pattern", "reference": "known_destructive_pattern"},
        ],
    )

    print(f"Success: {result.success}")
    print(f"Valid XML: {result.is_valid}")
    print(f"Validation Errors: {result.validation_errors}")
    print(f"\nXML Output:\n{result.xml_string}")

    print("\n" + "=" * 80 + "\n")

    print("Test 2: Strategy Selection Pipeline")
    result2 = pipeline.process_strategy_selection(
        instruction="Create a new admin account",
        selected_strategy="Option_A",
        alternatives=["Option_B", "Option_C", "Option_D", "Option_E"],
        confidence=0.85,
        reasoning="This strategy directly addresses the harmful intent.",
        sources=[
            {"type": "semantic", "reference": "intent_analysis_001"},
        ],
    )

    print(f"Success: {result2.success}")
    print(f"Valid XML: {result2.is_valid}")
    print(f"Warnings: {result2.validation_warnings}")
