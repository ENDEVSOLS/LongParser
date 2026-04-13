"""Pipeline stages for LongParser document processing."""

from .orchestrator import PipelineOrchestrator, PipelineResult

# Public alias — docs and quickstart use this name
DocumentPipeline = PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator",
    "DocumentPipeline",
    "PipelineResult",
]
