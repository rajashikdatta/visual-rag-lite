"""
Visual RAG-Lite Framework
A lightweight and efficient system for grounded document question answering.
"""

__version__ = "1.0.0"
__author__ = "Visual RAG-Lite Team"

from .parser import DocumentParser
from .retriever import MultimodalRetriever
from .generator import GroundedGenerator
from .pipeline import VisualRAGLitePipeline

__all__ = [
    "DocumentParser",
    "MultimodalRetriever",
    "GroundedGenerator",
    "VisualRAGLitePipeline",
]
