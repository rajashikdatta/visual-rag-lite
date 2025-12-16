"""
Phase 1: Layout-Aware Document Parsing

This module implements the document parsing pipeline using PaddleOCR for OCR
and layout analysis, followed by visual-semantic chunking to create coherent
document chunks while preserving structural information.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Try to import PPStructure (may not be available in all versions)
try:
    from paddleocr import PPStructure
    HAS_PP_STRUCTURE = True
except ImportError:
    HAS_PP_STRUCTURE = False
    PPStructure = None


@dataclass
class DocumentChunk:
    """
    Represents a semantically coherent chunk of a document.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        text: OCR-extracted text content
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        chunk_type: Type of content (e.g., 'paragraph', 'table', 'figure', 'heading')
        image_region: Cropped image region corresponding to this chunk
        metadata: Additional information about the chunk
    """
    chunk_id: str
    text: str
    bbox: Tuple[int, int, int, int]
    chunk_type: str
    image_region: np.ndarray
    metadata: Dict = None


class DocumentParser:
    """
    Handles document image parsing using PaddleOCR and implements
    layout-aware chunking strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DocumentParser with OCR engine and chunking parameters.
        
        Args:
            config: Configuration dictionary containing OCR and chunking settings
        """
        self.config = config
        ocr_config = config.get('ocr', {})
        chunking_config = config.get('chunking', {})
        
        # Initialize PaddleOCR with PP-Structure for layout analysis
        self.use_structure = ocr_config.get('use_pp_structure', True) and HAS_PP_STRUCTURE
        
        if self.use_structure and HAS_PP_STRUCTURE:
            # PP-Structure provides layout analysis capabilities
            self.ocr_engine = PPStructure(
                lang=ocr_config.get('lang', 'en'),
                use_angle_cls=ocr_config.get('use_angle_cls', True)
            )
        else:
            # Standard PaddleOCR without layout analysis
            if not HAS_PP_STRUCTURE and ocr_config.get('use_pp_structure', True):
                print("Warning: PPStructure not available, falling back to standard PaddleOCR")
            self.ocr_engine = PaddleOCR(
                lang=ocr_config.get('lang', 'en'),
                use_angle_cls=ocr_config.get('use_angle_cls', True)
            )
        
        # Chunking parameters
        self.max_chunk_size = chunking_config.get('max_chunk_size', 512)
        self.overlap = chunking_config.get('overlap', 50)
        self.preserve_structure = chunking_config.get('preserve_structure', True)
        self.group_headings = chunking_config.get('group_headings', True)
        self.group_captions = chunking_config.get('group_captions', True)
        
    def parse_document(self, image_path: str) -> List[DocumentChunk]:
        """
        Main parsing pipeline: Extract text and layout, then create chunks.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            List of DocumentChunk objects representing the parsed document
        """
        # Step 1: Load the document image
        image = self._load_image(image_path)
        
        # Step 2: Perform OCR and layout analysis
        ocr_results = self._perform_ocr(image)
        
        # Step 3: Apply visual-semantic chunking
        chunks = self._create_chunks(ocr_results, image)
        
        return chunks
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess the document image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array in RGB format
        """
        # Load image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Convert BGR to RGB (OpenCV loads in BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _perform_ocr(self, image: np.ndarray) -> List[Dict]:
        """
        Perform OCR and layout analysis on the document image.
        
        Args:
            image: Document image as numpy array
            
        Returns:
            List of OCR results with text, bounding boxes, and layout information
        """
        ocr_results = []
        
        if self.use_structure:
            # Use PP-Structure for advanced layout analysis
            structure_results = self.ocr_engine(image)
            
            for idx, result in enumerate(structure_results):
                # Extract layout type (text, table, figure, title, etc.)
                layout_type = result.get('type', 'text')
                
                # Extract bounding box
                bbox = result.get('bbox', [0, 0, 0, 0])
                
                # Extract text content
                if 'res' in result:
                    # For text regions, combine all text lines
                    text_lines = []
                    for line in result['res']:
                        if isinstance(line, dict) and 'text' in line:
                            text_lines.append(line['text'])
                        elif isinstance(line, tuple) and len(line) > 1:
                            text_lines.append(line[1][0])
                    text = ' '.join(text_lines)
                else:
                    text = result.get('text', '')
                
                ocr_results.append({
                    'id': f'ocr_{idx}',
                    'text': text,
                    'bbox': bbox,
                    'type': layout_type,
                    'confidence': result.get('confidence', 1.0)
                })
        else:
            # Standard OCR without layout analysis
            ocr_output = self.ocr_engine.ocr(image, cls=True)
            
            for page_idx, page_result in enumerate(ocr_output):
                if page_result is None:
                    continue
                    
                for idx, line in enumerate(page_result):
                    bbox_points = line[0]
                    text_info = line[1]
                    
                    # Convert bbox points to [x_min, y_min, x_max, y_max] format
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    ocr_results.append({
                        'id': f'ocr_{page_idx}_{idx}',
                        'text': text_info[0],
                        'bbox': bbox,
                        'type': 'text',
                        'confidence': text_info[1]
                    })
        
        return ocr_results
    
    def _create_chunks(self, ocr_results: List[Dict], image: np.ndarray) -> List[DocumentChunk]:
        """
        Apply visual-semantic chunking to create coherent document chunks.
        
        This method groups OCR results based on spatial proximity and semantic context,
        preserving document structure (headings with paragraphs, captions with figures, etc.)
        
        Args:
            ocr_results: List of OCR results with text and layout information
            image: Original document image
            
        Returns:
            List of DocumentChunk objects
        """
        if not ocr_results:
            return []
        
        chunks = []
        
        # Sort OCR results by vertical position (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        # Group nearby elements and maintain structural relationships
        current_group = []
        previous_bottom = 0
        group_id = 0
        
        for result in sorted_results:
            bbox = result['bbox']
            current_top = bbox[1]
            
            # Check if this element should be grouped with previous elements
            # based on vertical proximity
            vertical_gap = current_top - previous_bottom
            avg_height = (bbox[3] - bbox[1]) if len(current_group) == 0 else \
                         np.mean([r['bbox'][3] - r['bbox'][1] for r in current_group])
            
            # Threshold for grouping (1.5 times average height)
            should_group = (vertical_gap < 1.5 * avg_height) if current_group else True
            
            # Special handling for headings and captions
            if self.preserve_structure:
                current_type = result['type']
                
                # Start new group for tables and figures
                if current_type in ['table', 'figure']:
                    if current_group:
                        chunk = self._create_chunk_from_group(current_group, image, group_id)
                        chunks.append(chunk)
                        group_id += 1
                        current_group = []
                    
                    # Create dedicated chunk for table/figure
                    chunk = self._create_chunk_from_group([result], image, group_id)
                    chunks.append(chunk)
                    group_id += 1
                    previous_bottom = bbox[3]
                    continue
            
            if should_group and len(current_group) > 0:
                # Check if combined text doesn't exceed max_chunk_size
                combined_text = ' '.join([r['text'] for r in current_group + [result]])
                
                if len(combined_text.split()) <= self.max_chunk_size:
                    current_group.append(result)
                else:
                    # Create chunk from current group and start new group
                    chunk = self._create_chunk_from_group(current_group, image, group_id)
                    chunks.append(chunk)
                    group_id += 1
                    current_group = [result]
            elif current_group:
                # Vertical gap too large, create chunk and start new group
                chunk = self._create_chunk_from_group(current_group, image, group_id)
                chunks.append(chunk)
                group_id += 1
                current_group = [result]
            else:
                # Start first group
                current_group = [result]
            
            previous_bottom = bbox[3]
        
        # Don't forget the last group
        if current_group:
            chunk = self._create_chunk_from_group(current_group, image, group_id)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_group(self, group: List[Dict], image: np.ndarray, 
                                  chunk_id: int) -> DocumentChunk:
        """
        Create a DocumentChunk from a group of OCR results.
        
        Args:
            group: List of OCR results to combine into a chunk
            image: Original document image
            chunk_id: Unique identifier for this chunk
            
        Returns:
            DocumentChunk object
        """
        # Combine text from all elements in the group
        combined_text = ' '.join([r['text'] for r in group])
        
        # Calculate bounding box that encompasses all elements
        all_bboxes = [r['bbox'] for r in group]
        x_min = min([bbox[0] for bbox in all_bboxes])
        y_min = min([bbox[1] for bbox in all_bboxes])
        x_max = max([bbox[2] for bbox in all_bboxes])
        y_max = max([bbox[3] for bbox in all_bboxes])
        combined_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        
        # Crop image region for this chunk
        image_region = image[combined_bbox[1]:combined_bbox[3], 
                            combined_bbox[0]:combined_bbox[2]]
        
        # Determine chunk type (prioritize specific types like table, figure)
        chunk_types = [r['type'] for r in group]
        if 'table' in chunk_types:
            chunk_type = 'table'
        elif 'figure' in chunk_types:
            chunk_type = 'figure'
        elif 'title' in chunk_types or 'heading' in chunk_types:
            chunk_type = 'heading'
        else:
            chunk_type = 'text'
        
        # Create metadata
        metadata = {
            'num_elements': len(group),
            'avg_confidence': np.mean([r.get('confidence', 1.0) for r in group]),
            'element_types': chunk_types
        }
        
        return DocumentChunk(
            chunk_id=f'chunk_{chunk_id}',
            text=combined_text,
            bbox=combined_bbox,
            chunk_type=chunk_type,
            image_region=image_region,
            metadata=metadata
        )
