"""
Visual RAG-Lite Main Pipeline

This module implements the complete Visual RAG-Lite inference pipeline that
integrates all three phases: parsing, retrieval, and generation.
"""

import time
from typing import Dict, Tuple, Optional
from pathlib import Path
import yaml

from .parser import DocumentParser
from .retriever import MultimodalRetriever
from .generator import GroundedGenerator


class VisualRAGLitePipeline:
    """
    End-to-end Visual RAG-Lite pipeline for document question answering.
    
    This class implements the algorithm described in the paper, processing
    document images and questions to produce grounded answers with citations.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the Visual RAG-Lite pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Initialize components
        print("Initializing Visual RAG-Lite Pipeline...")
        
        # Phase 1: Document Parser
        print("\n[Phase 1] Initializing Document Parser...")
        self.parser = DocumentParser(self.config)
        
        # Phase 2: Multimodal Retriever
        print("\n[Phase 2] Initializing Multimodal Retriever...")
        self.retriever = MultimodalRetriever(self.config)
        
        # Phase 3: Grounded Generator
        print("\n[Phase 3] Initializing Grounded Generator...")
        self.generator = GroundedGenerator(self.config, training_mode=False)
        
        print("\n✓ Pipeline initialized successfully!")
        
        # Cache for parsed documents
        self._document_cache = {}
        self._index_built = False
    
    def process_document(self, document_image_path: str, force_reparse: bool = False):
        """
        Process a document image: parse and index it.
        
        Args:
            document_image_path: Path to the document image
            force_reparse: If True, reparse even if cached
        """
        doc_path = Path(document_image_path)
        
        # Check cache
        if not force_reparse and str(doc_path) in self._document_cache:
            print(f"Using cached parsing results for {doc_path.name}")
            chunks = self._document_cache[str(doc_path)]
        else:
            # Phase 1: Parse the document
            print(f"\n[Phase 1] Parsing document: {doc_path.name}")
            start_time = time.time()
            chunks = self.parser.parse_document(str(doc_path))
            parse_time = time.time() - start_time
            print(f"✓ Parsed {len(chunks)} chunks in {parse_time:.2f}s")
            
            # Cache the results
            self._document_cache[str(doc_path)] = chunks
        
        # Phase 2: Build the retrieval index
        print(f"\n[Phase 2] Building retrieval index...")
        start_time = time.time()
        self.retriever.build_index(chunks)
        index_time = time.time() - start_time
        print(f"✓ Index built in {index_time:.2f}s")
        
        self._index_built = True
    
    def answer_question(self, question: str, document_image_path: Optional[str] = None) -> Dict:
        """
        Answer a question about a document using the Visual RAG-Lite framework.
        
        This implements the main algorithm from the paper:
        1. Parse the document (if not already done)
        2. Retrieve relevant chunks
        3. Generate grounded answer with citation
        
        Args:
            question: Natural language question
            document_image_path: Optional path to document (if not already processed)
            
        Returns:
            Dictionary containing answer, citation, retrieved chunks, and timing info
        """
        # Process document if provided and not yet indexed
        if document_image_path and not self._index_built:
            self.process_document(document_image_path)
        
        if not self._index_built:
            raise ValueError("No document has been processed. Call process_document() first.")
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        
        # Phase 2: Retrieve relevant chunks
        print("\n[Phase 2] Retrieving relevant chunks...")
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(question)
        retrieval_time = time.time() - retrieval_start
        
        print(f"✓ Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s")
        for i, result in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {result.chunk.chunk_id} (score: {result.score:.4f})")
            print(f"     Preview: {result.chunk.text[:100]}...")
        
        # Phase 3: Generate grounded answer
        print("\n[Phase 3] Generating answer...")
        generation_start = time.time()
        answer, citation = self.generator.generate(question, retrieved_chunks)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - total_start_time
        
        print(f"✓ Answer generated in {generation_time:.2f}s")
        print(f"\n{'='*60}")
        print(f"Answer: {answer}")
        print(f"Citation: {citation}")
        print(f"{'='*60}")
        print(f"Total inference time: {total_time:.2f}s")
        
        return {
            'question': question,
            'answer': answer,
            'citation': citation,
            'retrieved_chunks': retrieved_chunks,
            'timing': {
                'retrieval_ms': retrieval_time * 1000,
                'generation_ms': generation_time * 1000,
                'total_ms': total_time * 1000
            }
        }
    
    def batch_answer(self, questions: list, document_image_path: Optional[str] = None) -> list:
        """
        Answer multiple questions about a document.
        
        Args:
            questions: List of questions
            document_image_path: Optional path to document (if not already processed)
            
        Returns:
            List of answer dictionaries
        """
        # Process document once
        if document_image_path and not self._index_built:
            self.process_document(document_image_path)
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n\nProcessing question {i}/{len(questions)}")
            result = self.answer_question(question)
            results.append(result)
        
        return results
    
    def save_index(self, index_path: str, chunks_path: str):
        """
        Save the retrieval index and chunks to disk for fast loading.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks (as pickle or json)
        """
        import pickle
        
        if not self._index_built:
            raise ValueError("No index to save. Process a document first.")
        
        # Save FAISS index
        self.retriever.save_index(index_path)
        
        # Save chunks
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.retriever.chunks, f)
        
        print(f"✓ Index and chunks saved")
    
    def load_index(self, index_path: str, chunks_path: str):
        """
        Load a pre-built retrieval index and chunks from disk.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks file
        """
        import pickle
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Load FAISS index
        self.retriever.load_index(index_path, chunks)
        
        self._index_built = True
        print(f"✓ Index and chunks loaded")


def main():
    """
    Example usage of the Visual RAG-Lite pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual RAG-Lite Inference")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--document', type=str, required=True,
                       help='Path to document image')
    parser.add_argument('--question', type=str, required=True,
                       help='Question to answer')
    parser.add_argument('--save-index', type=str, default=None,
                       help='Path to save index for reuse')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VisualRAGLitePipeline(config_path=args.config)
    
    # Answer question
    result = pipeline.answer_question(args.question, args.document)
    
    # Optionally save index
    if args.save_index:
        pipeline.save_index(
            f"{args.save_index}/index.faiss",
            f"{args.save_index}/chunks.pkl"
        )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
