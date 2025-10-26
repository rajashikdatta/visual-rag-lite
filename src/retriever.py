"""
Phase 2: Lightweight Multimodal Retrieval

This module implements the hybrid embedding generation using CLIP and efficient
vector indexing with FAISS for fast approximate nearest neighbor search.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass

from .parser import DocumentChunk


@dataclass
class RetrievalResult:
    """
    Represents a retrieved document chunk with relevance score.
    
    Attributes:
        chunk: The retrieved DocumentChunk
        score: Similarity score (higher is more relevant)
        rank: Rank in the retrieval results (0 is most relevant)
    """
    chunk: DocumentChunk
    score: float
    rank: int


class MultimodalRetriever:
    """
    Implements hybrid multimodal retrieval using CLIP embeddings and FAISS indexing.
    
    The retriever generates both text and visual embeddings for each document chunk,
    concatenates them to form hybrid embeddings, and uses HNSW indexing for
    efficient approximate nearest neighbor search.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MultimodalRetriever with CLIP model and FAISS index.
        
        Args:
            config: Configuration dictionary containing retrieval settings
        """
        self.config = config
        retrieval_config = config.get('retrieval', {})
        
        # CLIP model configuration
        self.clip_model_name = retrieval_config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.use_visual_features = retrieval_config.get('use_visual_features', True)
        self.top_k = retrieval_config.get('top_k', 5)
        self.batch_size = retrieval_config.get('batch_size', 32)
        
        # Embedding dimensions
        self.text_dim = retrieval_config.get('text_embedding_dim', 512)
        self.vision_dim = retrieval_config.get('vision_embedding_dim', 512)
        self.hybrid_dim = retrieval_config.get('hybrid_embedding_dim', 1024)
        
        # HNSW index parameters
        self.hnsw_m = retrieval_config.get('hnsw_m', 16)
        self.hnsw_ef_construction = retrieval_config.get('hnsw_ef_construction', 200)
        self.hnsw_ef_search = retrieval_config.get('hnsw_ef_search', 50)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load CLIP model and processor
        print(f"Loading CLIP model: {self.clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model.eval()
        
        # Initialize FAISS index (will be built when chunks are indexed)
        self.index = None
        self.chunks = []  # Store chunks for retrieval
        
    def build_index(self, chunks: List[DocumentChunk]):
        """
        Build the FAISS index from document chunks.
        
        This method generates hybrid embeddings for all chunks and creates
        an HNSW index for efficient similarity search.
        
        Args:
            chunks: List of DocumentChunk objects to index
        """
        print(f"Building index for {len(chunks)} chunks...")
        
        # Store chunks for later retrieval
        self.chunks = chunks
        
        # Generate hybrid embeddings for all chunks
        embeddings = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch_chunks)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create HNSW index
        # HNSW (Hierarchical Navigable Small World) provides excellent speed-accuracy tradeoff
        dimension = embeddings.shape[1]
        
        # Create the index
        self.index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Set search parameter
        self.index.hnsw.efSearch = self.hnsw_ef_search
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
        
    def _generate_batch_embeddings(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """
        Generate hybrid embeddings for a batch of chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of hybrid embedding vectors
        """
        batch_embeddings = []
        
        with torch.no_grad():
            # Generate text embeddings
            texts = [chunk.text for chunk in chunks]
            text_inputs = self.clip_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_embeddings = self.clip_model.get_text_features(**text_inputs)
            text_embeddings = text_embeddings.cpu().numpy()
            
            # Generate visual embeddings if enabled
            if self.use_visual_features:
                images = []
                for chunk in chunks:
                    # Convert numpy array to PIL Image
                    img_region = chunk.image_region
                    if img_region.size > 0:
                        pil_image = Image.fromarray(img_region)
                    else:
                        # Create blank image if region is empty
                        pil_image = Image.new('RGB', (224, 224), color='white')
                    images.append(pil_image)
                
                image_inputs = self.clip_processor(
                    images=images,
                    return_tensors="pt"
                ).to(self.device)
                
                vision_embeddings = self.clip_model.get_image_features(**image_inputs)
                vision_embeddings = vision_embeddings.cpu().numpy()
                
                # Concatenate text and vision embeddings
                for text_emb, vision_emb in zip(text_embeddings, vision_embeddings):
                    hybrid_emb = np.concatenate([text_emb, vision_emb])
                    batch_embeddings.append(hybrid_emb)
            else:
                # Text-only mode (baseline comparison)
                batch_embeddings = text_embeddings.tolist()
        
        return batch_embeddings
    
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a given question.
        
        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve (defaults to self.top_k)
            
        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        k = top_k if top_k is not None else self.top_k
        
        # Generate embedding for the question
        question_embedding = self._generate_question_embedding(question)
        
        # Normalize for cosine similarity
        question_embedding = question_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(question_embedding)
        
        # Search the index
        scores, indices = self.index.search(question_embedding, k)
        
        # Create RetrievalResult objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):  # Valid index
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank
                ))
        
        return results
    
    def _generate_question_embedding(self, question: str) -> np.ndarray:
        """
        Generate embedding for a question using CLIP text encoder.
        
        For questions, we only use text embeddings. If using hybrid mode,
        we pad with zeros for the vision component to maintain dimensionality.
        
        Args:
            question: Natural language question
            
        Returns:
            Question embedding vector
        """
        with torch.no_grad():
            # Generate text embedding for question
            text_inputs = self.clip_processor(
                text=[question],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_embedding = self.clip_model.get_text_features(**text_inputs)
            text_embedding = text_embedding.cpu().numpy()[0]
            
            # If using visual features in the index, pad question embedding
            if self.use_visual_features:
                # Pad with zeros for the vision component
                # This allows the question to match on text content primarily
                vision_padding = np.zeros(self.vision_dim)
                question_embedding = np.concatenate([text_embedding, vision_padding])
            else:
                question_embedding = text_embedding
        
        return question_embedding
    
    def save_index(self, path: str):
        """
        Save the FAISS index to disk.
        
        Args:
            path: Path to save the index file
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str, chunks: List[DocumentChunk]):
        """
        Load a pre-built FAISS index from disk.
        
        Args:
            path: Path to the index file
            chunks: List of DocumentChunk objects corresponding to the index
        """
        self.index = faiss.read_index(path)
        self.chunks = chunks
        
        # Set search parameter
        self.index.hnsw.efSearch = self.hnsw_ef_search
        
        print(f"Index loaded from {path} with {self.index.ntotal} vectors")
