"""
Training Script for Visual RAG-Lite

This script handles the training of the Visual RAG-Lite model and baselines
using LoRA for parameter-efficient fine-tuning.
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from tqdm import tqdm

from src.generator import GroundedGenerator
from src.parser import DocumentParser
from src.retriever import MultimodalRetriever


def load_docvqa_dataset(data_path: str, split: str = 'train', max_samples: int = None) -> List[Dict]:
    """
    Load DocVQA dataset from disk.
    
    Args:
        data_path: Path to DocVQA data directory
        split: Dataset split ('train', 'val', or 'test')
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        List of examples with questions, answers, and document paths
    """
    # Determine the correct filename based on split
    if split == 'train':
        filename = 'train_v1.0_withQT.json'
    elif split == 'val':
        filename = 'val_v1.0_withQT.json'
    elif split == 'test':
        filename = 'test_v1.0.json'
    else:
        raise ValueError(f"Unknown split: {split}")
    
    annotations_file = Path(data_path) / filename
    
    if not annotations_file.exists():
        print(f"Warning: {annotations_file} not found. Using dummy data.")
        return create_dummy_dataset(num_samples=min(max_samples or 100, 100))
    
    print(f"Loading {split} data from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    samples = data['data']
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    return samples


def create_dummy_dataset(num_samples: int = 100) -> List[Dict]:
    """
    Create a dummy dataset for testing.
    
    Args:
        num_samples: Number of dummy samples to create
        
    Returns:
        List of dummy examples
    """
    dummy_data = []
    for i in range(num_samples):
        dummy_data.append({
            'question_id': i,
            'question': f'What is shown in section {i}?',
            'answers': [f'Answer {i}', f'Alternative answer {i}'],
            'document_id': f'doc_{i % 10}',
            'image_path': f'data/docvqa/images/doc_{i % 10}.png'
        })
    return dummy_data


def prepare_training_data(examples: List[Dict], parser: DocumentParser,
                         retriever: MultimodalRetriever, 
                         generator: GroundedGenerator) -> Dataset:
    """
    Prepare training data by parsing documents and retrieving context.
    
    Args:
        examples: List of training examples
        parser: DocumentParser instance
        retriever: MultimodalRetriever instance
        generator: GroundedGenerator instance
        
    Returns:
        HuggingFace Dataset ready for training
    """
    training_samples = []
    
    # Group examples by document to avoid re-parsing
    docs_to_examples = {}
    for ex in examples:
        doc_id = ex['document_id']
        if doc_id not in docs_to_examples:
            docs_to_examples[doc_id] = []
        docs_to_examples[doc_id].append(ex)
    
    print(f"Processing {len(docs_to_examples)} unique documents...")
    
    for doc_id, doc_examples in tqdm(docs_to_examples.items()):
        # Parse document once
        image_path = doc_examples[0]['image_path']
        
        if not Path(image_path).exists():
            print(f"Skipping {doc_id}: image not found")
            continue
        
        try:
            chunks = parser.parse_document(image_path)
            retriever.build_index(chunks)
            
            # Process all questions for this document
            for ex in doc_examples:
                question = ex['question']
                answer = ex['answers'][0]  # Use first answer as ground truth
                
                # Retrieve relevant chunks
                retrieved = retriever.retrieve(question, top_k=5)
                
                # Format context
                context_parts = []
                for result in retrieved:
                    chunk = result.chunk
                    context_parts.append(f"[{chunk.chunk_id}] {chunk.text}")
                context = "\n\n".join(context_parts)
                
                # Create training prompt
                prompt = generator.prompt_template.format(
                    context=context,
                    question=question
                )
                
                # Add answer and citation
                # Format: prompt + answer [citation: chunk_id]
                top_chunk_id = retrieved[0].chunk.chunk_id
                full_text = f"{prompt}{answer} [citation: {top_chunk_id}]"
                
                # Tokenize
                tokens = generator.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=generator.max_length,
                    return_tensors="pt"
                )
                
                training_samples.append({
                    'input_ids': tokens['input_ids'][0],
                    'attention_mask': tokens['attention_mask'][0],
                    'labels': tokens['input_ids'][0]  # For causal LM
                })
        
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            continue
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(training_samples)
    return dataset


def train_visual_rag_lite(config_path: str, data_path: str, output_dir: str,
                          use_lora: bool = True, max_train_samples: int = None,
                          max_val_samples: int = None):
    """
    Train the Visual RAG-Lite model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to DocVQA dataset
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA for PEFT
        max_train_samples: Maximum training samples (for testing)
        max_val_samples: Maximum validation samples (for testing)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override LoRA setting
    config['generation']['use_lora'] = use_lora
    
    print(f"Training Visual RAG-Lite {'with LoRA' if use_lora else 'with full fine-tuning'}")
    
    # Check system resources
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA is not available. Training on CPU will be very slow.")
        print("   Consider using a GPU or reducing the model size.\n")
    
    # Initialize components
    print("\nInitializing components...")
    try:
        parser = DocumentParser(config)
        retriever = MultimodalRetriever(config)
        print("Loading generator model (this may take a few minutes)...")
        generator = GroundedGenerator(config, training_mode=True)
    except Exception as e:
        print(f"\n❌ Error initializing components: {e}")
        print("\nTroubleshooting:")
        print("  - If you're running out of memory, try using a smaller model")
        print("  - Ensure you have enough RAM (16GB+ recommended)")
        print("  - Check your internet connection for model downloads")
        raise
    
    # Load datasets
    print("\nLoading training data...")
    train_examples = load_docvqa_dataset(data_path, split='train', max_samples=max_train_samples)
    val_examples = load_docvqa_dataset(data_path, split='val', max_samples=max_val_samples)
    
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")
    
    # Prepare training data
    print("\nPreparing training data (this may take a while)...")
    train_dataset = prepare_training_data(train_examples, parser, retriever, generator)
    val_dataset = prepare_training_data(val_examples, parser, retriever, generator)
    
    print(f"Prepared {len(train_dataset)} training samples")
    print(f"Prepared {len(val_dataset)} validation samples")
    
    # Train the model
    print("\nStarting training...")
    generator.train(train_dataset, val_dataset)
    
    # Save the model
    save_path = Path(output_dir) / ('lora_adapter' if use_lora else 'full_model')
    save_path.mkdir(parents=True, exist_ok=True)
    generator.save_model(str(save_path))
    
    print(f"\n✓ Training complete! Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Visual RAG-Lite")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to DocVQA dataset')
    parser.add_argument('--output', type=str, default='models/checkpoints',
                       help='Output directory for model checkpoints')
    parser.add_argument('--no-lora', action='store_true',
                       help='Use full fine-tuning instead of LoRA')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Maximum training samples (for testing with small dataset)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                       help='Maximum validation samples (for testing with small dataset)')
    
    args = parser.parse_args()
    
    train_visual_rag_lite(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        use_lora=not args.no_lora,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )


if __name__ == "__main__":
    main()
