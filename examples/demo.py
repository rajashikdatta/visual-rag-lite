"""
Example Usage of Visual RAG-Lite Framework

This script demonstrates how to use the Visual RAG-Lite framework
for document question answering.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VisualRAGLitePipeline


def example_single_question():
    """Example: Answer a single question about a document."""
    print("="*60)
    print("Example 1: Single Question Answering")
    print("="*60)
    
    # Initialize pipeline with config
    pipeline = VisualRAGLitePipeline(config_path='config/config.yaml')
    
    # Example document and question
    document_path = "data/docvqa/images/example_document.png"
    question = "What is the total revenue for Q4?"
    
    # Answer the question
    result = pipeline.answer_question(question, document_path)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Citation: {result['citation']}")
    print(f"Inference time: {result['timing']['total_ms']:.2f}ms")


def example_multiple_questions():
    """Example: Answer multiple questions about the same document."""
    print("\n" + "="*60)
    print("Example 2: Multiple Questions")
    print("="*60)
    
    # Initialize pipeline
    pipeline = VisualRAGLitePipeline(config_path='config/config.yaml')
    
    # Process document once
    document_path = "data/docvqa/images/example_document.png"
    pipeline.process_document(document_path)
    
    # Ask multiple questions
    questions = [
        "What is the company name?",
        "What is the total revenue?",
        "What is the year of this report?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = pipeline.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Citation: {result['citation']}")


def example_save_and_load_index():
    """Example: Save and load index for faster processing."""
    print("\n" + "="*60)
    print("Example 3: Save and Load Index")
    print("="*60)
    
    # Initialize pipeline
    pipeline = VisualRAGLitePipeline(config_path='config/config.yaml')
    
    document_path = "data/docvqa/images/example_document.png"
    
    # Process and save index
    print("\nProcessing document and saving index...")
    pipeline.process_document(document_path)
    pipeline.save_index(
        index_path="cache/example_index.faiss",
        chunks_path="cache/example_chunks.pkl"
    )
    
    # Create new pipeline and load index
    print("\nLoading saved index...")
    pipeline2 = VisualRAGLitePipeline(config_path='config/config.yaml')
    pipeline2.load_index(
        index_path="cache/example_index.faiss",
        chunks_path="cache/example_chunks.pkl"
    )
    
    # Answer question using loaded index
    question = "What is the main topic?"
    result = pipeline2.answer_question(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")


def example_text_only_mode():
    """Example: Use text-only retrieval (ablation study)."""
    print("\n" + "="*60)
    print("Example 4: Text-Only Mode (No Visual Features)")
    print("="*60)
    
    import yaml
    
    # Load config and disable visual features
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['retrieval']['use_visual_features'] = False
    
    # Initialize pipeline with modified config
    pipeline = VisualRAGLitePipeline(config_dict=config)
    
    document_path = "data/docvqa/images/example_document.png"
    question = "What is the main conclusion?"
    
    result = pipeline.answer_question(question, document_path)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Citation: {result['citation']}")
    print("\n(This uses only text embeddings, no visual features)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*20 + "VISUAL RAG-LITE EXAMPLES")
    print("="*80)
    
    try:
        # Note: These examples assume you have a document at the specified path
        # If not, they will fail gracefully
        
        print("\nNote: These examples require document images to be present.")
        print("For demonstration purposes, create dummy data or use your own documents.\n")
        
        # Uncomment the examples you want to run:
        
        # example_single_question()
        # example_multiple_questions()
        # example_save_and_load_index()
        # example_text_only_mode()
        
        print("\nTo run examples, uncomment the example functions in the main() function.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure document images are in the correct location.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
