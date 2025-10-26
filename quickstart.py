"""
Quick Start Script for Visual RAG-Lite

This script provides a simple command-line interface to test the framework
with minimal setup.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.pipeline import VisualRAGLitePipeline
    from src.utils import print_system_info, set_seed
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nPlease run: python setup.py")
    print("to verify your installation.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Visual RAG-Lite Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a question about a document
  python quickstart.py --document path/to/doc.png --question "What is the total?"
  
  # Use custom config
  python quickstart.py --config myconfig.yaml --document doc.png --question "..."
  
  # Show system information
  python quickstart.py --system-info
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--document', type=str,
                       help='Path to document image')
    parser.add_argument('--question', type=str,
                       help='Question to answer')
    parser.add_argument('--system-info', action='store_true',
                       help='Show system information and exit')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Show system info if requested
    if args.system_info:
        print_system_info()
        return
    
    # Validate arguments
    if not args.document or not args.question:
        parser.print_help()
        print("\n❌ Error: Both --document and --question are required")
        sys.exit(1)
    
    if not Path(args.document).exists():
        print(f"❌ Error: Document not found: {args.document}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print("Using default config might fail. Create config/config.yaml first.")
    
    try:
        print("\n" + "="*60)
        print("Visual RAG-Lite - Quick Start")
        print("="*60)
        
        # Initialize pipeline
        print(f"\nInitializing pipeline with config: {args.config}")
        pipeline = VisualRAGLitePipeline(config_path=args.config)
        
        # Process document and answer question
        print(f"\nDocument: {args.document}")
        print(f"Question: {args.question}")
        print("\nProcessing...")
        
        result = pipeline.answer_question(args.question, args.document)
        
        # Display results
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"\n📝 Answer: {result['answer']}")
        print(f"\n📍 Citation: {result['citation']}")
        print(f"\n⏱️  Inference Time: {result['timing']['total_ms']:.2f}ms")
        
        # Show retrieved chunks
        print(f"\n📚 Retrieved Chunks ({len(result['retrieved_chunks'])}):")
        for i, chunk_result in enumerate(result['retrieved_chunks'], 1):
            chunk = chunk_result.chunk
            print(f"\n  {i}. {chunk.chunk_id} (score: {chunk_result.score:.4f})")
            print(f"     Type: {chunk.chunk_type}")
            print(f"     Text preview: {chunk.text[:100]}...")
        
        print("\n" + "="*60)
        print("✓ Done!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
