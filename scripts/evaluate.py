"""
Evaluation Script for Visual RAG-Lite

This script evaluates the Visual RAG-Lite model and baselines on DocVQA datasets,
generating results tables as shown in the paper.
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List
import time
from tqdm import tqdm

from src.pipeline import VisualRAGLitePipeline
from src.evaluation import DocVQAEvaluator, EvaluationResult


def load_test_dataset(data_path: str, dataset_name: str = 'docvqa') -> List[Dict]:
    """
    Load test dataset.
    
    Args:
        data_path: Path to data directory
        dataset_name: Name of dataset ('docvqa' or 'infographicvqa')
        
    Returns:
        List of test examples
    """
    test_file = Path(data_path) / dataset_name / 'test_annotations.json'
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found. Using dummy data.")
        return create_dummy_test_data(num_samples=50)
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    return data['data']


def create_dummy_test_data(num_samples: int = 50) -> List[Dict]:
    """Create dummy test data for demonstration."""
    dummy_data = []
    for i in range(num_samples):
        dummy_data.append({
            'question_id': i,
            'question': f'What is the value in row {i}?',
            'answers': [f'Value {i}'],
            'document_id': f'test_doc_{i % 5}',
            'image_path': f'data/docvqa/images/test_doc_{i % 5}.png'
        })
    return dummy_data


def run_evaluation(pipeline: VisualRAGLitePipeline, test_examples: List[Dict],
                   evaluator: DocVQAEvaluator, model_name: str) -> Dict:
    """
    Run evaluation on test set.
    
    Args:
        pipeline: VisualRAGLitePipeline instance
        test_examples: List of test examples
        evaluator: DocVQAEvaluator instance
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary with predictions and metrics
    """
    predictions = []
    references = []
    
    # Group by document
    docs_to_examples = {}
    for ex in test_examples:
        doc_id = ex['document_id']
        if doc_id not in docs_to_examples:
            docs_to_examples[doc_id] = []
        docs_to_examples[doc_id].append(ex)
    
    print(f"\nEvaluating {model_name} on {len(test_examples)} questions...")
    print(f"Processing {len(docs_to_examples)} unique documents...")
    
    total_time = 0
    
    for doc_id, doc_examples in tqdm(docs_to_examples.items()):
        image_path = doc_examples[0]['image_path']
        
        if not Path(image_path).exists():
            print(f"Skipping {doc_id}: image not found")
            continue
        
        try:
            # Process document once
            pipeline.process_document(image_path)
            
            # Answer all questions for this document
            for ex in doc_examples:
                question = ex['question']
                gt_answer = ex['answers'][0]
                
                # Get prediction
                start_time = time.time()
                result = pipeline.answer_question(question)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                predictions.append({
                    'question_id': ex['question_id'],
                    'answer': result['answer'],
                    'citation': result['citation'],
                    'timing': {
                        'total_ms': inference_time * 1000
                    }
                })
                
                references.append({
                    'question_id': ex['question_id'],
                    'answer': gt_answer
                })
        
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            continue
    
    # Compute metrics
    print(f"\nComputing metrics...")
    eval_result = evaluator.evaluate(predictions, references)
    
    return {
        'model_name': model_name,
        'predictions': predictions,
        'metrics': eval_result,
        'total_time': total_time
    }


def generate_results_table(results_list: List[Dict], output_path: str):
    """
    Generate results table similar to Table 1 in the paper.
    
    Args:
        results_list: List of evaluation results for different models
        output_path: Path to save the results table
    """
    # Create markdown table
    table_lines = []
    table_lines.append("| Model | ANLS | Accuracy (%) | Trainable Params (M) | Inference Latency (ms/sample) | Model Size (MB) |")
    table_lines.append("|-------|------|--------------|----------------------|-------------------------------|-----------------|")
    
    for result in results_list:
        metrics = result['metrics']
        model_name = result['model_name']
        
        table_lines.append(
            f"| {model_name} | {metrics.anls:.2f} | {metrics.accuracy*100:.1f} | "
            f"{metrics.trainable_params_m:.1f} | {metrics.avg_latency_ms:.0f} | "
            f"{metrics.model_size_mb:.0f} |"
        )
    
    table_text = "\n".join(table_lines)
    
    # Save table
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Visual RAG-Lite Evaluation Results\n\n")
        f.write("## Main Results (DocVQA Test Set)\n\n")
        f.write(table_text)
        f.write("\n\n")
    
    print(f"\nResults table saved to {output_path}")
    print("\n" + table_text)


def generate_ablation_table(results_list: List[Dict], output_path: str):
    """
    Generate ablation study table similar to Table 2 in the paper.
    
    Args:
        results_list: List of ablation results
        output_path: Path to save the ablation table
    """
    # Create markdown table
    table_lines = []
    table_lines.append("| Configuration | ANLS | Accuracy (%) |")
    table_lines.append("|---------------|------|--------------|")
    
    for result in results_list:
        metrics = result['metrics']
        config_name = result['model_name']
        
        table_lines.append(
            f"| {config_name} | {metrics.anls:.2f} | {metrics.accuracy*100:.1f} |"
        )
    
    table_text = "\n".join(table_lines)
    
    # Append to output file
    with open(output_path, 'a') as f:
        f.write("## Ablation Studies\n\n")
        f.write(table_text)
        f.write("\n\n")
    
    print("\n" + table_text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual RAG-Lite")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--dataset', type=str, default='docvqa',
                       choices=['docvqa', 'infographicvqa'],
                       help='Dataset name')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='results/experiments',
                       help='Output directory for results')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation studies')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    test_examples = load_test_dataset(args.data, args.dataset)
    
    # Initialize evaluator
    evaluator = DocVQAEvaluator(config)
    
    # Run main evaluation
    results_list = []
    
    print("\n" + "="*60)
    print("VISUAL RAG-LITE EVALUATION")
    print("="*60)
    
    # Evaluate Visual RAG-Lite (LoRA)
    print("\n[1/1] Evaluating Visual RAG-Lite (LoRA)...")
    pipeline = VisualRAGLitePipeline(config_dict=config)
    
    if args.model_path:
        pipeline.generator.load_model(args.model_path)
    
    result = run_evaluation(pipeline, test_examples, evaluator, "Visual RAG-Lite (LoRA)")
    evaluator.print_results(result['metrics'])
    results_list.append(result)
    
    # Save individual results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'{args.dataset}_results.json'
    with open(results_file, 'w') as f:
        # Convert EvaluationResult to dict
        serializable_results = []
        for r in results_list:
            r_copy = r.copy()
            r_copy['metrics'] = {
                'anls': r['metrics'].anls,
                'accuracy': r['metrics'].accuracy,
                'f1': r['metrics'].f1,
                'total_samples': r['metrics'].total_samples,
                'correct_predictions': r['metrics'].correct_predictions,
                'avg_latency_ms': r['metrics'].avg_latency_ms,
                'model_size_mb': r['metrics'].model_size_mb,
                'trainable_params_m': r['metrics'].trainable_params_m
            }
            # Remove predictions to reduce file size
            r_copy.pop('predictions', None)
            serializable_results.append(r_copy)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate results table
    table_path = output_dir / f'{args.dataset}_results_table.md'
    generate_results_table(results_list, str(table_path))
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
