"""
Evaluation Metrics and Utilities

This module implements evaluation metrics for DocQA tasks including ANLS, accuracy,
and efficiency measurements (latency, model size, trainable parameters).
"""

import time
import numpy as np
from typing import List, Dict, Tuple
import Levenshtein
from pathlib import Path
import torch
import json
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    anls: float
    accuracy: float
    f1: float
    total_samples: int
    correct_predictions: int
    avg_latency_ms: float
    model_size_mb: float
    trainable_params_m: float


class ANLSMetric:
    """
    Average Normalized Levenshtein Similarity (ANLS) metric.
    
    ANLS is the standard metric for DocVQA tasks. It uses normalized Levenshtein
    distance and applies a threshold for counting as correct.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize ANLS metric.
        
        Args:
            threshold: Threshold for considering an answer correct (default: 0.5)
        """
        self.threshold = threshold
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute ANLS score for a set of predictions and references.
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
            
        Returns:
            ANLS score between 0 and 1
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self._anls_single(pred, ref)
            scores.append(score)
        
        return np.mean(scores)
    
    def _anls_single(self, prediction: str, reference: str) -> float:
        """
        Compute ANLS for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Ground truth answer
            
        Returns:
            ANLS score between 0 and 1
        """
        # Normalize strings (lowercase, strip whitespace)
        pred = prediction.lower().strip()
        ref = reference.lower().strip()
        
        # Handle empty strings
        if len(ref) == 0:
            return 1.0 if len(pred) == 0 else 0.0
        
        # Compute Levenshtein distance
        distance = Levenshtein.distance(pred, ref)
        
        # Normalize by length of reference
        max_len = max(len(pred), len(ref))
        if max_len == 0:
            return 1.0
        
        normalized_distance = distance / max_len
        
        # Compute normalized similarity
        nl_similarity = 1.0 - normalized_distance
        
        # Apply threshold
        if nl_similarity < self.threshold:
            return 0.0
        else:
            return nl_similarity


class AccuracyMetric:
    """
    Exact match accuracy metric.
    """
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute accuracy (exact match) for predictions.
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
            
        Returns:
            Accuracy between 0 and 1
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
        
        correct = 0
        for pred, ref in zip(predictions, references):
            # Normalize and compare
            if pred.lower().strip() == ref.lower().strip():
                correct += 1
        
        return correct / len(predictions)


class F1Metric:
    """
    Token-level F1 score metric.
    """
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute token-level F1 score.
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
            
        Returns:
            Average F1 score
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            f1 = self._f1_single(pred, ref)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _f1_single(self, prediction: str, reference: str) -> float:
        """
        Compute F1 for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Ground truth answer
            
        Returns:
            F1 score
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Calculate precision and recall
        common_tokens = pred_tokens.intersection(ref_tokens)
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


class EfficiencyMetrics:
    """
    Utilities for measuring efficiency metrics: latency, model size, trainable parameters.
    """
    
    @staticmethod
    def measure_latency(inference_fn, num_samples: int = 100) -> float:
        """
        Measure average inference latency.
        
        Args:
            inference_fn: Function to measure (should take no arguments)
            num_samples: Number of samples to average over
            
        Returns:
            Average latency in milliseconds
        """
        latencies = []
        
        # Warmup
        for _ in range(5):
            inference_fn()
        
        # Measure
        for _ in range(num_samples):
            start = time.time()
            inference_fn()
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return np.mean(latencies)
    
    @staticmethod
    def get_model_size(model_path: str) -> float:
        """
        Get the size of a saved model in megabytes.
        
        Args:
            model_path: Path to model checkpoint directory
            
        Returns:
            Model size in MB
        """
        path = Path(model_path)
        
        if not path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        total_size = 0
        
        # Sum up all files in the directory
        if path.is_dir():
            for file in path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
        else:
            total_size = path.stat().st_size
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        return size_mb
    
    @staticmethod
    def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """
        Count trainable and total parameters in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        return trainable_params, total_params


class DocVQAEvaluator:
    """
    Main evaluator for DocVQA tasks.
    
    Computes all metrics (ANLS, accuracy, F1) and efficiency measurements.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        eval_config = config.get('evaluation', {})
        
        # Initialize metrics
        anls_threshold = eval_config.get('anls_threshold', 0.5)
        self.anls_metric = ANLSMetric(threshold=anls_threshold)
        self.accuracy_metric = AccuracyMetric()
        self.f1_metric = F1Metric()
        
        # Efficiency measurement settings
        self.measure_latency = eval_config.get('measure_latency', True)
        self.measure_memory = eval_config.get('measure_memory', True)
        self.num_latency_samples = eval_config.get('num_latency_samples', 100)
    
    def evaluate(self, predictions: List[Dict], references: List[Dict],
                 model=None, model_path: str = None) -> EvaluationResult:
        """
        Perform complete evaluation with all metrics.
        
        Args:
            predictions: List of prediction dicts with 'answer' key
            references: List of reference dicts with 'answer' key
            model: Optional PyTorch model for parameter counting
            model_path: Optional path to saved model for size measurement
            
        Returns:
            EvaluationResult object with all metrics
        """
        # Extract answers
        pred_answers = [p['answer'] for p in predictions]
        ref_answers = [r['answer'] for r in references]
        
        # Compute task metrics
        print("Computing ANLS...")
        anls = self.anls_metric.compute(pred_answers, ref_answers)
        
        print("Computing Accuracy...")
        accuracy = self.accuracy_metric.compute(pred_answers, ref_answers)
        
        print("Computing F1...")
        f1 = self.f1_metric.compute(pred_answers, ref_answers)
        
        # Count correct predictions
        correct = sum(1 for p, r in zip(pred_answers, ref_answers)
                     if p.lower().strip() == r.lower().strip())
        
        # Compute efficiency metrics
        avg_latency = 0.0
        if self.measure_latency and 'timing' in predictions[0]:
            # Use timing from predictions if available
            latencies = [p['timing']['total_ms'] for p in predictions if 'timing' in p]
            avg_latency = np.mean(latencies) if latencies else 0.0
        
        model_size = 0.0
        if model_path:
            model_size = EfficiencyMetrics.get_model_size(model_path)
        
        trainable_params = 0.0
        if model:
            trainable, total = EfficiencyMetrics.count_trainable_parameters(model)
            trainable_params = trainable / 1e6  # Convert to millions
        
        result = EvaluationResult(
            anls=anls,
            accuracy=accuracy,
            f1=f1,
            total_samples=len(predictions),
            correct_predictions=correct,
            avg_latency_ms=avg_latency,
            model_size_mb=model_size,
            trainable_params_m=trainable_params
        )
        
        return result
    
    def print_results(self, result: EvaluationResult):
        """
        Pretty print evaluation results.
        
        Args:
            result: EvaluationResult object
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Samples: {result.total_samples}")
        print(f"Correct Predictions: {result.correct_predictions}")
        print("\nTask Performance:")
        print(f"  ANLS:     {result.anls:.4f}")
        print(f"  Accuracy: {result.accuracy*100:.2f}%")
        print(f"  F1:       {result.f1:.4f}")
        print("\nEfficiency:")
        print(f"  Avg Latency:        {result.avg_latency_ms:.2f} ms/sample")
        print(f"  Model Size:         {result.model_size_mb:.2f} MB")
        print(f"  Trainable Params:   {result.trainable_params_m:.2f} M")
        print("="*60 + "\n")
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            result: EvaluationResult object
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"Results saved to {output_path}")
