"""
Utility Functions for Visual RAG-Lite

This module contains helper functions for data processing, visualization,
and other common tasks.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_json(file_path: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'frozen': total - trainable,
        'total': total,
        'trainable_percent': (trainable / total * 100) if total > 0 else 0
    }


def plot_results_comparison(results_list: List[Dict], output_path: str):
    """
    Plot comparison of different models' performance.
    
    Args:
        results_list: List of result dictionaries
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visual RAG-Lite: Performance Comparison', fontsize=16)
    
    model_names = [r['model_name'] for r in results_list]
    anls_scores = [r['metrics']['anls'] for r in results_list]
    accuracies = [r['metrics']['accuracy'] * 100 for r in results_list]
    latencies = [r['metrics']['avg_latency_ms'] for r in results_list]
    model_sizes = [r['metrics']['model_size_mb'] for r in results_list]
    
    # ANLS scores
    axes[0, 0].bar(model_names, anls_scores, color='steelblue')
    axes[0, 0].set_title('ANLS Score')
    axes[0, 0].set_ylabel('ANLS')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Accuracy
    axes[0, 1].bar(model_names, accuracies, color='coral')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Latency
    axes[1, 0].bar(model_names, latencies, color='seagreen')
    axes[1, 0].set_title('Inference Latency')
    axes[1, 0].set_ylabel('Latency (ms/sample)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Model size
    axes[1, 1].bar(model_names, model_sizes, color='mediumpurple')
    axes[1, 1].set_title('Model Size')
    axes[1, 1].set_ylabel('Size (MB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def print_config(config: Dict, indent: int = 0):
    """
    Pretty print configuration dictionary.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    return info


def print_system_info():
    """Print system and environment information."""
    import platform
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    
    device_info = get_device_info()
    print(f"\nCUDA available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"CUDA version: {device_info['cuda_version']}")
        print(f"GPU count: {device_info['cuda_device_count']}")
        print(f"GPU name: {device_info['cuda_device_name']}")
    print("="*60 + "\n")
