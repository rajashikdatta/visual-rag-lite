# Results Directory

This directory stores experimental results, evaluation metrics, and generated tables.

## Structure

```
results/
├── experiments/
│   ├── docvqa_results.json
│   ├── docvqa_results_table.md
│   ├── infographicvqa_results.json
│   └── infographicvqa_results_table.md
└── figures/
    ├── performance_comparison.png
    └── ablation_study.png
```

## Result Files

### JSON Results

Detailed results including predictions and metrics:
```json
{
  "model_name": "Visual RAG-Lite (LoRA)",
  "metrics": {
    "anls": 0.75,
    "accuracy": 0.661,
    "f1": 0.72,
    "avg_latency_ms": 240,
    "model_size_mb": 45,
    "trainable_params_m": 22.5
  }
}
```

### Markdown Tables

Results formatted as tables for inclusion in papers and reports.
