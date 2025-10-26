# Models Directory

This directory stores model checkpoints and pretrained models.

## Structure

```
models/
├── checkpoints/
│   ├── visual_rag_lite/      # LoRA fine-tuned model
│   ├── full_finetuned/        # Full fine-tuning baseline
│   └── text_only/             # Text-only RAG baseline
└── pretrained/
    └── (base models cached here)
```

## Model Files

### LoRA Adapters

When training with LoRA, only the adapter weights are saved:
- `adapter_config.json`: LoRA configuration
- `adapter_model.bin`: Adapter weights (~45 MB)

### Full Models

Full fine-tuned models include all parameters (~7.6 GB for 3.8B model).

## Loading Models

```python
from src.generator import GroundedGenerator

# Load LoRA model
generator = GroundedGenerator(config, training_mode=False)
generator.load_model('models/checkpoints/visual_rag_lite')
```
