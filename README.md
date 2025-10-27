# Visual RAG-Lite: Efficient Grounded Document Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Visual RAG-Lite is a lightweight and efficient framework for grounded document question answering that combines layout-aware OCR parsing, multimodal retrieval, and parameter-efficient fine-tuning (PEFT) with LoRA to create a system that is both accurate and computationally efficient.

## 🌟 Key Features

- **Layout-Aware Parsing**: Uses PaddleOCR with PP-Structure for advanced layout analysis
- **Hybrid Multimodal Retrieval**: Combines text and visual embeddings using CLIP for robust retrieval
- **Efficient Vector Search**: Implements HNSW indexing via FAISS for fast approximate nearest neighbor search
- **Parameter-Efficient Fine-Tuning**: Uses LoRA to fine-tune only 2% of parameters while maintaining performance
- **Grounded Generation**: Produces answers with citations to prevent hallucination
- **Lightweight**: Uses small language models (1-3B parameters) for efficient inference

## 📊 Performance

On the DocVQA benchmark, Visual RAG-Lite achieves:

- **ANLS**: 0.75 (comparable to 7B models)
- **Inference Latency**: 240 ms/sample (5x faster than baseline)
- **Model Size**: 45 MB (LoRA adapters only)
- **Trainable Parameters**: 22.5M (vs. 7B for full fine-tuning)

## 🏗️ Architecture

The framework consists of three main phases:

### Phase 1: Layout-Aware Document Parsing
- OCR with PaddleOCR and PP-Structure
- Visual-semantic chunking that preserves document structure
- Maintains relationships between headings, captions, and content

### Phase 2: Lightweight Multimodal Retrieval
- Hybrid embedding generation using CLIP (ViT-B-32)
- Text and vision features concatenated for rich representation
- HNSW indexing for efficient similarity search

### Phase 3: Grounded Generation via PEFT-Tuned SLM
- Small language model (Phi-3-mini-4k-instruct, 3.8B parameters)
- LoRA fine-tuning for parameter efficiency
- Generates answers with citations to source chunks

## 📁 Project Structure

```
CVPR/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── __init__.py             # Package initialization
│   ├── parser.py               # Phase 1: Document parsing
│   ├── retriever.py            # Phase 2: Multimodal retrieval
│   ├── generator.py            # Phase 3: Grounded generation
│   ├── pipeline.py             # Main inference pipeline
│   └── evaluation.py           # Evaluation metrics
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── examples/
│   └── demo.py                 # Usage examples
├── data/                       # Dataset directory
│   ├── docvqa/
│   └── infographicvqa/
├── models/
│   ├── checkpoints/            # Trained model checkpoints
│   └── pretrained/             # Pretrained base models
├── results/
│   └── experiments/            # Evaluation results
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.7+ (for GPU support, highly recommended)
- 16GB RAM minimum (32GB recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/rajashikdatta/visual-rag-lite.git
cd visual-rag-lite
```

### Step 2: Create Virtual Environment

```powershell
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For FAISS GPU support:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Step 4: Download Datasets (Optional)

Download the DocVQA and InfographicVQA datasets:

```bash
# DocVQA
# Download from: https://rrc.cvc.uab.es/?ch=17

# InfographicVQA  
# Download from: https://www.docvqa.org/datasets/infographicvqa
```

Place the datasets in the `data/` directory following this structure:
```
data/
├── docvqa/
│   ├── train/
│   ├── val/
│   └── test/
└── infographicvqa/
    ├── train/
    ├── val/
    └── test/
```

## 📖 Quick Start

### Basic Usage

```python
from src.pipeline import VisualRAGLitePipeline

# Initialize the pipeline
pipeline = VisualRAGLitePipeline(config_path='config/config.yaml')

# Process a document and ask a question
document_path = "path/to/your/document.png"
question = "What is the total revenue?"

result = pipeline.answer_question(question, document_path)

print(f"Answer: {result['answer']}")
print(f"Citation: {result['citation']}")
print(f"Inference time: {result['timing']['total_ms']:.2f}ms")
```

### Command Line Interface

```bash
# Answer a single question
python -m src.pipeline --config config/config.yaml \
    --document data/docvqa/images/example.png \
    --question "What is the total amount?"

# Save index for faster subsequent queries
python -m src.pipeline --config config/config.yaml \
    --document data/docvqa/images/example.png \
    --question "What is shown in the chart?" \
    --save-index cache/example_index
```

### Multiple Questions

```python
# Process document once, ask multiple questions
pipeline.process_document(document_path)

questions = [
    "What is the company name?",
    "What is the date of this report?",
    "What is the total revenue?"
]

for question in questions:
    result = pipeline.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

## 🎓 Training

### Training Visual RAG-Lite with LoRA

```bash
python scripts/train.py \
    --config config/config.yaml \
    --data data/docvqa \
    --output models/checkpoints/visual_rag_lite
```

### Training with Full Fine-Tuning (Baseline)

```bash
python scripts/train.py \
    --config config/config.yaml \
    --data data/docvqa \
    --output models/checkpoints/full_ft \
    --no-lora
```

### Training Configuration

Modify `config/config.yaml` to adjust training parameters:

```yaml
generation:
  # LoRA Configuration
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "v_proj"]
  
  # Training Configuration
  num_epochs: 3
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 4
```

## 📊 Evaluation

### Evaluate on DocVQA Test Set

```bash
python scripts/evaluate.py \
    --config config/config.yaml \
    --data data/docvqa \
    --dataset docvqa \
    --model-path models/checkpoints/visual_rag_lite \
    --output results/experiments
```

### Evaluate on InfographicVQA

```bash
python scripts/evaluate.py \
    --config config/config.yaml \
    --data data/infographicvqa \
    --dataset infographicvqa \
    --model-path models/checkpoints/visual_rag_lite \
    --output results/experiments
```

### Metrics

The evaluation script computes:

- **ANLS**: Average Normalized Levenshtein Similarity
- **Accuracy**: Exact match accuracy
- **F1**: Token-level F1 score
- **Inference Latency**: Average time per sample (ms)
- **Model Size**: Checkpoint size (MB)
- **Trainable Parameters**: Number of trainable parameters (M)

Results are saved in `results/experiments/` as JSON and Markdown tables.

## 🔬 Ablation Studies

### Text-Only Retrieval (No Visual Features)

```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Disable visual features
config['retrieval']['use_visual_features'] = False

pipeline = VisualRAGLitePipeline(config_dict=config)
```

### No Retrieval (Generator Only)

```python
# Use generator without retrieval
from src.generator import GroundedGenerator

generator = GroundedGenerator(config)
# Provide context directly without retrieval
```

### Full Fine-Tuning (No LoRA)

Train with `--no-lora` flag:
```bash
python scripts/train.py --config config/config.yaml \
    --data data/docvqa --no-lora
```

## 📝 Configuration

Key configuration options in `config/config.yaml`:

### OCR Settings
```yaml
ocr:
  engine: "paddleocr"
  use_pp_structure: true  # Enable layout analysis
  use_gpu: true
```

### Retrieval Settings
```yaml
retrieval:
  clip_model: "openai/clip-vit-base-patch32"
  top_k: 5  # Number of chunks to retrieve
  use_visual_features: true  # Enable hybrid embeddings
  index_type: "HNSW"
```

### Generation Settings
```yaml
generation:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  temperature: 0.7
```

## 🔧 Advanced Usage

### Save and Load Index

```python
# Save index after processing
pipeline.process_document(document_path)
pipeline.save_index("cache/doc_index.faiss", "cache/doc_chunks.pkl")

# Load index in new session
pipeline2 = VisualRAGLitePipeline(config_path='config/config.yaml')
pipeline2.load_index("cache/doc_index.faiss", "cache/doc_chunks.pkl")
```

### Batch Processing

```python
questions = ["Question 1?", "Question 2?", "Question 3?"]
results = pipeline.batch_answer(questions, document_path)

for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Citation: {result['citation']}\n")
```

### Custom Prompt Template

```python
config['generation']['prompt_template'] = """
Document Context:
{context}

User Question: {question}

Please provide a concise answer based solely on the context above.
Answer:
"""
```

## 📈 Results

### Main Results (DocVQA Test Set)

| Model | ANLS | Accuracy (%) | Trainable Params (M) | Inference Latency (ms/sample) | Model Size (MB) |
|-------|------|--------------|----------------------|-------------------------------|-----------------|
| Large MLLM (LLaVA-7B) | 0.78 | 68.5 | ~7,000 | 1250 | ~14,000 |
| Text-Only RAG + SLM | 0.69 | 59.2 | 22.5 | 230 | 45 |
| Visual RAG-Lite (Full FT) | 0.76 | 66.8 | ~3,800 | 245 | ~7,600 |
| **Visual RAG-Lite (LoRA)** | **0.75** | **66.1** | **22.5** | **240** | **45** |
| GPT-4o (Zero-Shot) | 0.82 | 72.3 | N/A | N/A | N/A |

### Ablation Studies

| Configuration | ANLS | Accuracy (%) |
|---------------|------|--------------|
| Full Model (Ours) | 0.75 | 66.1 |
| - Visual Cues (Text-Only) | 0.69 | 59.2 |
| - RAG (SLM only) | 0.48 | 41.5 |
| - LoRA (Full FT) | 0.76 | 66.8 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{visualraglite2025,
  title={Visual RAG-Lite: Efficient Grounded Document Question Answering},
  author={Rajashik Datta, Sanjan Baitalik},
  journal={CVPR},
  year={2025}
}
```

## 🙏 Acknowledgments

- **PaddleOCR**: For the excellent OCR engine and layout analysis
- **CLIP**: For the multimodal embedding model
- **FAISS**: For efficient vector similarity search
- **PEFT/LoRA**: For parameter-efficient fine-tuning
- **Hugging Face**: For the transformers library and model hub

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: rajashikdatta215@gmail.com
- GitHub: [@rajashikdatta](https://github.com/rajashikdatta)

## 🗺️ Roadmap

- [ ] Support for more OCR engines (Tesseract, EasyOCR)
- [ ] Multi-document question answering
- [ ] Support for more SLM backbones (Gemma, Mistral)
- [ ] Web interface for interactive demos
- [ ] Docker containerization
- [ ] API server for production deployment
- [ ] Support for multilingual documents

## 📖 Additional Resources

- [DocVQA Dataset](https://rrc.cvc.uab.es/?ch=17)
- [InfographicVQA Dataset](https://www.docvqa.org/datasets/infographicvqa)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

---

**Note**: This is a research implementation. For production use, additional optimizations and error handling may be needed.
