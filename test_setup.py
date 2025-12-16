"""
Quick test script to verify the setup is working
"""
import sys
import torch
from pathlib import Path

print("=" * 60)
print("Visual RAG-Lite Setup Test")
print("=" * 60)

# Test 1: Python version
print(f"\n1. Python Version: {sys.version}")

# Test 2: PyTorch
print(f"\n2. PyTorch:")
print(f"   - Version: {torch.__version__}")
print(f"   - CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - CUDA Version: {torch.version.cuda}")
    print(f"   - Device Count: {torch.cuda.device_count()}")
    print(f"   - Current Device: {torch.cuda.current_device()}")
    print(f"   - Device Name: {torch.cuda.get_device_name(0)}")

# Test 3: Import modules
print(f"\n3. Testing imports...")
try:
    from src.parser import DocumentParser
    print("   ✓ DocumentParser imported")
except Exception as e:
    print(f"   ✗ DocumentParser failed: {e}")

try:
    from src.retriever import MultimodalRetriever
    print("   ✓ MultimodalRetriever imported")
except Exception as e:
    print(f"   ✗ MultimodalRetriever failed: {e}")

try:
    from src.generator import GroundedGenerator
    print("   ✓ GroundedGenerator imported")
except Exception as e:
    print(f"   ✗ GroundedGenerator failed: {e}")

# Test 4: Check data
print(f"\n4. Checking data directories...")
data_path = Path("data/docvqa")
if data_path.exists():
    files = list(data_path.glob("*.json"))
    print(f"   ✓ DocVQA data found: {len(files)} JSON files")
    for f in files:
        print(f"     - {f.name}")
else:
    print(f"   ✗ DocVQA data not found at {data_path}")

# Test 5: Try loading a small model (tokenizer only)
print(f"\n5. Testing model loading (tokenizer only)...")
try:
    from transformers import AutoTokenizer
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"   Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded successfully")
    print(f"   - Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"   ✗ Failed to load tokenizer: {e}")

print("\n" + "=" * 60)
print("Setup test complete!")
print("=" * 60)
