"""
Setup and Installation Verification Script

This script checks the installation and sets up the Visual RAG-Lite framework.
"""

import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_python_version():
    """Check if Python version is compatible."""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    else:
        print("✓ Python version is compatible")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print_section("Checking Dependencies")
    
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'paddleocr',
        'faiss',
        'PIL',
        'numpy',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed")
        return True


def check_gpu():
    """Check GPU availability."""
    print_section("Checking GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA is not available - will use CPU")
            print("  For better performance, install CUDA and PyTorch with GPU support")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def create_directories():
    """Create necessary directories."""
    print_section("Creating Directories")
    
    directories = [
        'data/docvqa',
        'data/infographicvqa',
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'models/pretrained',
        'results/experiments',
        'results/figures',
        'logs',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
    
    print("\n✓ All directories created")


def verify_config():
    """Verify configuration file exists."""
    print_section("Verifying Configuration")
    
    config_path = Path('config/config.yaml')
    
    if config_path.exists():
        print(f"✓ Configuration file found: {config_path}")
        return True
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False


def test_import():
    """Test importing the package."""
    print_section("Testing Package Import")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from src import DocumentParser, MultimodalRetriever, GroundedGenerator, VisualRAGLitePipeline
        print("✓ Successfully imported DocumentParser")
        print("✓ Successfully imported MultimodalRetriever")
        print("✓ Successfully imported GroundedGenerator")
        print("✓ Successfully imported VisualRAGLitePipeline")
        
        print("\n✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_section("Next Steps")
    
    print("""
1. Download datasets:
   - DocVQA: https://rrc.cvc.uab.es/?ch=17
   - InfographicVQA: https://www.docvqa.org/datasets/infographicvqa
   
2. Place datasets in the data/ directory

3. Run a quick test:
   python examples/demo.py

4. Train the model:
   python scripts/train.py --config config/config.yaml --data data/docvqa

5. Evaluate the model:
   python scripts/evaluate.py --config config/config.yaml --data data/docvqa

For more information, see README.md
""")


def main():
    """Main setup function."""
    print("\n" + "*"*60)
    print("*" + " "*58 + "*")
    print("*" + " "*10 + "Visual RAG-Lite Setup Verification" + " "*14 + "*")
    print("*" + " "*58 + "*")
    print("*"*60)
    
    checks = []
    
    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("GPU Support", check_gpu()))
    
    # Setup tasks
    create_directories()
    checks.append(("Configuration", verify_config()))
    checks.append(("Package Import", test_import()))
    
    # Summary
    print_section("Setup Summary")
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
        if not passed and name in ["Python Version", "Dependencies"]:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
    else:
        print("\n⚠ Setup completed with some issues.")
        print("Please resolve the failed checks before proceeding.")
        print("\nFor help, see README.md or open an issue on GitHub.")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
