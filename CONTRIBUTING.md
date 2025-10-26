# Contributing to Visual RAG-Lite

Thank you for your interest in contributing to Visual RAG-Lite! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/visual-rag-lite.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and ensure code quality
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run setup verification
python setup.py
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Maximum line length: 100 characters
- Use Black for code formatting: `black src`
- Use flake8 for linting: `flake8 src`

Before submitting a PR, please format your code:
```bash
black src scripts examples
flake8 src scripts examples
```

## Testing

Currently, the project uses manual testing. We welcome contributions to add automated tests!

To test your changes:
```bash
# Run setup verification
python setup.py

# Test imports
python -c "from src import DocumentParser, MultimodalRetriever, GroundedGenerator"

# Run examples
python examples/demo.py
```

## Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if you add new features
- Add examples for new functionality

Example docstring:
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Write clear commit messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
3. **Update documentation**: If you change functionality, update docs
4. **Test your changes**: Ensure nothing breaks
5. **Describe your changes**: Write a clear PR description

### PR Title Format

- `feat: Add new feature`
- `fix: Fix bug in module`
- `docs: Update documentation`
- `refactor: Refactor code`
- `test: Add tests`
- `chore: Update dependencies`

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Add unit tests and integration tests
- [ ] Add more baseline models (LLaVA, Gemma, etc.)
- [ ] Improve error handling and logging
- [ ] Add data preprocessing utilities
- [ ] Create Jupyter notebook tutorials

### Medium Priority
- [ ] Add support for more OCR engines (Tesseract, EasyOCR)
- [ ] Implement multi-document QA
- [ ] Add web interface/demo
- [ ] Docker containerization
- [ ] Performance optimizations

### Documentation
- [ ] Add more usage examples
- [ ] Create video tutorials
- [ ] Write blog posts about the framework
- [ ] Improve API documentation

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to reproduce**: Minimal code to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: 
   - OS and version
   - Python version
   - Package versions (`pip list`)
   - GPU info (if applicable)
6. **Error messages**: Full error traceback

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person

## Questions?

If you have questions:
- Open an issue with the `question` label
- Check existing issues and discussions
- Read the README.md and documentation

Thank you for contributing to Visual RAG-Lite! 🎉
