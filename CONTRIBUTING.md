# Contributing to Squirrel Detection Project

Thank you for your interest in contributing to the Squirrel Detection Project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/yourusername/squirrel-detection-pi/issues) section
2. Create a new issue with a clear and descriptive title
3. Include detailed steps to reproduce the bug
4. Provide system information (OS, Python version, DeGirum version, etc.)
5. Include error messages and logs if applicable

### Suggesting Enhancements

1. Check if the enhancement has already been suggested
2. Create a new issue with the "enhancement" label
3. Describe the feature and its benefits
4. Provide use cases and examples if possible

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following the coding standards below
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to your branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📋 Development Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/squirrel-detection-pi.git
   cd squirrel-detection-pi
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8 black isort
   ```

4. **Set up pre-commit hooks (optional):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

## 📝 Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting and [isort](https://pycqa.github.io/isort/) for import sorting.

Format your code:
```bash
black .
isort .
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 .
```

## 📚 Documentation

- Update the README.md if you add new features
- Add docstrings to new functions and classes
- Update the project structure section if you add new files
- Include examples in docstrings

## 🔧 Hardware Testing

If you're contributing hardware-specific features:

1. Test on actual Raspberry Pi with Hailo hardware
2. Document any hardware-specific requirements
3. Include performance benchmarks if applicable
4. Test both GUI and headless modes

## 🚀 Release Process

1. Update version numbers in:
   - `setup.py`
   - `README.md` (if needed)
   - Any other version references

2. Update the changelog

3. Create a release tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

## 📞 Getting Help

- Check the [Issues](https://github.com/yourusername/squirrel-detection-pi/issues) section
- Review the [README.md](README.md) for setup instructions
- Contact the maintainers if you need assistance

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Squirrel Detection Project! 🐿️ 