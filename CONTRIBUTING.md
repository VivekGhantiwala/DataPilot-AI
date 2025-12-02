# Contributing to Data Analysis AI Project

First off, thank you for considering contributing to this project! It's people like you that make this project such a great tool for the data science community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Fork the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/VivekGhantiwala/data-analysis-ai-project.git
   cd data-analysis-ai-project
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/data-analysis-ai-project.git
   ```

## 💡 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title** for the issue
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** to demonstrate the steps
- **Describe the behavior you observed** and what you expected
- **Include screenshots** if applicable
- **Include your environment details** (OS, Python version, package versions)

**Bug Report Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g., Windows 11, Ubuntu 22.04]
 - Python version: [e.g., 3.11]
 - Package version: [e.g., 1.0.0]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Simple issues for newcomers
- `help wanted` - Issues that need attention
- `documentation` - Help improve our docs

## 🛠️ Development Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Run the tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Run the linter:**
   ```bash
   flake8 src/
   black src/ --check
   isort src/ --check-only
   ```

## 📝 Style Guidelines

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Maximum line length: 100 characters
- Use double quotes for strings
- Use type hints for function arguments and return values
- Write docstrings for all public modules, functions, classes, and methods

### Code Formatting

We use the following tools:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Before committing, run:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md if you change functionality

Example docstring:
```python
def function_name(param1: str, param2: int = 10) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Can span multiple lines
    and include more details about the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 10.
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> result = function_name("test", 20)
        >>> print(result)
        True
    """
    pass
```

## 💬 Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```
feat(automl): add support for CatBoost classifier

fix(preprocessing): handle edge case in outlier detection

docs(readme): update installation instructions

test(eda): add tests for correlation analysis
```

## 🔀 Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Keep your branch updated:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows the style guidelines
- [ ] Self-review of the code has been performed
- [ ] Code is commented, particularly in complex areas
- [ ] Documentation has been updated
- [ ] Changes don't generate new warnings
- [ ] Tests have been added for new functionality
- [ ] All tests pass locally (`pytest tests/ -v`)
- [ ] Linting passes (`flake8 src/`)
- [ ] Any dependent changes have been merged

### PR Review Process

1. A maintainer will review your PR
2. They may request changes or ask questions
3. Make requested changes and push to your branch
4. Once approved, a maintainer will merge your PR

## 🎉 Recognition

Contributors will be recognized in:

- The project's README.md
- Release notes when their changes are included
- Our contributors page

Thank you for contributing! 🙏
