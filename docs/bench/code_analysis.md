# Code Analysis Tools

SynergyX offers comprehensive code analysis capabilities for software development workflows.

## Python AST Analysis

Static analysis of Python code using Abstract Syntax Trees (AST).

### Metrics
- Function and class counts
- Import analysis
- Complexity calculations
- Code structure insights

### Features
- Syntax validation
- Decorator detection
- Method analysis within classes
- Docstring extraction

## Complexity Analysis

Cyclomatic complexity measurement using the Radon library.

### Metrics
- Cyclomatic Complexity
- Maintainability Index
- Halstead Metrics
- Lines of Code

### Complexity Levels
- **A**: 1-5 (Simple)
- **B**: 6-10 (Well structured)
- **C**: 11-20 (Complex)
- **D**: 21-50 (More complex)
- **E**: 51-100 (Unstable)
- **F**: >100 (Unmaintainable)

## Code Linting

Basic style and quality checks for Python code.

### Checks
- Line length validation
- Trailing whitespace detection
- Mixed indentation detection
- Basic syntax validation
- Integration with external linters (Ruff, Pylint)

### Severity Levels
- **Error**: Code cannot run
- **Warning**: Potential issues
- **Info**: Style suggestions