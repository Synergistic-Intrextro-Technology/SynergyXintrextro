"""Test configuration for pytest."""

import pytest
import tempfile
import shutil
from pathlib import Path

# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """name,age,salary,department
John Doe,30,50000,Engineering
Jane Smith,25,60000,Marketing
Bob Johnson,35,45000,Engineering
Alice Brown,28,55000,Sales
Charlie Wilson,42,70000,Engineering"""


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return """[
    {"name": "John", "age": 30, "score": 85.5, "active": true},
    {"name": "Jane", "age": 25, "score": 92.3, "active": true},
    {"name": "Bob", "age": 35, "score": 78.1, "active": false}
]"""


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
import os
import sys
from typing import List, Dict

def fibonacci(n: int) -> int:
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(("add", a, b, result))
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = a - b
        self.history.append(("subtract", a, b, result))
        return result

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.subtract(10, 4))
'''


@pytest.fixture 
def sample_text():
    """Sample text for analysis."""
    return """
Natural language processing (NLP) is a subfield of artificial intelligence (AI) 
that focuses on the interaction between computers and human language. It involves 
developing algorithms and models that can understand, interpret, and generate 
human language in a meaningful way.

NLP has numerous applications in today's technology landscape. These include 
machine translation, sentiment analysis, chatbots, text summarization, and 
information extraction. The field has seen significant advances with the 
development of deep learning techniques and large language models.

One of the main challenges in NLP is dealing with the ambiguity and complexity 
of human language. Words can have multiple meanings depending on context, and 
the same idea can be expressed in many different ways. Despite these challenges, 
NLP continues to evolve and improve, enabling more sophisticated AI applications.
"""