"""Code analysis tools."""

import ast
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit, h_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    logging.warning("Radon not available, code complexity analysis will be limited")

from .base import AnalysisTool

logger = logging.getLogger(__name__)


class PythonASTAnalyzer(AnalysisTool):
    """Tool for Python AST-based code analysis."""
    
    @property
    def name(self) -> str:
        return "analyze_python_code"
    
    @property
    def description(self) -> str:
        return "Analyze Python code structure, complexity, and metrics"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to analyze"
                },
                "include_complexity": {
                    "type": "boolean",
                    "description": "Include cyclomatic complexity analysis",
                    "default": True
                }
            },
            "required": ["code"]
        }
    
    async def execute(self, code: str, include_complexity: bool = True, **kwargs) -> Dict[str, Any]:
        """Execute Python code analysis."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "valid_syntax": False
            }
        
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        
        result = {
            "valid_syntax": True,
            "functions": analyzer.functions,
            "classes": analyzer.classes,
            "imports": analyzer.imports,
            "lines_of_code": len(code.split('\n')),
            "function_count": len(analyzer.functions),
            "class_count": len(analyzer.classes),
            "import_count": len(analyzer.imports)
        }
        
        # Add complexity analysis if radon is available
        if include_complexity and RADON_AVAILABLE:
            try:
                complexity_result = cc_visit(code)
                result["complexity"] = [
                    {
                        "name": block.name,
                        "complexity": block.complexity,
                        "type": block.__class__.__name__,
                        "lineno": block.lineno,
                        "endline": block.endline
                    }
                    for block in complexity_result
                ]
                
                # Calculate maintainability index
                mi_result = mi_visit(code, multi=True)
                result["maintainability_index"] = mi_result
                
                # Calculate Halstead metrics
                h_result = h_visit(code)
                result["halstead"] = {
                    "difficulty": h_result.difficulty,
                    "effort": h_result.effort,
                    "volume": h_result.volume
                } if h_result else None
                
            except Exception as e:
                logger.warning(f"Complexity analysis failed: {e}")
                result["complexity_error"] = str(e)
        
        return result


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing Python code structure."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            "name": node.name,
            "lineno": node.lineno,
            "args": [arg.arg for arg in node.args.args],
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node)
        })
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.functions.append({
            "name": node.name,
            "lineno": node.lineno,
            "args": [arg.arg for arg in node.args.args],
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "async": True
        })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append({
            "name": node.name,
            "lineno": node.lineno,
            "bases": [self._get_name(base) for base in node.bases],
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": []
        })
        
        # Analyze methods within the class
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.classes[-1]["methods"].append({
                    "name": item.name,
                    "lineno": item.lineno,
                    "args": [arg.arg for arg in item.args.args],
                    "async": isinstance(item, ast.AsyncFunctionDef)
                })
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "name": alias.name,
                "alias": alias.asname,
                "type": "import",
                "lineno": node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                "module": node.module,
                "name": alias.name,
                "alias": alias.asname,
                "type": "from_import",
                "lineno": node.lineno
            })
        self.generic_visit(node)
    
    def _get_name(self, node):
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _get_decorator_name(self, node):
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)


class CodeLinterTool(AnalysisTool):
    """Tool for basic code linting."""
    
    @property
    def name(self) -> str:
        return "lint_code"
    
    @property
    def description(self) -> str:
        return "Run basic linting checks on code"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to lint"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "enum": ["python", "javascript", "typescript"],
                    "default": "python"
                }
            },
            "required": ["code"]
        }
    
    async def execute(self, code: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Execute code linting."""
        if language == "python":
            return await self._lint_python(code)
        else:
            return {
                "error": f"Linting not implemented for {language}",
                "supported_languages": ["python"]
            }
    
    async def _lint_python(self, code: str) -> Dict[str, Any]:
        """Lint Python code."""
        issues = []
        
        # Basic syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "severity": "error"
            })
            return {
                "issues": issues,
                "issue_count": len(issues),
                "has_errors": True
            }
        
        # Basic style checks
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:
                issues.append({
                    "type": "line_too_long",
                    "message": f"Line too long ({len(line)} > 88 characters)",
                    "line": i,
                    "severity": "warning"
                })
            
            # Check trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append({
                    "type": "trailing_whitespace",
                    "message": "Trailing whitespace",
                    "line": i,
                    "severity": "info"
                })
            
            # Check mixed tabs and spaces
            if '\t' in line and '    ' in line:
                issues.append({
                    "type": "mixed_indentation",
                    "message": "Mixed tabs and spaces",
                    "line": i,
                    "severity": "warning"
                })
        
        # Try to run external linters if available
        external_results = await self._run_external_linters(code)
        if external_results:
            issues.extend(external_results)
        
        has_errors = any(issue["severity"] == "error" for issue in issues)
        
        return {
            "issues": issues,
            "issue_count": len(issues),
            "has_errors": has_errors,
            "errors": sum(1 for i in issues if i["severity"] == "error"),
            "warnings": sum(1 for i in issues if i["severity"] == "warning"),
            "info": sum(1 for i in issues if i["severity"] == "info")
        }
    
    async def _run_external_linters(self, code: str) -> List[Dict[str, Any]]:
        """Try to run external linters like ruff or pylint."""
        issues = []
        
        # Try ruff first (faster)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['ruff', 'check', '--output-format', 'json', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 or result.stdout:
                import json
                try:
                    ruff_issues = json.loads(result.stdout)
                    for issue in ruff_issues:
                        issues.append({
                            "type": "ruff_" + issue.get("code", "unknown"),
                            "message": issue.get("message", ""),
                            "line": issue.get("location", {}).get("row", 0),
                            "column": issue.get("location", {}).get("column", 0),
                            "severity": "warning",
                            "source": "ruff"
                        })
                except json.JSONDecodeError:
                    pass
            
            os.unlink(temp_file)
        
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            # Ruff not available or failed
            pass
        
        return issues