"""Data analysis tools for CSV, JSON, and DataFrame processing."""

import asyncio
import json
import tempfile
from io import StringIO
from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import numpy as np
from pathlib import Path

from .base import AnalysisTool

logger = logging.getLogger(__name__)


class DataAnalyzerTool(AnalysisTool):
    """Tool for analyzing CSV and JSON data."""
    
    @property
    def name(self) -> str:
        return "analyze_data"
    
    @property
    def description(self) -> str:
        return "Analyze CSV or JSON data files with descriptive statistics and insights"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "CSV data content or JSON data as string"
                },
                "format": {
                    "type": "string",
                    "description": "Data format",
                    "enum": ["csv", "json"],
                    "default": "csv"
                },
                "include_outliers": {
                    "type": "boolean",
                    "description": "Include outlier detection",
                    "default": True
                }
            },
            "required": ["data"]
        }
    
    async def execute(self, data: str, format: str = "csv", include_outliers: bool = True, **kwargs) -> Dict[str, Any]:
        """Execute data analysis."""
        try:
            # Load data into DataFrame
            if format == "csv":
                df = pd.read_csv(StringIO(data))
            elif format == "json":
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                else:
                    df = pd.DataFrame([json_data])
            else:
                return {"error": f"Unsupported format: {format}"}
            
            # Basic info
            result = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            # Descriptive statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                desc_stats = df[numeric_columns].describe()
                result["descriptive_statistics"] = desc_stats.to_dict()
                
                # Correlation matrix
                if len(numeric_columns) > 1:
                    corr_matrix = df[numeric_columns].corr()
                    result["correlation_matrix"] = corr_matrix.to_dict()
            
            # Categorical analysis
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                categorical_info = {}
                for col in categorical_columns:
                    value_counts = df[col].value_counts().head(10)
                    categorical_info[col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": value_counts.to_dict(),
                        "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None
                    }
                result["categorical_analysis"] = categorical_info
            
            # Outlier detection
            if include_outliers and len(numeric_columns) > 0:
                outliers = {}
                for col in numeric_columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(df) * 100),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    }
                
                result["outliers"] = outliers
            
            # Data quality assessment
            result["data_quality"] = {
                "completeness": float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100)
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {"error": str(e)}


class DataExplainerTool(AnalysisTool):
    """Tool for generating natural language explanations of data."""
    
    @property
    def name(self) -> str:
        return "explain_data"
    
    @property
    def description(self) -> str:
        return "Generate natural language explanation of data analysis results"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "analysis_result": {
                    "type": "object",
                    "description": "Result from data analysis tool"
                }
            },
            "required": ["analysis_result"]
        }
    
    async def execute(self, analysis_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate explanation of data analysis."""
        try:
            explanations = []
            
            # Dataset overview
            shape = analysis_result.get("shape", [0, 0])
            explanations.append(f"This dataset contains {shape[0]:,} rows and {shape[1]} columns.")
            
            # Missing values
            missing_values = analysis_result.get("missing_values", {})
            if missing_values:
                total_missing = sum(missing_values.values())
                if total_missing > 0:
                    missing_cols = [col for col, count in missing_values.items() if count > 0]
                    explanations.append(f"There are missing values in {len(missing_cols)} columns: {', '.join(missing_cols)}.")
                else:
                    explanations.append("The dataset has no missing values.")
            
            # Data types
            dtypes = analysis_result.get("dtypes", {})
            if dtypes:
                numeric_cols = [col for col, dtype in dtypes.items() if dtype in ['int64', 'float64']]
                text_cols = [col for col, dtype in dtypes.items() if dtype == 'object']
                
                if numeric_cols:
                    explanations.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}.")
                if text_cols:
                    explanations.append(f"Text/categorical columns ({len(text_cols)}): {', '.join(text_cols)}.")
            
            # Statistical insights
            desc_stats = analysis_result.get("descriptive_statistics", {})
            if desc_stats:
                for col, stats in desc_stats.items():
                    mean_val = stats.get("mean", 0)
                    std_val = stats.get("std", 0)
                    min_val = stats.get("min", 0)
                    max_val = stats.get("max", 0)
                    
                    explanations.append(f"Column '{col}': average {mean_val:.2f}, "
                                      f"ranges from {min_val:.2f} to {max_val:.2f}.")
                    
                    if std_val > mean_val:
                        explanations.append(f"'{col}' shows high variability (std dev {std_val:.2f}).")
            
            # Correlations
            corr_matrix = analysis_result.get("correlation_matrix", {})
            if corr_matrix:
                strong_correlations = []
                for col1 in corr_matrix:
                    for col2 in corr_matrix[col1]:
                        if col1 != col2:
                            corr_val = corr_matrix[col1][col2]
                            if abs(corr_val) > 0.7:
                                strong_correlations.append(f"{col1} and {col2} ({corr_val:.2f})")
                
                if strong_correlations:
                    explanations.append(f"Strong correlations found: {', '.join(strong_correlations)}.")
            
            # Outliers
            outliers = analysis_result.get("outliers", {})
            if outliers:
                outlier_cols = [col for col, info in outliers.items() if info["count"] > 0]
                if outlier_cols:
                    outlier_details = [f"{col} ({outliers[col]['count']} outliers)" for col in outlier_cols]
                    explanations.append(f"Outliers detected in: {', '.join(outlier_details)}.")
            
            # Data quality
            data_quality = analysis_result.get("data_quality", {})
            if data_quality:
                completeness = data_quality.get("completeness", 100)
                duplicates = data_quality.get("duplicate_rows", 0)
                
                explanations.append(f"Data completeness: {completeness:.1f}%.")
                if duplicates > 0:
                    explanations.append(f"Found {duplicates} duplicate rows.")
            
            # Categorical insights
            categorical_analysis = analysis_result.get("categorical_analysis", {})
            if categorical_analysis:
                for col, info in categorical_analysis.items():
                    unique_count = info.get("unique_values", 0)
                    most_frequent = info.get("most_frequent")
                    explanations.append(f"Column '{col}' has {unique_count} unique values, "
                                      f"most frequent: '{most_frequent}'.")
            
            full_explanation = " ".join(explanations)
            
            return {
                "explanation": full_explanation,
                "insights": explanations,
                "summary": self._generate_summary(analysis_result)
            }
        
        except Exception as e:
            logger.error(f"Data explanation failed: {e}")
            return {"error": str(e)}
    
    def _generate_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a brief summary."""
        shape = analysis_result.get("shape", [0, 0])
        data_quality = analysis_result.get("data_quality", {})
        completeness = data_quality.get("completeness", 100)
        
        if completeness > 95:
            quality_desc = "high quality"
        elif completeness > 80:
            quality_desc = "good quality"
        else:
            quality_desc = "needs cleaning"
        
        return f"Dataset with {shape[0]:,} rows and {shape[1]} columns, {quality_desc} data."


class CSVUploaderTool(AnalysisTool):
    """Tool for handling CSV file uploads and parsing."""
    
    @property
    def name(self) -> str:
        return "upload_csv"
    
    @property
    def description(self) -> str:
        return "Upload and parse CSV file for analysis"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to CSV file"
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter",
                    "default": ","
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8"
                },
                "sample_rows": {
                    "type": "integer",
                    "description": "Number of sample rows to return",
                    "default": 5
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, file_path: str, delimiter: str = ",", encoding: str = "utf-8", 
                     sample_rows: int = 5, **kwargs) -> Dict[str, Any]:
        """Upload and analyze CSV file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            # Check file size (limit to 10MB)
            file_size = file_path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                return {"error": f"File too large: {file_size / 1024 / 1024:.1f}MB (max 10MB)"}
            
            # Read CSV
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            
            # Get sample data
            sample_data = df.head(sample_rows).to_dict('records')
            
            return {
                "success": True,
                "file_info": {
                    "name": file_path.name,
                    "size_bytes": file_size,
                    "size_mb": file_size / 1024 / 1024
                },
                "data_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict()
                },
                "sample_data": sample_data,
                "csv_content": df.to_csv(index=False)  # Return for further analysis
            }
        
        except Exception as e:
            logger.error(f"CSV upload failed: {e}")
            return {"error": str(e)}