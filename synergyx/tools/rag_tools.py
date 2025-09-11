"""RAG (Retrieval-Augmented Generation) tool."""

import asyncio
from typing import Any, Dict, List, Optional
import logging

from ..tools.base import AnalysisTool
from ..rag.simple_rag import SimpleRAG
from ..config.manager import get_config

logger = logging.getLogger(__name__)


class RAGSearchTool(AnalysisTool):
    """Tool for searching documents using RAG."""
    
    def __init__(self):
        self.config = get_config()
        self.rag: Optional[SimpleRAG] = None
    
    @property
    def name(self) -> str:
        return "search_documents"
    
    @property
    def description(self) -> str:
        return "Search through document collection for relevant information"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or question"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Execute document search."""
        try:
            # Initialize RAG if needed
            if self.rag is None:
                self.rag = SimpleRAG(self.config)
                await self.rag.initialize()
            
            # Search for relevant documents
            results = await self.rag.search(query, top_k=top_k)
            
            if not results:
                return {
                    "query": query,
                    "results": [],
                    "message": "No relevant documents found"
                }
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "score": score,
                    "metadata": doc.metadata,
                    "doc_id": doc.doc_id
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_found": len(results)
            }
        
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return {"error": str(e)}


class RAGAnswerTool(AnalysisTool):
    """Tool for answering questions using RAG."""
    
    def __init__(self):
        self.config = get_config()
        self.rag: Optional[SimpleRAG] = None
    
    @property
    def name(self) -> str:
        return "answer_from_documents"
    
    @property
    def description(self) -> str:
        return "Answer questions using information from document collection"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to answer"
                },
                "context_limit": {
                    "type": "integer",
                    "description": "Maximum number of context documents to use",
                    "default": 3
                }
            },
            "required": ["question"]
        }
    
    async def execute(self, question: str, context_limit: int = 3, **kwargs) -> Dict[str, Any]:
        """Execute question answering using RAG."""
        try:
            # Initialize RAG if needed
            if self.rag is None:
                self.rag = SimpleRAG(self.config)
                await self.rag.initialize()
            
            # Get answer context
            answer_data = await self.rag.answer_question(question)
            
            if not answer_data.get("context"):
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information in the documents to answer this question.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # The context is provided to the LLM via the tool calling mechanism
            # The actual answer generation happens in the main chat engine
            return {
                "question": question,
                "context": answer_data["context"],
                "sources": answer_data["sources"],
                "retrieved_docs": answer_data["retrieved_docs"],
                "answer_context": answer_data["context"]  # For LLM to use in generating response
            }
        
        except Exception as e:
            logger.error(f"RAG answer failed: {e}")
            return {"error": str(e)}


class RAGIndexTool(AnalysisTool):
    """Tool for managing RAG index."""
    
    def __init__(self):
        self.config = get_config()
        self.rag: Optional[SimpleRAG] = None
    
    @property
    def name(self) -> str:
        return "rebuild_document_index"
    
    @property
    def description(self) -> str:
        return "Rebuild the document search index from the docs directory"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "force_rebuild": {
                    "type": "boolean",
                    "description": "Force rebuild even if index exists",
                    "default": False
                }
            }
        }
    
    async def execute(self, force_rebuild: bool = False, **kwargs) -> Dict[str, Any]:
        """Execute index rebuild."""
        try:
            # Initialize RAG
            if self.rag is None:
                self.rag = SimpleRAG(self.config)
            
            # Build index
            await self.rag.build_index()
            
            return {
                "success": True,
                "message": f"Document index rebuilt with {len(self.rag.documents)} chunks",
                "document_count": len(self.rag.documents),
                "docs_directory": str(self.rag.docs_dir)
            }
        
        except Exception as e:
            logger.error(f"RAG index rebuild failed: {e}")
            return {"error": str(e)}