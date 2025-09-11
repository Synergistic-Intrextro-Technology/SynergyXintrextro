"""Simple RAG (Retrieval-Augmented Generation) implementation."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import pickle
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.manager import Config

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available, using sklearn for similarity search")


class Document:
    """Represents a document in the RAG system."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                 doc_id: Optional[str] = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id(content)
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for document."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id")
        )


class SimpleRAG:
    """Simple RAG implementation with optional FAISS backend."""
    
    def __init__(self, config: Config):
        self.config = config
        self.docs_dir = Path(config.get("tools.rag.docs_directory", "./docs"))
        self.index_path = Path(config.get("tools.rag.index_path", "./data/rag_index"))
        self.chunk_size = config.get("tools.rag.chunk_size", 500)
        self.chunk_overlap = config.get("tools.rag.chunk_overlap", 50)
        self.top_k = config.get("tools.rag.top_k", 5)
        self.similarity_threshold = config.get("tools.rag.similarity_threshold", 0.1)
        self.use_faiss = config.get("tools.rag.use_faiss", True) and FAISS_AVAILABLE
        
        self.documents: List[Document] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self.faiss_index = None
        
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the RAG system by loading or building index."""
        if await self._load_index():
            logger.info("Loaded existing RAG index")
        else:
            await self.build_index()
    
    async def build_index(self) -> None:
        """Build the RAG index from documents directory."""
        logger.info(f"Building RAG index from {self.docs_dir}")
        
        # Load documents
        await self._load_documents()
        
        if not self.documents:
            logger.warning("No documents found to index")
            return
        
        # Create chunks
        chunks = []
        for doc in self.documents:
            doc_chunks = self._create_chunks(doc.content, doc.metadata, doc.doc_id)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(self.documents)} documents")
        
        # Convert chunks to documents
        self.documents = chunks
        
        # Build vector index
        await self._build_vector_index()
        
        # Save index
        await self._save_index()
        
        logger.info("RAG index built successfully")
    
    async def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Search for relevant documents."""
        if not self.documents or self.vectorizer is None:
            await self.initialize()
        
        if not self.documents:
            return []
        
        top_k = top_k or self.top_k
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        if self.use_faiss and self.faiss_index is not None:
            # Use FAISS for search
            scores, indices = self.faiss_index.search(
                query_vector.toarray().astype(np.float32), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= self.similarity_threshold:
                    results.append((self.documents[idx], float(score)))
            
        else:
            # Use sklearn cosine similarity
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= self.similarity_threshold:
                    results.append((self.documents[idx], score))
        
        return results
    
    async def answer_question(self, question: str, context_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Answer a question using retrieved documents."""
        if context_docs is None:
            search_results = await self.search(question)
            context_docs = [doc for doc, _ in search_results]
        
        # Combine retrieved context
        context = "\n\n".join([doc.content for doc in context_docs[:3]])  # Use top 3
        
        return {
            "question": question,
            "context": context,
            "retrieved_docs": len(context_docs),
            "sources": [
                {
                    "doc_id": doc.doc_id,
                    "metadata": doc.metadata
                }
                for doc in context_docs[:3]
            ]
        }
    
    async def _load_documents(self) -> None:
        """Load documents from the docs directory."""
        self.documents = []
        
        if not self.docs_dir.exists():
            logger.warning(f"Documents directory not found: {self.docs_dir}")
            return
        
        # Load text and markdown files
        supported_extensions = {'.txt', '.md', '.markdown'}
        
        for file_path in self.docs_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    metadata = {
                        "source_file": str(file_path.relative_to(self.docs_dir)),
                        "file_type": file_path.suffix,
                        "file_size": file_path.stat().st_size
                    }
                    
                    doc = Document(content=content, metadata=metadata)
                    self.documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def _create_chunks(self, content: str, metadata: Dict[str, Any], doc_id: str) -> List[Document]:
        """Create overlapping chunks from document content."""
        chunks = []
        
        # Simple word-based chunking
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "parent_doc_id": doc_id,
                "chunk_index": len(chunks),
                "chunk_start": i,
                "chunk_end": i + len(chunk_words)
            })
            
            chunk = Document(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=f"{doc_id}_chunk_{len(chunks)}"
            )
            chunks.append(chunk)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    async def _build_vector_index(self) -> None:
        """Build vector index from documents."""
        if not self.documents:
            return
        
        # Extract text content
        texts = [doc.content for doc in self.documents]
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        # Build FAISS index if available
        if self.use_faiss:
            try:
                # Convert to dense format for FAISS
                dense_vectors = self.doc_vectors.toarray().astype(np.float32)
                
                # Create FAISS index
                dimension = dense_vectors.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(dense_vectors)
                self.faiss_index.add(dense_vectors)
                
                logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
            
            except Exception as e:
                logger.warning(f"FAISS index creation failed: {e}, falling back to sklearn")
                self.use_faiss = False
                self.faiss_index = None
    
    async def _save_index(self) -> None:
        """Save the RAG index to disk."""
        try:
            index_data = {
                "documents": [doc.to_dict() for doc in self.documents],
                "config": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "use_faiss": self.use_faiss
                }
            }
            
            # Save documents and config
            with open(self.index_path / "documents.json", 'w') as f:
                json.dump(index_data, f)
            
            # Save vectorizer
            if self.vectorizer:
                with open(self.index_path / "vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
            
            # Save doc vectors
            if self.doc_vectors is not None:
                with open(self.index_path / "doc_vectors.pkl", 'wb') as f:
                    pickle.dump(self.doc_vectors, f)
            
            # Save FAISS index
            if self.use_faiss and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.index_path / "faiss.index"))
            
            logger.info(f"Saved RAG index to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save RAG index: {e}")
    
    async def _load_index(self) -> bool:
        """Load existing RAG index from disk."""
        try:
            if not (self.index_path / "documents.json").exists():
                return False
            
            # Load documents and config
            with open(self.index_path / "documents.json", 'r') as f:
                index_data = json.load(f)
            
            self.documents = [Document.from_dict(doc_data) for doc_data in index_data["documents"]]
            
            # Load vectorizer
            vectorizer_path = self.index_path / "vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            # Load doc vectors
            vectors_path = self.index_path / "doc_vectors.pkl"
            if vectors_path.exists():
                with open(vectors_path, 'rb') as f:
                    self.doc_vectors = pickle.load(f)
            
            # Load FAISS index
            faiss_path = self.index_path / "faiss.index"
            if self.use_faiss and faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
            
            logger.info(f"Loaded RAG index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load RAG index: {e}")
            return False