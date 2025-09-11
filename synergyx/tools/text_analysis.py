"""Text analysis tools."""

import re
import asyncio
from typing import Any, Dict, List
from collections import Counter
import logging

from .base import AnalysisTool

logger = logging.getLogger(__name__)


class TextSummarizerTool(AnalysisTool):
    """Tool for text summarization."""
    
    @property
    def name(self) -> str:
        return "summarize_text"
    
    @property
    def description(self) -> str:
        return "Summarize long text into key points and main ideas"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Maximum number of sentences in summary",
                    "default": 3
                }
            },
            "required": ["text"]
        }
    
    async def execute(self, text: str, max_sentences: int = 3, **kwargs) -> Dict[str, Any]:
        """Execute text summarization using extractive approach."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= max_sentences:
            return {
                "summary": text,
                "sentence_count": len(sentences),
                "compression_ratio": 1.0,
                "method": "extractive"
            }
        
        # Simple extractive summarization based on sentence position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (beginning and end sentences are important)
            if i < 3:
                score += 3 - i
            if i >= len(sentences) - 3:
                score += i - len(sentences) + 4
            
            # Length score (medium length sentences preferred)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 2
            elif word_count > 5:
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        selected = [sent for _, sent in scored_sentences[:max_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in selected:
                summary_sentences.append(sentence)
        
        summary = " ".join(summary_sentences)
        
        return {
            "summary": summary,
            "original_sentences": len(sentences),
            "summary_sentences": len(summary_sentences),
            "compression_ratio": len(summary) / len(text),
            "method": "extractive"
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class SentimentAnalysisTool(AnalysisTool):
    """Tool for sentiment analysis."""
    
    def __init__(self):
        # Simple rule-based sentiment lexicon
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad',
            'angry', 'disappointed', 'frustrated', 'annoyed', 'disgusted',
            'poor', 'worst', 'dreadful', 'pathetic', 'useless', 'stupid'
        }
    
    @property
    def name(self) -> str:
        return "analyze_sentiment"
    
    @property
    def description(self) -> str:
        return "Analyze the sentiment of text (positive, negative, neutral)"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze for sentiment"
                }
            },
            "required": ["text"]
        }
    
    async def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """Execute sentiment analysis."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate sentiment score
        if positive_count + negative_count == 0:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / (positive_count + negative_count)
            
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "positive_words_found": positive_count,
            "negative_words_found": negative_count,
            "method": "rule_based"
        }


class KeywordExtractorTool(AnalysisTool):
    """Tool for keyword extraction."""
    
    def __init__(self):
        # Common stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'could'
        }
    
    @property
    def name(self) -> str:
        return "extract_keywords"
    
    @property
    def description(self) -> str:
        return "Extract important keywords and phrases from text"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract keywords from"
                },
                "max_keywords": {
                    "type": "integer",
                    "description": "Maximum number of keywords to extract",
                    "default": 10
                }
            },
            "required": ["text"]
        }
    
    async def execute(self, text: str, max_keywords: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute keyword extraction."""
        # Simple TF-based keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Extract n-grams (bigrams)
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
            bigrams.append(bigram)
        
        bigram_freq = Counter(bigrams)
        
        # Combine and rank keywords
        keywords = []
        
        # Single words
        for word, freq in word_freq.most_common(max_keywords):
            keywords.append({
                "keyword": word,
                "frequency": freq,
                "type": "unigram"
            })
        
        # Bigrams with frequency > 1
        for bigram, freq in bigram_freq.most_common(max_keywords // 2):
            if freq > 1:
                keywords.append({
                    "keyword": bigram,
                    "frequency": freq,
                    "type": "bigram"
                })
        
        # Sort by frequency and limit
        keywords.sort(key=lambda x: x["frequency"], reverse=True)
        keywords = keywords[:max_keywords]
        
        return {
            "keywords": keywords,
            "total_words": len(words),
            "unique_words": len(word_freq),
            "method": "tf_based"
        }