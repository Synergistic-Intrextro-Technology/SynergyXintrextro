"""Web content fetching and analysis tools."""

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin
import logging

import httpx
from bs4 import BeautifulSoup

from .base import AnalysisTool

logger = logging.getLogger(__name__)


class WebFetcherTool(AnalysisTool):
    """Tool for fetching and analyzing web content."""
    
    def __init__(self):
        self.user_agent = "SynergyX/0.1.0"
        self.timeout = 30
        self.max_content_length = 1000000  # 1MB
    
    @property
    def name(self) -> str:
        return "fetch_web_content"
    
    @property
    def description(self) -> str:
        return "Fetch and analyze content from web URLs"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from"
                },
                "extract_text_only": {
                    "type": "boolean",
                    "description": "Extract only text content, removing HTML",
                    "default": True
                },
                "summarize": {
                    "type": "boolean",
                    "description": "Generate a summary of the content",
                    "default": False
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to process",
                    "default": 10000
                }
            },
            "required": ["url"]
        }
    
    async def execute(self, url: str, extract_text_only: bool = True, 
                     summarize: bool = False, max_length: int = 10000, **kwargs) -> Dict[str, Any]:
        """Fetch and analyze web content."""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"error": "Invalid URL format"}
            
            if parsed_url.scheme not in ['http', 'https']:
                return {"error": "Only HTTP and HTTPS URLs are supported"}
            
            # Basic safety checks
            if self._is_blocked_domain(parsed_url.netloc):
                return {"error": "Domain is blocked for security reasons"}
            
            # Fetch content
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent}
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if not any(ct in content_type for ct in ["text/html", "text/plain", "application/xml"]):
                    return {"error": f"Unsupported content type: {content_type}"}
                
                # Check content length
                content_length = len(response.content)
                if content_length > self.max_content_length:
                    return {"error": f"Content too large: {content_length} bytes (max {self.max_content_length})"}
                
                html_content = response.text
            
            # Parse content
            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": content_length
            }
            
            if extract_text_only:
                text_content = self._extract_text(html_content)
                
                # Truncate if too long
                if len(text_content) > max_length:
                    text_content = text_content[:max_length] + "... (truncated)"
                
                result["text_content"] = text_content
                result["word_count"] = len(text_content.split())
                
                # Extract metadata
                metadata = self._extract_metadata(html_content)
                result["metadata"] = metadata
                
                # Extract links
                links = self._extract_links(html_content, url)
                result["links"] = links[:20]  # Limit to 20 links
                
                if summarize:
                    summary = self._simple_summarize(text_content)
                    result["summary"] = summary
            else:
                result["html_content"] = html_content
            
            return result
        
        except httpx.TimeoutException:
            return {"error": "Request timeout"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Web fetch failed for {url}: {e}")
            return {"error": str(e)}
    
    def _is_blocked_domain(self, domain: str) -> bool:
        """Check if domain should be blocked."""
        # Block internal/private domains
        blocked_patterns = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '10.',
            '192.168.',
            '172.16.',
            '172.17.',
            '172.18.',
            '172.19.',
            '172.20.',
            '172.21.',
            '172.22.',
            '172.23.',
            '172.24.',
            '172.25.',
            '172.26.',
            '172.27.',
            '172.28.',
            '172.29.',
            '172.30.',
            '172.31.'
        ]
        
        domain_lower = domain.lower()
        return any(pattern in domain_lower for pattern in blocked_patterns)
    
    def _extract_text(self, html_content: str) -> str:
        """Extract text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return html_content  # Return raw HTML as fallback
    
    def _extract_metadata(self, html_content: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {}
            
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                
                if name and content:
                    if name in ['description', 'keywords', 'author']:
                        metadata[name] = content
                    elif name.startswith('og:'):
                        metadata[name] = content
                    elif name.startswith('twitter:'):
                        metadata[name] = content
            
            # Headings
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    text = heading.get_text().strip()
                    if text:
                        headings.append({
                            'level': i,
                            'text': text
                        })
            
            metadata['headings'] = headings[:10]  # Limit to 10 headings
            
            return metadata
        
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    def _extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                if href.startswith(('http://', 'https://')):
                    absolute_url = href
                else:
                    absolute_url = urljoin(base_url, href)
                
                # Skip empty links or fragments only
                if href and not href.startswith('#'):
                    links.append({
                        'url': absolute_url,
                        'text': text,
                        'href': href
                    })
            
            return links
        
        except Exception as e:
            logger.warning(f"Link extraction failed: {e}")
            return []
    
    def _simple_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization."""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= max_sentences:
                return text
            
            # Score sentences by length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                
                # Position score (beginning sentences are important)
                if i < 3:
                    score += 3 - i
                
                # Length score (prefer medium-length sentences)
                word_count = len(sentence.split())
                if 10 <= word_count <= 30:
                    score += 2
                elif word_count > 5:
                    score += 1
                
                # Keyword score (sentences with common words)
                words = sentence.lower().split()
                common_words = ['important', 'key', 'main', 'primary', 'significant', 'major']
                if any(word in words for word in common_words):
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
            
            return '. '.join(summary_sentences) + '.'
        
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return text[:500] + "..." if len(text) > 500 else text