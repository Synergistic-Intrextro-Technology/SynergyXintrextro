# SynergyX Text Analysis Capabilities

SynergyX provides powerful text analysis tools for natural language processing tasks.

## Summarization

The text summarization tool uses extractive methods to identify and extract the most important sentences from a document.

### Features
- Automatic sentence ranking based on position and content
- Configurable summary length
- Compression ratio calculation
- Support for various text formats

### Usage
```python
from synergyx.tools.text_analysis import TextSummarizerTool

tool = TextSummarizerTool()
result = await tool.execute(
    text="Your long text here...",
    max_sentences=3
)
```

## Sentiment Analysis

Rule-based sentiment classification that analyzes emotional tone in text.

### Features
- Positive, negative, and neutral classification
- Confidence scoring
- Word-level analysis
- Extensible lexicon

### Sentiment Categories
- **Positive**: Indicates favorable sentiment
- **Negative**: Indicates unfavorable sentiment  
- **Neutral**: Indicates balanced or no clear sentiment

## Keyword Extraction

Automatic identification of important terms and phrases in text.

### Methods
- TF-based term frequency analysis
- N-gram extraction (unigrams and bigrams)
- Stop word filtering
- Frequency ranking

### Applications
- Content categorization
- Search optimization
- Topic identification
- Document clustering