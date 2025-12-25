# Multimodal Semantic Search Engine

A semantic search engine that enables natural language search across PDF documents, supporting both text and image content using OpenAI's CLIP model.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mohammedadnan2307/multimodal-search.git
cd multimodal-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Add PDFs** to the `data/pdfs/` directory

2. **Build the index**:
```bash
python cli.py index
```

3. **Start the web server**:
```bash
python cli.py serve
```

4. Open http://localhost:5000 in your browser

### CLI Commands

```bash
# Build index with CLIP embeddings
python cli.py index

# Build index with demo embedder (no GPU required)
python cli.py index --demo

# Search from command line
python cli.py search "push up technique" -k 5

# Search only images
python cli.py search "person exercising" -m image

# Show index statistics
python cli.py stats

# Start web server
python cli.py serve --port 5000
```

## API Reference

### Search

```http
POST /api/search
Content-Type: application/json

{
  "query": "boy running",
  "top_k": 10,
  "mode": "all"  // "all", "text", or "image"
}
```

### Response

```json
{
  "query": "boy running",
  "results": [
    {
      "id": "doc_p1_c0_0",
      "score": 0.8542,
      "modality": "text",
      "document": "fitness_guide",
      "page": 1,
      "content": "The push-up is a fundamental exercise..."
    }
  ]
}
```

## Configuration

Edit `config.py` to customize:

```python
CLIP_MODEL = "openai/clip-vit-base-patch32"   # CLIP model variant
EMBEDDING_DIM = 512                           # Embedding dimensions
CHUNK_SIZE = 500                              # Text chunk size
DEFAULT_TOP_K = 10                            # Default results count
```

## Technical Details

### Embeddings

The system supports two embedding modes:

1. **CLIP Embedder**: Uses OpenAI's CLIP model for true multimodal understanding. Requires PyTorch.

2. **Demo Embedder**: Hash-based embeddings with vocabulary matching for testing without GPU.

## Performance

| Metric | Value |
|--------|-------|
| Text embedding | ~30 chunks/sec (CPU) |
| Image embedding | ~15 images/sec (CPU) |
| Search latency | <50ms |
| Index size | ~2KB per item |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for CLIP)
- 4GB+ RAM recommended
- GPU optional (CPU works fine)

## License

MIT License - see [LICENSE](LICENSE) for details.
