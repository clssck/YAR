<div align="center">

# 🏴‍☠️ YAR — Yet Another RAG

<p align="center">
  <b>A production-grade RAG framework with graph-based knowledge retrieval</b>
</p>

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/PostgreSQL-pgvector%20%2B%20AGE-336791?style=for-the-badge&logo=postgresql&logoColor=white">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</div>

</div>

---

## 🙏 Acknowledgments

**YAR began as a fork of [LightRAG](https://github.com/HKUDS/LightRAG)** by the brilliant team at HKUDS. Their research on graph-based RAG retrieval and the "Simple and Fast" philosophy laid the foundation for this project.

However, over time, YAR has diverged significantly:

- **Different storage backends** — PostgreSQL-native with pgvector and Apache AGE, S3 integration
- **Different API architecture** — FastAPI with comprehensive document management
- **Different UI** — Complete React rewrite with modern tooling
- **Different focus** — Production deployment over research flexibility

The codebases have drifted far enough apart that maintaining YAR as a fork is no longer practical. We've rebranded and moved forward independently, while remaining grateful for the excellent foundation LightRAG provided.

**If you're looking for the original research-focused implementation, please visit [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG).**

---

## ⚡ What is YAR?

YAR incorporates **graph structures into text indexing and retrieval**. It employs a dual-level retrieval system:

- **Low-level**: Entity and relationship extraction from documents
- **High-level**: Topic and concept clustering across the knowledge graph

This enables contextually rich answers that understand connections between concepts, not just keyword matches.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/clssck/YAR.git
cd YAR

# Install with uv (recommended)
uv sync --extra api
source .venv/bin/activate

# Configure environment
cp env.example .env
# Edit .env with your API keys and database credentials

# Start the server
yar-server
```

Open [http://localhost:9600](http://localhost:9600) for the Web UI.

### Python API

```python
import asyncio
from yar import LightRAG, QueryParam
from yar.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()

    # Insert documents
    await rag.ainsert("Your document content here...")

    # Query with hybrid retrieval
    result = await rag.aquery(
        "What are the main concepts?",
        param=QueryParam(mode="hybrid")
    )
    print(result)

asyncio.run(main())
```

---

## 📦 Features

| Feature | Description |
|---------|-------------|
| **Graph-Based Retrieval** | Dual-level entity and topic extraction |
| **PostgreSQL Native** | pgvector for embeddings, Apache AGE for graphs |
| **S3 Storage** | Native S3/R2/MinIO support for documents |
| **Modern Web UI** | React + TypeScript with graph visualization |
| **Document Pipeline** | PDF, DOCX, PPTX, Markdown extraction via Kreuzberg |
| **Citation Tracking** | Source attribution in responses |
| **Multi-Workspace** | Isolated knowledge graphs per workspace |

---

## 🛠️ Configuration

Key environment variables (see `env.example` for full list):

```ini
# LLM
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# PostgreSQL (recommended)
POSTGRES_HOST=localhost
POSTGRES_DB=yar
POSTGRES_USER=yar
POSTGRES_PASSWORD=...

# S3 Storage (optional)
S3_ENDPOINT_URL=https://...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=yar-documents
```

---

## 📚 Documentation

- [API Reference](./yar/api/README.md)
- [Docker Deployment](./docs/DockerDeployment.md)
- [Algorithm Details](./docs/Algorithm.md)

---

## 🤝 Contributing

```bash
# Install dev dependencies
uv sync --extra test

# Run tests
pytest tests/

# Lint & format
ruff check . && ruff format .
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built with ☕ by <a href="https://github.com/clssck">clssck</a></sub><br>
  <sub>Standing on the shoulders of <a href="https://github.com/HKUDS/LightRAG">LightRAG</a></sub>
</div>
