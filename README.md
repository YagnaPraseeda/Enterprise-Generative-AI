# 🧠 Enterprise Generative AI Knowledge Platform

An enterprise-grade AI platform featuring **RAG pipelines**, a **Model Garden**, **AI governance controls**, and a **premium chat interface** — designed to demonstrate production-ready GenAI architecture.

---

## 🏗️ Architecture

```
                    ┌──────────────────────┐
                    │   Streamlit Frontend │
                    │   (Chat / Compare)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   FastAPI Backend     │
                    │   /api/chat           │
                    │   /api/documents      │
                    │   /api/models         │
                    │   /api/governance     │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
    ┌─────▼─────┐      ┌──────▼──────┐      ┌──────▼──────┐
    │ RAG Engine │      │Model Garden │      │ Governance  │
    │            │      │             │      │   Engine    │
    │ • Retrieve │      │ • Registry  │      │ • Toxicity  │
    │ • Generate │      │ • Compare   │      │ • Halluc.   │
    │ • Cite     │      │ • Metrics   │      │ • Prompt    │
    └─────┬─────┘      └─────────────┘      └─────────────┘
          │
    ┌─────▼─────┐
    │ ChromaDB  │
    │ (Vectors) │
    └───────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Document Processing** | Upload PDF, DOCX, TXT → automatic chunking & embedding |
| 🔍 **RAG Pipeline** | Retrieval-augmented generation with source citations |
| 🤖 **Model Garden** | Multiple LLM support with comparison & usage tracking |
| 🛡️ **AI Governance** | Toxicity filtering, hallucination detection, prompt guardrails |
| 📊 **Evaluation** | Faithfulness, relevance, and context precision metrics |
| 💬 **Chat Interface** | Premium Streamlit UI with conversation memory |
| 🐳 **Docker Ready** | Full containerization with docker-compose |
| ⚡ **CI/CD** | GitHub Actions pipeline with lint, test, and build |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/enterprise-genai-platform.git
cd enterprise-genai-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive API documentation.

### 4. Start the Frontend

```bash
cd frontend
streamlit run app.py
```

Visit **http://localhost:8501** for the chat interface.

### 5. Load Sample Documents

In the Streamlit sidebar, click **"📚 Load Samples"** to ingest the included enterprise documents, then start asking questions!

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:8501
```

---

## 📁 Project Structure

```
Enterprise Generative AI/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry point
│   ├── schemas.py              # Pydantic models
│   └── routes/
│       ├── chat.py             # Chat/RAG endpoints
│       ├── documents.py        # Document management
│       ├── models.py           # Model Garden endpoints
│       └── governance.py       # Governance checks
├── core/                       # Core AI components
│   ├── document_processor.py   # PDF/DOCX/TXT loading & chunking
│   ├── vector_store.py         # ChromaDB vector database
│   ├── rag_engine.py           # RAG pipeline
│   ├── model_garden.py         # Model management
│   └── model_registry.json     # Default model configs
├── governance/                 # AI safety layer
│   ├── toxicity_filter.py      # Detoxify-based toxic content filter
│   ├── hallucination_detector.py # Embedding similarity scoring
│   ├── prompt_guard.py         # Prompt injection detection
│   └── governance_engine.py    # Unified governance orchestrator
├── evaluation/                 # Quality metrics
│   └── metrics.py              # Faithfulness, relevance, precision
├── frontend/                   # Streamlit UI
│   ├── app.py                  # Main chat application
│   ├── components.py           # Reusable UI components
│   └── styles.css              # Premium dark theme
├── data/
│   └── sample_docs/            # Built-in enterprise documents
├── tests/                      # Test suite
├── config/                     # Settings management
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile                  # Container build
├── docker-compose.yml          # Multi-service deployment
└── requirements.txt            # Python dependencies
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Submit a RAG query |
| `GET` | `/api/chat/history` | Get conversation history |
| `POST` | `/api/documents/upload` | Upload a document |
| `POST` | `/api/documents/ingest-samples` | Load sample documents |
| `GET` | `/api/documents/stats` | Vector store statistics |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/models/compare` | Compare model responses |
| `GET` | `/api/models/{name}/metrics` | Model usage metrics |
| `POST` | `/api/governance/check` | Run governance checks |
| `GET` | `/api/health` | Health check |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_governance.py -v
python -m pytest tests/test_documents.py -v
python -m pytest tests/test_api.py -v
```

---

## 🛡️ Governance Controls

### Toxicity Detection
Uses **Detoxify** (BERT-based) to scan inputs and outputs across 6 toxicity categories.

### Hallucination Detection
Compares LLM responses against source documents using **embedding cosine similarity** to flag ungrounded claims.

### Prompt Injection Guard
Pattern-matching detection for:
- Instruction override attacks
- System prompt extraction
- Role impersonation
- Jailbreak attempts
- Data exfiltration

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, Uvicorn |
| **LLM Framework** | LangChain |
| **LLM Provider** | Groq (Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B, Mixtral 8x7B) |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, no API key needed) |
| **Governance** | Detoxify, scikit-learn |
| **Frontend** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Configuration** | Pydantic Settings |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
