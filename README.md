# Book Recommender System

NLP-powered book recommender that combines semantic embeddings with a Netflix-style browsing interface. Given a CSV dataset of ~6 200 books, the system cleans, enriches (3 genres + 3 keywords per book via LLM), embeds, and serves recommendations through a FastAPI backend and a Gradio GUI.

## Architecture

```
NLP_Project-1/
├── config.yaml            # All tunables (data paths, models, API/GUI ports)
├── run.py                 # One-command launcher for API + Bridge + GUI
├── requirements.txt       # Python dependencies
│
├── data/
│   ├── books.csv          # Raw dataset (title, author, description, rating, …)
│   ├── books_clean.csv    # After removing incomplete rows
│   ├── books_enriched.csv # + genres (3) + keywords (3) per book via LLM
│   └── embeddings.npz     # Pre-computed BGE-large-en-v1.5 vectors
│
├── recommender/
│   ├── clean.py           # Step 1 — drop books missing required fields
│   ├── enrich.py          # Step 2 — LLM-generated genres & keywords (OpenRouter)
│   ├── embed.py           # Step 3 — sentence-transformer embeddings
│   └── api.py             # FastAPI server (POST /recommend, GET /books, …)
│
└── gui/
    └── app.py             # Gradio frontend (Netflix-style dark theme)
```

## Setup

```bash
# Clone & enter the project
cd NLP_Project-1

# Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set your OpenRouter API key (only needed for enrichment)
echo 'OPENROUTER_API_KEY=sk-...' > .env
```

## Pipeline (run once)

```bash
# 1. Clean raw data
python -m recommender.clean

# 2. Enrich with LLM-generated genres + keywords (3 + 3 per book)
python -m recommender.enrich

# 3. Compute embeddings
python -m recommender.embed
```

## Running the App

**Option A — single command (recommended):**

```bash
python run.py
```

Starts three servers:

| Server         | Port | Purpose                                      |
|----------------|------|----------------------------------------------|
| FastAPI API    | 8000 | NLP recommendations, book catalog            |
| Bridge server  | 7861 | JS ↔ Python communication for interactivity  |
| Gradio GUI     | 7860 | Netflix-style browsing interface              |

Open `http://localhost:7860` in a browser. The GUI loads immediately; search becomes available once the API finishes loading the model (~30-60 s on first run).

`run.py` auto-cleans stale ports from previous runs and computes embeddings if missing.

**Option B — separate terminals:**

```bash
# Terminal 1: API server
python -m recommender.api

# Terminal 2: Gradio GUI
python -m gui.app
```

## Configuration

All tunables live in `config.yaml`:

| Section     | Key             | Default                        | Purpose                                        |
|-------------|-----------------|--------------------------------|------------------------------------------------|
| `data`      | `raw`           | `data/books.csv`               | Path to the raw dataset                        |
| `data`      | `clean`         | `data/books_clean.csv`         | Output of the cleaning step                    |
| `data`      | `enriched`      | `data/books_enriched.csv`      | Output of the enrichment step                  |
| `data`      | `embeddings`    | `data/embeddings.npz`          | Pre-computed embedding vectors                 |
| `llm`       | `model`         | `arcee-ai/trinity-large-preview:free` | OpenRouter model for enrichment         |
| `llm`       | `batch_size`    | `5`                            | Books per LLM request                          |
| `embedding` | `model`         | `BAAI/bge-large-en-v1.5`      | Sentence-transformer model                     |
| `scoring`   | `rating_weight` | `0.1`                          | 0 = pure similarity, 1 = pure rating           |
| `api`       | `port`          | `8000`                         | FastAPI server port                            |
| `gui`       | `port`          | `7860`                         | Gradio GUI port                                |

## Dependencies

See `requirements.txt`. Key packages:

- **pandas** — data manipulation
- **gradio** — GUI framework
- **fastapi / uvicorn** — API server
- **sentence-transformers** — BGE-large embeddings
- **httpx** — HTTP client (GUI → API, LLM enrichment)
- **pyyaml / python-dotenv** — config and secrets
