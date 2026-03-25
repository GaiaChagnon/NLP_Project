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

## Application Notes

### Startup Sequence

`run.py` orchestrates three concurrent servers in a specific order:

1. **Embeddings check** — if `data/embeddings.npz` is missing, the embed step runs synchronously (one-time cost: ~2-5 min depending on hardware). Subsequent launches skip this.
2. **Port cleanup** — `lsof`-based kill of any stale processes on the API, bridge, and GUI ports. This is macOS/Linux-specific; on Windows, manual port cleanup may be required.
3. **FastAPI API** — launched in a daemon thread. The `/health` endpoint is polled every 2 s with a 120 s timeout. Model loading (sentence-transformer into memory) is the bottleneck (~30-60 s).
4. **Bridge server** — a lightweight FastAPI instance on port 7861 that proxies JS click events (book select, add/remove from list) back into Python rendering functions, bypassing Gradio's SSE-based state roundtrip.
5. **Gradio GUI** — starts immediately and is usable for browsing before the API is ready. Search and recommendations silently return empty results until the API responds on `/health`.

### Architecture Decisions

- **Three-server design** — Gradio's built-in interactivity uses server-sent events (SSE) which introduce latency for click-driven UI updates. The bridge server on port 7861 handles card clicks, list mutations, and page navigation via direct `fetch()` calls from injected JavaScript, giving near-instant DOM updates. Gradio still owns the search form, filters, and state initialization.
- **Embedding at import time** — the API module loads the sentence-transformer model and the full embedding matrix at import time (module-level statements in `api.py`). This means the FastAPI thread blocks until the model is ready, which is why the GUI launches first and polls `/health` asynchronously.
- **Cosine similarity + rating boost** — `final_score = (1 - rating_weight) * cosine_sim + rating_weight * normalised_rating`. The default `rating_weight` of 0.001 makes the ranking almost purely semantic; increase it toward 0.1-0.3 to surface popular books more aggressively.
- **Cover image fallback chain** — each book card tries Open Library (`-L.jpg`, high-res) first. On 404 or a detected non-book image (aspect ratio < 1.1 or width < 30 px), it falls back to the Google Books thumbnail. If both fail, the card is hidden via `display:none` so the grid stays clean.

### NLP Pipeline Details

| Step | Module | Input → Output | Key Behaviour |
|------|--------|----------------|---------------|
| Clean | `recommender.clean` | `books.csv` → `books_clean.csv` | Drops rows missing description, author, thumbnail, or having rating = 0. Safe to re-run on already-enriched data. |
| Enrich | `recommender.enrich` | `books_clean.csv` → `books_enriched.csv` | Sends batches of 5 books to OpenRouter; asks for exactly 3 genres + 3 keywords per book (temperature 0.3). Saves progress every 10 batches, so interrupted runs resume from last checkpoint. |
| Embed | `recommender.embed` | `books_enriched.csv` → `embeddings.npz` | Builds a composite text per book: `title. by authors. Genres: ... Keywords: ... Category: ... description[:400]`. Encodes with BGE-large-en-v1.5, L2-normalised. Output shape is `(N, 1024)`. |

### Recommendation Flow

1. **User query** enters the Gradio search box.
2. GUI applies local filters (page count, rating, sort) to narrow the candidate ISBN set.
3. Filtered ISBNs + query string are sent to `POST /recommend` on the FastAPI backend.
4. The API encodes the query with the same BGE model (prefixed with the retrieval prompt), computes dot-product similarity against the candidate subset, blends in the rating boost, and returns the top-N results.
5. If the API is unreachable (still loading or crashed), the GUI falls back to a naive substring search over title/author/genre/keyword/description fields.

### Personal List & "Recommended for You"

- The user's list is maintained entirely client-side in a JavaScript array (`_userList`). It is not persisted across page reloads.
- When the list is non-empty, the home page builds a composite query from the liked books' genres and keywords (up to 60 terms, semicolon-joined) and calls `/recommend` to generate the "Recommended for You" hero banner and carousel row.
- Books already in the list are excluded from recommendations.

### Known Constraints

- **No persistence** — the personal list lives in the browser's JS runtime. Refreshing the page resets it.
- **Single-user design** — all state is per-tab; no authentication or multi-user session management.
- **Port conflicts** — the `lsof`-based port cleanup in `run.py` uses `kill -9`, which is Unix-only. Windows users need to free ports 7860/7861/8000 manually.
- **Memory footprint** — the BGE-large model (~1.3 GB) plus the embedding matrix (~25 MB for 6 000 books at 1024-dim float32) are held in RAM for the lifetime of the API process. On machines with < 4 GB free RAM, expect swap pressure.
- **Cold start** — first launch downloads the BGE model (~1.3 GB) and computes all embeddings. Allow 5-10 min on a laptop with no GPU.
- **LLM enrichment rate limits** — OpenRouter free-tier models have per-minute token caps. Enriching all ~6 200 books takes 20-40 min at batch_size=5 with the default 0.5 s inter-batch delay. Increase `batch_delay` if you hit 429 errors.

### Extending the System

- **Swap the embedding model** — change `embedding.model` in `config.yaml` and re-run `python -m recommender.embed`. Any sentence-transformers-compatible model works; smaller models (e.g., `all-MiniLM-L6-v2`, 384-dim) trade quality for ~4x faster encoding and ~3x less memory.
- **Add a new dataset** — replace `data/books.csv` with any CSV containing at minimum: `isbn13`, `title`, `authors`, `description`, `thumbnail`, `average_rating`. Then re-run the full pipeline (clean → enrich → embed).
- **Persist the user list** — the bridge server's `/add` and `/remove` endpoints already return the updated list. A `localStorage` write in the JS bridge or a small SQLite backend would make it survive page reloads.
