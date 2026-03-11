# Recommender Module

Core NLP pipeline and API server for the book recommender system. Handles data cleaning, LLM enrichment, embedding computation, and recommendation serving.

## Pipeline Steps

### 1. Cleaning (`clean.py`)

```bash
python -m recommender.clean
```

- **Input:** `data/books.csv` (raw)
- **Output:** `data/books_clean.csv`
- **Logic:** Drops rows missing any of: `description`, `authors`, `thumbnail`, `average_rating > 0`. Preserves enrichment columns (`genres`, `keywords`) if they exist in the source.

### 2. Enrichment (`enrich.py`)

```bash
python -m recommender.enrich
```

- **Input:** `data/books_clean.csv`
- **Output:** `data/books_enriched.csv`
- **Logic:** Calls an OpenRouter LLM to generate **3 genres** and **3 thematic keywords** per book (6 enrichment values total). Results are semicolon-separated (e.g., `"Fantasy; Adventure; Coming-of-Age"`). Supports resume from partial progress — safe to interrupt and rerun.
- **Requires:** `OPENROUTER_API_KEY` in `.env`

### 3. Embedding (`embed.py`)

```bash
python -m recommender.embed
```

- **Input:** `data/books_enriched.csv`
- **Output:** `data/embeddings.npz` (numpy archive: `embeddings` matrix + `isbn13` array)
- **Logic:** Builds a compact text representation per book (title, author, genres, keywords, categories, truncated description) and encodes it with `BAAI/bge-large-en-v1.5`. Embeddings are L2-normalised for direct cosine similarity via dot product.

## API Server (`api.py`)

```bash
python -m recommender.api
```

Starts a FastAPI server on the port specified in `config.yaml` (default: 8000).

### Endpoints

| Method | Path             | Description                                    |
|--------|------------------|------------------------------------------------|
| GET    | `/`              | Status page with endpoint listing              |
| GET    | `/health`        | Health check (used by `run.py` readiness probe)|
| POST   | `/recommend`     | NLP-powered recommendations (cosine + rating)  |
| GET    | `/books`         | Filterable book catalog                        |
| GET    | `/books/{isbn}`  | Single book detail by ISBN-13                  |

### POST `/recommend`

**Request body:**
```json
{
    "query": "epic fantasy with magic systems",
    "book_ids": [9780006163831, 9780006480099],
    "n": 10
}
```

- `query` (required) — free-text search prompt
- `book_ids` (optional) — restrict candidates to these ISBNs; `null` = search all
- `n` (optional, default 10) — number of results

**Response:**
```json
{
    "recommendations": [
        {
            "isbn13": 9780006480099,
            "title": "Assassin's Apprentice",
            "authors": "Robin Hobb",
            "thumbnail": "http://...",
            "description": "...",
            "average_rating": 4.15,
            "ratings_count": 133972,
            "published_year": 1996,
            "num_pages": 460,
            "genres": "Fantasy; Adventure; Coming-of-Age",
            "keywords": "magic; apprenticeship; destiny",
            "categories": "American fiction",
            "score": 0.8234,
            "similarity": 0.7891
        }
    ],
    "total_candidates": 6229
}
```

### Scoring Formula

```
final_score = (1 - rating_weight) * cosine_similarity
            +      rating_weight  * normalised_rating
```

`rating_weight` defaults to 0.1 (see `config.yaml`), giving a mild boost to higher-rated books while keeping semantic relevance dominant.

### GET `/books`

Query parameters: `min_rating`, `category`, `year_from`, `year_to`. Returns columns: isbn13, title, authors, categories, genres, keywords, thumbnail, average_rating, ratings_count, published_year, num_pages.

### GET `/books/{isbn}`

Returns the full metadata dict for a single book, or 404 if not found.
