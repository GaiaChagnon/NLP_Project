# Data

Book dataset and derived artifacts for the recommender pipeline.

## Files

| File                 | Rows  | Columns                              | Source               |
|----------------------|-------|--------------------------------------|----------------------|
| `books.csv`          | ~7 000 | isbn13, isbn10, title, subtitle, authors, categories, thumbnail, description, published_year, average_rating, num_pages, ratings_count | Original dataset |
| `books_clean.csv`    | ~6 229 | Same as above                        | `recommender.clean`  |
| `books_enriched.csv` | ~6 229 | + `genres` (3 semicolon-separated), `keywords` (3 semicolon-separated) | `recommender.enrich` |
| `embeddings.npz`     | —     | `embeddings` (N x 1024 float32), `isbn13` (N,) | `recommender.embed`  |

## Enrichment Columns

Each book in `books_enriched.csv` has 6 LLM-generated enrichment values:

- **genres** — 3 genre labels, semicolon-separated (e.g., `"Fantasy; Adventure; Coming-of-Age"`)
- **keywords** — 3 thematic keywords, semicolon-separated (e.g., `"magic; apprenticeship; destiny"`)

These are used for:
1. Genre-based browsing rows in the GUI
2. Enriched embedding text for better semantic search
3. "Recommended for You" composite queries from liked books

## Filtered Titles

The GUI excludes a small number of titles with known cover-image mismatches at load time. These books remain in the CSV but are not displayed. See `_REMOVE_TITLES` in `gui/app.py`.
