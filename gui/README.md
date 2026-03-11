# GUI Module

Netflix-style book browsing and recommendation interface built with Gradio.

## Running

```bash
# Standalone (requires API server running separately)
python -m gui.app

# Or via the unified launcher (starts API + Bridge + GUI)
python run.py
```

Opens at `http://localhost:7860` by default (configurable in `config.yaml` → `gui.port`).

## Features

### Hero Banner
- Large featured section at the top of the page with blurred background, cover art, title, rating, description.
- Shows the top recommendation when the user has saved books, otherwise the top-rated book (highest rating with 500+ reviews).

### Home View (Browse)
- **Recommended for You** — appears when you have saved books; combines their genres + keywords into a composite query sent to the recommender API.
- **Top Rated** — highest average rating among books with 500+ reviews (ensures both quality and broad appeal).
- **Genre Rows** — top 12 genres by book count (Fantasy, Mystery, Thriller, etc.), each a scrollable carousel with navigation arrows.
- **Round-robin distribution** — genre rows are filled one slot at a time across all genres so every category gets evenly distributed quality.
- **No duplicates** — a book never appears in more than one section.

### Cover Images
- Primary source: **Open Library Covers API** (`-L` size, 300-500px, 98% coverage).
- Fallback: Google Books thumbnail (128px, tied to volume ID — always correct).
- `onload` crosscheck: if Open Library returns a non-book image (wrong aspect ratio from ISBN mismatch), auto-swaps to Google Books.
- Cards with no available cover from either source are hidden automatically.

### Search
- Type any free-text query (title, author, genre, keyword, theme).
- Filters are applied first (min/max pages, min rating) to narrow the candidate set.
- The filtered ISBNs + query go to the `/recommend` API endpoint.
- Results display in a ranked 6-column grid with rank badges.
- Sort options: Relevance (API score), Alphabetical, Rating, Newest.
- Falls back to local text matching when the API server is unavailable.

### Book Detail
- Click any book thumbnail to open its detail view.
- Shows: cover image, title, subtitle, author, publication year, page count, average rating with count, full description.
- Displays all enrichment tags: **genres** (red), **keywords** (gold), **categories** (blue).
- "Add to Your List" / "Remove" toggle button.

### Your List
- Click "Your List (N)" in the nav bar to view saved books.
- State is managed client-side via the JavaScript bridge (resets on page reload).

### Hover Effect
- Hovering a book thumbnail blurs the image and overlays: title, author, star rating, and a truncated description.
- Pure CSS implementation.

## Architecture

The GUI loads `data/books_enriched.csv` directly for catalog browsing (no API needed for the home page). Search and "Recommended for You" delegate to the FastAPI `/recommend` endpoint via `httpx`.

Interactive elements (book clicks, list add/remove, navigation) communicate through a lightweight FastAPI bridge server (port 7861) that bypasses Gradio's SSE API. The JavaScript bridge (`_LOAD_JS`) makes `fetch` calls to the bridge, which calls rendering functions in `app.py` and returns plain JSON with HTML.

User state ("Your List") is managed client-side in a JavaScript `_userList` array, passed explicitly in each bridge request. The bridge server intercepts Gradio button clicks to use this JS state instead of Gradio's `gr.State`.

## Customisation

| What               | Where                            | Default           |
|--------------------|----------------------------------|-----------------  |
| GUI port           | `config.yaml` → `gui.port`      | 7860              |
| API URL            | Derived from `config.yaml` → `api.port` | localhost:8000 |
| Genre rows shown   | `TOP_GENRES` in `gui/app.py`     | Top 12 by count   |
| Books per carousel | `ITEMS_PER_ROW` in `gui/app.py`  | 20                |
| Dark theme colors  | `CSS` string in `gui/app.py`     | Netflix palette   |
