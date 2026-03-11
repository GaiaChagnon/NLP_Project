"""FastAPI backend — serves book recommendations over cosine similarity + rating boost."""

import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from recommender import load_config

cfg = load_config()

app = FastAPI(title="Book Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pre-computed data ──────────────────────────────────
print("[api] Loading enriched book catalog ...")
df = pd.read_csv(cfg["data"]["enriched"]).fillna("")

emb_path = cfg["data"]["embeddings"]
if not os.path.exists(emb_path):
    raise FileNotFoundError(
        f"{emb_path} not found — run `python -m recommender.embed` first"
    )

print(f"[api] Loading embeddings from {emb_path} ...")
npz = np.load(emb_path, allow_pickle=True)
embeddings = npz["embeddings"].astype(np.float32)
isbn_arr = npz["isbn13"]

isbn_to_idx = {int(v): i for i, v in enumerate(isbn_arr)}
book_lookup = {int(row["isbn13"]): row.to_dict() for _, row in df.iterrows()}

# ── Embedding model (shared across requests) ───────────────
print(f"[api] Loading sentence-transformer model ({cfg['embedding']['model']}) ...")
model = SentenceTransformer(cfg["embedding"]["model"])
query_prefix = cfg["embedding"].get("query_prefix", "")
print("[api] Model loaded — server ready.")

# ── Scoring parameters ─────────────────────────────────────
rating_w = cfg["scoring"]["rating_weight"]
r_min, r_max = cfg["scoring"]["min_rating"], cfg["scoring"]["max_rating"]


# ── Schemas ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str
    book_ids: list[int] | None = None  # None = use all books
    n: int = 10


# ── Endpoints ───────────────────────────────────────────────
@app.get("/")
def root():
    """Status page when visiting the API root in a browser."""
    return {
        "service": "BookFlix Recommender API",
        "status": "ok",
        "total_books": len(book_lookup),
        "endpoints": ["/recommend (POST)", "/books (GET)", "/books/{isbn} (GET)", "/health (GET)", "/docs (GET)"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    # Resolve candidate set
    if req.book_ids is None:
        indices = list(range(len(isbn_arr)))
        isbns = [int(v) for v in isbn_arr]
    else:
        indices, isbns, seen = [], [], set()
        for bid in req.book_ids:
            if bid in isbn_to_idx and bid not in seen:
                indices.append(isbn_to_idx[bid])
                isbns.append(bid)
                seen.add(bid)

    if not indices:
        raise HTTPException(400, "None of the provided book_ids matched the dataset")

    # Cosine similarity (embeddings are already L2-normalised)
    q_emb = model.encode([query_prefix + req.query], normalize_embeddings=True)[0]
    sims = embeddings[indices] @ q_emb

    # Mild rating boost
    norm_ratings = np.array([
        (book_lookup.get(b, {}).get("average_rating", 0) - r_min) / max(r_max - r_min, 1e-9)
        for b in isbns
    ])
    scores = (1 - rating_w) * sims + rating_w * norm_ratings

    top = np.argsort(scores)[::-1][: req.n]

    results = []
    for t in top:
        b = book_lookup.get(isbns[t], {})
        results.append({
            "isbn13": isbns[t],
            "title": b.get("title", ""),
            "authors": b.get("authors", ""),
            "thumbnail": b.get("thumbnail", ""),
            "description": str(b.get("description", ""))[:300],
            "average_rating": float(b.get("average_rating", 0)),
            "ratings_count": int(b.get("ratings_count", 0)),
            "published_year": int(b.get("published_year", 0)),
            "num_pages": int(b.get("num_pages", 0)),
            "genres": b.get("genres", ""),
            "keywords": b.get("keywords", ""),
            "categories": b.get("categories", ""),
            "score": round(float(scores[t]), 4),
            "similarity": round(float(sims[t]), 4),
        })

    return {"recommendations": results, "total_candidates": len(indices)}


@app.get("/books")
def list_books(
    min_rating: float = Query(0, ge=0, le=5),
    category: str = Query(""),
    year_from: int = Query(0),
    year_to: int = Query(9999),
):
    """Return filterable book catalog so the frontend can build its permissible list."""
    out = df.copy()
    if min_rating > 0:
        out = out[out["average_rating"] >= min_rating]
    if category:
        mask = out["categories"].str.contains(category, case=False, na=False)
        if "genres" in out.columns:
            mask |= out["genres"].str.contains(category, case=False, na=False)
        out = out[mask]
    if year_from > 0:
        out = out[out["published_year"] >= year_from]
    if year_to < 9999:
        out = out[out["published_year"] <= year_to]

    cols = ["isbn13", "title", "authors", "categories", "genres", "keywords",
            "thumbnail", "average_rating", "ratings_count", "published_year",
            "num_pages"]
    cols = [c for c in cols if c in out.columns]
    return {"books": out[cols].to_dict(orient="records"), "count": len(out)}


@app.get("/books/{isbn}")
def get_book(isbn: int):
    """Return full metadata for a single book by ISBN-13."""
    if isbn not in book_lookup:
        raise HTTPException(404, "Book not found")
    return book_lookup[isbn]


if __name__ == "__main__":
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])
