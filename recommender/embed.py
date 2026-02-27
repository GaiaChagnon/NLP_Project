"""Compute BGE embeddings for every enriched book and save to .npz."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recommender import load_config


def _build_text(row, max_chars: int) -> str:
    """Compact representation that fits the BGE-large 512-token window."""
    title = row["title"]
    if pd.notna(row.get("subtitle")) and str(row.get("subtitle", "")).strip():
        title += f": {row['subtitle']}"

    parts = [title, f"by {row['authors']}"]
    for col, label in [("genres", "Genres"), ("keywords", "Keywords"), ("categories", "Category")]:
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(f"{label}: {val}")
    parts.append(str(row.get("description", ""))[:max_chars])
    return ". ".join(parts)


def embed():
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    df = pd.read_csv(cfg["data"]["enriched"]).fillna("")

    model = SentenceTransformer(emb_cfg["model"])
    texts = [_build_text(row, emb_cfg["description_max_chars"]) for _, row in df.iterrows()]

    print(f"Embedding {len(texts)} books with {emb_cfg['model']} ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

    np.savez(
        cfg["data"]["embeddings"],
        embeddings=embeddings,
        isbn13=df["isbn13"].values,
    )
    print(f"Saved {cfg['data']['embeddings']}  shape={embeddings.shape}")


if __name__ == "__main__":
    embed()
