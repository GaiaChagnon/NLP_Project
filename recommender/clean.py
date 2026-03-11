"""Remove books that lack a description, author, thumbnail, or rating.

Preserves enrichment columns (genres, keywords — 3+3 per book) if they
exist in the source file, so re-cleaning enriched data is safe.
"""

import pandas as pd
from recommender import load_config


def clean():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["raw"])
    initial = len(df)

    required = ["description", "authors", "thumbnail", "average_rating"]
    df = df.dropna(subset=required)
    for col in ["description", "authors", "thumbnail"]:
        df = df[df[col].astype(str).str.strip() != ""]
    df = df[df["average_rating"] > 0]

    # Normalise enrichment columns when present (genres: 3, keywords: 3)
    for col in ("genres", "keywords"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.reset_index(drop=True)
    df.to_csv(cfg["data"]["clean"], index=False)
    print(f"Clean: {initial} -> {len(df)} books  ({cfg['data']['clean']})")


if __name__ == "__main__":
    clean()
