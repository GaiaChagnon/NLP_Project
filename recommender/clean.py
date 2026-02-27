"""Remove books that lack a description, author, thumbnail, or rating."""

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

    df = df.reset_index(drop=True)
    df.to_csv(cfg["data"]["clean"], index=False)
    print(f"Clean: {initial} -> {len(df)} books  ({cfg['data']['clean']})")


if __name__ == "__main__":
    clean()
