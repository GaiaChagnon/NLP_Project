"""Add LLM-generated genres and keywords to each book via OpenRouter."""

import json
import os
import time

import httpx
import pandas as pd
from dotenv import load_dotenv

from recommender import load_config

load_dotenv()


def _classify_batch(client: httpx.Client, model: str, books: list[dict]) -> list[dict] | None:
    """Ask the LLM for 3 genres + 3 keywords per book. Returns parsed JSON list."""
    book_list = "\n".join(
        f'{i + 1}. "{b["title"]}" by {b["authors"]} — {str(b.get("description", ""))[:200]}'
        for i, b in enumerate(books)
    )
    prompt = (
        "For each book below, return exactly 3 genres and 3 thematic keywords.\n"
        "Reply ONLY with a JSON array (one object per book, same order):\n"
        '[{"genres":["g1","g2","g3"],"keywords":["k1","k2","k3"]}]\n\n'
        f"{book_list}"
    )
    resp = client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    start, end = text.find("["), text.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return None


def enrich():
    cfg = load_config()
    llm = cfg["llm"]
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")

    df = pd.read_csv(cfg["data"]["clean"])
    out_path = cfg["data"]["enriched"]

    # Resume from partial progress
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        if "genres" in prev.columns:
            done = prev.dropna(subset=["genres"])
            done = done[done["genres"].astype(str).str.strip() != ""]
            done_map = dict(zip(done["isbn13"], zip(done["genres"], done["keywords"])))
            df["genres"] = df["isbn13"].map(lambda x: done_map.get(x, ("", ""))[0])
            df["keywords"] = df["isbn13"].map(lambda x: done_map.get(x, ("", ""))[1])
        else:
            df["genres"], df["keywords"] = "", ""
    else:
        df["genres"], df["keywords"] = "", ""

    todo = df.index[df["genres"].astype(str).str.strip() == ""].tolist()
    total = len(df)
    print(f"To enrich: {len(todo)} / {total}")

    client = httpx.Client(
        base_url=llm["base_url"],
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )

    bs = llm["batch_size"]
    for i in range(0, len(todo), bs):
        batch_idx = todo[i : i + bs]
        books = [df.loc[idx].to_dict() for idx in batch_idx]

        for attempt in range(llm["max_retries"]):
            try:
                results = _classify_batch(client, llm["model"], books)
                if results and len(results) >= len(books):
                    for j, idx in enumerate(batch_idx):
                        df.at[idx, "genres"] = "; ".join(results[j].get("genres", []))
                        df.at[idx, "keywords"] = "; ".join(results[j].get("keywords", []))
                    break
            except Exception as e:
                wait = llm["retry_delay"] * (attempt + 1)
                print(f"  Batch {i // bs} attempt {attempt + 1} failed: {e} — retry in {wait}s")
                time.sleep(wait)

        # Save progress periodically
        if (i // bs) % 10 == 0:
            df.to_csv(out_path, index=False)
            done_n = total - len(df.index[df["genres"].astype(str).str.strip() == ""])
            print(f"  Progress: {done_n}/{total}")

        time.sleep(llm.get("batch_delay", 0.5))

    client.close()
    df.to_csv(out_path, index=False)
    print(f"Enriched: {total} books -> {out_path}")


if __name__ == "__main__":
    enrich()
