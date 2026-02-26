# Book Recommender System

NLP-based book recommender system. Loads book metadata, builds recommendation logic, and exposes a Gradio GUI for queries.

## Requirements

- Load and preprocess the dataset
- Build recommendation logic (method TBD)
- Allow user requests for recommendations
- GUI with Gradio (title input, text search, filters)
- End-to-end execution

## Structure

```
data/           # Dataset (books.csv) and preprocessing
recommender/    # Recommendation logic
gui/            # Gradio interface
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

*(To be added once implementation is complete.)*

## Data

`data/books.csv` — metadata: title, authors, categories, description, thumbnail, published_year, average_rating, etc.
