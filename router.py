from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path(__file__).with_name("dataset.csv")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None
_centroids: Dict[str, np.ndarray] | None = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _init_router() -> None:
    global _embedder, _centroids

    if _embedder is not None and _centroids is not None:
        return

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"dataset.csv not found at {DATASET_PATH}. Run generate_dataset.py first.")

    df = pd.read_csv(DATASET_PATH)
    required_cols = {"query", "domain"}
    if not required_cols.issubset(df.columns):
        raise ValueError("dataset.csv must contain columns: query, domain")

    _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    centroids: Dict[str, np.ndarray] = {}
    for domain, domain_df in df.groupby("domain"):
        queries = domain_df["query"].astype(str).tolist()
        embeddings = _embedder.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        centroid = embeddings.mean(axis=0)
        centroids[str(domain)] = _normalize(centroid)

    _centroids = centroids


def route_query(query: str) -> str:
    """Route query to the most similar domain centroid in embedding space."""
    _init_router()
    assert _embedder is not None
    assert _centroids is not None

    query_embedding = _embedder.encode(
        [str(query)],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]

    best_domain = "general"
    best_score = float("-inf")
    for domain, centroid in _centroids.items():
        score = float(np.dot(query_embedding, centroid))
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain
