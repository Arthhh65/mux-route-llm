from __future__ import annotations

import os
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm

from router import route_query

DATASET_PATH = Path("dataset.csv")
OUTPUT_PATH = Path("answers.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral:latest")
MAX_QUERIES = 80


def generate_single_answer(model: str, prompt: str, query: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.2, "num_predict": 120},
    )
    return response["message"]["content"].strip()


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.csv not found. Run generate_dataset.py first.")

    df = pd.read_csv(DATASET_PATH).head(MAX_QUERIES)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        query = str(row["query"])
        routed_domain = route_query(query)

        mux_prompt = (
            f"You are an expert in {routed_domain}. "
            "Answer clearly in 3-4 lines."
        )
        mono_prompt = "Answer clearly in 3-4 lines."

        mux_answer = generate_single_answer(MODEL_NAME, mux_prompt, query)
        mono_answer = generate_single_answer(MODEL_NAME, mono_prompt, query)

        rows.append(
            {
                "query": query,
                "mux_answer": mux_answer,
                "mono_answer": mono_answer,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved answers to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
