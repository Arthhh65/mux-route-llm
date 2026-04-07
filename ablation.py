from __future__ import annotations

import os
from pathlib import Path
from typing import List

import ollama
import pandas as pd
from tqdm import tqdm

from judge import score_answers
from router import route_query

DATASET_PATH = Path("dataset.csv")
MUX_MODEL = os.getenv("MUX_MODEL", "qwen2.5:7b")
MONO_MODEL = os.getenv("MONO_MODEL", "gemma:2b")

SCORES_FULL = Path("scores_full.csv")
SCORES_NO_PROMPT = Path("scores_no_prompt.csv")
SCORES_NO_ROUTING = Path("scores_no_routing.csv")


def generate_single_answer(model: str, prompt: str, query: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.2, "num_predict": 180},
    )
    return response["message"]["content"].strip()


def build_variant_answers(variant: str) -> Path:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.csv not found. Run generate_dataset.py first.")

    df = pd.read_csv(DATASET_PATH)

    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {variant}"):
        query = str(row["query"])

        if variant == "full":
            routed_domain = route_query(query)
            mux_prompt = (
                f"You are an expert in {routed_domain}. "
                "Provide a detailed, professional answer."
            )
        elif variant == "no_prompt":
            mux_prompt = "Answer briefly and simply."
        elif variant == "no_routing":
            mux_prompt = "You are an expert in general. Provide a detailed, professional answer."
        else:
            raise ValueError(f"Unknown variant: {variant}")

        mono_prompt = "Answer briefly and simply."

        variant_answer = generate_single_answer(MUX_MODEL, mux_prompt, query)
        mono_answer = generate_single_answer(MONO_MODEL, mono_prompt, query)

        rows.append(
            {
                "query": query,
                "mux_answer": variant_answer,
                "mono_answer": mono_answer,
            }
        )

    answers_path = Path(f"answers_{variant}.csv")
    pd.DataFrame(rows).to_csv(answers_path, index=False)
    return answers_path


def main() -> None:
    variants = [
        ("full", SCORES_FULL),
        ("no_prompt", SCORES_NO_PROMPT),
        ("no_routing", SCORES_NO_ROUTING),
    ]

    for variant, score_path in variants:
        answers_path = build_variant_answers(variant)
        score_answers(answers_path, score_path)
        print(f"Saved {variant} scores to: {score_path.resolve()}")


if __name__ == "__main__":
    main()
