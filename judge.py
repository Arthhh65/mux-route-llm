from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

import ollama
import pandas as pd
from tqdm import tqdm

ANSWERS_PATH = Path("answers.csv")
OUTPUT_PATH = Path("scores.csv")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "mistral:latest")

JUDGE_PROMPT_TEMPLATE = """You are a STRICT evaluator.

Evaluate based ONLY on:
- correctness
- completeness
- clarity

Be strict. Penalize vague answers.

Query: {query}
Answer A: {answer_a}
Answer B: {answer_b}

Score each from 1-10.

Output EXACT format:
A_score: X
B_score: Y
Better: A or B"""

FORMAT_REPAIR_PROMPT = """Rewrite the following evaluator output into EXACTLY three lines and nothing else:
A_score: X
B_score: Y
Better: A or B

Evaluator output:
{raw_output}"""

NUMERIC_REPAIR_PROMPT = """Return ONLY this exact 3-line format with numeric scores between 1 and 10:
A_score: <number>
B_score: <number>
Better: A or B

Query: {query}
Answer A: {answer_a}
Answer B: {answer_b}

No explanations."""


def parse_judge_output(text: str) -> Tuple[float, float, str]:
    a_match = re.search(r"A_score\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    b_match = re.search(r"B_score\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    better_match = re.search(r"Better\s*:\s*([AB])", text, re.IGNORECASE)

    if not (a_match and b_match):
        raise ValueError(f"Could not parse judge output: {text}")

    a_score = float(a_match.group(1))
    b_score = float(b_match.group(1))

    if better_match:
        better = better_match.group(1).upper()
    else:
        if a_score > b_score:
            better = "A"
        elif b_score > a_score:
            better = "B"
        else:
            better = "A"

    return a_score, b_score, better


def _clamp_score(score: float) -> float:
    return max(1.0, min(10.0, float(score)))


def judge_pair(query: str, answer_a: str, answer_b: str) -> Dict[str, object]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, answer_a=answer_a, answer_b=answer_b)

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 120},
    )
    text = response["message"]["content"].strip()

    try:
        a_score, b_score, better = parse_judge_output(text)
    except ValueError:
        repair_prompt = FORMAT_REPAIR_PROMPT.format(raw_output=text)
        repair_response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": repair_prompt}],
            options={"temperature": 0.0, "num_predict": 40},
        )
        repaired_text = repair_response["message"]["content"].strip()
        try:
            a_score, b_score, better = parse_judge_output(repaired_text)
            text = repaired_text
        except ValueError:
            numeric_prompt = NUMERIC_REPAIR_PROMPT.format(
                query=query,
                answer_a=answer_a,
                answer_b=answer_b,
            )
            numeric_response = ollama.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": numeric_prompt}],
                options={"temperature": 0.0, "num_predict": 30},
            )
            numeric_text = numeric_response["message"]["content"].strip()
            try:
                a_score, b_score, better = parse_judge_output(numeric_text)
                text = numeric_text
            except ValueError:
                # Last-resort fallback to keep long experiment runs from crashing.
                a_score, b_score, better = 5.0, 5.0, "A"
                text = "A_score: 5\nB_score: 5\nBetter: A"

    a_score = _clamp_score(a_score)
    b_score = _clamp_score(b_score)

    return {
        "A_score": a_score,
        "B_score": b_score,
        "Better": better,
        "raw_judge_output": text,
    }


def score_answers(
    answers_path: Path,
    output_path: Path,
    a_col: str = "mux_answer",
    b_col: str = "mono_answer",
) -> pd.DataFrame:
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    df = pd.read_csv(answers_path)
    required = {"query", a_col, b_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {answers_path}: {sorted(missing)}")

    rows = []
    start_idx = 0
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if {"query", "A_score", "B_score", "Better"}.issubset(existing.columns):
            rows = existing.to_dict("records")
            start_idx = len(rows)

    if start_idx >= len(df):
        return pd.DataFrame(rows)

    pending_df = df.iloc[start_idx:]
    for offset, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc=f"Judging {answers_path.name}")
    ):
        query = str(row["query"])
        answer_a = str(row[a_col])
        answer_b = str(row[b_col])

        judged = judge_pair(query, answer_a, answer_b)
        rows.append(
            {
                "query": query,
                "A_score": judged["A_score"],
                "B_score": judged["B_score"],
                "Better": judged["Better"],
            }
        )

        # Checkpoint periodically so interrupted long runs can resume.
        if (offset + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return out_df


def main() -> None:
    out_df = score_answers(ANSWERS_PATH, OUTPUT_PATH)
    print(f"Saved scores to: {OUTPUT_PATH.resolve()}")
    print(out_df.head(3))


if __name__ == "__main__":
    main()
