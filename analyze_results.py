from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAIN_SCORES_PATH = Path("scores.csv")
SCORES_FULL_PATH = Path("scores_full.csv")
SCORES_NO_PROMPT_PATH = Path("scores_no_prompt.csv")
SCORES_NO_ROUTING_PATH = Path("scores_no_routing.csv")
RESULTS_PATH = Path("results.txt")


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    required_cols = {"query", "A_score", "B_score", "Better"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def compute_quality_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    mux_avg = float(np.mean(df["A_score"]))
    mono_avg = float(np.mean(df["B_score"]))
    win_rate = float(np.mean(df["A_score"] > df["B_score"]) * 100.0)
    return mux_avg, mono_avg, win_rate


def main() -> None:
    main_df = load_scores(MAIN_SCORES_PATH)
    mux_avg, mono_avg, win_rate = compute_quality_metrics(main_df)

    full_df = load_scores(SCORES_FULL_PATH)
    no_prompt_df = load_scores(SCORES_NO_PROMPT_PATH)
    no_routing_df = load_scores(SCORES_NO_ROUTING_PATH)

    full_avg = float(np.mean(full_df["A_score"]))
    no_prompt_avg = float(np.mean(no_prompt_df["A_score"]))
    no_routing_avg = float(np.mean(no_routing_df["A_score"]))

    drop_no_prompt = full_avg - no_prompt_avg
    drop_no_routing = full_avg - no_routing_avg

    quality_table = (
        "Quality Comparison\n"
        f"MUX: {mux_avg:.2f}\n"
        f"Mono: {mono_avg:.2f}\n\n"
        f"Win Rate: {win_rate:.2f}%\n"
    )

    ablation_table = (
        "Ablation Results\n"
        f"Full: {full_avg:.2f}\n"
        f"No Prompt: {no_prompt_avg:.2f} (Drop vs Full: {drop_no_prompt:.2f})\n"
        f"No Routing: {no_routing_avg:.2f} (Drop vs Full: {drop_no_routing:.2f})\n"
    )

    summary = (
        "We evaluate on 150 queries using LLM-as-Judge scoring for correctness, "
        "completeness, and clarity. The judge compares MUX-Route (Answer A) and "
        "a mono baseline (Answer B) on a 1-10 scale. Results show overall quality "
        "differences and quantify ablation impact from removing domain prompts "
        "or routing.\n\n"
    )

    output_text = summary + quality_table + "\n" + ablation_table

    print(quality_table)
    print(ablation_table)

    RESULTS_PATH.write_text(output_text, encoding="utf-8")
    print(f"Saved paper-ready summary to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
