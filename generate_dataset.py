from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

OUTPUT_PATH = Path("dataset.csv")

DOMAIN_COUNTS: Dict[str, int] = {
    "healthcare": 30,
    "finance": 30,
    "legal": 30,
    "technology": 30,
    "general": 20,
    "cross_domain": 10,
}


def build_healthcare_queries() -> List[str]:
    concerns = [
        "persistent fever",
        "lower back pain",
        "chest pain",
        "stomach pain",
        "knee pain",
        "headache symptoms",
        "seasonal allergy symptoms",
        "flu symptoms",
        "dehydration symptoms",
        "insomnia symptoms",
    ]
    contexts = [
        "for two days",
        "after exercise",
        "during pregnancy",
        "in a 7-year-old child",
        "in an elderly parent",
        "while fasting",
        "after traveling",
        "with no prior history",
        "with mild dizziness",
        "that worsens at night",
    ]

    queries: List[str] = []
    for i in range(30):
        concern = concerns[i % len(concerns)]
        context = contexts[i % len(contexts)]
        queries.append(
            f"What should I do about {concern} {context}, and when should I seek urgent medical care?"
        )
    return queries


def build_finance_queries() -> List[str]:
    targets = [
        "retirement",
        "a house down payment",
        "my emergency fund",
        "college savings",
        "short-term goals",
        "long-term wealth",
    ]
    instruments = [
        "index stock funds",
        "government bonds",
        "high-yield savings",
        "a balanced ETF portfolio",
        "dividend stocks",
        "a money market fund",
    ]
    budgets = [
        "$200 per month",
        "$500 per month",
        "$1,000 per month",
        "$5,000 lump sum",
        "$20,000 lump sum",
    ]

    queries: List[str] = []
    for i in range(30):
        target = targets[i % len(targets)]
        instrument = instruments[i % len(instruments)]
        budget = budgets[i % len(budgets)]
        queries.append(
            f"How should I invest {budget} in {instrument} for {target} while managing risk and taxes?"
        )
    return queries


def build_legal_queries() -> List[str]:
    documents = [
        "employment agreement",
        "rental agreement",
        "freelance agreement",
        "vendor agreement",
        "non-disclosure agreement",
        "service agreement",
    ]
    issues = [
        "termination clause",
        "liability limits",
        "payment terms",
        "intellectual property rights",
        "data protection obligations",
        "dispute resolution",
    ]

    queries: List[str] = []
    for i in range(30):
        document = documents[i % len(documents)]
        issue = issues[i % len(issues)]
        queries.append(
            f"Can you explain the {issue} in this {document} and what legal rights I should verify before signing?"
        )
    return queries


def build_technology_queries() -> List[str]:
    tools = [
        "Python",
        "a Linux laptop",
        "a Windows laptop",
        "Git",
        "Docker",
        "a cloud notebook",
    ]
    problems = [
        "import error",
        "runtime error",
        "memory error",
        "dependency conflict",
        "slow performance",
        "failed deployment",
    ]

    queries: List[str] = []
    for i in range(30):
        tool = tools[i % len(tools)]
        problem = problems[i % len(problems)]
        queries.append(
            f"How can I troubleshoot a {problem} in {tool} and set up a reliable debugging workflow?"
        )
    return queries


def build_general_queries() -> List[str]:
    topics = [
        "time management",
        "public speaking",
        "team communication",
        "healthy habits",
        "career planning",
        "learning strategy",
        "travel planning",
        "conflict resolution",
        "productivity routines",
        "decision making",
    ]

    queries: List[str] = []
    for i in range(20):
        topic = topics[i % len(topics)]
        queries.append(
            f"What are practical steps to improve {topic} in daily life with measurable progress over one month?"
        )
    return queries


def build_cross_domain_queries() -> List[str]:
    return [
        "How do health insurance law changes affect hospital billing costs for chronic pain treatment?",
        "Should I invest money in healthcare stocks after new drug approval regulations?",
        "What legal rights do I have if a fintech app misreports my credit data?",
        "Can Python help analyze medical trial data for financial risk decisions?",
        "How does a startup agreement cover patient data security in a health app?",
        "What are the tax and legal implications of selling medical device software licenses?",
        "How should a hospital choose laptop security standards under privacy law requirements?",
        "Can stock market volatility impact funding for public healthcare technology projects?",
        "What symptoms of burnout in software teams can increase legal compliance mistakes?",
        "How can agreement terms protect investors when a healthcare AI model has errors?",
    ]


def build_dataset() -> pd.DataFrame:
    queries_by_domain: Dict[str, List[str]] = {
        "healthcare": build_healthcare_queries(),
        "finance": build_finance_queries(),
        "legal": build_legal_queries(),
        "technology": build_technology_queries(),
        "general": build_general_queries(),
        "cross_domain": build_cross_domain_queries(),
    }

    records = []
    for domain, expected_count in DOMAIN_COUNTS.items():
        queries = queries_by_domain[domain]
        if len(queries) != expected_count:
            raise ValueError(
                f"Domain {domain} has {len(queries)} queries; expected {expected_count}."
            )
        for query in queries:
            records.append({"query": query, "domain": domain})

    df = pd.DataFrame(records)
    if len(df) != 150:
        raise ValueError(f"Dataset has {len(df)} rows; expected 150.")
    return df


def main() -> None:
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset created:")
    print(df["domain"].value_counts().sort_index())
    print(f"Total queries: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
