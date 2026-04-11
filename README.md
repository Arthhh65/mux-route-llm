\# MUX-Route: Domain-Aware LLM Multiplexing



Using one large model for every query is inefficient. Not every question needs the same level of reasoning or the same domain knowledge. This project explores a simple idea: route queries to the right domain and apply the right context before generating answers.



MUX-Route is a domain-aware routing system designed to improve response quality while keeping inference efficient.



\---



\## What this project does



\- Generates a structured multi-domain dataset  

\- Routes queries using embedding-based similarity  

\- Applies domain-aware prompting  

\- Generates responses using LLMs  

\- Evaluates outputs using an LLM-as-judge framework  

\- Analyzes performance through ablation experiments  



This is a complete pipeline from data generation to evaluation and analysis.



\---



\## Core idea



Instead of forcing one model to handle everything:



\- Classify the query into a domain  

\- Route it to the closest domain representation  

\- Apply domain-specific prompting  

\- Generate a more relevant response  



Simple idea. Better results.



\---



\## How it works



1\. Generate synthetic queries across multiple domains  

2\. Convert queries into embeddings  

3\. Compare against domain centroids  

4\. Select the best matching domain  

5\. Apply domain-specific instructions  

6\. Generate answers using LLMs  

7\. Evaluate responses using a judge model  



The system focuses on structured decision-making rather than increasing model size.



\---



\## Project structure



generate\_dataset.py  

Creates a synthetic multi-domain dataset  



router.py  

Routes queries using embedding similarity  



generate\_answers.py  

Generates responses using routed prompts  



judge.py  

Evaluates responses using LLM-as-judge scoring  



ablation.py  

Runs ablation experiments on system components  



analyze\_results.py  

Computes metrics and summarizes results  



results.txt  

Contains final evaluation summary  



\---



\## Results



The system is evaluated using an LLM-as-Judge framework based on:



\- correctness  

\- completeness  

\- clarity  



\### Sample Metrics



\- MUX Average Score: 9.45  

\- Mono Baseline Score: 8.70  

\- Win Rate: 78.75%  



\### Key observations



\- MUX routing achieves higher average quality compared to a mono baseline  

\- Strong win rate improvement across queries  

\- Domain-aware prompting contributes measurable gains  

\- Routing improves consistency without increasing model size  



These results show that structured routing can match or outperform single-model setups.



\---



\## Quick Start



Install dependencies:

pip install -r requirements.txt



Generate dataset:

python generate\_dataset.py



Generate answers:

python generate\_answers.py



Evaluate results:

python judge.py



Run analysis:

python analyze\_results.py



\---



\## Limitations



\- Uses synthetic dataset, not real-world data  

\- Evaluation is based on LLM-as-judge, not human scoring  

\- Does not measure hallucination or safety explicitly  



These are areas for future improvement.



\---



\## Why this matters



Most systems rely on a single model and assume all queries are equal. They are not.



Routing introduces a simple decision layer that improves both efficiency and response quality without adding heavy complexity.



\---



\## Status



The system is fully implemented and evaluated. The core pipeline is complete.



\---



\## License



To be added

