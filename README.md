"# MUX-Route: Domain-Aware LLM Multiplexing"



MUX-Route: Domain-Aware LLM Multiplexing



Using one large model for every query is expensive and unnecessary. A question about taxes does not need the same treatment as a medical query. This project explores a simple idea: route queries to the right domain and let specialized handling do the work.



MUX-Route is a domain-aware routing system that selects the appropriate context and model behavior based on the query. The goal is to improve response quality while keeping the system efficient.



What this project does

Generates a structured multi-domain dataset

Routes queries using embedding-based similarity

Applies domain-aware context before inference

Builds a pipeline for evaluating response quality



This is not a toy script. It is a full pipeline that moves from data generation to evaluation.



Core idea



Instead of forcing one model to handle everything:



Classify the query into a domain

Route it to the closest domain representation

Apply domain-specific prompting

Generate a more relevant response



Simple idea. Surprisingly effective.



Current components

Dataset generation

Creates a balanced multi-domain dataset with controlled query types

Router

Uses sentence embeddings and centroid similarity to assign domains

Modular design

Each stage is separated, making it easy to extend or replace

How it works

Generate synthetic queries across multiple domains

Convert queries into embeddings

Compare against domain centroids

Select the best matching domain

Apply domain-specific instructions

Generate the final answer



No unnecessary complexity. Just structured decision-making.



Project structure



generate\_dataset.py

Creates the dataset used for experiments



router.py

Handles domain routing using embedding similarity



requirements.txt

Lists dependencies required to run the project



Work in progress



The following components are being added step by step:



Answer generation pipeline

LLM-as-judge evaluation system

Ablation experiments

Result analysis



The repository is being built in stages to reflect actual development, not a last-minute dump.



Why this matters



Most systems rely on a single model and hope for the best. This approach assumes all queries are equal. They are not.



Routing adds a layer of decision-making. That small change can improve both quality and efficiency.



Status



This project is under active development alongside a research submission. The implementation is being organized and uploaded in stages.



License



To be added

