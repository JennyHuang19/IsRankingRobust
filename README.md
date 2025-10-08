# Robustness of LLM Ranking Systems

This repository accompanies our ICML 2025 paper, presented as an Oral at the 2nd Workshop on Models of Human Feedback for AI Alignment:
**"Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings"**

### Overview

LLM leaderboards, such as Chatbot Arena, rank models based on thousands of human or AI preference comparisons. But how stable are these rankings really?

We show that removing as little as 0.003% of preference data can flip the top of the leaderboard.
To investigate this phenomenon, we introduce a fast method for checking whether the rankings of top-performing LLMs are stable to the removal of a small number of preference evaluations, focusing on Bradley–-Terry based systems, like Chatbot Arena.

Our method also efficiently identifies the most influential prompt/response pairs in each arena.

### What’s Inside

- **`package/`** — Core implementation of the robustness-check algorithm.  
- **`notebooks/`** — Interactive scripts for reproducing sensitivity analyses on Chatbot Arena, Webdev Arena, Vision Arena, Search Arena, and MT Bench.
- **`results/`** — Data files generated from running robustness checks on each arena. Each file includes the dropped prompts/answers and the number of points that needed to be removed to alter the rankings in that arena.
