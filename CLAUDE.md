# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository currently contains only `PRD.md` — a product requirements document. No application code, `requirements.txt`, or `.env` file has been created yet. There is no build/lint/test tooling to run until the implementation described below exists.

## What this project is

A Streamlit agent ("매크로 테마 밸류체인 탐색 에이전트") that takes a macro/theme news input (e.g. "엔비디아 H200 수요 폭증") and screens for supply-chain stocks that haven't rallied yet:

1. Call an OpenAI ChatGPT model to extract a value-chain graph (materials → parts → equipment → end product) from the input issue, as structured JSON.
2. Map each value-chain node to a real listed ticker, validating names/codes against `fdr.StockListing('KRX')` (KRX-focused; FinanceDataReader has limited non-Korean coverage, so scope stays domestic for the MVP).
3. Build the extracted nodes/relations into a graph (`networkx`) rooted at the lead stock ("대장주") and compute hop distance to derive 1차/2차/3차 beneficiary tiers — this is the project's "GraphRAG" step: a lightweight LLM-extraction + graph-traversal pattern, not full Microsoft GraphRAG (no community detection/embeddings in MVP).
4. Pull price data per ticker via `FinanceDataReader`, with a mandatory `time.sleep(1)` between calls to avoid rate limiting.
5. Exclude any stock already up +15% or more (user-adjustable threshold) from recommendations — mark it "already reacted" rather than dropping it silently.
6. Rank and present the remaining 2차/3차 candidates with an LLM-generated rationale for each.

Full requirements, the pipeline diagram, UI layout, and constraints live in `PRD.md` — read it before implementing `app.py`.

## Planned architecture (per PRD §6–9, §13)

- `app.py` — single Streamlit app implementing the pipeline above (sidebar for issue input/threshold/period, main area for value-chain viz + results table).
- `requirements.txt` — `streamlit`, `openai`, `python-dotenv`, `finance-datareader`, `networkx`, `pandas`, `plotly` (per PRD §8).
- `.env` (not committed) — `OPENAI_API_KEY`, loaded via `python-dotenv`. Never hardcode the key.

## Constraints to preserve when implementing

- OpenAI API key must come from `.env`, never hardcoded.
- Every `FinanceDataReader` call in a per-ticker loop must be followed by `time.sleep(1)`.
- Stocks already up ≥15% (default threshold, must stay user-adjustable via UI) are excluded from the recommended list but still shown/labeled, not silently dropped.
- LLM-derived tickers must be validated against actual KRX listings before being treated as real — LLM hallucination of nonexistent tickers/relationships is a known risk called out in the PRD.
- This is a screening/reference tool, not investment advice — the PRD requires a disclaimer in the UI.
