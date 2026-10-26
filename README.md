# Self-Correcting RAG Engine with LangGraph & ChromaDB

A 100% local, self-correcting Retrieval-Augmented Generation (RAG) system built with LangGraph, ChromaDB, Ollama (`llama3.2`), and Great Expectations.

The pipeline validates data quality at ingestion, grades retrieved context for relevance, detects hallucinations, and rewrites search queries when context or answers fail validation checks. Everything runs on your machine, with no API keys or cloud services.

## Features

1. **Data quality guardrails at ingestion:** Great Expectations checks that text chunks are non-null and within expected length thresholds before they are indexed.
2. **Local vector storage and embeddings:** documents are split, embedded locally with HuggingFace (`all-MiniLM-L6-v2`), and persisted in ChromaDB.
3. **Relevance grading node:** filters out irrelevant chunks using local LLM inference through Ollama (`llama3.2`).
4. **Hallucination detection:** checks each generated answer against the retrieved facts to confirm it is grounded.
5. **Dynamic self-correction:** when the retrieved context or the answer fails a check, the graph loops back and rewrites the search query.

## How it works

```
Documents -> split -> validate (Great Expectations) -> embed -> ChromaDB

Question -> retrieve -> grade relevance -> generate answer -> check grounding -> answer
                ^               |                                   |
                +-- rewrite query <-- (no relevant context)         |
                +-- rewrite query <-- (answer not grounded) --------+
```

[TODO: confirm this flow matches your LangGraph nodes and edges. If you cap the number of rewrite attempts, say so here (for example, "up to 3 retries").]

## Tech stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector store | ChromaDB |
| LLM (local) | Ollama, `llama3.2` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Data validation | Great Expectations |
| Language | Python |

## Getting started

**Prerequisites**

- Python [TODO: version]
- [Ollama](https://ollama.com) installed and running

**Setup**

```bash
git clone https://github.com/Mrudula0619/Project_3_RAG.git
cd Project_3_RAG
pip install -r requirements.txt   # [TODO: confirm this file exists]
ollama pull llama3.2
```

**Run**

```bash
# [TODO: replace with your real commands, for example:]
# python ingest.py       # validate, embed, and index documents
# python main.py "your question here"
```

## Project structure

```
# [TODO: paste the output of `tree -L 2` or list your main files]
```

## Example

```
# [TODO: add one real example: a question, the retrieved context,
#  and a case where the query was rewritten. This is the best part
#  of the README for a recruiter to see.]
```

## Limitations

- [TODO: one or two honest limits, such as document types supported or model size on local hardware]
