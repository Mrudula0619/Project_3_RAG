# Self-Correcting RAG Engine with LangGraph & ChromaDB

A 100% local, self-correcting Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **ChromaDB**, **Ollama (`llama3.2`)**, and **Great Expectations**.

The pipeline validates data quality at ingestion, grades retrieved context for relevance, detects hallucinations, and dynamically rewrites search queries when context or answers fail validation checks.

---

## ✨ Key Features

1. **Data Quality Ingestion Guardrails:** Uses Great Expectations to ensure text chunks are non-null and within expected length thresholds before vector indexing.
2. **Local Vector Storage & Embeddings:** Embeds document splits locally using HuggingFace (`all-MiniLM-L6-v2`) and persists vectors into ChromaDB.
3. **Relevance Grading Node:** Filters out irrelevant document chunks using local LLM inference via Ollama (`llama3.2`).
4. **Hallucination Detection Guardrail:** Evaluates LLM generations against retrieved facts to ensure outputs are grounded.
5. **Dynamic Self-Correction:** Automatically loops back to rewrite search queries upThe error `fatal: pathspec 'README.md' did not match any files` means Git cannot find a file named `README.md` in the directory you are currently running the command from.

Here is how to resolve it:

### 1. Check the exact filename case and spelling
Git is case-sensitive on many environments. If your file is named `readme.md`, `Readme.md`, or `README.txt`, Git won't find `README.md`.

Run this command to see all files (including hidden ones) in your folder:

```bash
ls -la
