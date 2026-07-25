A local AI research assistant using Retrieval-Augmented Generation (RAG) — upload PDFs and ask questions grounded in their content, with summarization, citation generation, paper comparison, and literature review, powered by a fully local LLM (Ollama + Mistral).

> Upload research papers as PDFs and ask questions grounded in their actual
> content — no hallucinated answers, no cloud API required. Runs entirely
> locally via Ollama. Also generates summaries, citations, paper comparisons,
> and literature reviews from the same retrieval pipeline.

# AI Research Assistant (RAG) — MVP

A minimal but real, working version of the project: upload PDFs, ask questions,
get answers grounded in the papers with sources shown.

## What's in here

```
research-assistant/
├── app.py            # Streamlit UI — run this
├── rag_pipeline.py   # The actual RAG logic (heavily commented, read this to learn how RAG works)
├── requirements.txt  # Python dependencies
└── README.md         # You are here
```

## Setup (step by step)

1. **Install Python 3.10+** if you don't have it (check with `python3 --version`).

2. **Create a virtual environment** (keeps this project's packages separate
   from everything else on your machine):
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (runs the LLM locally, no API key, no internet needed
   at question time): download from https://ollama.com/download and run
   the installer. It installs a background service that starts
   automatically.

5. **Pull a model** (one-time download, run in any terminal — this doesn't
   need your venv activated):
   ```bash
   ollama pull mistral
   ```
   This downloads ~4GB.

   If you use a different model name, update `OLLAMA_MODEL` at the top of
   `rag_pipeline.py` to match.

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Your browser should open automatically at `http://localhost:8501`.

7. **Try it:** Upload a PDF of a research paper in the sidebar, click
   "Add to library," then ask a question about it in the main panel. The
   first answer may be slow (~10-30s) while Ollama loads the model into
   memory; after that it's faster.

## How it works (the short version)

Read `rag_pipeline.py` top to bottom — every function has a comment block
explaining what it does and why. The short version:

`PDF → extract text → split into chunks → convert chunks to vectors → store
in ChromaDB → (later) embed your question → find closest chunks → give those
chunks to Claude → Claude answers using only that text`

This "only that text" instruction is what makes the answers grounded instead
of hallucinated — the model isn't allowed to use outside knowledge.

## Known limitations of this MVP (on purpose — fix these as you go)

- **Chunking is character-based**, not sentence/paragraph aware. It can cut
  sentences mid-word. A good first improvement: chunk by paragraph or use a
  library like `langchain`'s `RecursiveCharacterTextSplitter`.
- **No reranking** — we just take the top-k nearest chunks by embedding
  similarity. Adding a reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  usually improves answer quality noticeably — this maps directly to
  "Research Contribution #3" in your project doc.
- **Single collection for all papers** — good for cross-paper questions, but
  you may want per-paper filtering later (ChromaDB supports `where=` filters
  on metadata — you already store `paper_title` in metadata for this).
- **No authentication / single user** — fine for a demo or personal use;
  add FastAPI + JWT only if you need multiple users with private libraries.

## Roadmap — order to add the remaining project features

This maps the original feature list to concrete build steps, roughly easiest
→ hardest:

1. **Paper summarization** ✅ *(done — see the "Summarize a Paper" tab)* —
   `summarize_paper()` in `rag_pipeline.py` pulls all chunks for one paper in
   original order and asks the LLM for a summary in one of three styles.
   Note: long papers get truncated to the first ~12,000 characters to stay
   within the model's context window (`MAX_SUMMARY_INPUT_CHARS`) — a good
   next upgrade is "map-reduce" summarization (summarize chunks in groups,
   then summarize the summaries) so nothing gets cut off.
2. **Citation generator** ✅ *(done — see the "Generate Citation" tab)* —
   `generate_citation()` reads a paper's first couple of chunks (title page
   info) and asks the LLM to format it as IEEE/APA/MLA/BibTeX. Since PDF
   text extraction doesn't always cleanly separate title/authors/year, the
   model is instructed to use placeholders like `[Author]` instead of
   guessing — always double-check the result against the actual paper.
3. **Paper comparison table** ✅ *(done — see the "Compare Papers" tab)* —
   `compare_papers()` asks the LLM to return JSON across your chosen
   criteria for 2+ papers, which gets rendered as a table. Local models
   occasionally don't return valid JSON — the UI falls back to showing the
   raw text response rather than crashing when that happens. If you see
   the fallback often, try comparing fewer papers or fewer criteria at once
   (less for the model to juggle in one response).
4. **Semantic search UI** — *(added, then removed)* — this was tried as a
   thin wrapper around `retrieve_chunks()`, but raw top-k chunks without an
   LLM to filter/explain them read as noisy and often felt irrelevant to
   the query. `retrieve_chunks()` is still in `rag_pipeline.py` and still
   powers "Ask a Question" — it's just not exposed as its own UI tab. If
   you revisit this later, pairing it with a reranker (see Limitations
   above) would likely fix the relevance issue.
5. **Literature review generator** ✅ *(done — see the "Literature Review"
   tab)* — `generate_literature_review()` retrieves the top matches for a
   topic across your WHOLE library (top_k=12, vs. 5 for regular Q&A) and
   asks the model to synthesize them into flowing prose with inline source
   markers, rather than answering one narrow question.
6. **Research gap identification** ✅ *(done — see the "Research Gaps"
   tab)* — `identify_research_gaps()` retrieves chunks weighted toward
   limitations/future-work language and asks the model to suggest gaps,
   with the prompt explicitly instructing hedged language ("may be worth
   exploring") instead of confident claims. The UI also shows a persistent
   warning banner reinforcing that these are AI-generated starting points
   for brainstorming, not verified conclusions — keep that framing if you
   extend this feature further.
7. **Explainable retrieval (page numbers)** — you're already tagging pages
   during extraction (`[PAGE N]`) — parse that tag back out per chunk and
   show it next to each source in the UI.
8. **Knowledge graph** — bigger lift. Extract entities (authors, methods,
   datasets) per paper with an LLM call, store as nodes/edges (e.g. in
   `networkx`), visualize with `pyvis` or `streamlit-agraph`.
9. **Research trend analysis** — once you have structured metadata (dataset
   names, methods, publication years) from step 8, this is just aggregation
   + charts (`st.bar_chart`, `st.line_chart`).

For your "Possible Research Contributions" section, steps 2 (chunking
strategy) and 3 (reranking) from the limitations list above are the easiest
to turn into a measurable experiment — you can swap one component, keep
everything else fixed, and compare answer accuracy on a fixed set of test
questions.
