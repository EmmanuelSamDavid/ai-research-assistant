"""
rag_pipeline.py
================
This file contains the "brain" of the Research Assistant. It has 5 jobs:

1. EXTRACT text from a PDF                (extract_text_from_pdf)
2. CHUNK that text into small pieces       (chunk_text)
3. EMBED chunks into vectors + STORE them  (add_paper)
4. RETRIEVE the most relevant chunks       (retrieve_chunks)
5. ASK the LLM to answer using those chunks (ask_question)

If you're new to RAG, here's the mental model:
  - We can't paste a whole 20-page paper into every question (too long, too
    expensive). So instead we chop papers into small "chunks" of text.
  - Each chunk gets converted into a list of numbers (a "vector" / "embedding")
    that captures its meaning.
  - When you ask a question, we embed the question too, then find the chunks
    whose vectors are "closest" in meaning to your question.
  - We hand ONLY those relevant chunks to the LLM and say "answer using
    only this text." That's what makes the answers grounded instead of
    hallucinated.
"""

import os
import re
import json
import uuid
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# SETUP: these run once when the app starts
# ---------------------------------------------------------------------------

# The embedding model turns text into vectors. This one is small, free,
# and runs on CPU -- good for getting started. (~80MB download on first run)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ChromaDB is our vector database. PersistentClient means it saves to disk
# in the "chroma_db" folder, so your papers survive between app restarts.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")

# Which local model Ollama should use for answering questions.
# Make sure you've run `ollama pull mistral` (or another model) first --
# see README for setup. You can swap this to "llama3.1", "llama3.1:8b", etc.
OLLAMA_MODEL = "mistral"

# Ollama defaults to a small 2048-token context window, which can truncate
# longer prompts (like a whole paper's chunks for summarization). Mistral
# supports up to 8192 comfortably, so we ask for that explicitly.
OLLAMA_OPTIONS = {"num_ctx": 8192}


# ---------------------------------------------------------------------------
# STEP 1: Extract text from PDF
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Reads a PDF and returns all its text as one big string."""
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # We tag each page so we can later tell the user "this came from page 8"
        full_text += f"\n[PAGE {page_num + 1}]\n{text}"
    doc.close()
    return full_text


# ---------------------------------------------------------------------------
# STEP 2: Chunk the text
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks.

    Why overlap? If a sentence gets cut in half at a chunk boundary, the
    overlap ensures the full idea still appears intact in at least one chunk.

    chunk_size / overlap are in characters (simple to start with -- you can
    upgrade to token-based or sentence-based chunking later, see README).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# STEP 3: Embed + store a paper
# ---------------------------------------------------------------------------
def add_paper(file_path: str, paper_title: str) -> int:
    """
    Full ingestion pipeline for one PDF:
    extract -> chunk -> embed -> store in ChromaDB.
    Returns the number of chunks stored.
    """
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    # Turn all chunks into vectors in one batch (faster than one at a time)
    embeddings = embedder.encode(chunks).tolist()

    # ChromaDB needs a unique ID per chunk, plus optional metadata
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"paper_title": paper_title, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# STEP 4: Retrieve relevant chunks for a question
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Embeds the question and finds the top_k most similar chunks
    across ALL uploaded papers (multi-document retrieval).
    """
    query_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "paper_title": meta["paper_title"]})
    return retrieved


# ---------------------------------------------------------------------------
# STEP 5: Ask the LLM, grounded in retrieved chunks
# ---------------------------------------------------------------------------
def ask_question(question: str, top_k: int = 5) -> dict:
    """
    The full RAG answer flow:
    1. Retrieve relevant chunks
    2. Build a prompt that includes ONLY those chunks
    3. Ask Claude to answer using only that context
    Returns the answer text plus the sources used (for citation display).
    """
    chunks = retrieve_chunks(question, top_k=top_k)

    if not chunks:
        return {"answer": "No papers have been uploaded yet.", "sources": []}

    # Build the context block the LLM will read
    context_block = "\n\n".join(
        f"[Source {i+1} - {c['paper_title']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research assistant. Answer the user's question using ONLY "
        "the information in the provided sources below. If the sources don't "
        "contain enough information to answer, say so clearly instead of "
        "guessing. When you use information from a source, mention which "
        "source number it came from, e.g. '(Source 2)'."
    )

    user_prompt = f"SOURCES:\n{context_block}\n\nQUESTION: {question}"

    # Ollama must be running in the background (the app you installed starts
    # a local server automatically). This call goes to http://localhost:11434
    # -- nothing leaves your machine.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    answer_text = response["message"]["content"]

    return {
        "answer": answer_text,
        "sources": [{"paper_title": c["paper_title"], "excerpt": c["text"][:200]} for c in chunks],
    }


def list_uploaded_papers() -> list[str]:
    """Returns the unique list of paper titles currently in the database."""
    all_items = collection.get()
    titles = {m["paper_title"] for m in all_items["metadatas"]}
    return sorted(titles)


# ---------------------------------------------------------------------------
# STEP 6: Summarize a single paper
# ---------------------------------------------------------------------------
def _get_paper_chunks_in_order(paper_title: str) -> list[str]:
    """
    Fetches every chunk belonging to one paper, ordered the way it appeared
    in the original PDF (using the chunk_index we stored at ingestion time).
    """
    results = collection.get(where={"paper_title": paper_title})
    pairs = list(zip(results["metadatas"], results["documents"]))
    pairs.sort(key=lambda p: p[0]["chunk_index"])
    return [doc for _, doc in pairs]


# Max characters of paper text we'll feed the LLM in one go. This is a
# simple safeguard against exceeding the model's context window -- for long
# papers, this means the summary is based on the first ~15 pages worth of
# chunks rather than the entire document. A more advanced approach (worth
# trying as a research contribution) is "map-reduce" summarization: summarize
# each chunk group separately, then summarize the summaries.
MAX_SUMMARY_INPUT_CHARS = 12000

SUMMARY_STYLES = {
    "Short (100-150 words)": (
        "Write a concise summary of 100-150 words covering what this paper "
        "is about and its main finding."
    ),
    "Detailed (structured)": (
        "Write a structured summary with these headings: Key Contributions, "
        "Methodology, Results, Limitations, Future Work. Use 1-3 sentences "
        "per heading, based only on what's in the excerpts."
    ),
    "Bullet points": (
        "Summarize this paper as 5-8 bullet points covering its main "
        "contributions, method, and findings."
    ),
}


def summarize_paper(paper_title: str, style: str = "Detailed (structured)") -> str:
    """
    Generates a summary of one paper in the library, in the requested style.
    Unlike ask_question, this doesn't use semantic retrieval -- it feeds the
    paper's own chunks (in original order) directly to the LLM, since a
    summary needs the paper's overall structure, not just the bits most
    similar to a search query.
    """
    chunks = _get_paper_chunks_in_order(paper_title)
    if not chunks:
        return f"No content found for '{paper_title}'."

    combined_text = "\n\n".join(chunks)
    if len(combined_text) > MAX_SUMMARY_INPUT_CHARS:
        combined_text = combined_text[:MAX_SUMMARY_INPUT_CHARS] + "\n\n[...paper truncated for length...]"

    instruction = SUMMARY_STYLES.get(style, SUMMARY_STYLES["Detailed (structured)"])

    system_prompt = (
        "You are a research assistant summarizing an academic paper. Base "
        "your summary only on the excerpts provided -- do not invent "
        "details that aren't there."
    )
    user_prompt = f"{instruction}\n\nPAPER EXCERPTS:\n{combined_text}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# STEP 7: Generate a citation for a paper
# ---------------------------------------------------------------------------
CITATION_FORMATS = ["IEEE", "APA", "MLA", "BibTeX"]


def generate_citation(paper_title: str, citation_format: str = "IEEE") -> str:
    """
    Generates a formatted citation for one paper. Title/author/year info is
    almost always on the first page, so we only need the paper's first
    couple of chunks -- no need to read the whole document for this.

    Caveat: PDFs don't always make authors/year easy to parse (e.g. authors
    listed as initials only, or year buried in a footer). The prompt asks
    the model to use a placeholder like [Author] or [Year] instead of
    guessing, so double-check the result against the actual paper before
    using it.
    """
    chunks = _get_paper_chunks_in_order(paper_title)
    if not chunks:
        return f"No content found for '{paper_title}'."

    front_matter = "\n\n".join(chunks[:2])  # title page + a bit more is enough

    system_prompt = (
        "You are a citation formatting assistant. Extract the title, "
        "authors, and publication year from the given excerpt and format "
        "them as a single citation. If any field isn't clearly present in "
        "the text, use a placeholder like [Author] or [Year] instead of "
        "guessing. Output ONLY the citation itself, with no explanation."
    )
    user_prompt = (
        f"Format a {citation_format} citation from this excerpt "
        f"(the source filename was '{paper_title}', which may hint at the "
        f"title if it's not clearly stated in the text):\n\n{front_matter}"
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# STEP 8: Compare multiple papers side by side
# ---------------------------------------------------------------------------
DEFAULT_COMPARISON_CRITERIA = ["Dataset", "Method/Model", "Key Result", "Limitations"]

# Characters of each paper's text sent to the model for comparison. Kept
# smaller than the summarization limit because we're now sending MULTIPLE
# papers in one prompt and need to stay within the context window.
COMPARISON_CHARS_PER_PAPER = 6000


def compare_papers(paper_titles: list[str], criteria: list[str] = None) -> dict:
    """
    Asks the LLM to fill in a comparison table across the given papers and
    criteria. Returns a dict shaped either:
      {"table": {paper_title: {criterion: value, ...}, ...}, "raw": None}
    or, if the model's output couldn't be parsed as JSON:
      {"table": None, "raw": "<the model's raw text response>"}
    The caller should check which one it got (see app.py).
    """
    if criteria is None:
        criteria = DEFAULT_COMPARISON_CRITERIA

    paper_excerpts = {}
    for title in paper_titles:
        chunks = _get_paper_chunks_in_order(title)
        combined = "\n\n".join(chunks)
        if len(combined) > COMPARISON_CHARS_PER_PAPER:
            combined = combined[:COMPARISON_CHARS_PER_PAPER] + "\n\n[...truncated...]"
        paper_excerpts[title] = combined

    excerpts_block = "\n\n".join(
        f"=== PAPER: {title} ===\n{text}" for title, text in paper_excerpts.items()
    )
    criteria_str = ", ".join(criteria)

    system_prompt = (
        "You are a research assistant comparing academic papers. Respond "
        "with ONLY a JSON object -- no explanation, no markdown code "
        "fences, no text before or after the JSON. If a criterion isn't "
        "discussed in a paper's excerpt, use the value \"Not specified\" "
        "instead of guessing."
    )
    user_prompt = (
        f"Compare these papers across these criteria: {criteria_str}.\n\n"
        f"{excerpts_block}\n\n"
        f"Respond with JSON in exactly this shape (one entry per paper):\n"
        f'{{"<paper title>": {{"{criteria[0]}": "...", ...}}, ...}}'
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    raw = response["message"]["content"].strip()
    # Local models sometimes wrap JSON in ```json fences despite instructions
    # not to -- strip those before attempting to parse.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        table = json.loads(cleaned)
        return {"table": table, "criteria": criteria, "raw": None}
    except json.JSONDecodeError:
        # Parsing failed -- hand back the raw text so the UI can still show
        # *something* useful instead of crashing.
        return {"table": None, "criteria": criteria, "raw": raw}


# ---------------------------------------------------------------------------
# STEP 9: Generate a literature review on a topic
# ---------------------------------------------------------------------------
def generate_literature_review(topic: str, top_k: int = 12) -> dict:
    """
    Synthesizes a literature-review paragraph on a topic by retrieving the
    most relevant chunks across ALL uploaded papers (not just one), then
    asking the LLM to weave them into flowing prose with source markers --
    rather than just answering a single question.

    top_k is higher than ask_question's default (12 vs 5) since a review
    should draw on more of the library, not just the single best match.
    """
    chunks = retrieve_chunks(topic, top_k=top_k)

    if not chunks:
        return {"review": "No papers have been uploaded yet.", "sources": []}

    context_block = "\n\n".join(
        f"[Source {i+1} - {c['paper_title']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research assistant writing a literature review section. "
        "Synthesize the provided excerpts into a coherent, flowing review "
        "(not a list) on the given topic. Group related ideas together and "
        "note agreements or contrasts between sources where relevant. Cite "
        "sources inline by number, e.g. '(Source 2)'. Use ONLY information "
        "in the excerpts -- do not add outside knowledge."
    )
    user_prompt = (
        f"TOPIC: {topic}\n\nSOURCES:\n{context_block}\n\n"
        f"Write a literature review of roughly 250-400 words synthesizing "
        f"these sources on the topic above."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    return {
        "review": response["message"]["content"],
        "sources": [{"paper_title": c["paper_title"], "excerpt": c["text"][:200]} for c in chunks],
    }


# ---------------------------------------------------------------------------
# STEP 10: Suggest potential research gaps on a topic
# ---------------------------------------------------------------------------
def identify_research_gaps(topic: str, top_k: int = 12) -> dict:
    """
    Suggests possible research gaps for a topic, based on what the uploaded
    papers say (or don't say) about limitations and future work.

    IMPORTANT framing: this is explicitly a SUGGESTION, not a fact. Spotting
    genuine research gaps requires human judgment and broader awareness of
    the field than a handful of uploaded PDFs can provide. The prompt below
    instructs the model to hedge accordingly, and the UI should keep that
    framing visible too (see app.py) -- don't strip the caveats out.
    """
    # Nudge retrieval toward limitations/future-work language, since that's
    # where gaps are usually discussed (or conspicuously absent).
    chunks = retrieve_chunks(f"{topic} limitations future work open problems", top_k=top_k)

    if not chunks:
        return {"gaps": "No papers have been uploaded yet.", "sources": []}

    context_block = "\n\n".join(
        f"[Source {i+1} - {c['paper_title']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research assistant helping a student brainstorm potential "
        "research directions. Based ONLY on the provided excerpts, suggest "
        "areas that appear underexplored, methods with stated limitations, "
        "or combinations of approaches not yet tried. Frame every point as "
        "a possible suggestion for further investigation, not a definitive "
        "claim -- use hedging language like 'may be worth exploring' or "
        "'these excerpts suggest limited work on...'. Cite sources by "
        "number. If the excerpts don't clearly support any gaps, say so "
        "honestly instead of inventing some."
    )
    user_prompt = (
        f"TOPIC: {topic}\n\nSOURCES:\n{context_block}\n\n"
        f"Suggest 3-5 potential research gaps or promising directions based "
        f"on limitations or open problems mentioned in these excerpts."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OLLAMA_OPTIONS,
    )

    return {
        "gaps": response["message"]["content"],
        "sources": [{"paper_title": c["paper_title"], "excerpt": c["text"][:200]} for c in chunks],
    }
