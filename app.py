"""
app.py
======
The user-facing web app. Run this with:  streamlit run app.py

This is intentionally simple: one page, upload papers on the left (sidebar),
chat with them on the right (main area). Once this works end-to-end, you can
add the fancier features from the project doc (summarization, comparison
tables, research gap detection, etc.) as new functions in rag_pipeline.py
and new UI sections here.
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import (
    add_paper,
    ask_question,
    list_uploaded_papers,
    summarize_paper,
    SUMMARY_STYLES,
    generate_citation,
    CITATION_FORMATS,
    compare_papers,
    DEFAULT_COMPARISON_CRITERIA,
    generate_literature_review,
    identify_research_gaps,
)
import pandas as pd

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant")
st.caption("Upload research papers, then ask questions grounded in their content.")

# --- Sidebar: upload papers -------------------------------------------------
with st.sidebar:
    st.header("Upload Papers")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Add to library"):
            with st.spinner("Reading, chunking, and embedding the paper..."):
                # Save the uploaded file to a temp path so PyMuPDF can open it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                paper_title = uploaded_file.name.replace(".pdf", "")
                num_chunks = add_paper(tmp_path, paper_title)
                os.unlink(tmp_path)

            st.success(f"Added '{paper_title}' ({num_chunks} chunks indexed)")

    st.divider()
    st.subheader("Library")
    papers = list_uploaded_papers()
    if papers:
        for p in papers:
            st.write(f"- {p}")
    else:
        st.write("No papers uploaded yet.")

# --- Main area: all feature tabs ---------------------------------------------
ask_tab, summarize_tab, cite_tab, compare_tab, review_tab, gaps_tab = st.tabs(
    [
        "Ask a Question",
        "Summarize a Paper",
        "Generate Citation",
        "Compare Papers",
        "Literature Review",
        "Research Gaps",
    ]
)

with ask_tab:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Your question", placeholder="e.g. What datasets are used across these papers?")

    if st.button("Ask") and question:
        with st.spinner("Retrieving relevant sections and generating an answer..."):
            result = ask_question(question)
        st.session_state.chat_history.append((question, result))

    # Show conversation, most recent first
    for q, result in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.write(result["answer"])
        with st.expander(f"Sources used ({len(result['sources'])})"):
            for i, s in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1} — {s['paper_title']}**")
                st.caption(s["excerpt"] + "...")
        st.divider()

with summarize_tab:
    papers = list_uploaded_papers()

    if not papers:
        st.write("Upload a paper first to summarize it.")
    else:
        selected_paper = st.selectbox("Choose a paper", papers)
        selected_style = st.selectbox("Summary style", list(SUMMARY_STYLES.keys()))

        if st.button("Summarize"):
            with st.spinner("Reading the paper and writing a summary..."):
                summary = summarize_paper(selected_paper, style=selected_style)
            st.markdown(f"### Summary — {selected_paper}")
            st.write(summary)

with cite_tab:
    papers = list_uploaded_papers()

    if not papers:
        st.write("Upload a paper first to generate a citation.")
    else:
        cite_paper = st.selectbox("Choose a paper", papers, key="cite_paper_select")
        cite_format = st.selectbox("Citation format", CITATION_FORMATS)

        if st.button("Generate Citation"):
            with st.spinner("Reading the paper's title page..."):
                citation = generate_citation(cite_paper, citation_format=cite_format)
            st.markdown(f"### {cite_format} Citation")
            st.code(citation, language=None)
            st.caption("Double-check author names and year against the actual PDF — extraction isn't always perfect.")

with compare_tab:
    papers = list_uploaded_papers()

    if len(papers) < 2:
        st.write("Upload at least 2 papers to compare them.")
    else:
        chosen_papers = st.multiselect("Choose papers to compare (2 or more)", papers)
        criteria_input = st.text_input(
            "Comparison criteria (comma-separated)",
            value=", ".join(DEFAULT_COMPARISON_CRITERIA),
        )

        if st.button("Compare"):
            if len(chosen_papers) < 2:
                st.warning("Pick at least 2 papers first.")
            else:
                criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]
                with st.spinner("Reading each paper and building the comparison..."):
                    result = compare_papers(chosen_papers, criteria=criteria)

                if result["table"] is not None:
                    # result["table"] looks like {paper_title: {criterion: value}}
                    # Flip it into a DataFrame with criteria as rows, papers as columns
                    df = pd.DataFrame(result["table"])
                    st.dataframe(df)
                else:
                    st.warning(
                        "Couldn't parse a structured table from the model's response "
                        "this time — showing the raw answer instead. Try again, or "
                        "reduce the number of criteria."
                    )
                    st.write(result["raw"])

with review_tab:
    st.caption(
        "Draws on the most relevant chunks across your ENTIRE library (not just "
        "one paper) and asks the model to synthesize them into a review paragraph, "
        "rather than just answering a single question."
    )
    review_topic = st.text_input(
        "Topic for the review",
        placeholder="e.g. deep learning approaches to lung disease classification",
    )

    if st.button("Generate Review") and review_topic:
        with st.spinner("Retrieving across your library and writing the review..."):
            result = generate_literature_review(review_topic)
        st.markdown(f"### Literature Review — {review_topic}")
        st.write(result["review"])
        with st.expander(f"Sources used ({len(result['sources'])})"):
            for i, s in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1} — {s['paper_title']}**")
                st.caption(s["excerpt"] + "...")

with gaps_tab:
    st.info(
        "⚠️ These are AI-generated **suggestions**, not verified facts. Spotting a "
        "genuine research gap requires human judgment and knowledge of the wider "
        "field beyond what's in your uploaded papers — treat this as a brainstorming "
        "starting point, not a conclusion.",
        icon="⚠️",
    )
    gap_topic = st.text_input(
        "Topic to explore for gaps",
        placeholder="e.g. explainable AI for lung disease classification",
    )

    if st.button("Suggest Research Gaps") and gap_topic:
        with st.spinner("Reading limitations and future-work sections across your library..."):
            result = identify_research_gaps(gap_topic)
        st.markdown(f"### Possible Research Gaps — {gap_topic}")
        st.write(result["gaps"])
        with st.expander(f"Sources used ({len(result['sources'])})"):
            for i, s in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1} — {s['paper_title']}**")
                st.caption(s["excerpt"] + "...")
