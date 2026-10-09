import os
import streamlit as st
from dotenv import load_dotenv

from document_processor import process_pdf
from vector_store import build_vector_store
from agent import answer_with_agent

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Visual polish: spacing, hierarchy, readable chat width ----
st.markdown(
    """
    <style>
    :root {
        --dm-bg: #0b1020;
        --dm-panel: #111a2e;
        --dm-panel-2: #172239;
        --dm-border: #2a3650;
        --dm-text: #edf2ff;
        --dm-muted: #a7b3cc;
        --dm-accent: #8b9cff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 70% 0%, rgba(81, 102, 190, .10), transparent 38%),
            var(--dm-bg);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .82);
    }

    [data-testid="stSidebar"] {
        background: #0e1628;
        border-right: 1px solid var(--dm-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.025);
        border: 1px solid var(--dm-border);
        border-radius: 12px;
        padding: .65rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dm-hero {
        padding: 1.25rem 1.4rem;
        margin: 0 0 1.4rem 0;
        border: 1px solid var(--dm-border);
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(26, 39, 67, .95), rgba(17, 26, 46, .82));
    }

    .dm-eyebrow {
        color: #aebaff;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .4rem;
    }

    .dm-title {
        color: var(--dm-text);
        font-size: clamp(1.65rem, 3vw, 2.25rem);
        line-height: 1.15;
        font-weight: 750;
        margin: 0;
    }

    .dm-subtitle {
        color: var(--dm-muted);
        font-size: .96rem;
        margin-top: .55rem;
        max-width: 720px;
        line-height: 1.55;
    }

    .dm-section-label {
        color: var(--dm-muted);
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: .35rem 0 .65rem;
    }

    .dm-empty {
        padding: 2rem 1.5rem;
        border: 1px dashed #394763;
        border-radius: 16px;
        background: rgba(255,255,255,.018);
        text-align: center;
        margin: 1rem 0;
    }

    .dm-empty-icon {
        font-size: 2rem;
        margin-bottom: .55rem;
    }

    .dm-empty-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--dm-text);
    }

    .dm-empty-copy {
        color: var(--dm-muted);
        font-size: .9rem;
        margin-top: .35rem;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(124, 145, 190, .14);
        border-radius: 16px;
        padding: .95rem 1.05rem;
        margin-bottom: .85rem;
        background: rgba(20, 30, 51, .72);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(37, 49, 78, .72);
    }

    [data-testid="stChatInput"] {
        border-color: #354465;
        border-radius: 16px;
        background: #111a2e;
    }

    [data-testid="stChatInput"] textarea {
        line-height: 1.5;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--dm-border);
        border-radius: 12px;
        background: rgba(255,255,255,.018);
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 650;
    }

    .dm-small-note {
        color: var(--dm-muted);
        font-size: .78rem;
        line-height: 1.5;
    }

    .dm-brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        margin: .1rem 0 .45rem;
    }
    .dm-brand-icon {
        display: grid;
        place-items: center;
        width: 2.55rem;
        height: 2.55rem;
        border-radius: 12px;
        color: #fff;
        font-size: 1.35rem;
        background: linear-gradient(135deg, #687cff, #9c78ff);
        box-shadow: 0 8px 22px rgba(104,124,255,.22);
    }
    .dm-brand-title {
        color: var(--dm-text);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -.035em;
        line-height: 1.2;
    }
    .dm-brand-title span { color: #a6b0ff; }
    .dm-brand-subtitle {
        color: var(--dm-muted);
        font-size: .58rem;
        letter-spacing: .14em;
        margin-top: .22rem;
        font-weight: 700;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 9px;
    }
    div.stButton > button[kind="primary"] {
        border: 0;
        color: white;
        background: linear-gradient(100deg, #6578ff, #8b75f5);
        box-shadow: 0 8px 20px rgba(101,120,255,.18);
    }
    div.stButton > button[kind="primary"]:hover {
        border: 0;
        filter: brightness(1.08);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.025);
        border: 1px solid var(--dm-border);
        padding: .8rem 1rem;
        border-radius: 13px;
    }
    [data-testid="stMetricLabel"] { color: var(--dm-muted); }
    .dm-hero {
        position: relative;
        overflow: hidden;
        background: radial-gradient(ellipse at 92% 10%, rgba(119,132,255,.18), transparent 35%),
                    linear-gradient(120deg, rgba(26,39,67,.98), rgba(17,26,46,.88));
        box-shadow: 0 14px 40px rgba(0,0,0,.12);
    }
    .dm-hero:after {
        content: "✦";
        position: absolute;
        right: 2rem;
        top: .4rem;
        color: rgba(166,176,255,.18);
        font-size: 5rem;
        pointer-events: none;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #8492ff;
        box-shadow: 0 0 0 1px rgba(132,146,255,.22);
    }

    @media (max-width: 800px) {
        .block-container {
            padding: 1rem .8rem 1.5rem;
        }
        .dm-hero {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_names" not in st.session_state:
    st.session_state.processed_names = []
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="dm-brand"><span class="dm-brand-icon">✦</span><div><div class="dm-brand-title">DocuMind <span>AI</span></div><div class="dm-brand-subtitle">DOCUMENT INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.caption("Your personal document research workspace")
    st.divider()

    st.markdown("#### 1. Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF documents to build a searchable workspace.",
        label_visibility="visible",
    )

    process_clicked = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<p class="dm-small-note">PDF text is extracted locally. Relevant text is sent to Gemini when you ask a question.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.processed_names:
        st.divider()
        st.markdown("#### Indexed documents")
        for name in st.session_state.processed_names:
            st.markdown(f"📎 `{name}`")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ---- Main header ----
st.markdown(
    """
    <div class="dm-hero">
      <div class="dm-eyebrow">Multimodal document intelligence</div>
      <div class="dm-title">Ask your documents anything.</div>
      <div class="dm-subtitle">
        Search PDFs, understand complex reports, and get answers grounded in document evidence.
        Every response can be checked against its source pages.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Document processing ----
if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF in the sidebar first.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY. Add it to your .env file and restart Streamlit.")
    else:
        all_chunks = []
        failed = []
        original_pdfs = {}
        progress = st.progress(0, text="Preparing documents...")
        for i, uploaded in enumerate(uploaded_files):
            try:
                pdf_bytes = uploaded.getvalue()
                extracted = process_pdf(
                        pdf_bytes=pdf_bytes,
                        filename=uploaded.name,
                        use_ocr=False,
                    )
                all_chunks.extend(extracted)
                original_pdfs[uploaded.name] = pdf_bytes
            except Exception as exc:
                failed.append(f"{uploaded.name}: {exc}")
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processing {i + 1} of {len(uploaded_files)} documents",
            )
        progress.empty()

        if all_chunks:
            try:
                with st.spinner("Creating searchable document index..."):
                    st.session_state.vector_store = build_vector_store(all_chunks)
                st.session_state.processed_names = list(original_pdfs.keys())
                st.session_state.pdf_files = original_pdfs
                st.session_state.chat_history = []
                st.success(f"Ready to chat — indexed {len(all_chunks)} pages/chunks from your documents.")
            except Exception as exc:
                st.error(f"Could not build the vector index: {exc}")
        else:
            st.error("No extractable text was found. This version works best with text-based PDFs; scanned-page OCR is not enabled yet.")
        for message in failed:
            st.warning(message)

# ---- Chat content ----
chat_col, info_col = st.columns([2.25, 1], gap="large")

with chat_col:
    st.markdown('<div class="dm-section-label">Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        if st.session_state.vector_store is None:
            st.markdown(
                """
                <div class="dm-empty">
                  <div class="dm-empty-icon">🗂️</div>
                  <div class="dm-empty-title">Your workspace is ready</div>
                  <div class="dm-empty-copy">
                    Upload one or more PDFs from the sidebar, then select <b>Process documents</b>
                    to start asking questions.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="dm-empty">
                  <div class="dm-empty-icon">✨</div>
                  <div class="dm-empty-title">Documents indexed successfully</div>
                  <div class="dm-empty-copy">
                    Ask a question below. Try a summary, a specific fact, or a comparison across documents.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Render all history in order. The chat input stays below the conversation.
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander(f"Sources ({len(item['sources'])})"):
                    for source in item["sources"]:
                        st.markdown(
                            f"**{source['filename']} · Page {source['page']}**"
                        )
                        st.caption(source["snippet"])
                        st.divider()

with info_col:
    st.markdown('<div class="dm-section-label">Workspace</div>', unsafe_allow_html=True)
    if st.session_state.vector_store is None:
        st.markdown("**No index yet**")
        st.caption("Your uploaded documents will appear here after processing.")
    else:
        st.markdown("**Index ready**")
        st.caption(f"{len(st.session_state.processed_names)} document(s) available for questions.")
        st.success("Semantic search is active.")
    st.markdown("#### Suggested questions")
    examples = [
        "Summarize the main points.",
        "What skills or requirements are mentioned?",
        "List important dates and figures.",
        "What information is missing or unclear?",
    ]
    for example in examples:
        if st.button(example, key=f"suggest_{example}", use_container_width=True,
                     disabled=st.session_state.vector_store is None):
            st.session_state.pending_question = example
            st.rerun()
    st.markdown(
        '<p class="dm-small-note">Answers can be imperfect. Open the source panel to verify important details.</p>',
        unsafe_allow_html=True,
    )

# Chat input is deliberately at the bottom so it never appears between old and new messages.
typed_question = st.chat_input(
    "Ask a question about your indexed PDFs...",
    disabled=st.session_state.vector_store is None,
)
question = typed_question or st.session_state.pop("pending_question", None)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching evidence and generating an answer..."):
            result = answer_with_agent(
                question,
                st.session_state.vector_store,
                pdf_files=st.session_state.pdf_files,
            )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.get("answer", "No answer was generated."),
                "sources": result.get("sources", []),
            }
        )
    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"I couldn't answer that question because an error occurred: {exc}",
                "sources": [],
            }
        )
    st.rerun()
