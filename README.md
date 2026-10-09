# DocuMind AI — Multimodal Document Intelligence & RAG Agent

A beginner-friendly PDF question-answering application built with Streamlit, PyMuPDF, LangChain, ChromaDB, Google Gemini, and LangGraph.

## Features

- Upload one or multiple PDFs.
- Extract text while preserving source filenames and page numbers.
- Optional OCR fallback for scanned pages.
- Chunk documents and index them in ChromaDB using Google embeddings.
- Ask questions with retrieval-augmented generation (RAG).
- Display source excerpts and ask Gemini to cite pages.
- Optional LangGraph workflow toggle.
- Clear separation of UI, document processing, chunking, retrieval, and agent logic.

## Architecture

1. `app.py`: Streamlit user interface.
2. `document_processor.py`: extracts page text and optionally OCRs pages with little text.
3. `chunker.py`: splits page documents into overlapping chunks.
4. `vector_store.py`: creates embeddings and a Chroma vector store.
5. `rag_pipeline.py`: retrieves evidence and generates a grounded answer.
6. `agent.py`: wraps the current RAG operation in a LangGraph workflow.

## Requirements

- Python 3.10–3.12 recommended. Confirm dependency support before using a newer Python release.
- Google AI Studio API key.
- Tesseract OCR is optional but recommended for scanned PDFs.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `replace_with_your_api_key` with your API key.

Run the app:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

### Tesseract OCR (optional)

Install the Tesseract OCR application for Windows, then ensure `tesseract.exe` is on your PATH. If it is installed outside PATH, configure `pytesseract.pytesseract.tesseract_cmd` in `document_processor.py`.

If OCR is unavailable, digital PDFs still work. Scanned PDFs may return little or no text.

## Usage

1. Start the app.
2. Upload PDF documents in the sidebar.
3. Choose whether to enable OCR.
4. Click **Process documents** and wait for indexing.
5. Ask questions in the chat box.
6. Expand **View sources** to inspect the retrieved evidence.
7. Toggle **Use LangGraph agent workflow** to run the current workflow through LangGraph.

## Important implementation notes

- The initial LangGraph graph is intentionally small: it wraps the same RAG function in one graph node. This makes the code easy to learn. A later version can add separate routing, table extraction, visual-page inspection, and citation-verification nodes.
- This starter supports text retrieval and OCR fallback. It does not yet implement Gemini Vision diagram analysis or dedicated table extraction. Those are planned extension points, not features to claim as complete.
- Source snippets and page metadata are returned from retrieved chunks. Model-generated inline citations should be checked against the source panel; the starter does not guarantee every citation claim is entailed by its cited page.
- Chroma is currently created in memory for the active app session. Restarting the app or processing new documents may require re-indexing.
- Do not commit `.env` or API keys to GitHub.
- Gemini model availability, quotas, API names, and package compatibility can change. If a model is unavailable for your API key, update the model name in `.env` to one currently enabled in Google AI Studio.

## Troubleshooting

### Missing `GOOGLE_API_KEY`
Check that `.env` exists in the project root and contains a valid key. Restart Streamlit after editing it.

### API/model error
Verify your key, quota, network, and model identifiers. The embedding model and chat model are configured separately.

### OCR does not detect scanned text
Install the Tesseract application (not just the Python `pytesseract` package), check that it is on PATH, and ensure the PDF page is readable.

### Dependency installation fails
Use a supported Python version such as Python 3.11, create a fresh virtual environment, and install requirements again.

## Suggested next improvements

1. Add a dedicated table extraction tool.
2. Add a page-image inspection tool using Gemini vision for diagrams and drawings.
3. Add a citation-verification node in LangGraph.
4. Add a small evaluation set with expected answers and source pages.
5. Add persistent Chroma storage and per-user document isolation.
6. Add tests for extraction, chunking, retrieval, and unsupported questions.
