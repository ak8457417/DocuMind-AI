import re
from typing import Any, Dict, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_pipeline import answer_question
from vision_processor import analyze_pdf_page


class DocuMindState(TypedDict, total=False):
    question: str
    vector_store: Any
    pdf_files: Dict[str, bytes]
    route: Literal["text", "table", "vision"]
    retrieval_type: Literal["text", "table"]
    result: Dict[str, Any]
    answer: str
    sources: list


VISION_KEYWORDS = (
    "diagram", "architecture", "chart", "figure", "flowchart", "drawing",
    "visual", "image", "graph", "schematic", "illustration", "what does this show",
    "explain this page", "describe this page",
)
TABLE_KEYWORDS = (
    "table", "tabular", "rows", "columns", "spreadsheet", "breakdown",
    "figures in", "data in the table",
)


def route_question(state: DocuMindState) -> Dict[str, str]:
    """Choose visual analysis, table retrieval, or regular text retrieval."""
    question = state.get("question", "").lower()
    if any(keyword in question for keyword in VISION_KEYWORDS):
        return {"route": "vision", "retrieval_type": "text"}
    retrieval_type = "table" if any(word in question for word in TABLE_KEYWORDS) else "text"
    return {"route": retrieval_type, "retrieval_type": retrieval_type}


def _requested_page(question: str):
    """Convert an explicitly mentioned 1-based page number to a 0-based index."""
    match = re.search(r"\bpage\s+(\d+)\b", question, flags=re.IGNORECASE)
    return int(match.group(1)) - 1 if match else None


def _choose_visual_source(question: str, vector_store, pdf_files: Dict[str, bytes]):
    """Choose the requested page, or use retrieval to find a likely relevant page."""
    requested_page = _requested_page(question)
    if requested_page is not None:
        # If the question names a file, prefer that file; otherwise use the first uploaded PDF.
        for filename, pdf_bytes in pdf_files.items():
            if filename.lower() in question.lower():
                return filename, pdf_bytes, requested_page
        filename = next(iter(pdf_files))
        return filename, pdf_files[filename], requested_page

    # Without a page number, retrieve relevant text and use its source/page as a hint.
    try:
        matches = vector_store.similarity_search(question, k=4)
        for doc in matches:
            metadata = getattr(doc, "metadata", {}) or {}
            filename = metadata.get("source") or metadata.get("filename")
            page = metadata.get("page")
            if filename in pdf_files and page is not None:
                return filename, pdf_files[filename], int(page)
    except Exception:
        pass

    filename = next(iter(pdf_files))
    return filename, pdf_files[filename], 0


def analyze_visual_content(state: DocuMindState) -> Dict[str, Any]:
    """Analyze a relevant original PDF page with Gemini Vision."""
    pdf_files = state.get("pdf_files") or {}
    if not pdf_files:
        result = answer_question(
            question=state["question"],
            vector_store=state["vector_store"],
        )
        result["answer"] = (
            "I couldn't access the original PDF pages for visual analysis, so I answered "
            "using extracted text instead.\n\n" + result.get("answer", "")
        )
    else:
        filename, pdf_bytes, page_number = _choose_visual_source(
            state["question"], state["vector_store"], pdf_files
        )
        result = analyze_pdf_page(
            pdf_bytes=pdf_bytes,
            page_number=page_number,
            question=state["question"],
            filename=filename,
        )

    return {
        "result": result,
        "answer": result.get("answer", "No answer was generated."),
        "sources": result.get("sources", []),
    }


def retrieve_and_answer(state: DocuMindState) -> Dict[str, Any]:
    result = answer_question(
        question=state["question"],
        vector_store=state["vector_store"],
        content_type=state.get("retrieval_type"),
    )
    return {
        "result": result,
        "answer": result.get("answer", "No answer was generated."),
        "sources": result.get("sources", []),
    }


def build_agent():
    graph = StateGraph(DocuMindState)
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("analyze_visual_content", analyze_visual_content)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        lambda state: state["route"],
        {
            "text": "retrieve_and_answer",
            "table": "retrieve_and_answer",
            "vision": "analyze_visual_content",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("analyze_visual_content", END)
    return graph.compile()


_AGENT = None


def answer_with_agent(question: str, vector_store, pdf_files=None) -> Dict[str, Any]:
    """Run the agent. pdf_files should map uploaded PDF filenames to their raw bytes."""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()

    final_state = _AGENT.invoke({
        "question": question,
        "vector_store": vector_store,
        "pdf_files": pdf_files or {},
    })
    return {
        "answer": final_state.get("answer", "No answer was generated."),
        "sources": final_state.get("sources", []),
    }
