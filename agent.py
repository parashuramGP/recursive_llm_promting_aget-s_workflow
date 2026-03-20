"""
LangGraph PDF Iterative Improvement Agent
------------------------------------------
1. Asks the user for a PDF file path
2. Reads and extracts the PDF text
3. Passes it through 5 LangGraph nodes
4. Each node sends the previous response back to Gemini asking to improve it
5. Prints the final polished output
"""

import sys
from pathlib import Path
from typing import TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph

GEMINI_API_KEY = "AIzaSyA2uEUB7zV0cZT_ec9xy8K1tur2rPc_mjY"

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def read_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


class AgentState(TypedDict):
    pdf_path: str
    pdf_content: str
    current_response: str
    iteration: int


def node_1_initial_analysis(state: AgentState) -> AgentState:
    """Node 1 - Initial analysis of the PDF content."""
    print("\nNode 1: Initial Analysis ...")
    prompt = (
        "You are an expert document analyst. "
        "Read the following document and provide a clear, structured summary. "
        "Highlight the key points, main themes, and important details.\n\n"
        f"DOCUMENT:\n{state['pdf_content']}"
    )
    response = call_gemini(prompt)
    print("   Done")
    return {"current_response": response, "iteration": 1}


def node_2_improve(state: AgentState) -> AgentState:
    """Node 2 - First improvement: improve clarity and structure."""
    print("Node 2: Improving Clarity and Structure ...")
    prompt = (
        "Below is an analysis of a document. "
        "Please make it better - improve the structure, make the language clearer, "
        "and ensure all key insights are well-organised. "
        "Return only the improved version.\n\n"
        f"{state['current_response']}"
    )
    response = call_gemini(prompt)
    print("   Done")
    return {"current_response": response, "iteration": 2}


def node_3_improve(state: AgentState) -> AgentState:
    """Node 3 - Second improvement: deepen insights and add context."""
    print("Node 3: Deepening Insights and Adding Context ...")
    prompt = (
        "Here is an analysis. Take it further - add more depth, expand on important "
        "points, provide context where needed, and make the content more informative "
        "and comprehensive. Return only the improved version.\n\n"
        f"{state['current_response']}"
    )
    response = call_gemini(prompt)
    print("   Done")
    return {"current_response": response, "iteration": 3}


def node_4_improve(state: AgentState) -> AgentState:
    """Node 4 - Third improvement: professional polish and flow."""
    print("Node 4: Professional Polish and Flow ...")
    prompt = (
        "Review the following analysis critically. Refine the language to be "
        "professional and polished, improve the logical flow between sections, "
        "strengthen the arguments, and ensure it reads like a high-quality report. "
        "Return only the improved version.\n\n"
        f"{state['current_response']}"
    )
    response = call_gemini(prompt)
    print("   Done")
    return {"current_response": response, "iteration": 4}


def node_5_final_polish(state: AgentState) -> AgentState:
    """Node 5 - Final pass: make it the absolute best version."""
    print("Node 5: Final Polish (Best Version) ...")
    prompt = (
        "This is the final improvement pass. Take the following analysis and perfect it. "
        "It should be the definitive, highest-quality version - excellent flow, "
        "crystal-clear insights, comprehensive coverage, and impeccable language. "
        "Return only the final version.\n\n"
        f"{state['current_response']}"
    )
    response = call_gemini(prompt)
    print("   Done")
    return {"current_response": response, "iteration": 5}


def build_agent():
    """Construct and compile the 5-node LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("node_1_initial_analysis", node_1_initial_analysis)
    graph.add_node("node_2_improve", node_2_improve)
    graph.add_node("node_3_improve", node_3_improve)
    graph.add_node("node_4_improve", node_4_improve)
    graph.add_node("node_5_final_polish", node_5_final_polish)

    graph.add_edge(START, "node_1_initial_analysis")
    graph.add_edge("node_1_initial_analysis", "node_2_improve")
    graph.add_edge("node_2_improve", "node_3_improve")
    graph.add_edge("node_3_improve", "node_4_improve")
    graph.add_edge("node_4_improve", "node_5_final_polish")
    graph.add_edge("node_5_final_polish", END)

    return graph.compile()


def main():
    print("=" * 60)
    print("   LangGraph PDF Iterative Improvement Agent (Gemini)")
    print("=" * 60)

    pdf_path = input("\nEnter the path to your PDF file: ").strip().strip('"').strip("'")

    if not pdf_path:
        print("No path provided. Exiting.")
        sys.exit(1)

    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    if Path(pdf_path).is_dir():
        print(f"That is a folder, not a PDF file: {pdf_path}")
        print("Please provide the full path to a .pdf file.")
        print("Example: C:\\Users\\YourName\\Documents\\myfile.pdf")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"The file does not have a .pdf extension: {pdf_path}")
        sys.exit(1)

    print(f"\nReading PDF: {pdf_path}")

    pdf_content = read_pdf(pdf_path)

    if not pdf_content.strip():
        print("PDF is empty or could not be read.")
        sys.exit(1)

    print(f"Extracted {len(pdf_content):,} characters from the PDF")
    print("\nStarting the 5-node improvement pipeline...\n")

    initial_state: AgentState = {
        "pdf_path": pdf_path,
        "pdf_content": pdf_content,
        "current_response": "",
        "iteration": 0,
    }

    agent = build_agent()
    final_state = agent.invoke(initial_state)

    print("\n" + "=" * 60)
    print("   FINAL OUTPUT (after 5 improvement passes)")
    print("=" * 60)
    print(final_state["current_response"])
    print("\n" + "=" * 60)
    print(f"   Total nodes completed: {final_state['iteration']}/5")
    print("=" * 60)


if __name__ == "__main__":
    main()
