
import json
import os
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import operator

# Double check the flow of the code .
class StoryState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    curr_page: int
    book_id: str # This wasnt there before but will be needed
    iteration_count: int
    max_iterations: int
    chapter_context: dict
    retrieved_docs: List[Document]

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# A global cache for vector stores to avoid reloading from disk on every request
vector_store_cache = {}

# Search query seems to be missing

def get_vector_store(book_id: str, data_dir: str) -> Chroma: # Completlely new
    """Loads a persistent Chroma vector store, creating it if it doesn't exist."""
    if book_id in vector_store_cache:
        return vector_store_cache[book_id]

    book_data_dir = os.path.join(data_dir, book_id)
    persist_directory = os.path.join(book_data_dir, "chroma_db")
    pdf_path = os.path.join(book_data_dir, "source.pdf")

    if os.path.exists(persist_directory):
        print(f"Loading existing vector store for book_id: {book_id}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    elif os.path.exists(pdf_path):
        print(f"Creating new vector store for book_id: {book_id}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=persist_directory)
    else:
        raise FileNotFoundError(f"Could not find source PDF or vector store for book_id: {book_id}")

    vector_store_cache[book_id] = vector_store
    return vector_store

def initialize_question(state: StoryState) -> Dict: # This looks to be modified
    """Initializes the state for a new question."""
    question = state["messages"][-1].content
    return {
        "question": question,
        "iteration_count": 0,
        "max_iterations": 3,
        "retrieved_docs": []
    }

def get_current_chapter_context(state: StoryState) -> Dict:
    """Loads chapter context from the book's JSON file."""
    curr_page = state["curr_page"]
    book_id = state["book_id"] # Not there before
    data_dir = "data" # Assuming a 'data' directory at the root
    json_path = os.path.join(data_dir, book_id, "story_global_view.json") # Need to modify this to a database

    try:
        with open(json_path, "r") as f:
            story_data = json.load(f)
        for chapter in story_data.get("chapter", []):
            if chapter["pages"][0] <= curr_page <= chapter["pages"][1]:
                return {"chapter_context": chapter}
        # Fallback to first chapter
        return {"chapter_context": story_data.get("chapter", [{}])[0]}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"chapter_context": {}}

def semantic_search(state: StoryState) -> Dict:
    """Performs spoiler-free semantic search."""
    query = state["question"]
    curr_page = state["curr_page"]
    book_id = state["book_id"]
    data_dir = "data"

    max_allowed_page = state.get("chapter_context", {}).get("pages", [0, 9999])[1]

    try:
        vector_store = get_vector_store(book_id, data_dir)
        unfiltered_results = vector_store.similarity_search(query, k=10)
        
        filtered_results = [doc for doc in unfiltered_results if doc.metadata.get("page", 9999) <= max_allowed_page]
        
        print(f"Search query: '{query}' -> Found {len(unfiltered_results)} results, filtered to {len(filtered_results)} spoiler-free results.")
        return {"retrieved_docs": filtered_results[:4]}
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return {"retrieved_docs": []}

def generate_answer_or_refine(state: StoryState) -> Dict:
    """The main reasoning node. Decides whether to answer or refine the query."""
    if state["iteration_count"] >= state["max_iterations"]:
        return generate_final_answer(state)

    prompt = """You are an expert library assistant specializing in helping readers understand stories without spoilers.

**Current Chapter Information:**
{chapter_context}

**Retrieved Information from Book:**
{search_results}

**Reader's Question:**
{question}

**Your Task:**
Analyze if you have sufficient information to provide a complete, accurate answer. 
1. If you can answer completely, provide a clear, helpful response.
2. If you need more specific information, respond with exactly "NEED_MORE_CONTEXT:" followed by a new, focused search query.

**Rules:**
- Base your answer ONLY on the provided context.
- NEVER use outside knowledge or spoil future events.
"""
    
    search_results_text = "\n".join([f"- [Page {doc.metadata.get('page')}]: {doc.page_content[:250]}..." for doc in state["retrieved_docs"]])
    
    response = llm.invoke([
        SystemMessage(content=prompt.format(
            chapter_context=json.dumps(state["chapter_context"], indent=2),
            search_results=search_results_text or "No relevant passages found.",
            question=state["question"]
        ))
    ])
    
    return {"messages": [AIMessage(content=response.content)], "iteration_count": state["iteration_count"] + 1}

def generate_final_answer(state: StoryState) -> Dict:
    """Generates a final answer when the refinement loop ends."""
    # This node is simplified; it could be a more complex summarization
    last_message = state["messages"][-1].content
    return {"messages": [AIMessage(content=last_message)]}

def extract_refined_query(state: StoryState) -> Dict:
    """Extracts the new query from the agent's response."""
    last_message = state["messages"][-1].content
    new_query = last_message.split("NEED_MORE_CONTEXT:", 1)[1].strip()
    return {"question": new_query}

# --- Graph Construction ---

def decision_router(state: StoryState) -> str:
    """Routes the conversation based on the agent's last message."""
    if state["iteration_count"] >= state["max_iterations"]:
        return "final_answer"
    if "NEED_MORE_CONTEXT:" in state["messages"][-1].content:
        return "refine_search"
    return "final_answer"

def create_rag_agent():
    """Builds and compiles the LangGraph agent."""
    builder = StateGraph(StoryState)
    
    builder.add_node("initialize_question", initialize_question)
    builder.add_node("get_chapter_context", get_current_chapter_context)
    builder.add_node("semantic_search", semantic_search)
    builder.add_node("generate_answer_or_refine", generate_answer_or_refine)
    builder.add_node("extract_refined_query", extract_refined_query)
    builder.add_node("final_answer", generate_final_answer)

    builder.add_edge(START, "initialize_question")
    builder.add_edge("initialize_question", "get_chapter_context")
    builder.add_edge("get_chapter_context", "semantic_search")
    builder.add_edge("semantic_search", "generate_answer_or_refine")
    builder.add_conditional_edges(
        "generate_answer_or_refine",
        decision_router,
        {
            "refine_search": "extract_refined_query",
            "final_answer": "final_answer"
        }
    )
    builder.add_edge("extract_refined_query", "semantic_search")
    builder.add_edge("final_answer", END)
    
    return builder.compile()

# Create a single, reusable agent instance
interview_agent = create_rag_agent()
