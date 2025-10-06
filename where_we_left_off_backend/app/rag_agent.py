# rag_agent.py

import os
import json
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

from motor.motor_asyncio import AsyncIOMotorClient

LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # Use text-embedding-3-large if you want to see better results but as far as I can see, small is good enough. Should test with larger books though
DEFAULT_MAX_ITERATIONS = 3
SEARCH_K = 10
RESULTS_LIMIT = 4

llm = ChatOpenAI(model=LLM_MODEL)
vector_store_cache = {}
# mongo_client = None
# db = None


def deduplicate_docs(existing: List[Document], new: List[Document]) -> List[Document]:
    if not existing:
        return new

    seen = set()
    for doc in existing:
        page = doc.metadata.get("page", -1)
        # Use first 200 chars for hash to identify unique content
        content_hash = hash(doc.page_content[:200]) if doc.page_content else 0
        seen.add((page, content_hash))

    unique_new = []
    for doc in new:
        page = doc.metadata.get("page", -1)
        content_hash = hash(doc.page_content[:200]) if doc.page_content else 0
        if (page, content_hash) not in seen:
            unique_new.append(doc)
            seen.add((page, content_hash))

    return existing + unique_new


class StoryState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    curr_page: int
    book_id: str
    iteration_count: int
    max_iterations: int
    chapter_context: dict
    retrieved_docs: Annotated[List[Document], deduplicate_docs]
    db: Any


# def init_mongo(connection_string: str = None):  # Didnt I already inisitalize a DB
#     global mongo_client, db
#     if connection_string is None:
#         connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
#     mongo_client = AsyncIOMotorClient(connection_string)
#     db = mongo_client.story_processor


async def get_chapter_from_mongo(db , book_id: str, curr_page: int) -> dict:
    """Fetch chapter context from MongoDB based on current page"""

    try:
        # Query the global chapters collection
        cursor = db.chapters_global.find({"book_id": book_id})
        chapters = await cursor.to_list(length=None)

        # Find the chapter containing curr_page
        for chapter in chapters:
            start_page, end_page = chapter.get("pages", [-1, -1])
            if start_page <= curr_page <= end_page:
                return {
                    "chapter_id": chapter.get("chapter_id"),
                    "title": chapter.get("title", "Unknown"),
                    "summary_global": chapter.get("summary_global", ""),
                    "summary_local": chapter.get("summary_local", ""),
                    "characters": chapter.get("characters", []),
                    "pages": chapter.get("pages", [start_page, end_page]),
                }

        # Fallback: return first chapter if no match
        if chapters:
            first = chapters[0]
            return {
                "chapter_id": first.get("chapter_id"),
                "title": first.get("title", "Unknown"),
                "summary_global": first.get("summary_global", ""),
                "summary_local": first.get("summary_local", ""),
                "characters": first.get("characters", []),
                "pages": first.get("pages", [0, 0]),
            }
    except Exception as e:
        print(f"Error fetching chapter from MongoDB: {e}")

    return {}

async def create_vector_store(book_id: str , pdf_path: str , persist_directory: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
    splits = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), # Use text-embedding-3-large if you want to see better results but as far as I can see, small is good enough. Should test with larger books though
        persist_directory=persist_directory,
    )
    vector_store_cache[book_id] = vector_store



# THis function should be in main me thinks
# def get_vector_store(book_id: str, data_dir: str = "data") -> Chroma:
#     """Loads or creates a persistent Chroma vector store"""
#     if book_id in vector_store_cache:
#         return vector_store_cache[book_id]
#     # Example path: data/<book_id>/chroma_db
#     book_data_dir = os.path.join(data_dir, "processed" , book_id)
#     persist_directory = os.path.join(book_data_dir, "processed", "chroma_db")
#     pdf_path = os.path.join(
#         book_data_dir, "source.pdf"
#     )  # TODO: We have to input the path as well

#     if os.path.exists(persist_directory):
#         print(f"Loading existing vector store for book_id: {book_id}")
#         vector_store = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
#         )
#     elif os.path.exists(pdf_path):
#         print(f"Creating new vector store for book_id: {book_id}")
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, add_start_index=True
#         )
#         splits = text_splitter.split_documents(docs)
#         vector_store = Chroma.from_documents(
#             documents=splits,
#             embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
#             persist_directory=persist_directory,
#         )
#     else:
#         raise FileNotFoundError(
#             f"Could not find source PDF or vector store for book_id: {book_id}"
#         )

#     vector_store_cache[book_id] = vector_store
#     return vector_store


def get_vector_store(book_id: str, data_dir: str = "data") -> Chroma:
    if book_id in vector_store_cache:
        return vector_store_cache[book_id]
    
    book_data_dir = os.path.join(data_dir, "processed", book_id)
    persist_directory = os.path.join(book_data_dir, "chroma_db")  # Removed extra "processed"
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found for book_id: {book_id}. "
            f"The background processing may still be running. Check /books/status/{book_id}"
        )
    
    print(f"Loading vector store for book_id: {book_id}")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    
    vector_store_cache[book_id] = vector_store
    return vector_store



def initialize_question(state: StoryState) -> Dict:
    if state.get("messages"):
        question = state["messages"][-1].content
        return {
            "question": question,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            "retrieved_docs": [],
        }
    return {}


async def get_current_chapter_context(state: StoryState) -> Dict:
    """Fetch chapter context from MongoDB"""
    curr_page = state.get("curr_page", 0)
    book_id = state.get("book_id", "")
    db = state.get("db") 

    if not book_id:
        print("Warning: No book_id provided")
        return {"chapter_context": {}}

    chapter_context = await get_chapter_from_mongo(db , book_id, curr_page)
    return {"chapter_context": chapter_context}


def semantic_search(state: StoryState) -> Dict:
    """Perform spoiler-free semantic search"""
    query = state.get("question", "")
    book_id = state.get("book_id", "")
    chapter_context = state.get("chapter_context", {})

    if not query or not book_id:
        return {"retrieved_docs": []}

    # Get spoiler boundary from chapter context
    max_allowed_page = chapter_context.get("pages", [0, 9999])[1]

    try:
        vector_store = get_vector_store(book_id)
        unfiltered_results = vector_store.similarity_search(query, k=SEARCH_K)

        # Filter out spoilers
        filtered_results = [
            doc
            for doc in unfiltered_results
            if doc.metadata.get("page", 9999) <= max_allowed_page
        ]

        final_results = filtered_results[:RESULTS_LIMIT]

        print(
            f"Search: '{query}' -> {len(unfiltered_results)} results, "
            f"{len(filtered_results)} spoiler-free, returning {len(final_results)}"
        )

        return {"retrieved_docs": final_results}

    except Exception as e:
        print(f"Error during semantic search: {e}")
        return {"retrieved_docs": []}


def generate_answer_or_refine(state: StoryState) -> Dict:
    """Main reasoning node - decides to answer or refine query"""
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # Force final answer if max iterations reached
    if iteration_count >= max_iterations:
        print(f"Max iterations ({max_iterations}) reached")
        return _generate_forced_answer(state)

    chapter_context = state.get("chapter_context", {})
    retrieved_docs = state.get("retrieved_docs", [])
    question = state.get("question", "")

    # Format search results
    search_results_text = (
        "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', 'Unknown')}]: {doc.page_content[:250]}..."
                for doc in retrieved_docs
            ]
        )
        if retrieved_docs
        else "No relevant passages found."
    )

    # Format chapter info
    chapter_info = f"""
Title: {chapter_context.get('title', 'Unknown')}
Summary: {chapter_context.get('summary_global', chapter_context.get('summary_local', 'N/A'))}
Key Characters: {', '.join([c.get('name', 'Unknown') for c in chapter_context.get('characters', [])[:5]])}
"""

    prompt = f"""You are an expert library assistant helping readers understand stories without spoilers.

**Current Chapter Information:**
{chapter_info}

**Retrieved Information:**
{search_results_text}

**Reader's Question:**
{question}

**Your Task:**
1. If you can answer completely with the available information:
   - Provide a clear, helpful response
   - Base your answer ONLY on the provided context
   
2. If you need more specific information:
   - Start with exactly: "NEED_MORE_CONTEXT:"
   - Follow with a focused search query for the missing information

**Rules:**
- Never reference future events (spoilers)
- Never use outside knowledge
- Be conversational and helpful
"""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "iteration_count": iteration_count + 1,
    }


def _generate_forced_answer(state: StoryState) -> Dict:
    """Generate final answer when max iterations reached"""
    chapter_context = state.get("chapter_context", {})
    retrieved_docs = state.get("retrieved_docs", [])
    question = state.get("question", "")

    search_summary = (
        "\n".join([f"- {doc.page_content[:200]}..." for doc in retrieved_docs[:3]])
        if retrieved_docs
        else "No specific passages available."
    )

    prompt = f"""As a librarian expert, based on available information, provide your best answer to the reader's question.

**Chapter:** {chapter_context.get('title', 'Unknown')}
**Summary:** {chapter_context.get('summary_local', 'N/A')}
**Retrieved passages:** {search_summary}

**Question:** {question}

Provide a helpful response based ONLY on available information. If some aspects cannot be answered, clearly state what you can and cannot answer."""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "messages": [
            AIMessage(content=f"Based on available information: {response.content}")
        ]
    }


def extract_refined_query(state: StoryState) -> Dict:
    """Extract refined query from AI response"""
    last_message = state["messages"][-1].content
    
    if "NEED_MORE_CONTEXT:" in last_message:
        parts = last_message.split("NEED_MORE_CONTEXT:", 1)
        if len(parts) > 1:
            new_query = parts[1].strip()
            print(f"Refining search with: {new_query}")
            return {"question": new_query}
    
    return {}


def decision_router(state: StoryState) -> str:
    if not state.get("messages"): # If there are no messages in the state, it means no interaction has occurred yet.
        return "final_answer"
    
    last_message = state["messages"][-1].content
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    
    # Check max iterations
    if iteration_count >= max_iterations:
        return "final_answer"
    
    # Check if refinement needed
    if "NEED_MORE_CONTEXT:" in last_message:
        return "refine_search"
    
    return "final_answer"


def create_rag_agent():
    """Build and compile the LangGraph RAG agent"""
    builder = StateGraph(StoryState)
    
    # Add nodes
    builder.add_node("initialize_question", initialize_question)
    builder.add_node("get_chapter_context", get_current_chapter_context)
    builder.add_node("semantic_search", semantic_search)
    builder.add_node("generate_answer_or_refine", generate_answer_or_refine)
    builder.add_node("extract_refined_query", extract_refined_query)
    
    # Define flow
    builder.add_edge(START, "initialize_question")
    builder.add_edge("initialize_question", "get_chapter_context")
    builder.add_edge("get_chapter_context", "semantic_search")
    builder.add_edge("semantic_search", "generate_answer_or_refine")
    
    builder.add_conditional_edges(
        "generate_answer_or_refine",
        decision_router,
        {
            "refine_search": "extract_refined_query",
            "final_answer": END
        }
    )
    
    builder.add_edge("extract_refined_query", "semantic_search")
    
    return builder.compile()


rag_agent = create_rag_agent()

async def query_book(
    book_id: str,
    question: str,
    curr_page: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> str:
    """
    Main API function to query a book
    
    Args:
        book_id: Unique identifier for the book
        question: Reader's question
        curr_page: Current page the reader is on
        max_iterations: Maximum refinement iterations (default: 3)
    
    Returns:
        Final answer string
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "book_id": book_id,
        "curr_page": curr_page,
        "max_iterations": max_iterations
    }
    
    config = {"configurable": {"thread_id": f"{book_id}_{curr_page}"}}
    
    result = await rag_agent.ainvoke(initial_state, config)
    
    # Extract final answer
    if result.get("messages"):
        return result["messages"][-1].content
    
    return "I apologize, but I couldn't generate an answer to your question."
