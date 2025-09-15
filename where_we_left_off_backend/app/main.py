
import os
import uuid
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from .models import ChatRequest, ChatResponse, BookProcessingResponse
from .processing import process_book
from .rag_agent import interview_agent

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Project Velcro Backend",
    description="API for processing story books and chatting with a RAG agent.",
    version="1.0.0",
)

# --- Setup Data Directories ---

DATA_DIR = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# In-memory dictionary to track the status of background tasks
background_tasks_status = {}

# --- API Endpoints ---

@app.post("/books/upload", response_model=BookProcessingResponse,
          summary="Upload and Process a Book")
async def upload_and_process_book(background_tasks: BackgroundTasks, file: UploadFile = File(...) ):
    """
    Accepts a PDF file, saves it, and starts a background task to process it.

    When you upload a PDF, the system will:
    1. Save the PDF to the server.
    2. Trigger a multi-stage analysis process in the background (this can take several minutes).
    3. Immediately return a `book_id` and `task_id` for you to track the progress.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    book_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    # Define paths
    book_processing_dir = os.path.join(PROCESSED_DIR, book_id)
    os.makedirs(book_processing_dir, exist_ok=True)
    pdf_path = os.path.join(book_processing_dir, "source.pdf")

    # Save the uploaded file
    try:
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Add the processing task to run in the background
    background_tasks.add_task(process_book, pdf_path, book_id, PROCESSED_DIR)
    
    # Store task status
    background_tasks_status[task_id] = "in_progress"

    return {
        "message": "Book upload successful. Processing has started in the background.",
        "book_id": book_id,
        "task_id": task_id
    }

@app.get("/books/status/{book_id}", summary="Check Book Processing Status")
def get_book_status(book_id: str):
    """
    Checks if the analysis for a given `book_id` is complete.
    
    The final analysis file (`story_global_view.json`) is checked to determine completion.
    """
    status_file = os.path.join(PROCESSED_DIR, book_id, "story_global_view.json")
    if os.path.exists(status_file):
        return {"status": "complete", "book_id": book_id}
    
    # Check if the processing was initiated but is not yet complete
    if os.path.exists(os.path.join(PROCESSED_DIR, book_id)):
         return {"status": "in_progress", "book_id": book_id}

    raise HTTPException(status_code=404, detail="Book ID not found.")

@app.post("/chat/{book_id}", response_model=ChatResponse, summary="Chat with the Book Agent")
def chat_with_book(
    book_id: str,
    request: ChatRequest,
):
    """
    The main endpoint for interacting with the RAG agent.

    You provide the `book_id` of the processed book, your `question`, and the `curr_page` you are on.
    The agent will use this context to provide a spoiler-free answer.
    """
    # First, verify that the book has been processed
    status_file = os.path.join(PROCESSED_DIR, book_id, "story_global_view.json")
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Book has not been processed or book_id is invalid.")

    # Prepare the initial state for the LangGraph agent
    initial_state = {
        "messages": [HumanMessage(content=request.question)],
        "book_id": book_id,
        "curr_page": request.curr_page,
    }

    # The config is needed for the agent to properly thread conversations if needed
    config = {"configurable": {"thread_id": f"book-{book_id}"}}

    try:
        # Invoke the agent and get the final state
        final_state = interview_agent.invoke(initial_state, config=config)
        
        # Extract the last AI message as the answer
        answer = "No answer found."
        for message in reversed(final_state["messages"]):
            if isinstance(message, AIMessage):
                answer = message.content
                break
        
        # Clean up the response if it was a refinement query
        if "NEED_MORE_CONTEXT:" in answer:
            answer = "I was unable to find a definitive answer with the available context."

        return ChatResponse(answer=answer)
    except Exception as e:
        # Log the exception details for debugging
        print(f"Error invoking RAG agent for book {book_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your question.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Project Velcro Backend. Visit /docs for API documentation."}
