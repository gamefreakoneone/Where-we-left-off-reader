
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file before anything else
load_dotenv()

from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient

from .models import ChatRequest, ChatResponse, BookProcessingResponse
from .processing import process_book
from .rag_agent import create_vector_store, rag_agent 
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

# --- Constants ---
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "story_processor"
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MongoDB connection during app lifecycle."""
    app.mongodb_client = AsyncIOMotorClient(MONGO_URI)
    app.db = app.mongodb_client[DB_NAME]
    # init_mongo(MONGO_URI)  # Initialize mongo for rag_agent module
    print("MongoDB connection established.")
    yield

    app.mongodb_client.close()
    print("MongoDB connection closed.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Project Velcro Backend",
    description="API for processing story books and chatting with a RAG agent.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
# This allows the frontend (running on localhost:3000) to communicate with the backend.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


@app.get("/books", summary="List All Books")
async def list_books():
    """
    Retrieves a list of all books that have been uploaded.
    """
    books_cursor = app.db.books.find({}, {"_id": 0, "pdf_path": 0, "vector_store_path": 0}) # Exclude internal and path fields
    books = await books_cursor.to_list(length=None)
    return books


# --- API Endpoints ---

@app.post("/books/upload", response_model=BookProcessingResponse, summary="Upload and Process a Book")
async def upload_and_process_book(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF, saves it, and starts background tasks for processing and vector store creation.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    book_id = str(uuid.uuid4())
    
    book_processing_dir = os.path.join(PROCESSED_DIR, book_id)
    os.makedirs(book_processing_dir, exist_ok=True)
    pdf_path = os.path.join(book_processing_dir, "source.pdf")
    persist_directory = os.path.join(book_processing_dir, "chroma_db")

    try:
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    book_collection = app.db.books
    await book_collection.insert_one({
        "book_id": book_id,
        "filename": file.filename,
        "pdf_path": pdf_path,
        "vector_store_path": persist_directory,
        "status": "processing",
        "created_at": datetime.utcnow()
    })

    # Add background tasks for book processing and vector store creation
    background_tasks.add_task(process_book, pdf_path, book_id, app.db)
    background_tasks.add_task(create_vector_store, book_id, pdf_path, persist_directory)

    return {
        "message": "Book upload successful. Processing has started in the background.",
        "book_id": book_id,
        "task_id": book_id  # Using book_id as the task identifier for now. Maybe figure outsomething better later
    }

@app.get("/books/status/{book_id}", summary="Check Book Processing Status")
async def get_book_status(book_id: str):
    """
    Checks the processing status of a book by querying MongoDB.
    """
    # Check for completion by looking for data in the global chapters collection.
    # This is a reliable indicator that the second pass of processing is done.
    global_chapters_collection = app.db.chapters_global
    count = await global_chapters_collection.count_documents({"book_id": book_id})

    if count > 0:
        # If global chapters exist, processing is complete.
        # Update the book's status to 'complete' for faster lookups next time.
        book_collection = app.db.books
        await book_collection.update_one(
            {"book_id": book_id},
            {"$set": {"status": "complete"}}
        )
        return {"status": "complete", "book_id": book_id}

    # If not complete, check the books collection for the initial status.
    book_collection = app.db.books
    book_doc = await book_collection.find_one({"book_id": book_id})
    if book_doc:
        return {"status": book_doc.get("status", "processing"), "book_id": book_id}

    raise HTTPException(status_code=404, detail="Book ID not found.")


@app.get("/books/pdf/{book_id}", summary="Get PDF file for a book")
async def get_pdf_file(book_id: str):
    """Serves the PDF file for a given book ID."""
    book_meta = await app.db.books.find_one({"book_id": book_id})
    if not book_meta or "pdf_path" not in book_meta:
        raise HTTPException(status_code=404, detail="PDF path not found for this book.")

    pdf_path = book_meta["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server.")

    return FileResponse(pdf_path, media_type='application/pdf', filename=book_meta.get("filename"))


@app.get("/books/data/{book_id}", summary="Get Processed Book Data")
async def get_book_data(book_id: str):
    """
    Retrieves all processed data for a given book_id, including book metadata
    and all chapter information from the global processing pass.
    """
    # Fetch book metadata
    book_meta = await app.db.books.find_one({"book_id": book_id})
    if not book_meta:
        raise HTTPException(status_code=404, detail="Book metadata not found.")

    # Fetch all global chapters for the book
    chapters_cursor = app.db.chapters_global.find({"book_id": book_id})
    chapters = await chapters_cursor.to_list(length=None) # Use length=None to get all documents

    if not chapters:
        raise HTTPException(status_code=404, detail="No processed chapter data found for this book. Processing may still be in progress.")

    # To remove the internal MongoDB _id before sending it to the frontend
    for chap in chapters:
        chap.pop('_id', None)

    return {
        "book_id": book_id,
        "book_title": book_meta.get("filename", "Unknown Title"),
        "chapter": chapters,
    }


@app.post("/chat/{book_id}", response_model=ChatResponse, summary="Chat with the Book Agent")
async def chat_with_book(book_id: str, request: ChatRequest):
    """
    Handles chat interactions with the RAG agent for a specific book.
    """
    # Verify that the book has been fully processed.
    global_chapters_collection = app.db.chapters_global
    count = await global_chapters_collection.count_documents({"book_id": book_id})
    if count == 0:
        raise HTTPException(status_code=404, detail="Book has not been processed yet or book_id is invalid.")

    # Prepare the initial state for the LangGraph agent
    initial_state = {
        "messages": [HumanMessage(content=request.question)],
        "book_id": book_id,
        "curr_page": request.curr_page,
        "db": app.db, 
    }
    
    config = {"configurable": {"thread_id": f"book-{book_id}"}}

    try:
        # Invoke the asynchronous RAG agent
        final_state = await rag_agent.ainvoke(initial_state, config=config)
        
        # Extract the last AI message as the answer
        answer = "No answer found."
        for message in reversed(final_state["messages"]):
            if isinstance(message, AIMessage):
                answer = message.content
                break
        
        if "NEED_MORE_CONTEXT:" in answer:
            answer = "I was unable to find a definitive answer with the available context."

        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error invoking RAG agent for book {book_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your question.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Project Velcro Backend. Visit /docs for API documentation."}
