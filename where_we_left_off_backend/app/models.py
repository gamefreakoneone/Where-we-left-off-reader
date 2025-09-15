from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    question: str
    curr_page: int

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    answer: str

class BookProcessingResponse(BaseModel):
    """Response model for the book upload endpoint."""
    message: str
    book_id: str
    task_id: str
