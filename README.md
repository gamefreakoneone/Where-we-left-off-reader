# Project Velcro - "Where we left off"

An AI-powered story analysis and reading companion that helps users understand and track narrative elements as they read books. The application provides character relationship tracking, story summaries, and a Q&A system with spoiler protection.

## 🚀 Features

- **PDF Upload & Processing**: Upload any PDF book for AI-powered analysis
- **Character Tracking**: Monitor character development, relationships, and status throughout the story
- **Relationship Graphs**: Visualize character relationships and how they evolve
- **Chapter Summaries**: Get local and global story summaries for each chapter
- **Spoiler-Free Q&A**: Ask questions about the story with protection against future spoilers
- **Reading Progress Sync**: Keep track of your reading progress and drop in fresh stories

## 🛠️ Tech Stack

### Frontend
- **Next.js 15.5.2** (with Turbopack)
- **React 19.1.0**
- **TypeScript**
- **Tailwind CSS**
- **React-PDF** (for PDF viewing)
- **ReactFlow** (for relationship graphs)

### Backend
- **FastAPI** (Python web framework)
- **LangGraph** (AI agent workflows)
- **OpenAI API** (for text analysis)
- **ChromaDB** (vector store)
- **PyMuPDF** (PDF processing)
- **LangChain** (LLM orchestration)

### Core Processing
- **Python 3.8+**
- **Pydantic** (data validation)
- **Python-dotenv** (environment management)

##  Project Structure

```
Project-Velcro/
├── where_we_left_off/          # Frontend Next.js application
│   ├── app/                    # Next.js 13+ App Router
│   │   ├── components/         # React components
│   │   ├── books/[bookId]/     # Dynamic book page route
│   │   ├── layout.tsx          # Root layout
│   │   └── page.tsx            # Home page
│   ├── package.json            # Frontend dependencies
│   └── ...
├── where_we_left_off_backend/  # FastAPI backend
│   └── requirements.txt        # Backend dependencies
├── where_we_left_off_core/     # Core Python processing logic
│   ├── core_development.py     # Main analysis engine
│   ├── story_reader_mvp_implementation_plan_v_1.md  # Implementation plan
│   └── ...
└── Books/                      # PDF upload directory
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pnpm package manager
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Project-Velcro
   ```

2. **Set up the backend:**
   ```bash
   # Navigate to backend directory
   cd where_we_left_off_backend

   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install Python dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   # Edit .env and add your OpenAI API key and other configuration
   ```

3. **Set up the frontend:**
   ```bash
   # Navigate to frontend directory
   cd ../where_we_left_off

   # Install dependencies using pnpm
   pnpm install
   ```

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
UPLOAD_DIR=../Books/
```

### Running the Application

1. **Start the backend server:**
   ```bash
   cd where_we_left_off_backend
   uvicorn app.main:app --reload --port 8000
   ```

2. **In a new terminal, start the frontend:**
   ```bash
   cd where_we_left_off
   pnpm run dev
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://127.0.0.1:8000

## 🔧 How It Works

### Two-Pass Processing Architecture

1. **Pass 1 - Chapter Analysis:**
   - PDF segmentation and chapter extraction
   - AI-powered analysis of each chapter for characters, relationships, and summaries
   - Generation of structured data per chapter

2. **Pass 2 - Story Consolidation:**
   - Entity resolution and canonicalization of characters
   - Relationship tracking across chapters
   - Generation of global story summaries and character status

### Spoiler Protection

The application ensures you don't get spoilers from chapters beyond your current reading position:
- All queries are filtered by the current chapter/page
- Semantic search results are limited to content you've already read
- Q&A system refuses to answer questions about future plot points

### Components

- **Frontend (`where_we_left_off`)**: Next.js application for user interaction
  - PDF viewer with bookmarking
  - Character relationship visualization
  - Q&A interface
  - Progress tracking

- **Backend (`where_we_left_off_backend`)**: FastAPI server
  - PDF upload and processing endpoints
  - Book status management
  - Analysis data retrieval
  - Q&A agent

- **Core (`where_we_left_off_core`)**: The original processing logic where I experimented with ideas. Use it as reference
  - PDF parsing and chapter segmentation
  - AI analysis of text content
  - Character and relationship tracking
  - Vector indexing for semantic search

## 📖 Usage

1. **Upload a PDF**: Go to the home page and upload your book
2. **Wait for Processing**: The system will analyze the book in the background
3. **Read Your Book**: Navigate through the PDF while exploring character relationships and summaries
4. **Ask Questions**: Use the Q&A panel to ask about characters, plot points, or relationships
5. **Track Progress**: The app remembers where you left off

## 🤖 AI-Powered Features

- **Character Analysis**: Automatic detection of characters, aliases, roles, and status
- **Relationship Mapping**: Tracking of character relationships and how they evolve
- **Story Summarization**: Chapter-level and global story summaries
- **Semantic Search**: Intelligent search through the book content
- **Q&A with Context**: Answers questions based on current reading level

## 🔒 Spoiler Safety

The application implements strict spoiler protection:
- All results are filtered by the current chapter
- Questions about future events are politely declined
- Relationship data is only shown up to your current reading position

## 📊 Data Model

### Chapter Record Structure
```json
{
  "chapter_id": 1,
  "title": "Chapter One",
  "pages": [5, 18],
  "summary_local": "≤160 words",
  "characters": [
    {
      "name": "Elizabeth",
      "status": "introduced",
      "chapter_role": "POV",
      "relationships": [
        {
          "with_name": "Darcy",
          "type": "rival",
          "justification": "≤30 words"
        }
      ]
    }
  ]
}
```

### Global Story Structure
```json
{
  "book_id": "uuid",
  "chapters": [
    {
      "chapter_id": 1,
      "title": "Chapter One",
      "pages": [5,18],
      "summary": {"local": "...", "so_far": "≤250 words"},
      "characters": [
        {
          "canonical_id": "c_elizabeth_bennet",
          "aliases": ["Elizabeth", "Lizzy", "Ms. Bennet"],
          "status": "active",
          "chapter_role": "POV",
          "relationships": [
            {
              "with": "c_fitzwilliam_darcy",
              "type": "rival",
              "justification": "...",
              "importance": 0.7
            }
          ]
        }
      ]
    }
  ]
}
```

### Development Scripts

In the frontend directory:

```bash
# Development server
pnpm run dev

# Build for production
pnpm run build

# Start production server
pnpm run start
```

## 🚀 Deployment

### Backend (FastAPI)

The backend can be deployed to any platform that supports Python applications (Heroku, VPS, etc.).

### Frontend (Next.js)

The frontend is ready for deployment to Vercel, Netlify, or any hosting service that supports Next.js.

## Acknowledgments

- Built with Next.js, React, FastAPI, and OpenAI
- Uses PyMuPDF for PDF processing
- Powered by LangGraph for AI agent workflows
- My really bad attention span thanks to TikTok and reels.
- Nier Automata Soundtrack for blocking the voices in my head
- If you are still reading this, refer to me a company at this point man. (◕ᴥ◕ʋ)