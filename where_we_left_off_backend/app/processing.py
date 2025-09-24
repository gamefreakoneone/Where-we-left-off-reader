
import re
import json
import fitz  # PyMuPDF
import asyncio
import os
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError, InternalServerError

class Relationship(BaseModel):
    with_name: str = Field(..., description="Other character's name")
    type: str = Field(..., description="e.g., ally | mentor | antagonist | family | rival | colleague | unknown")
    justification: str = Field(..., description="<= 50 words")

class Character(BaseModel):
    name: str = Field(..., description="Name of the character")
    aliases: List[str] = Field(..., description="List of aliases for the character")
    status: str = Field(..., description="e.g: active | missing | dead | resolved | tentative")
    chapter_role: str = Field(..., description="e.g: POV | supporting | antagonist | cameo | unknown")
    character_actions: str = Field(..., description="Key actions in this chapter (<= 100 words)")
    relationships: List[Relationship]

class ChapterFill(BaseModel):
    summary_local: str = Field(..., description="Summary of the chapter (<= 160 words)")
    characters: List[Character]

class RelationshipGlobal(BaseModel):
    with_name: str
    type: str
    justification: str = Field(..., description="<= 100 words")
    importance: int = Field(..., description="0-5 scale")

class CharacterGlobal(BaseModel):
    name: str
    aliases: List[str]
    status: str
    chapter_role: str
    character_actions: str = Field(..., description="Key actions in the ongoing story (<= 200 words)")
    relationships: List[RelationshipGlobal]

class ChapterGlobal(BaseModel):
    summary_global: str = Field(..., description="Summary of the ongoing story (<= 250 words)")
    characters: List[CharacterGlobal]


# --- Core PDF Processing Logic (Adapted from your script) ---

FRONT_MATTER_STOPWORDS = {
    "contents", "table of contents", "copyright", "title page",
    "about the author", "dedication", "preface", "foreword", "acknowledgements",
}

def is_probable_chapter(title: str) -> bool:
    t = title.strip().lower()
    if t in FRONT_MATTER_STOPWORDS:
        return False
    if re.match(r"^\s*(chapter\s+\d+|[ivxlcdm]+\.)\b", t):
        return True
    if re.match(r"^\s*\d+(\.\d+)*\b", t):
        return True
    return len(title.strip()) >= 6

def load_toc(doc: fitz.Document):
    toc = doc.get_toc(simple=True)
    clean = []
    for level, title, page in toc:
        if page is None or page < 1:
            continue
        clean.append((level, title.strip(), page))
    return clean

def derive_chapter_ranges(doc: fitz.Document):
    toc = load_toc(doc)
    if not toc:
        return []

    level_counts = {level: level_counts.get(level, 0) + 1 for level, title, _ in toc if is_probable_chapter(title)}
    chapter_level = min(level_counts, key=lambda k: (-level_counts[k], k)) if level_counts else 1

    filtered = [(lvl, title, page) for (lvl, title, page) in toc if lvl >= chapter_level]
    chapters = []
    for i, (lvl, title, page1_based) in enumerate(filtered):
        if lvl != chapter_level:
            continue
        start = max(0, page1_based - 1)
        end_0based = doc.page_count - 1
        for j in range(i + 1, len(filtered)):
            lvl_j, _, page_j_1b = filtered[j]
            if lvl_j <= chapter_level:
                end_0based = max(0, page_j_1b - 2)
                break
        if start <= end_0based:
            chapters.append({"level": lvl, "title": title, "start_page": start, "end_page": end_0based})

    return [c for c in chapters if is_probable_chapter(c["title"])]

def extract_text_range(doc: fitz.Document, start_page: int, end_page: int) -> str:
    return "\n".join(doc[p].get_text("text", sort=True).strip() for p in range(start_page, end_page + 1)).strip()

def make_chapter_skeletons(chapters):
    return {"chapter": [{
        "chapter_id": i, "title": ch["title"], "pages": [ch["start_page"], ch["end_page"]],
        "summary_local": "", "characters": []
    } for i, ch in enumerate(chapters, start=1)]}



chapter_fill_prompt = "You are a library assistant skilled at extracting structured information from story book chapters. Fill in the structured data fields based *only* on the provided chapter text. Do not add outside knowledge or spoilers. Be concise and accurate."
chapter_fill_global_prompt = "You are a library assistant skilled at synthesizing story information. Given the story so far (summary and characters) and a new chapter, update the global summary and character list to reflect the new developments. Be concise and avoid spoilers."

async def fill_chapter_with_model_async(client: AsyncOpenAI, chapter_text: str, semaphore: asyncio.Semaphore) -> ChapterFill:
    # This function remains largely the same, but we remove the retry logic here
    # for simplicity in this example. A production system should have robust retries.
    async with semaphore:
        resp = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": chapter_fill_prompt},
                {"role": "user", "content": chapter_text},
            ],
            response_model=ChapterFill,
        )
        return resp

async def create_global_view(client: AsyncOpenAI, previous_summary: str, previous_characters: dict, current_chapter: dict) -> ChapterGlobal:
    context = {
        "previous_story_summary": previous_summary,
        "all_known_characters_so_far": list(previous_characters.values()),
        "current_chapter_title": current_chapter["title"],
        "current_chapter_summary": current_chapter["summary_local"],
        "characters_in_current_chapter": current_chapter["characters"],
    }
    resp = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": chapter_fill_global_prompt},
            {"role": "user", "content": json.dumps(context, indent=2)}
        ],
        response_model=ChapterGlobal,
    )
    return resp

def _normalize_and_validate_global(ch: ChapterGlobal) -> ChapterGlobal:
    if not ch or not ch.characters:
        return ch
    for c in ch.characters:
        c.aliases = list(set(alias.strip() for alias in (c.aliases or []) if alias.strip()))
        if c.relationships:
            for r in c.relationships:
                r.importance = max(0, min(5, int(r.importance or 0)))
    return ch

# --- Main Processing Orchestrator ---

async def process_book(pdf_path: str, book_id: str, data_dir: str):
    """
    The main background task to process a PDF file.
    It orchestrates the two-pass processing and saves the results.
    """
    print(f"Starting processing for book_id: {book_id}")
    book_data_dir = os.path.join(data_dir, book_id)
    os.makedirs(book_data_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return

    # === First Pass: Chapter Extraction and Local Summaries ===
    chapters = derive_chapter_ranges(doc)
    if not chapters:
        print("No chapters derived. Aborting.")
        doc.close()
        return

    first_pass_data = make_chapter_skeletons(chapters)
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(4) # Limit concurrency

    async def first_pass_task(i, chapter_info):
        text = extract_text_range(doc, chapter_info["pages"][0], chapter_info["pages"][1])
        if not text.strip():
            return i, None
        try:
            filled_data = await fill_chapter_with_model_async(aclient, text, semaphore=sem)
            return i, filled_data
        except Exception as e:
            print(f"Chapter {chapter_info['chapter_id']} (pass 1) failed: {e}")
            return i, None

    tasks = [first_pass_task(i, ch) for i, ch in enumerate(first_pass_data["chapter"])]
    results = await asyncio.gather(*tasks)

    for i, result in results:
        if result:
            first_pass_data["chapter"][i]["summary_local"] = result.summary_local
            first_pass_data["chapter"][i]["characters"] = [c.model_dump() for c in result.characters]

    # === Second Pass: Global View Synthesis ===
    second_pass_output = {"chapter": []}
    cumulative_summary = ""
    cumulative_characters = {}

    for i, chapter_data in enumerate(first_pass_data["chapter"]):
        print(f"Second Pass - Processing chapter {i+1}/{len(first_pass_data['chapter'])}")
        try:
            global_view = await create_global_view(
                client=aclient,
                previous_summary=cumulative_summary,
                previous_characters=cumulative_characters,
                current_chapter=chapter_data
            )
            validated_global_view = _normalize_and_validate_global(global_view)
            cumulative_summary = validated_global_view.summary_global
            cumulative_characters = {char.name: char.model_dump() for char in validated_global_view.characters}

            second_pass_output["chapter"].append({
                **chapter_data,
                "summary_global": cumulative_summary,
                "characters": list(cumulative_characters.values()),
            })
        except Exception as e:
            print(f"Chapter {chapter_data['chapter_id']} (pass 2) failed: {e}")
            # Append with only first pass data if second pass fails
            second_pass_output["chapter"].append(chapter_data)


    # === Save Final Output ===
    output_filename = os.path.join(book_data_dir, "story_global_view.json")
    with open(output_filename, "w") as f:
        json.dump(second_pass_output, f, indent=2)

    doc.close()
    print(f"Processing complete for book_id: {book_id}. Output saved to {output_filename}")
