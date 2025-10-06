import re
import json
import fitz  # PyMuPDF
import asyncio
import os
import random
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIError,
    APIConnectionError,
    InternalServerError,
)
from motor.motor_asyncio import AsyncIOMotorClient


class Relationship(BaseModel):
    with_name: str = Field(..., description="Other character's name")
    type: str = Field(
        ...,
        description="e.g., ally | mentor | antagonist | family | rival | colleague | unknown",
    )
    justification: str = Field(..., description="<= 50 words")


class Character(BaseModel):
    name: str = Field(..., description="Name of the character")
    aliases: List[str] = Field(..., description="List of aliases for the character")
    status: str = Field(
        ..., description="e.g: active | missing | dead | resolved | tentative"
    )
    chapter_role: str = Field(
        ..., description="e.g: POV | supporting | antagonist | cameo | unknown"
    )
    character_actions: str = Field(
        ..., description="Key actions in this chapter (<= 100 words)"
    )
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
    character_actions: str = Field(
        ..., description="Key actions in the ongoing story (<= 200 words)"
    )
    relationships: List[RelationshipGlobal]


class ChapterGlobal(BaseModel):
    summary_global: str = Field(
        ..., description="Summary of the ongoing story (<= 250 words)"
    )
    characters: List[CharacterGlobal]


FRONT_MATTER_STOPWORDS = {
    "contents",
    "table of contents",
    "copyright",
    "title page",
    "about the author",
    "dedication",
    "preface",
    "foreword",
    "acknowledgements",
}


def is_probable_chapter(title: str):
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

    level_counts = {}
    for level, title, _ in toc:
        if is_probable_chapter(title):
            level_counts[level] = level_counts.get(level, 0) + 1
    chapter_level = (
        min(level_counts, key=lambda k: (-level_counts[k], k)) if level_counts else 1
    )

    filtered = [
        (lvl, title, page) for (lvl, title, page) in toc if lvl >= chapter_level
    ]
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
            chapters.append(
                {
                    "level": lvl,
                    "title": title,
                    "start_page": start,
                    "end_page": end_0based,
                }
            )

    return [c for c in chapters if is_probable_chapter(c["title"])]


def extract_text_range(doc: fitz.Document, start_page: int, end_page: int):
    return "\n".join(
        doc[p].get_text("text", sort=True).strip()
        for p in range(start_page, end_page + 1)
    ).strip()


def make_chapter_skeletons(book_id: str, chapters):
    return {
        book_id: {
            "book_id": book_id,
            "chapters": [
                {
                    "chapter_id": i,
                    "title": ch["title"],
                    "pages": [ch["start_page"], ch["end_page"]],
                    "summary_local": "",
                    "characters": [],
                }
                for i, ch in enumerate(chapters, start=1)
            ],
        }
    }


chapter_fill_prompt = (
    "You are an expert library assistant who is skilled at extracting structured information from story book chapters. You are given the text of a chapter and must fill in the structured data fields, such that it meets the following criteria:"
    "1. The summary_local field must contain a concise summary of the chapter in the context of the data provided. Even if you are aware about the story you are dealing with, do not add additional information that can potentially spoil the future chapters, limited to 160 words.\n"
    "2. For the characters entry, it is a list of Character objects, each with the following fields:\n"
    "   - name: Name of the character\n"
    "   - aliases: List of aliases for the character. Do not include generic pronouns such as 'he', 'she', or 'they' or role words such as teacher, guard , protagonist , antagonist etc\n"
    "   - status: e.g: active | missing | dead | resolved | tentative\n"
    "   - chapter_role: e.g: POV | supporting | antagonist | cameo | unknown\n"
    "   - character_actions: Key actions or events involving the character in this chapter (<= 100 words)\n"
    "   - relationships: List of Relationship objects\n"
    "3. For the relationships entry, it is a list of Relationship objects, each with the following fields:\n"
    "   - with_name: Other character's name\n"
    "   - type: Relationship type (e.g., ally | mentor | antagonist | family | rival | colleague | unknown)\n"
    "   - justification: an explanation in the context of the chapter why the relationship exists ( <= 50 words)\n"
)

chapter_fill_global = """
You are an expert library assistant who is skilled at extracting structured information from story book chapters. You are given the text of a chapter and must fill in the structured data fields, such that it meets the following criteria:
1. The summary_global field must contain a concise summary of the ongoing story in the context of the previous chapter summary and current chapter summary provided. Even if you are aware about the story you are dealing with, do not add additional information that can potentially spoil the future chapters, limited to 250 words.
2. The characters field must include all relevant characters introduced or developed in the chapter, along with their updated attributes. This includes :
   - Name: Name of the character
   - Aliases: List of character aliases
   - Status: Current status of the character in the context of the ongoing story with reference to previous chapters and current chapter
   - Chapter role: Role of the character in the chapter in the context of the ongoing story
   - Character actions: Key actions or events involving the character in the ongoing story
   - Relationships: List of relationships with other characters
3. For the relationships field, it is a list of Relationship objects, each with the following fields:
    - with_name: Other character's name
    - type: Relationship type (e.g., ally | mentor | antagonist | family | rival | colleague | unknown)
    - justification: an explanation in the context of the chapter why the relationship exists ( <= 100 words)
    - importance: Importance of the relationship on a scale of 0-5, where 0 is negligible and 5 is critical to the story. Update the importance level as the story progresses.
"""


async def fill_chapter_with_model_async(
    client: AsyncOpenAI,
    chapter_text: str,
    *,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
):
    backoff = 1
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.responses.parse(
                    model="gpt-5-mini",
                    input=[
                        {"role": "system", "content": chapter_fill_prompt},
                        {"role": "user", "content": chapter_text},
                    ],
                    text_format=ChapterFill,
                )
            return resp.output_parsed
        except (RateLimitError, APIError, APIConnectionError, InternalServerError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30)


async def create_global_view(
    client: AsyncOpenAI,
    previous_summary: str,
    previous_characters: dict,
    current_chapter: dict,
):
    context = {
        "previous_story_summary": previous_summary,
        "all_known_characters_so_far": list(previous_characters.values()),
        "current_chapter_title": current_chapter["title"],
        "current_chapter_summary": current_chapter["summary_local"],
        "characters_in_current_chapter": current_chapter["characters"],
    }

    resp = await client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": chapter_fill_global},
            {"role": "user", "content": json.dumps(context, indent=2)},
        ],
        text_format=ChapterGlobal,
    )
    return resp.output_parsed


def _normalize_and_validate_global(ch: ChapterGlobal):
    if not ch or not ch.characters:
        return ch

    for c in ch.characters:
        seen_aliases = set()
        unique_aliases = []
        for alias in c.aliases or []:
            stripped_alias = (alias or "").strip()
            if stripped_alias and stripped_alias.lower() not in seen_aliases:
                seen_aliases.add(stripped_alias.lower())
                unique_aliases.append(stripped_alias)
        c.aliases = unique_aliases

        if not c.relationships:
            continue
        for r in c.relationships:
            r.importance = max(0, min(5, int(r.importance or 0)))
            if r.justification and len(r.justification.split()) > 100:
                r.justification = " ".join(r.justification.split()[:100]) + "..."
    return ch


# # --- MongoDB Setup ---
# def get_mongo_client(connection_string: str = None):
#     """Create and return MongoDB client"""
#     if connection_string is None:
#         connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
#     return AsyncIOMotorClient(connection_string)


async def save_local_chapters_to_mongo(db, book_id: str, chapters_data: dict):
    """Save first pass (local) chapter data to MongoDB"""
    collection = db.chapters_local

    book_data = chapters_data[book_id]
    for chapter in book_data["chapters"]:
        doc = {
            "book_id": book_id,
            "chapter_id": chapter["chapter_id"],
            "title": chapter["title"],
            "pages": chapter["pages"],
            "summary_local": chapter["summary_local"],
            "characters": chapter["characters"],
            "processed_at": datetime.utcnow(),
        }
        await collection.update_one(
            {"book_id": book_id, "chapter_id": chapter["chapter_id"]},
            {"$set": doc},
            upsert=True,
        )
    print(f"Saved {len(book_data['chapters'])} local chapters to MongoDB")


async def save_global_chapters_to_mongo(db, book_id: str, chapters_data: dict):
    """Save second pass (global) chapter data to MongoDB"""
    collection = db.chapters_global

    book_data = chapters_data[book_id]
    for chapter in book_data["chapters"]:
        doc = {
            "book_id": book_id,
            "chapter_id": chapter["chapter_id"],
            "title": chapter["title"],
            "pages": chapter["pages"],
            "summary_local": chapter.get("summary_local", ""),
            "summary_global": chapter.get("summary_global", ""),
            "characters": chapter["characters"],
            "processed_at": datetime.utcnow(),
        }
        await collection.update_one(
            {"book_id": book_id, "chapter_id": chapter["chapter_id"]},
            {"$set": doc},
            upsert=True,
        )
    print(f"Saved {len(book_data['chapters'])} global chapters to MongoDB")


async def process_book(
    pdf_path: str, book_id: str,  db
):
    """
    Main task to process a PDF file and store results in MongoDB.
    """
    print(f"Starting processing for book_id: {book_id}")

    # Connect to MongoDB
    # mongo_client = get_mongo_client(mongo_connection_string)
    # db = mongo_client.story_processor

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return

    # First Pass: Local Chapter Summaries and Characters
    chapters = derive_chapter_ranges(doc)
    if not chapters:
        print("No chapters derived. Aborting.")
        doc.close()
        return

    first_pass_data = make_chapter_skeletons(book_id, chapters)
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(4)

    async def first_pass_task(i, chapter_info):
        text = extract_text_range(
            doc, chapter_info["pages"][0], chapter_info["pages"][1]
        )
        if not text.strip():
            return i, None
        try:
            filled_data = await fill_chapter_with_model_async(
                aclient, text, semaphore=sem
            )
            return i, filled_data
        except Exception as e:
            print(f"Chapter {chapter_info['chapter_id']} (pass 1) failed: {e}")
            return i, None

    tasks = [
        first_pass_task(i, ch)
        for i, ch in enumerate(first_pass_data[book_id]["chapters"])
    ]
    results = await asyncio.gather(*tasks)

    for i, result in results:
        if result:
            first_pass_data[book_id]["chapters"][i][
                "summary_local"
            ] = result.summary_local
            first_pass_data[book_id]["chapters"][i]["characters"] = [
                c.model_dump() for c in result.characters
            ]

    # Save first pass data to MongoDB
    await save_local_chapters_to_mongo(db, book_id, first_pass_data)

    # === Second Pass: Global View ===
    second_pass_output = {book_id: {"book_id": book_id, "chapters": []}}
    cumulative_summary = ""
    cumulative_characters = {}

    for i, chapter_data in enumerate(first_pass_data[book_id]["chapters"]):
        print(
            f"Second Pass - Processing chapter {i+1}/{len(first_pass_data[book_id]['chapters'])}"
        )
        try:
            global_view = await create_global_view(
                client=aclient,
                previous_summary=cumulative_summary,
                previous_characters=cumulative_characters,
                current_chapter=chapter_data,
            )
            validated_global_view = _normalize_and_validate_global(global_view)
            cumulative_summary = validated_global_view.summary_global
            cumulative_characters = {
                char.name: char.model_dump()
                for char in validated_global_view.characters
            }

            second_pass_output[book_id]["chapters"].append(
                {
                    **chapter_data,
                    "summary_global": cumulative_summary,
                    "characters": list(cumulative_characters.values()),
                }
            )
        except Exception as e:
            print(f"Chapter {chapter_data['chapter_id']} (pass 2) failed: {e}")
            second_pass_output[book_id]["chapters"].append(chapter_data)

    # Save second pass data to MongoDB
    await save_global_chapters_to_mongo(db, book_id, second_pass_output)

    doc.close()
    # mongo_client.close()
    print(f"Processing complete for book_id: {book_id}. Data saved to MongoDB")
