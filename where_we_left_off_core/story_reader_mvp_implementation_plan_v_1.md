# Story Reader MVP – Implementation Plan v1.0

A production-lean blueprint derived from the refined two‑pass architecture, with a hackathon‑friendly single‑pass variant. Includes schemas, code skeletons, and acceptance checks.

---

---

## 1) Core data shapes (strict JSON + Pydantic)

**Enums (controlled vocabulary):**
- `CharacterStatus`: `introduced | active | missing | dead | resolved | tentative`
- `Role`: `POV | supporting | antagonist | cameo | unknown`
- `RelType`: `ally | mentor | antagonist | family | rival | colleague | unknown`

**Per‑chapter (Pass 1) record**
```json
{
  "chapter": {
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
}
```

**Consolidated chapter entry (chapter_summary_status)**
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
              "importance": {number in the range from 0 - 1 indicating importance to character}
            }
          ]
        }
      ]
    }
  ]
}
```

> **Tip:** In SQL mode, keep identical shapes inside `metadata_json` columns and add relational tables for analytics (chapters, characters, relationships, beats, mentions, chunks).

---

## 2) Chapter segmentation (PDF → page ranges)

```python
# backend/segmentation.py
from typing import List, Dict, Tuple
import re
import pdfplumber

CHAPTER_RE = re.compile(r"^(chapter\s+([0-9ivxlcdm]+)|\bchapter\b)\b", re.I)

def extract_chapter_ranges(pdf_path: str) -> List[Dict]:
    """Heuristic segmentation. Prefer bookmarks/ToC if available; fallback to heading regex & font jumps."""
    ranges: List[Tuple[int, str]] = []  # (page_index, maybe_title)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip().splitlines()[:6]
            heading = next((ln for ln in text if CHAPTER_RE.search(ln)), None)
            if heading:
                ranges.append((i, heading))
    # Close ranges
    chapters = []
    for j, (start, title) in enumerate(ranges):
        end = (ranges[j+1][0]-1) if j+1 < len(ranges) else len(pdf.pages)-1
        chapters.append({"chapter_id": j+1, "title": title, "pages": [start+1, end+1]})  # 1-based pages
    return chapters
```

Persist to `data/chapters.json` and expose via `/api/chapters`.

---

## 3) Pass 1 – Stateless per‑chapter extraction

**Prompt shape (LLM‑agnostic)**
```
System: You extract structured story data with strict JSON. Do not include future context beyond the provided text.
User: <chapter_text>
Tools: Use the provided JSON schema. Fill enums exactly as given. Limit summary_local to 160 words. Include evidence_pages.
Output: JSON only. No prose.
```

**Skeleton**
```python
# backend/pass1_extract.py
from typing import Dict, Any
from schemas import ChapterRecord  # Pydantic model with strict enums/limits
from provider import call_llm_json  # wraps OpenAI/Gemini JSON-mode

def extract_chapter(chapter_text: str, chap_meta: Dict) -> Dict[str, Any]:
    schema = ChapterRecord.model_json_schema()
    instructions = {
        "enums": {
            "CharacterStatus": ["introduced","active","missing","dead","resolved","tentative"],
            "Role": ["POV","supporting","antagonist","cameo","unknown"],
            "RelType": ["ally","mentor","antagonist","family","rival","colleague","unknown"],
            "Interaction": ["dialogue","conflict","help","gift","travel","plan","other"],
            "BeatType": ["setup","discovery","conflict","twist","resolution","other"]
        },
        "constraints": {"summary_local_max_words": 160, "justification_max_words": 30}
    }
    resp = call_llm_json(chapter_text, schema=schema, instructions=instructions)
    record = ChapterRecord.model_validate(resp)
    # attach page metadata
    record.chapter.pages = chap_meta["pages"]
    record.chapter.title = chap_meta.get("title","Chapter")
    record.chapter.chapter_id = chap_meta["chapter_id"]
    return record.model_dump()
```

Persist each to `data/pass1_chapter_records/chapter_{id}.json` with `run_id`, `model`, and `prompt_version`.

---

## 4) Entity Resolver – canonical IDs & aliases

**Rules:**
- `canonical_id = "c_" + slugify(primary_name)`; primary chosen by frequency or first introduction.
- Map new names to canonical via **(embedding cosine ≥ τ1) OR (Levenshtein ≤ τ2)**; else mark `tentative` and keep as alias for later sweep.

**Skeleton**
```python
# backend/entity_resolver.py
from typing import Dict, List
from rapidfuzz.distance import Levenshtein
import numpy as np

THRESH_COS = 0.82
THRESH_EDIT = 3

class EntityResolver:
    def __init__(self, embedder):
        self.embedder = embedder
        self.canon_vecs: Dict[str, np.ndarray] = {}
        self.alias_map: Dict[str, str] = {}  # alias -> canonical_id

    def add_canonical(self, name: str, canonical_id: str):
        self.alias_map[name.lower()] = canonical_id
        self.canon_vecs[canonical_id] = self.embedder.encode(name)

    def resolve(self, name: str) -> str | None:
        key = name.lower()
        if key in self.alias_map: return self.alias_map[key]
        vec = self.embedder.encode(name)
        best = None; best_sim = -1
        for cid, cvec in self.canon_vecs.items():
            sim = float(np.dot(vec, cvec) / (np.linalg.norm(vec)*np.linalg.norm(cvec)+1e-8))
            if sim > best_sim: best_sim, best = sim, cid
        if best_sim >= THRESH_COS or Levenshtein.distance(name, best.replace("c_","")) <= THRESH_EDIT:
            self.alias_map[key] = best
            return best
        return None  # unresolved → mark tentative
```

---

## 5) Pass 2 – Sequential consolidation (build “so far”)

**Strategy:**
- Iterate chapters 1..N. Maintain `global_state` of characters (canonical map) and relationships.
- For each chapter: merge Pass‑1 record → generate `summary.so_far` (≤250 words) + relationship deltas; keep provenance (beat/relationship IDs, evidence pages).

**Skeleton**
```python
# backend/pass2_consolidate.py
from typing import Dict, List
from utils import clip_words

def consolidate(pass1_records: List[Dict], resolver) -> Dict:
    state = {"book_id": "uuid", "chapters": [], "version": "v1.1.0"}
    global_chars: Dict[str, Dict] = {}

    for rec in pass1_records:
        chap = rec["chapter"]
        # Canonicalize characters
        canon_chars = []
        for ch in chap["characters"]:
            cid = resolver.resolve(ch["name"]) or f"c_{ch['name'].lower().replace(' ','_')}_tent"
            global_entry = global_chars.setdefault(cid, {"aliases": set(), "status": ch["status"]})
            global_entry["aliases"].add(ch["name"])
            global_entry["status"] = ch["status"]  # last known
            # Canonicalize relationships
            rels = []
            for r in ch.get("relationships", []):
                with_id = resolver.resolve(r["with_name"]) or f"c_{r['with_name'].lower().replace(' ','_')}_tent"
                rels.append({**r, "with": with_id, "with_name": None})
            canon_chars.append({
                "canonical_id": cid,
                "aliases": sorted(list(global_entry["aliases"])),
                "status": ch["status"],
                "chapter_role": ch.get("chapter_role","unknown"),
                "relationships": rels,
                "notes": ch.get("notes","")
            })
        # Compose so_far from prior so_far + local summary + key beats
        prev_so_far = state["chapters"][-1]["summary"]["so_far"] if state["chapters"] else ""
        so_far = clip_words(prev_so_far + " " + chap["summary_local"], 250)
        state["chapters"].append({
            "chapter_id": chap["chapter_id"],
            "title": chap["title"],
            "pages": chap["pages"],
            "summary": {"local": chap["summary_local"], "so_far": so_far},
            "beats": chap["beats"],
            "characters": canon_chars
        })
    # finalize alias sets as lists
    for c in global_chars.values(): c["aliases"] = sorted(list(c["aliases"]))
    return state
```

---

## 6) Consistency sweeper (lightweight)

Checks run every 3–5 chapters or at the end:
- Unresolved/tentative IDs
- Conflicting relationship types (A↔B flips repeatedly)
- Missing `evidence_pages`
- Oversized fields (truncate and log)

```python
# backend/sweeper.py
from typing import Dict

def sweep(book: Dict) -> Dict:
    issues = []
    seen_pairs = {}
    for chap in book["chapters"]:
        for ch in chap["characters"]:
            for r in ch.get("relationships", []):
                pair = tuple(sorted([ch["canonical_id"], r["with"]]))
                seen_pairs.setdefault(pair, set()).add(r["type"])
                if not r.get("evidence_pages"): issues.append(("missing_evidence", chap["chapter_id"], pair))
    for pair, types in seen_pairs.items():
        if len(types) > 2: issues.append(("contradictory_relationships", list(pair), list(types)))
    book.setdefault("_sweeper", {})["issues"] = issues
    return book
```

---

## 7) Indexing raw text (vector store)

- Chunk by page or scene.
- Metadata: `book_id, chapter_id, page_start, page_end, character_ids[], beat_ids[]`.
- Use OpenAI embeddings (e.g., `text-embedding-3-small`) or provider of choice. Store in FAISS/Chroma.

```python
# backend/indexer.py
import faiss, numpy as np

class VectorIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []
        self.meta = []
    def add(self, vecs: np.ndarray, metas: list):
        self.index.add(vecs)
        self.meta.extend(metas)
    def search(self, qvec: np.ndarray, k=5):
        D,I = self.index.search(qvec.reshape(1,-1), k)
        return [(float(D[0][i]), self.meta[I[0][i]]) for i in range(len(I[0]))]
```

---

## 8) Q&A Agent (LangGraph-style nodes)

**Nodes:**
- `router(query)` → {structured_lookup | vector_search}
- `structured_lookup(query, current_chapter)` → search `chapter_summary_status` (≤ current)
- `vector_search(query, current_chapter)` → filter chunks with `chapter_id <= current`
- `compose(answer_parts)` → draft
- `spoiler_gate(draft, current_chapter)` → redact future content

**Pseudo**
```python
# backend/qa_agent.py

def spoiler_gate(text: str, allowed_max_chapter: int) -> str:
    # Simple policy: ensure all cited chapters/pages ≤ allowed
    # In production, also re-run an LLM classifier on the draft with chapter bounds.
    return text

def answer(query: str, last_read_chapter: int, stores) -> str:
    # 1) Try structured
    s_hits = stores.chapter_state.search(query, max_chap=last_read_chapter)
    if s_hits.confidence >= 0.6:
        draft = compose_from_structured(s_hits)
        return spoiler_gate(draft, last_read_chapter)
    # 2) Fallback to vector
    v_hits = stores.vector.search(query, filter_max_chapter=last_read_chapter)
    draft = compose_from_vector(v_hits)
    return spoiler_gate(draft, last_read_chapter)
```

---

## 9) Frontend (Next.js/React) – essentials

- **PdfReader.tsx**: pdf.js viewer; push `last_read_page` to backend.
- **StorySoFarPanel.tsx**: show `summary.so_far` and a relationship list (chips: `A —[rival/ally]→ B`).
- **CharacterGraph.tsx**: Cytoscape graph; nodes = canonical characters; edges = latest relationship type up to current chapter.
- **QABox.tsx**: sends query + `last_read_chapter`; displays answer with page ranges.

**API endpoints**
- `GET /api/chapters` → chapter ranges
- `GET /api/state?upto=X` → `chapter_summary_status` up to X
- `POST /api/qa` {query, last_read_chapter}

---

## 10) Acceptance tests (spoiler‑safe + fidelity)

- **Spoiler guard**: Ask “What happens to X in Chapter N+1?” when current=N → must refuse politely.
- **Alias resolution**: “Lizzy vs Elizabeth vs Ms. Bennet” resolve to same canonical.
- **Relationship query**: “What is A’s relationship to B up to Chapter k?” returns type + brief justification + page refs.
- **Quote retrieval**: “Exact words when Y says Z?” triggers vector fallback and cites page range.
- **Schema compliance**: All Pass‑1/2 outputs validate against Pydantic models; enums are exact.

---

## 11) Config & Ops

- `.env`: `LLM_PROVIDER=...`, `OPENAI_API_KEY=...`, `EMBED_MODEL=text-embedding-3-small`, `DB_URL=sqlite:///story.db`.
- Log `run_id`, `model`, `prompt_version` in every artifact.
- Version `chapter_summary_status` (e.g., `v1.1.0`) and keep old copies for diff.

---

## 12) 48‑Hour Hackathon Plan (sequenced)

**Day 1**
1) Segment chapters and freeze `data/chapters.json`.
2) Implement Pass‑1 extractor (JSON mode) + write `pass1_chapter_records/`.
3) Build minimal EntityResolver (embedding + Levenshtein).
4) Implement Pass‑2 consolidator → produce `chapter_summary_status.json`.

**Day 2**
5) Build minimal frontend (pdf.js + StorySoFarPanel).
6) Implement `/api/state?upto=X` and `/api/qa` (structured lookup only).
7) Add spoiler_gate and 3 acceptance tests.
8) Stretch: index raw text + vector fallback for quotes.

---

## 13) Prompt snippets (ready to paste)

**Pass‑1 system**
> You are a precise information extractor. Output strictly valid JSON that conforms to the provided schema and enums. Do not infer or include any events outside the provided chapter text. Keep `summary_local` ≤160 words. Include `evidence_pages` for beats and relationships.

**Pass‑1 user (template)**
```
<CHAPTER_META>
- chapter_id: {id}
- title: {title}
- pages: {start}-{end}
</CHAPTER_META>

<TEXT>
{chapter_text}
</TEXT>

<SCHEMA>
{json_schema}
</SCHEMA>
```

**Pass‑2 system**
> You are a consolidator. Given the prior "so_far" summary and the current chapter’s structured record, produce an updated "so_far" (≤250 words) and return canonicalized character states and relationships with justifications and evidence pages. Do not invent future content.

---

## 14) Notes & gotchas

- Keep temperatures low on extraction/consolidation.
- Never pass prior raw text to Pass‑2; pass only compact structured state.
- Always filter UI/API answers by `chapter_id <= last_read_chapter`.
- Store deltas if possible; you can materialize `so_far` on read.

---

**End of Plan**

