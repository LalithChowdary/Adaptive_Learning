"""Hybrid RAG engine combining SQLite FTS5 BM25, TF-IDF cosine vectors, and RRF (k=60)."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import PACKAGE_DIR, catalogue_bridge
from .store import AdaptiveStore

KNOWLEDGE_DIR = PACKAGE_DIR / "knowledge"
RRF_K = 60

_PYTHON_SYNONYMS: dict[str, str] = {
    "off by one": "range stop exclusive start inclusive boundary len",
    "loop": "for while iteration state trace pass range body",
    "range": "range start stop step exclusive half-open empty",
    "accumulator": "accumulator initialize identity baseline total sum product count",
    "initial": "initialize initialization pre-loop baseline identity total product",
    "condition": "if elif else boolean inequality precedence short-circuit branch",
    "elif": "if elif else mutually exclusive ordered branch",
    "index": "list index zero-based negative IndexError len",
    "negative": "negative index -1 tail reverse backward sequence",
    "return": "def function return print None exit scope parameter",
    "print": "print stdout return None side effect expression",
    "none": "None implicit missing return print evaluation",
    "error": "IndexError TypeError ValueError UnboundLocalError trace bug",
    "why": "explain trace step-by-step misconception rule example",
}

_STATIC_CACHE_LOCK = threading.Lock()
_STATIC_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class KnowledgeChunk:
    doc_id: str
    source_type: str  # "curriculum" | "v22_exemplar" | "episodic"
    learner: str
    skill: str
    difficulty: str
    task_type: str
    title: str
    section: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created: float = 0.0

    @property
    def searchable_text(self) -> str:
        parts = [self.skill, self.difficulty, self.task_type, self.title, self.section, self.content]
        return " ".join(p for p in parts if p)


def _tokenize_alnum(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", str(text or ""))


def _build_safe_fts_query(query: str) -> str:
    tokens = []
    seen: set[str] = set()
    for tok in _tokenize_alnum(query):
        low = tok.lower()
        if len(low) < 2 and not low.isdigit():
            continue
        if low in {"and", "or", "not", "near"}:
            continue
        if low not in seen:
            seen.add(low)
            tokens.append(f'"{low}"')
        if len(tokens) >= 24:
            break
    return " OR ".join(tokens)


def _chunk_markdown_document(path: Path, skill: str) -> list[KnowledgeChunk]:
    raw = path.read_text(encoding="utf-8")
    doc_title = catalogue_bridge.SKILL_META.get(skill, {}).get("title", skill)
    lines = raw.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_sec = "Overview"
    buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title_cand = stripped.lstrip("# ").strip()
            if title_cand:
                doc_title = title_cand
            continue
        if stripped.startswith("## ") or stripped.startswith("### "):
            if buf:
                sections.append((current_sec, buf))
                buf = []
            current_sec = stripped.lstrip("# ").strip() or "Section"
        else:
            buf.append(line)
    if buf:
        sections.append((current_sec, buf))

    chunks: list[KnowledgeChunk] = []
    for sec_idx, (sec_name, sec_lines) in enumerate(sections):
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        words = text.split()
        max_words = 260
        overlap = 35
        if len(words) <= max_words:
            sub_texts = [text]
        else:
            sub_texts = []
            start = 0
            while start < len(words):
                end = min(len(words), start + max_words)
                sub_texts.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = max(start + 1, end - overlap)

        sec_low = sec_name.lower()
        inferred_diff = "mixed"
        if "easy" in sec_low:
            inferred_diff = "easy"
        elif "hard" in sec_low:
            inferred_diff = "hard"
        elif "medium" in sec_low:
            inferred_diff = "medium"

        for sub_idx, chunk_text in enumerate(sub_texts):
            doc_id = f"cur.{skill}.{sec_idx}.{sub_idx}"
            chunks.append(
                KnowledgeChunk(
                    doc_id=doc_id,
                    source_type="curriculum",
                    learner="",
                    skill=skill,
                    difficulty=inferred_diff,
                    task_type="",
                    title=doc_title,
                    section=sec_name,
                    content=chunk_text,
                    metadata={
                        "path": path.name,
                        "section_index": sec_idx,
                        "chunk_index": sub_idx,
                        "citation_label": f"{doc_title} — {sec_name}",
                    },
                    created=0.0,
                )
            )
    return chunks


def _summarize_v22_task_for_index(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    snippets = [str(s) for s in (task.get("snippets") or [])]
    line_count = sum(len(s.splitlines()) for s in snippets)
    snippet_preview = " | ".join(s.strip().replace("\n", " ; ")[:160] for s in snippets[:2])
    prompt = str(task.get("prompt") or "").strip()
    opt_count = len(task.get("options") or [])
    test_count = len(task.get("tests") or [])
    ref_sec = float(task.get("reference_seconds") or 30.0)
    content = (
        f"Skill {task['skill_id']} | Difficulty {task['difficulty']} | Format {task['task_type']} "
        f"| Target time {ref_sec:.0f}s | Prompt: {prompt}"
    )
    if snippet_preview:
        content += f" | Code: {snippet_preview}"

    meta = {
        "task_id": task["task_id"],
        "pool": str(task.get("pool") or "practice"),
        "reference_seconds": ref_sec,
        "snippet_count": len(snippets),
        "option_count": opt_count,
        "test_count": test_count,
        "line_count": line_count,
        "prompt": prompt,
        "snippets": snippets,
        "options": task.get("options") or [],
        "criteria": task.get("criteria") or [],
        "hints": task.get("hints") or [],
        "citation_label": f"v2.2 {task['task_id']} ({task['difficulty']} {task['task_type']})",
    }
    return content, meta


def _load_static_corpus_cache() -> dict[str, Any]:
    global _STATIC_CACHE
    with _STATIC_CACHE_LOCK:
        if _STATIC_CACHE is not None:
            return _STATIC_CACHE

        chunks: list[KnowledgeChunk] = []
        # 1. Curriculum Markdown corpus
        if KNOWLEDGE_DIR.is_dir():
            for md_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
                skill = md_path.stem
                if skill in catalogue_bridge.PPO_SKILLS_ORDER:
                    chunks.extend(_chunk_markdown_document(md_path, skill))

        # 2. v2.2 2,160-item exemplar corpus
        for task in catalogue_bridge._RAW_V22_TASKS:
            content, meta = _summarize_v22_task_for_index(task)
            chunks.append(
                KnowledgeChunk(
                    doc_id=f"ex.{task['task_id']}",
                    source_type="v22_exemplar",
                    learner="",
                    skill=str(task["skill_id"]),
                    difficulty=str(task["difficulty"]),
                    task_type=str(task["task_type"]),
                    title=str(task["task_id"]),
                    section=f"{task['difficulty']} · {task['task_type']}",
                    content=content,
                    metadata=meta,
                    created=0.0,
                )
            )

        texts = [c.searchable_text for c in chunks]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8192,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b",
        )
        matrix = vectorizer.fit_transform(texts) if texts else None
        corpus_sig = hashlib.sha256(
            f"{len(chunks)}:{len(catalogue_bridge._RAW_V22_TASKS)}".encode("utf-8")
        ).hexdigest()[:16]

        _STATIC_CACHE = {
            "chunks": chunks,
            "by_id": {c.doc_id: c for c in chunks},
            "idx_by_id": {c.doc_id: idx for idx, c in enumerate(chunks)},
            "vectorizer": vectorizer,
            "matrix": matrix,
            "signature": corpus_sig,
        }
        return _STATIC_CACHE


def _clean_citation_excerpt(text: str, max_chars: int = 210) -> str:
    """Turn Markdown chunk text into a clean plain-text preview stopping at sentence/word boundaries."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"^[-*•]\s+", "", s)
        s = re.sub(r"^\d+\.\s+", "", s)
        s = s.replace("**", "")
        lines.append(s)
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()
    if len(joined) <= max_chars:
        return joined
    clipped = joined[:max_chars]
    # Prefer ending cleanly at a sentence boundary
    for end_ch in (". ", "! ", "? "):
        pos = clipped.rfind(end_ch)
        if pos >= max_chars // 2:
            out = clipped[: pos + 1].strip()
            break
    else:
        sp = clipped.rfind(" ")
        out = (clipped[:sp] if sp > max_chars // 3 else clipped).rstrip(" ,;:-") + "…"
    # Never leave an odd number of backticks in the snippet
    if out.count("`") % 2 == 1:
        last_bt = out.rfind("`")
        if last_bt != -1:
            out = out[:last_bt].rstrip(" ,;:-")
    return out


def max_text_similarity(candidate_text: str, reference_texts: Iterable[str]) -> float:
    """Compute highest token-Jaccard similarity between candidate text and reference strings."""
    cand_tokens = {t.lower() for t in _tokenize_alnum(candidate_text) if len(t) >= 2}
    if not cand_tokens:
        return 0.0
    best = 0.0
    for ref in reference_texts:
        ref_tokens = {t.lower() for t in _tokenize_alnum(ref) if len(t) >= 2}
        if not ref_tokens:
            continue
        inter = len(cand_tokens & ref_tokens)
        union = len(cand_tokens | ref_tokens)
        if union:
            best = max(best, inter / union)
    return round(best, 4)


class HybridRAGEngine:
    """Hybrid FTS5 BM25 + TF-IDF Cosine + RRF (k=60) + Reranking over 3 adaptive corpora."""

    def __init__(self, store: AdaptiveStore):
        self.store = store
        self._lock = threading.RLock()
        self._cache = _load_static_corpus_cache()
        self._recent_audits: list[dict[str, Any]] = []
        self._audits_by_learner: dict[str, dict[str, Any]] = {}
        self._init_schema_and_seed()

    def _init_schema_and_seed(self) -> None:
        with self.store._lock:
            self.store.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks(
                    doc_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    learner TEXT NOT NULL DEFAULT '',
                    skill TEXT NOT NULL DEFAULT '',
                    difficulty TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    section TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS rag_chunks_filter
                    ON rag_chunks(source_type, learner, skill, difficulty, task_type);
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                    doc_id UNINDEXED,
                    source_type UNINDEXED,
                    learner UNINDEXED,
                    skill UNINDEXED,
                    difficulty UNINDEXED,
                    task_type UNINDEXED,
                    title,
                    section,
                    content,
                    tokenize='porter unicode61'
                );
                """
            )
            row = self.store.db.execute(
                "SELECT value FROM rag_meta WHERE key='static_signature'"
            ).fetchone()
            existing_static_count = self.store.db.execute(
                "SELECT COUNT(*) AS n FROM rag_chunks WHERE source_type IN ('curriculum', 'v22_exemplar')"
            ).fetchone()["n"]
            sig = self._cache["signature"]
            if row is not None and row["value"] == sig and existing_static_count >= len(self._cache["chunks"]):
                return

            with self.store.transaction():
                self.store.db.execute(
                    "DELETE FROM rag_chunks WHERE source_type IN ('curriculum', 'v22_exemplar')"
                )
                self.store.db.execute(
                    "DELETE FROM rag_chunks_fts WHERE source_type IN ('curriculum', 'v22_exemplar')"
                )
                chunk_rows = [
                    (
                        c.doc_id,
                        c.source_type,
                        c.learner,
                        c.skill,
                        c.difficulty,
                        c.task_type,
                        c.title,
                        c.section,
                        c.content,
                        json.dumps(c.metadata, ensure_ascii=False),
                        c.created,
                    )
                    for c in self._cache["chunks"]
                ]
                fts_rows = [
                    (
                        c.doc_id,
                        c.source_type,
                        c.learner,
                        c.skill,
                        c.difficulty,
                        c.task_type,
                        c.title,
                        c.section,
                        c.content,
                    )
                    for c in self._cache["chunks"]
                ]
                self.store.db.executemany(
                    """
                    INSERT INTO rag_chunks(
                        doc_id, source_type, learner, skill, difficulty, task_type,
                        title, section, content, metadata_json, created
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
                self.store.db.executemany(
                    """
                    INSERT INTO rag_chunks_fts(
                        doc_id, source_type, learner, skill, difficulty, task_type,
                        title, section, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )
                self.store.db.execute(
                    "INSERT INTO rag_meta(key, value) VALUES ('static_signature', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (sig,),
                )

    def stats(self, learner: str | None = None) -> dict[str, int]:
        with self.store._lock:
            rows = self.store.db.execute(
                "SELECT source_type, COUNT(*) AS n FROM rag_chunks GROUP BY source_type"
            ).fetchall()
            counts = {r["source_type"]: int(r["n"]) for r in rows}
            learner_ep = 0
            if learner:
                learner_ep = int(
                    self.store.db.execute(
                        "SELECT COUNT(*) AS n FROM rag_chunks WHERE source_type='episodic' AND learner=?",
                        (learner,),
                    ).fetchone()["n"]
                )
        return {
            "curriculum_chunks": counts.get("curriculum", 0),
            "v22_exemplars": counts.get("v22_exemplar", 0),
            "episodic_total": counts.get("episodic", 0),
            "episodic_learner": learner_ep,
            "total_chunks": sum(counts.values()),
        }

    def ingest_markdown(
        self,
        *,
        skill: str,
        title: str,
        markdown_text: str,
        doc_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest custom instructor Markdown documentation into SQLite FTS5 + vector retrieval."""
        if skill not in catalogue_bridge.PPO_SKILLS_ORDER:
            raise ValueError(f"Unknown skill {skill!r}")
        cleaned = str(markdown_text or "").strip()
        if not cleaned:
            raise ValueError("markdown_text must be non-empty.")

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(doc_key or title or "doc").strip()).strip("-")[:40] or "doc"
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
        # Split on Markdown headings
        paragraphs = [p.strip() for p in re.split(r"\n(?=#{1,3}\s+)", cleaned) if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        saved: list[dict[str, Any]] = []
        now = time.time()
        with self.store._lock, self.store.transaction():
            for idx, sec_text in enumerate(paragraphs):
                first_line = sec_text.splitlines()[0].lstrip("# ").strip() if sec_text.splitlines() else title
                section = first_line[:80] if sec_text.lstrip().startswith("#") else title[:80]
                body = "\n".join(sec_text.splitlines()[1:]).strip() if sec_text.lstrip().startswith("#") else sec_text
                body = body or sec_text
                doc_id = f"inst.{skill}.{slug}.{digest}.{idx}"
                meta = {
                    "custom": True,
                    "citation_label": f"{title} — {section}",
                }
                self.store.db.execute(
                    """
                    INSERT INTO rag_chunks(
                        doc_id, source_type, learner, skill, difficulty, task_type,
                        title, section, content, metadata_json, created
                    ) VALUES (?, 'curriculum', '', ?, 'mixed', '', ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        title=excluded.title, section=excluded.section,
                        content=excluded.content, metadata_json=excluded.metadata_json, created=excluded.created
                    """,
                    (doc_id, skill, title, section, body, json.dumps(meta, ensure_ascii=False), now),
                )
                self.store.db.execute("DELETE FROM rag_chunks_fts WHERE doc_id=?", (doc_id,))
                self.store.db.execute(
                    """
                    INSERT INTO rag_chunks_fts(
                        doc_id, source_type, learner, skill, difficulty, task_type,
                        title, section, content
                    ) VALUES (?, 'curriculum', '', ?, 'mixed', '', ?, ?, ?)
                    """,
                    (doc_id, skill, title, section, body),
                )
                saved.append({"doc_id": doc_id, "title": title, "section": section, "skill": skill})
        return saved

    def record_episodic(
        self,
        *,
        learner: str,
        skill: str,
        event_kind: str,
        summary: str,
        task_id: str = "",
        difficulty: str = "",
        task_type: str = "",
        metadata: dict[str, Any] | None = None,
        created: float | None = None,
    ) -> dict[str, Any] | None:
        """Store a learner-scoped episodic mistake/reveal/doubt document for future grounded feedback."""
        if not learner or skill not in catalogue_bridge.PPO_SKILLS_ORDER:
            return None
        cleaned = str(summary or "").strip()
        if not cleaned:
            return None

        ts = float(time.time() if created is None else created)
        raw_key = f"{learner}:{skill}:{event_kind}:{task_id}:{cleaned}:{ts:.6f}"
        doc_id = f"ep.{learner}.{skill}.{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:12]}"
        skill_name = catalogue_bridge.SKILL_META[skill]["name"]
        title = f"{skill_name} ({event_kind})"
        section = f"Learner History · {event_kind}"
        meta = dict(metadata or {})
        meta.setdefault("event_kind", event_kind)
        meta.setdefault("task_id", task_id)
        meta.setdefault("citation_label", f"Your prior {event_kind} on {skill_name}")

        with self.store._lock, self.store.transaction():
            self.store.db.execute(
                """
                INSERT INTO rag_chunks(
                    doc_id, source_type, learner, skill, difficulty, task_type,
                    title, section, content, metadata_json, created
                ) VALUES (?, 'episodic', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    content=excluded.content, metadata_json=excluded.metadata_json, created=excluded.created
                """,
                (
                    doc_id,
                    learner,
                    skill,
                    str(difficulty or ""),
                    str(task_type or ""),
                    title,
                    section,
                    cleaned,
                    json.dumps(meta, ensure_ascii=False),
                    ts,
                ),
            )
            self.store.db.execute("DELETE FROM rag_chunks_fts WHERE doc_id=?", (doc_id,))
            self.store.db.execute(
                """
                INSERT INTO rag_chunks_fts(
                    doc_id, source_type, learner, skill, difficulty, task_type,
                    title, section, content
                ) VALUES (?, 'episodic', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    learner,
                    skill,
                    str(difficulty or ""),
                    str(task_type or ""),
                    title,
                    section,
                    cleaned,
                ),
            )
        return {
            "doc_id": doc_id,
            "learner": learner,
            "skill": skill,
            "event_kind": event_kind,
            "created": ts,
        }

    def rewrite_query(
        self,
        query: str,
        *,
        skill: str | None = None,
        difficulty: str | None = None,
        task_type: str | None = None,
        extra_context: str = "",
    ) -> str:
        """Expand short/vague learner or prompt queries with domain & skill keywords."""
        raw = str(query or "").strip()
        expansions: list[str] = [raw] if raw else []
        if skill and skill in catalogue_bridge.SKILL_META:
            meta = catalogue_bridge.SKILL_META[skill]
            expansions.append(f"{meta['name']} {meta['summary']}")
        if difficulty and difficulty != "mixed":
            expansions.append(difficulty)
        if task_type and task_type != "mixed":
            expansions.append(task_type)

        low = raw.lower()
        for trigger, synonym_text in _PYTHON_SYNONYMS.items():
            if trigger in low:
                expansions.append(synonym_text)

        if extra_context:
            extra_tokens = _tokenize_alnum(extra_context)[:20]
            if extra_tokens:
                expansions.append(" ".join(extra_tokens))

        return " ".join(p for p in expansions if p).strip()

    def _allowed_skills(self, skill: str | None, *, include_prerequisites: bool) -> list[str] | None:
        if not skill or skill not in catalogue_bridge.PPO_SKILLS_ORDER:
            return None
        skills = [skill]
        if include_prerequisites:
            for pre in catalogue_bridge.PREREQUISITES.get(skill, ()):
                if pre not in skills:
                    skills.append(pre)
        return skills

    def _passes_metadata_filter(
        self,
        *,
        doc_source: str,
        doc_learner: str,
        doc_skill: str,
        doc_diff: str,
        doc_fmt: str,
        source_set: set[str] | None,
        learner: str | None,
        allowed_skills: list[str] | None,
        difficulty: str | None,
        task_type: str | None,
        exclude_doc_ids: set[str],
        doc_id: str,
    ) -> bool:
        if doc_id in exclude_doc_ids:
            return False
        if source_set is not None and doc_source not in source_set:
            return False
        if doc_source == "episodic":
            if not learner or doc_learner != learner:
                return False
        if allowed_skills is not None and doc_skill not in allowed_skills:
            return False
        if difficulty and difficulty != "mixed":
            if doc_source == "v22_exemplar" and doc_diff != difficulty:
                return False
        if task_type and task_type != "mixed":
            if doc_source == "v22_exemplar" and doc_fmt != task_type:
                return False
        return True

    def retrieve(
        self,
        query: str,
        *,
        purpose: str = "general",
        skill: str | None = None,
        learner: str | None = None,
        difficulty: str | None = None,
        task_type: str | None = None,
        source_types: Iterable[str] | None = ("curriculum",),
        include_prerequisites: bool = False,
        mastery_tier: str | None = None,
        extra_context: str = "",
        top_k: int = 4,
        exclude_task_ids: Iterable[str] = (),
        record_audit: bool = True,
    ) -> dict[str, Any]:
        """Run hybrid FTS5 BM25 + TF-IDF Cosine retrieval with RRF (k=60) and metadata reranking."""
        t0 = time.perf_counter()
        top_k = max(1, min(int(top_k), 20))
        rewritten = self.rewrite_query(
            query,
            skill=skill,
            difficulty=difficulty,
            task_type=task_type,
            extra_context=extra_context,
        )
        source_list = tuple(source_types) if source_types is not None else ("curriculum", "v22_exemplar", "episodic")
        source_set = set(source_list)
        allowed_skills = self._allowed_skills(skill, include_prerequisites=include_prerequisites)
        exclude_doc_ids = {f"ex.{tid}" if not str(tid).startswith("ex.") else str(tid) for tid in exclude_task_ids}

        # 1. SQLite FTS5 BM25 retrieval
        fts_ranks: dict[str, int] = {}
        fts_docs: dict[str, dict[str, Any]] = {}
        fts_query = _build_safe_fts_query(rewritten)
        if fts_query:
            sql_parts = [
                "SELECT c.doc_id, c.source_type, c.learner, c.skill, c.difficulty, c.task_type, "
                "c.title, c.section, c.content, c.metadata_json, c.created, "
                "bm25(rag_chunks_fts, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.0) AS rank_score "
                "FROM rag_chunks_fts f JOIN rag_chunks c ON c.doc_id = f.doc_id "
                "WHERE rag_chunks_fts MATCH ?"
            ]
            params: list[Any] = [fts_query]
            placeholders_src = ",".join("?" for _ in source_list)
            sql_parts.append(f"AND c.source_type IN ({placeholders_src})")
            params.extend(source_list)
            if allowed_skills is not None:
                placeholders_sk = ",".join("?" for _ in allowed_skills)
                sql_parts.append(f"AND c.skill IN ({placeholders_sk})")
                params.extend(allowed_skills)
            if difficulty and difficulty != "mixed" and "v22_exemplar" in source_set:
                sql_parts.append("AND (c.source_type != 'v22_exemplar' OR c.difficulty = ?)")
                params.append(difficulty)
            if task_type and task_type != "mixed" and "v22_exemplar" in source_set:
                sql_parts.append("AND (c.source_type != 'v22_exemplar' OR c.task_type = ?)")
                params.append(task_type)
            if "episodic" in source_set:
                if learner:
                    sql_parts.append("AND (c.source_type != 'episodic' OR c.learner = ?)")
                    params.append(learner)
                else:
                    sql_parts.append("AND c.source_type != 'episodic'")
            sql_parts.append("ORDER BY rank_score ASC LIMIT ?")
            params.append(max(48, top_k * 12))

            with self.store._lock:
                rows = self.store.db.execute(" ".join(sql_parts), params).fetchall()
            rank = 1
            for r in rows:
                doc_id = r["doc_id"]
                if not self._passes_metadata_filter(
                    doc_source=r["source_type"],
                    doc_learner=r["learner"],
                    doc_skill=r["skill"],
                    doc_diff=r["difficulty"],
                    doc_fmt=r["task_type"],
                    source_set=source_set,
                    learner=learner,
                    allowed_skills=allowed_skills,
                    difficulty=difficulty,
                    task_type=task_type,
                    exclude_doc_ids=exclude_doc_ids,
                    doc_id=doc_id,
                ):
                    continue
                fts_ranks[doc_id] = rank
                rank += 1
                fts_docs[doc_id] = {
                    "doc_id": doc_id,
                    "source_type": r["source_type"],
                    "learner": r["learner"],
                    "skill": r["skill"],
                    "difficulty": r["difficulty"],
                    "task_type": r["task_type"],
                    "title": r["title"],
                    "section": r["section"],
                    "content": r["content"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created": float(r["created"]),
                }

        # 2. Vector TF-IDF Cosine retrieval (static matrix + dynamic custom/episodic rows)
        vec_ranks: dict[str, int] = {}
        vec_docs: dict[str, dict[str, Any]] = {}
        vectorizer: TfidfVectorizer = self._cache["vectorizer"]
        static_mat = self._cache["matrix"]
        static_chunks: list[KnowledgeChunk] = self._cache["chunks"]
        q_vec = vectorizer.transform([rewritten or query or skill or "python"])

        scored_vec: list[tuple[float, str, dict[str, Any]]] = []
        if static_mat is not None and static_chunks:
            sims = (static_mat @ q_vec.T).toarray().ravel()
            for idx, score in enumerate(sims):
                ch = static_chunks[idx]
                if not self._passes_metadata_filter(
                    doc_source=ch.source_type,
                    doc_learner=ch.learner,
                    doc_skill=ch.skill,
                    doc_diff=ch.difficulty,
                    doc_fmt=ch.task_type,
                    source_set=source_set,
                    learner=learner,
                    allowed_skills=allowed_skills,
                    difficulty=difficulty,
                    task_type=task_type,
                    exclude_doc_ids=exclude_doc_ids,
                    doc_id=ch.doc_id,
                ):
                    continue
                if score > 0.0 or (ch.source_type == "curriculum" and ch.skill == skill):
                    scored_vec.append(
                        (
                            float(score),
                            ch.doc_id,
                            {
                                "doc_id": ch.doc_id,
                                "source_type": ch.source_type,
                                "learner": ch.learner,
                                "skill": ch.skill,
                                "difficulty": ch.difficulty,
                                "task_type": ch.task_type,
                                "title": ch.title,
                                "section": ch.section,
                                "content": ch.content,
                                "metadata": dict(ch.metadata),
                                "created": ch.created,
                            },
                        )
                    )

        # Also score dynamic rows (episodic for learner + custom instructor curriculum docs)
        dyn_rows = []
        if "episodic" in source_set and learner:
            with self.store._lock:
                dyn_rows.extend(
                    self.store.db.execute(
                        "SELECT doc_id, source_type, learner, skill, difficulty, task_type, "
                        "title, section, content, metadata_json, created "
                        "FROM rag_chunks WHERE source_type='episodic' AND learner=?",
                        (learner,),
                    ).fetchall()
                )
        if "curriculum" in source_set:
            with self.store._lock:
                dyn_rows.extend(
                    self.store.db.execute(
                        "SELECT doc_id, source_type, learner, skill, difficulty, task_type, "
                        "title, section, content, metadata_json, created "
                        "FROM rag_chunks WHERE source_type='curriculum' AND doc_id LIKE 'inst.%'"
                    ).fetchall()
                )
        if dyn_rows:
            dyn_texts = [
                f"{r['skill']} {r['difficulty']} {r['task_type']} {r['title']} {r['section']} {r['content']}"
                for r in dyn_rows
            ]
            dyn_mat = vectorizer.transform(dyn_texts)
            dyn_sims = (dyn_mat @ q_vec.T).toarray().ravel()
            for r, score in zip(dyn_rows, dyn_sims):
                if not self._passes_metadata_filter(
                    doc_source=r["source_type"],
                    doc_learner=r["learner"],
                    doc_skill=r["skill"],
                    doc_diff=r["difficulty"],
                    doc_fmt=r["task_type"],
                    source_set=source_set,
                    learner=learner,
                    allowed_skills=allowed_skills,
                    difficulty=difficulty,
                    task_type=task_type,
                    exclude_doc_ids=exclude_doc_ids,
                    doc_id=r["doc_id"],
                ):
                    continue
                scored_vec.append(
                    (
                        float(score) + (0.015 if r["source_type"] == "episodic" else 0.01),
                        r["doc_id"],
                        {
                            "doc_id": r["doc_id"],
                            "source_type": r["source_type"],
                            "learner": r["learner"],
                            "skill": r["skill"],
                            "difficulty": r["difficulty"],
                            "task_type": r["task_type"],
                            "title": r["title"],
                            "section": r["section"],
                            "content": r["content"],
                            "metadata": json.loads(r["metadata_json"] or "{}"),
                            "created": float(r["created"]),
                        },
                    )
                )

        scored_vec.sort(key=lambda item: item[0], reverse=True)
        for rank_idx, (sim_score, doc_id, doc_obj) in enumerate(scored_vec[: max(24, top_k * 8)], start=1):
            vec_ranks[doc_id] = rank_idx
            doc_obj["vector_score"] = round(sim_score, 4)
            vec_docs[doc_id] = doc_obj

        # 3. Reciprocal Rank Fusion (RRF k=60) + Metadata Pedagogical Reranking
        orig_tokens = {t.lower() for t in _tokenize_alnum(query) if len(t) >= 3}
        all_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
        fused: list[dict[str, Any]] = []
        for doc_id in all_ids:
            doc = dict(vec_docs.get(doc_id) or fts_docs[doc_id])
            rrf = 0.0
            if doc_id in fts_ranks:
                rrf += 1.0 / (RRF_K + fts_ranks[doc_id])
            if doc_id in vec_ranks:
                rrf += 1.0 / (RRF_K + vec_ranks[doc_id])

            rerank_boost = 0.0
            if skill and doc["skill"] == skill:
                rerank_boost += 0.012
            if difficulty and difficulty != "mixed" and doc["difficulty"] == difficulty:
                rerank_boost += 0.008
            if task_type and task_type != "mixed" and doc["task_type"] == task_type:
                rerank_boost += 0.006
            if orig_tokens:
                doc_tokens = {
                    t.lower()
                    for t in _tokenize_alnum(f"{doc['title']} {doc['section']} {doc['content']}")
                }
                if doc_tokens:
                    rerank_boost += 0.030 * (len(orig_tokens & doc_tokens) / len(orig_tokens))

            sec_low = doc["section"].lower()
            if mastery_tier in ("lacking", "unassessed"):
                if any(w in sec_low for w in ("core", "misconception", "pitfall", "learner history", "prereq")):
                    rerank_boost += 0.010
                if doc["source_type"] == "episodic":
                    rerank_boost += 0.009
            elif mastery_tier == "mastered":
                if any(w in sec_low for w in ("hard", "difficulty", "progression", "edge")):
                    rerank_boost += 0.008

            doc["fts_rank"] = fts_ranks.get(doc_id)
            doc["vector_rank"] = vec_ranks.get(doc_id)
            doc["rrf_score"] = round(rrf, 6)
            doc["final_score"] = round(rrf + rerank_boost, 6)
            fused.append(doc)

        fused.sort(key=lambda d: (d["final_score"], d["rrf_score"]), reverse=True)
        selected = fused[:top_k]

        # Build safe citation objects (never leak task answer_key or raw keys)
        citations: list[dict[str, Any]] = []
        for item in selected:
            meta = item.get("metadata") or {}
            if item["source_type"] == "episodic":
                ev_kind = str(meta.get("event_kind") or "history").replace("_", " ")
                short_label = f"Prior {ev_kind}"
                raw_excerpt = str(item.get("content") or "")
            elif item["source_type"] == "v22_exemplar":
                short_label = f"v2.2 {item['difficulty']} {item['task_type'].replace('_', ' ')}"
                raw_excerpt = str(meta.get("prompt") or "")
            else:
                short_label = item["section"] or item["title"]
                raw_excerpt = str(item.get("content") or "")
            citations.append(
                {
                    "doc_id": item["doc_id"],
                    "source_type": item["source_type"],
                    "skill": item["skill"],
                    "title": item["title"],
                    "section": item["section"],
                    "label": meta.get("citation_label") or f"{item['title']} — {item['section']}",
                    "short_label": short_label,
                    "excerpt": _clean_citation_excerpt(raw_excerpt),
                    "score": item["final_score"],
                }
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        audit_entry = {
            "purpose": purpose,
            "skill": skill,
            "learner": learner,
            "difficulty": difficulty,
            "task_type": task_type,
            "mastery_tier": mastery_tier,
            "original_query": str(query or "")[:200],
            "rewritten_query": rewritten[:280],
            "source_types": list(source_list),
            "fts_hits": len(fts_ranks),
            "vector_hits": len(vec_ranks),
            "fused_candidates": len(all_ids),
            "returned": len(selected),
            "citations": citations,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        if record_audit:
            with self._lock:
                self._recent_audits.append(audit_entry)
                if len(self._recent_audits) > 25:
                    self._recent_audits = self._recent_audits[-25:]
                if learner:
                    self._audits_by_learner[learner] = audit_entry

        return {
            "chunks": selected,
            "citations": citations,
            "audit": audit_entry,
        }

    def retrieve_exemplars(
        self,
        *,
        skill: str,
        difficulty: str,
        task_format: str,
        query: str = "",
        exclude_task_ids: Iterable[str] = (),
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve strictly difficulty-locked v2.2 exemplars for few-shot question synthesis."""
        res = self.retrieve(
            query or f"{skill} {difficulty} {task_format} question pattern",
            purpose="question_exemplars",
            skill=skill,
            difficulty=difficulty,
            task_type=task_format,
            source_types=("v22_exemplar",),
            include_prerequisites=False,
            top_k=top_k,
            exclude_task_ids=exclude_task_ids,
        )
        exemplars: list[dict[str, Any]] = []
        for ch in res["chunks"]:
            meta = ch.get("metadata") or {}
            tid = meta.get("task_id")
            full_task = catalogue_bridge.get_v22_task(str(tid)) if tid else None
            if full_task is not None:
                exemplars.append(full_task)
        return {
            "exemplars": exemplars,
            "citations": res["citations"],
            "audit": res["audit"],
        }

    def get_latest_audit(self, learner: str | None = None) -> dict[str, Any]:
        with self._lock:
            entry = self._audits_by_learner.get(learner) if learner else None
            if entry is None and self._recent_audits:
                entry = self._recent_audits[-1]
            return {
                "corpus": self.stats(learner=learner),
                "last_retrieval": dict(entry) if entry is not None else None,
                "recent_count": len(self._recent_audits),
            }
