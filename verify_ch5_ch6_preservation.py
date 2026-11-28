"""Verify that Chapters 5 and 6 in a built .docx match the baseline snapshot 100%.

Usage:
    python3 3/verify_ch5_ch6_preservation.py --snapshot
    python3 3/verify_ch5_ch6_preservation.py [path/to/report.docx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_DOCX = REPO / "outputs" / "Adaptive_Tutoring_Experiments_and_Results.docx"
DEFAULT_SNAPSHOT = ROOT / "report_sections" / "ch5_ch6_baseline_snapshot.json"

CH5_TITLE = "Chapter 5 — Evaluation Protocol"
CH6_TITLE = "Chapter 6 — Results and Performance Analysis"
H1_COLOR = RGBColor(18, 42, 76)


def _is_h1(p, *, exact: str | None = None, prefix: str | None = None) -> bool:
    txt = p.text.strip()
    if exact is not None and txt != exact:
        return False
    if prefix is not None and not txt.startswith(prefix):
        return False
    for r in p.runs:
        if not r.text.strip():
            continue
        is_15pt = r.font.size == Pt(15)
        is_navy = r.font.color is not None and r.font.color.rgb == H1_COLOR
        if r.bold is True and (is_15pt or is_navy):
            return True
    return False


def _has_page_break(p) -> bool:
    if p.text.strip():
        return False
    brs = p._element.xpath('.//*[local-name()="br" and @*[local-name()="type"]="page"]')
    return len(brs) > 0


def _extract_run_info(run) -> dict[str, Any]:
    return {
        "text": run.text,
        "font_name": run.font.name,
        "font_size_pt": run.font.size.pt if run.font.size is not None else None,
        "bold": run.bold,
        "italic": run.italic,
        "color_rgb": str(run.font.color.rgb) if (run.font.color is not None and run.font.color.rgb is not None) else None,
    }


def extract_ch5_ch6(docx_path: Path) -> dict[str, Any]:
    doc = Document(str(docx_path))
    body = doc.element.body

    body_items: list[tuple[str, int, Any]] = []
    p_idx = 0
    t_idx = 0
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            body_items.append(("p", p_idx, doc.paragraphs[p_idx]))
            p_idx += 1
        elif tag == "tbl":
            body_items.append(("tbl", t_idx, doc.tables[t_idx]))
            t_idx += 1

    ch5_starts = [idx for idx, (kind, _, obj) in enumerate(body_items) if kind == "p" and _is_h1(obj, exact=CH5_TITLE)]
    if len(ch5_starts) != 1:
        raise RuntimeError(f"Expected exactly 1 H1 '{CH5_TITLE}' in {docx_path}, found {len(ch5_starts)}")
    start_pos = ch5_starts[0]

    end_pos = len(body_items)
    for idx in range(start_pos + 1, len(body_items)):
        kind, _, obj = body_items[idx]
        if kind == "p" and _is_h1(obj, prefix="Chapter 7"):
            end_pos = idx
            break

    # Exclude trailing page-break paragraph that belongs to Chapter 7's add_h1(..., page_break=True)
    if end_pos > start_pos and body_items[end_pos - 1][0] == "p" and _has_page_break(body_items[end_pos - 1][2]):
        end_pos -= 1

    span = body_items[start_pos:end_pos]
    ch6_found = any(kind == "p" and _is_h1(obj, exact=CH6_TITLE) for kind, _, obj in span)
    if not ch6_found:
        raise RuntimeError(f"H1 '{CH6_TITLE}' not found inside Chapter 5..7 span of {docx_path}")

    elements: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []

    span_p_list = [(idx, obj) for idx, (kind, _, obj) in enumerate(span) if kind == "p"]
    p_pos_by_span_idx = {span_idx: i for i, (span_idx, _) in enumerate(span_p_list)}

    for span_idx, (kind, _, obj) in enumerate(span):
        if kind == "p":
            local_p_idx = len(paragraphs)
            elements.append({"type": "p", "index": local_p_idx})
            paragraphs.append({
                "text": obj.text,
                "style": obj.style.name if obj.style is not None else None,
                "alignment": str(obj.alignment) if obj.alignment is not None else None,
                "runs": [_extract_run_info(r) for r in obj.runs],
            })
            blips = obj._element.xpath('.//*[local-name()="blip"]/@*[local-name()="embed"]')
            for rid in blips:
                part = doc.part.related_parts[rid]
                sha = hashlib.sha256(part.blob).hexdigest()
                caption = ""
                if local_p_idx + 1 < len(span_p_list):
                    caption = span_p_list[local_p_idx + 1][1].text
                figures.append({
                    "sha256": sha,
                    "size_bytes": len(part.blob),
                    "content_type": part.content_type,
                    "caption": caption,
                })
        elif kind == "tbl":
            local_t_idx = len(tables)
            elements.append({"type": "tbl", "index": local_t_idx})
            tables.append({
                "rows": len(obj.rows),
                "cols": len(obj.columns),
                "cells": [[cell.text for cell in row.cells] for row in obj.rows],
            })

    sections = [
        {
            "page_width_in": round(sec.page_width.inches, 4),
            "page_height_in": round(sec.page_height.inches, 4),
            "top_margin_in": round(sec.top_margin.inches, 4),
            "bottom_margin_in": round(sec.bottom_margin.inches, 4),
            "left_margin_in": round(sec.left_margin.inches, 4),
            "right_margin_in": round(sec.right_margin.inches, 4),
        }
        for sec in doc.sections
    ]

    return {
        "ch5_title": CH5_TITLE,
        "ch6_title": CH6_TITLE,
        "counts": {
            "elements": len(elements),
            "paragraphs": len(paragraphs),
            "tables": len(tables),
            "figures": len(figures),
        },
        "sections_geometry": sections,
        "elements": elements,
        "paragraphs": paragraphs,
        "tables": tables,
        "figures": figures,
    }


def verify_docx(docx_path: Path, snapshot_path: Path) -> list[str]:
    errors: list[str] = []
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    actual = extract_ch5_ch6(docx_path)

    # 1. Check section geometry across entire document (8.50 x 11.00 in, 0.85 in margins)
    for idx, sec in enumerate(actual["sections_geometry"]):
        for key, target in (
            ("page_width_in", 8.5),
            ("page_height_in", 11.0),
            ("top_margin_in", 0.85),
            ("bottom_margin_in", 0.85),
            ("left_margin_in", 0.85),
            ("right_margin_in", 0.85),
        ):
            if abs(sec[key] - target) > 0.005:
                errors.append(f"Section[{idx}] {key}={sec[key]} != expected {target}")

    # 2. Check counts and element ordering
    if actual["counts"] != expected["counts"]:
        errors.append(f"Counts mismatch: actual={actual['counts']} vs expected={expected['counts']}")
    if actual["elements"] != expected["elements"]:
        errors.append("Paragraph/table interleaving order differs from baseline snapshot.")

    # 3. Check paragraphs and runs
    n_p = min(len(actual["paragraphs"]), len(expected["paragraphs"]))
    for i in range(n_p):
        ap = actual["paragraphs"][i]
        ep = expected["paragraphs"][i]
        if ap["text"] != ep["text"]:
            errors.append(f"Paragraph[{i}] text mismatch:\n  actual:   {ap['text']!r}\n  expected: {ep['text']!r}")
        if ap["style"] != ep["style"]:
            errors.append(f"Paragraph[{i}] style {ap['style']!r} != {ep['style']!r}")
        if ap["alignment"] != ep["alignment"]:
            errors.append(f"Paragraph[{i}] alignment {ap['alignment']!r} != {ep['alignment']!r}")
        if ap["runs"] != ep["runs"]:
            errors.append(f"Paragraph[{i}] runs mismatch:\n  actual:   {ap['runs']}\n  expected: {ep['runs']}")

    # 4. Check all 12 tables
    n_t = min(len(actual["tables"]), len(expected["tables"]))
    for i in range(n_t):
        at = actual["tables"][i]
        et = expected["tables"][i]
        if (at["rows"], at["cols"]) != (et["rows"], et["cols"]):
            errors.append(f"Table[{i}] dimensions {(at['rows'], at['cols'])} != {(et['rows'], et['cols'])}")
            continue
        for r in range(et["rows"]):
            for c in range(et["cols"]):
                if at["cells"][r][c] != et["cells"][r][c]:
                    errors.append(
                        f"Table[{i}] cell({r},{c}) mismatch: {at['cells'][r][c]!r} != {et['cells'][r][c]!r}"
                    )

    # 5. Check all 5 figures
    n_f = min(len(actual["figures"]), len(expected["figures"]))
    for i in range(n_f):
        af = actual["figures"][i]
        ef = expected["figures"][i]
        if af != ef:
            errors.append(f"Figure[{i}] mismatch:\n  actual:   {af}\n  expected: {ef}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot or verify Chapters 5-6 preservation in .docx.")
    parser.add_argument("docx_path", nargs="?", default=str(DEFAULT_DOCX), help="Path to .docx file to check")
    parser.add_argument("--snapshot", action="store_true", help="Write baseline snapshot JSON from docx_path")
    parser.add_argument("--snapshot-file", default=str(DEFAULT_SNAPSHOT), help="Path to snapshot JSON")
    args = parser.parse_args()

    docx_path = Path(args.docx_path)
    snapshot_path = Path(args.snapshot_file)

    if args.snapshot:
        snap = extract_ch5_ch6(docx_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Wrote baseline snapshot -> {snapshot_path} "
            f"(paragraphs={snap['counts']['paragraphs']}, tables={snap['counts']['tables']}, figures={snap['counts']['figures']})"
        )
        return 0

    if not snapshot_path.exists():
        print(f"ERROR: Snapshot file not found: {snapshot_path}", file=sys.stderr)
        return 2

    errors = verify_docx(docx_path, snapshot_path)
    if errors:
        print(f"FAIL: {len(errors)} discrepancy(ies) detected in {docx_path}:", file=sys.stderr)
        for err in errors[:30]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more.", file=sys.stderr)
        return 1

    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    c = snap["counts"]
    print(
        f"PASS: Chapters 5 & 6 in {docx_path} match baseline snapshot 100% "
        f"({c['paragraphs']} paragraphs, {c['tables']} tables, {c['figures']} figures, US Letter 0.85in margins)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
