"""Modular orchestrator for building outputs/Adaptive_Tutoring_Experiments_and_Results.docx.

Preserves Chapters 5 and 6 100% untouched (verified against
3/report_sections/ch5_ch6_baseline_snapshot.json on every build).
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report_sections.style import (  # noqa: E402
    add_page_number_footer,
    add_placeholder_page,
    configure_page_geometry,
    set_arabic_footer,
    set_roman_footer,
)
from verify_ch5_ch6_preservation import DEFAULT_SNAPSHOT, verify_docx  # noqa: E402

EVAL_DIR = ROOT / "eval_runs" / "v2.2-preregistered"
FIG_DIR = EVAL_DIR / "figures"
ASSETS_DIR = ROOT / "report_sections" / "assets"
OUT_DOCX = REPO / "outputs" / "Adaptive_Tutoring_Experiments_and_Results.docx"

BODY_MODULES = [
    ("ch1_introduction", "Chapter 1 — Introduction"),
    ("ch2_literature_survey", "Chapter 2 — Literature Survey"),
    ("ch3_datasets_methodology", "Chapter 3 — Datasets and Methodology"),
    ("ch4_implementation", "Chapter 4 — Implementation"),
    ("ch5_evaluation", "Chapter 5 — Evaluation Protocol"),
    ("ch6_results", "Chapter 6 — Results and Performance Analysis"),
    ("ch7_conclusion", "Chapter 7 — Limitations and Conclusion"),
    ("references", "References"),
]


def load_context(page_map: dict | None = None) -> dict:
    return {
        "T1": json.loads((EVAL_DIR / "tier1_run.json").read_text(encoding="utf-8")),
        "T4": json.loads((EVAL_DIR / "tier4_run.json").read_text(encoding="utf-8")),
        "T2": json.loads((EVAL_DIR / "tier2_offline_diagnostics.json").read_text(encoding="utf-8")),
        "BASE_DIR": ROOT,
        "REPO_DIR": REPO,
        "EVAL_DIR": EVAL_DIR,
        "FIG_DIR": FIG_DIR,
        "ASSETS_DIR": ASSETS_DIR,
        "page_map": page_map or {},
    }


def _import_section(module_name: str):
    return importlib.import_module(f"report_sections.{module_name}")


def build_doc(
    out_path: Path = OUT_DOCX,
    *,
    only_check_ch56: bool = False,
    allow_missing: bool = False,
    verify: bool = True,
    page_map: dict | None = None,
) -> Path:
    ctx = load_context(page_map=page_map)
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    sec0 = doc.sections[0]
    configure_page_geometry(sec0)

    if only_check_ch56:
        sec0.different_first_page_header_footer = True
        add_page_number_footer(sec0)
        _import_section("ch5_evaluation").build(doc, ctx)
        _import_section("ch6_results").build(doc, ctx)
        add_placeholder_page(doc, "Chapter 7 — Limitations and Conclusion", "[Smoke-check placeholder]")
    else:
        set_roman_footer(sec0, hide_first_page=True)
        ctx["manage_sections_externally"] = True
        try:
            _import_section("front_matter").build(doc, ctx)
        except ModuleNotFoundError as exc:
            if not allow_missing:
                raise RuntimeError(
                    "Missing section module 'report_sections.front_matter'. "
                    "Use --only-check-ch56 or --allow-missing for partial builds."
                ) from exc
            print(f"[warn] Missing front_matter ({exc}); inserting placeholder.", file=sys.stderr)
            add_placeholder_page(doc, "Front Matter", "[Placeholder — Front matter pending.]", page_break=False)

        body_sec = doc.add_section(WD_SECTION.NEW_PAGE)
        configure_page_geometry(body_sec)
        set_arabic_footer(body_sec, restart_at_1=True)
        ctx["arabic_section_started"] = True

        for mod_name, title in BODY_MODULES:
            try:
                mod = _import_section(mod_name)
            except ModuleNotFoundError as exc:
                if not allow_missing or mod_name in ("ch5_evaluation", "ch6_results"):
                    raise RuntimeError(
                        f"Missing section module 'report_sections.{mod_name}'. "
                        "Use --only-check-ch56 or --allow-missing for partial builds."
                    ) from exc
                print(f"[warn] Missing {mod_name} ({exc}); inserting placeholder.", file=sys.stderr)
                add_placeholder_page(doc, title, f"[Placeholder — {title} pending.]", page_break=(mod_name != "ch1_introduction"))
                continue
            mod.build(doc, ctx)

        if len(doc.sections) != 2:
            raise RuntimeError(f"Expected exactly 2 sections (Front Matter + Body), got {len(doc.sections)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"Saved Word report -> {out_path}")

    if verify:
        errors = verify_docx(out_path, DEFAULT_SNAPSHOT)
        if errors:
            for err in errors[:20]:
                print(f"[PRESERVATION FAIL] {err}", file=sys.stderr)
            raise SystemExit(1)
        print("Verified Chapters 5 & 6 match baseline snapshot 100%.")

    return out_path


_FOOTER_PAGE_RE = re.compile(r"^Page\s+([ivxlcdm]+|\d+)$")


def _int_to_roman(num: int) -> str:
    pairs = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    res = []
    n = num
    for val, sym in pairs:
        while n >= val:
            res.append(sym)
            n -= val
    return "".join(res)


def _export_docx_to_pdf_word(docx_path: Path, pdf_path: Path, timeout_sec: int = 90) -> None:
    """Export a .docx file to PDF using Microsoft Word on macOS."""
    docx_str = str(docx_path.absolute())
    pdf_str = str(pdf_path.absolute())
    docx_name = docx_path.name
    applescript = """\
on run argv
    set docxPosix to item 1 of argv
    set pdfPosix to item 2 of argv
    set docxName to item 3 of argv
    set docxFile to POSIX file docxPosix
    tell application "Microsoft Word"
        with timeout of 60 seconds
            open docxFile with read only
            repeat 40 times
                if exists (document docxName) then exit repeat
                delay 0.25
            end repeat
            if not (exists (document docxName)) then
                error "Timed out waiting for Word to open " & docxName
            end if
            set d to document docxName
            save as d file name pdfPosix file format format PDF
            close d saving do not save changes
        end timeout
    end tell
end run
"""
    scpt_path = pdf_path.parent / f"_word_export_{pdf_path.stem}.scpt"
    scpt_path.write_text(applescript, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["osascript", str(scpt_path), docx_str, pdf_str, docx_name],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    finally:
        scpt_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Microsoft Word PDF export failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Microsoft Word PDF export did not produce a valid file: {pdf_path}")


def _extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Read a PDF via pypdf and extract footer labels plus normalized non-footer text."""
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path), strict=False)
    pages: list[dict] = []
    for idx, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        label = None
        non_footer_lines = list(lines)
        if lines:
            m_first = _FOOTER_PAGE_RE.fullmatch(lines[0])
            m_last = _FOOTER_PAGE_RE.fullmatch(lines[-1])
            if m_first:
                label = m_first.group(1)
                non_footer_lines = lines[1:]
            elif m_last:
                label = m_last.group(1)
                non_footer_lines = lines[:-1]
        non_footer_text = "\n".join(non_footer_lines)
        norm_lines = [re.sub(r"\s+", " ", ln).strip() for ln in non_footer_lines]
        norm_text = re.sub(r"\s+", " ", non_footer_text).strip()
        pages.append({
            "pdf_index": idx + 1,
            "label": label,
            "non_footer_lines": non_footer_lines,
            "norm_lines": norm_lines,
            "non_footer_text": non_footer_text,
            "norm_text": norm_text,
        })
    return pages


def _canon_for_match(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([\u2013\u2014])\s*", r"\1", collapsed)


def _extract_page_map_from_pdf(pdf_path: Path) -> tuple[dict[str, str], list[dict]]:
    """Extract exact page labels for all TOC, List of Tables, and List of Figures entries."""
    from report_sections.front_matter import (
        DEFAULT_FIGURE_ENTRIES,
        DEFAULT_TABLE_ENTRIES,
        DEFAULT_TOC_ENTRIES,
    )

    pages = _extract_pdf_pages(pdf_path)
    for p in pages:
        p["canon_text"] = _canon_for_match(p["norm_text"])
    arabic_indices = [i for i, p in enumerate(pages) if p["label"] == "1"]
    if not arabic_indices:
        raise RuntimeError("Could not locate Arabic Page 1 in exported PDF.")
    arabic_start_idx = arabic_indices[0]

    page_map: dict[str, str] = {}

    # 1. TOC entries: split into Front Matter (Roman) vs Body (Arabic)
    body_started = False
    front_cur = 1  # skip Title Page (index 0)
    body_cur = arabic_start_idx
    for _, key, title in DEFAULT_TOC_ENTRIES:
        norm_title = re.sub(r"\s+", " ", title).strip()
        canon_title = _canon_for_match(title)
        if not body_started and key.startswith("Chapter 1"):
            body_started = True

        matched_idx = None
        if not body_started:
            for idx in range(front_cur, arabic_start_idx):
                if any(ln == norm_title for ln in pages[idx]["norm_lines"]):
                    matched_idx = idx
                    front_cur = idx
                    break
        else:
            for idx in range(body_cur, len(pages)):
                if canon_title in pages[idx]["canon_text"]:
                    matched_idx = idx
                    body_cur = idx
                    break

        if matched_idx is None:
            raise RuntimeError(f"Failed to locate TOC entry in PDF: {title!r}")
        lbl = pages[matched_idx]["label"]
        if not lbl:
            raise RuntimeError(f"Matched TOC entry {title!r} on PDF page {matched_idx + 1} with missing footer label.")
        page_map[key] = lbl

    # 2. List of Tables entries (strictly inside Arabic body)
    tbl_cur = arabic_start_idx
    for key, caption in DEFAULT_TABLE_ENTRIES:
        canon_cap = _canon_for_match(caption)
        canon_prefix = _canon_for_match(f"{key} \u2014")
        matched_idx = None
        for idx in range(tbl_cur, len(pages)):
            ct = pages[idx]["canon_text"]
            if canon_cap in ct or canon_prefix in ct:
                matched_idx = idx
                tbl_cur = idx
                break
        if matched_idx is None:
            raise RuntimeError(f"Failed to locate table caption in PDF: {caption!r}")
        page_map[key] = pages[matched_idx]["label"]

    # 3. List of Figures entries (strictly inside Arabic body)
    fig_cur = arabic_start_idx
    for key, caption in DEFAULT_FIGURE_ENTRIES:
        canon_cap = _canon_for_match(caption)
        canon_prefix = _canon_for_match(f"{key} \u2014")
        matched_idx = None
        for idx in range(fig_cur, len(pages)):
            ct = pages[idx]["canon_text"]
            if canon_cap in ct or canon_prefix in ct:
                matched_idx = idx
                fig_cur = idx
                break
        if matched_idx is None:
            raise RuntimeError(f"Failed to locate figure caption in PDF: {caption!r}")
        page_map[key] = pages[matched_idx]["label"]

    return page_map, pages


def _verify_pdf_layout_and_page_map(pdf_path: Path, expected_map: dict[str, str]) -> dict[str, str]:
    """Verify 0 blank pages, Roman/Arabic footers, zero '-' placeholders, and 100% page-map match."""
    from report_sections.front_matter import (
        DEFAULT_FIGURE_ENTRIES,
        DEFAULT_TABLE_ENTRIES,
        DEFAULT_TOC_ENTRIES,
    )

    actual_map, pages = _extract_page_map_from_pdf(pdf_path)

    blank_pages = [p["pdf_index"] for p in pages if not p["norm_text"]]
    if blank_pages:
        raise RuntimeError(f"Blank pages detected in PDF: {blank_pages}")

    if pages[0]["label"] is not None:
        raise RuntimeError(f"Title page should have no footer, found: {pages[0]['label']!r}")

    arabic_start_idx = next(i for i, p in enumerate(pages) if p["label"] == "1")
    for idx in range(1, arabic_start_idx):
        exp_roman = _int_to_roman(idx + 1)
        if pages[idx]["label"] != exp_roman:
            raise RuntimeError(
                f"PDF p.{idx + 1} expected Roman footer {exp_roman!r}, got {pages[idx]['label']!r}"
            )

    if "Chapter 1 \u2014 Introduction" not in pages[arabic_start_idx]["norm_text"]:
        raise RuntimeError("Arabic Page 1 does not start with 'Chapter 1 — Introduction'.")

    for offset, idx in enumerate(range(arabic_start_idx, len(pages)), start=1):
        if pages[idx]["label"] != str(offset):
            raise RuntimeError(
                f"PDF p.{idx + 1} expected Arabic footer {str(offset)!r}, got {pages[idx]['label']!r}"
            )

    # Check TOC/LoT/LoF pages for '-' placeholders and Table 3.3 dot leaders
    toc_start_idx = next(
        i for i in range(1, arabic_start_idx)
        if any(ln == "Table of Contents" for ln in pages[i]["norm_lines"])
    )
    placeholder_hits = []
    table33_lines = []
    for idx in range(toc_start_idx, arabic_start_idx):
        for ln in pages[idx]["non_footer_lines"]:
            if re.search(r"(?:\.{2,}|\s)-\s*$", ln):
                placeholder_hits.append((idx + 1, ln))
            if ln.startswith("Table 3.3"):
                table33_lines.append((idx + 1, ln))

    if placeholder_hits:
        raise RuntimeError(f"Found '-' placeholder(s) in TOC/LoT/LoF: {placeholder_hits}")

    if not table33_lines or not any("...." in ln for _, ln in table33_lines):
        raise RuntimeError(f"Table 3.3 did not retain dot leaders cleanly in List of Tables: {table33_lines}")

    if actual_map != expected_map:
        diffs = {
            k: (expected_map.get(k), actual_map.get(k))
            for k in set(expected_map) | set(actual_map)
            if expected_map.get(k) != actual_map.get(k)
        }
        raise RuntimeError(f"Pass 2 page map drifted from Pass 1: {diffs}")

    print(
        f"Verified PDF layout: {len(pages)} pages, 0 blank pages, Title Page footer=None, "
        f"Front Matter ii..{_int_to_roman(arabic_start_idx)}, Body 1..{len(pages) - arabic_start_idx}, "
        f"{len(DEFAULT_TOC_ENTRIES)}/{len(DEFAULT_TOC_ENTRIES)} TOC + "
        f"{len(DEFAULT_TABLE_ENTRIES)}/{len(DEFAULT_TABLE_ENTRIES)} LoT + "
        f"{len(DEFAULT_FIGURE_ENTRIES)}/{len(DEFAULT_FIGURE_ENTRIES)} LoF exact matches, "
        f"Table 3.3 dot leaders intact."
    )
    return actual_map


def build_with_auto_page_map(
    out_path: Path = OUT_DOCX,
    *,
    allow_missing: bool = False,
    verify: bool = True,
) -> Path:
    """Two-pass build: Pass 1 -> Word PDF + pypdf page extraction -> Pass 2 + Word PDF verification."""
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory(prefix="snu_thesis_pagemap_") as tmp_str:
        tmpdir = Path(tmp_str)
        pass1_docx = tmpdir / "pass1.docx"
        pass1_pdf = tmpdir / "pass1.pdf"

        print("[Pass 1] Building draft .docx with placeholder page numbers...")
        build_doc(out_path=pass1_docx, allow_missing=allow_missing, verify=verify, page_map={})

        print("[Pass 1] Exporting draft .docx to PDF via Microsoft Word...")
        _export_docx_to_pdf_word(pass1_docx, pass1_pdf)

        page_map, pages1 = _extract_page_map_from_pdf(pass1_pdf)
        print(f"[Pass 1] Extracted {len(page_map)} verified page numbers across {len(pages1)} PDF pages.")

        # Build Pass 2 (and converge once if replacing '-' shifts any page)
        for attempt in range(1, 3):
            print(f"[Pass 2.{attempt}] Building final .docx with extracted page_map -> {out_path}")
            build_doc(out_path=out_path, allow_missing=allow_missing, verify=verify, page_map=page_map)

            pass2_docx = tmpdir / f"pass2_verify_{attempt}.docx"
            pass2_pdf = tmpdir / f"pass2_verify_{attempt}.pdf"
            shutil.copy2(out_path, pass2_docx)

            print(f"[Pass 2.{attempt}] Verifying final layout & page numbers via Microsoft Word PDF...")
            _export_docx_to_pdf_word(pass2_docx, pass2_pdf)
            try:
                _verify_pdf_layout_and_page_map(pass2_pdf, page_map)
                break
            except RuntimeError as exc:
                if attempt == 1 and "drifted" in str(exc):
                    print(f"[Pass 2.{attempt}] Page shift detected; re-syncing page_map for Pass 3: {exc}")
                    page_map, _ = _extract_page_map_from_pdf(pass2_pdf)
                    continue
                raise

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SNU B.Tech CSE thesis .docx report.")
    parser.add_argument("--out", default=str(OUT_DOCX), help="Output .docx path")
    parser.add_argument("--only-check-ch56", action="store_true", help="Build only Chapters 5-6 for smoke verification")
    parser.add_argument("--allow-missing", action="store_true", help="Insert placeholders for missing non-Ch5/6 modules")
    parser.add_argument("--no-verify", action="store_true", help="Skip Chapters 5-6 preservation check")
    parser.add_argument("--page-map-json", default=None, help="Optional JSON file mapping section/table/figure labels to page numbers")
    parser.add_argument("--auto-page-map", action="store_true", help="Run two-pass Word PDF + pypdf extraction to populate and verify TOC/LoT/LoF page numbers")
    args = parser.parse_args()

    if args.auto_page_map:
        if args.only_check_ch56:
            raise SystemExit("--auto-page-map cannot be combined with --only-check-ch56")
        build_with_auto_page_map(
            out_path=Path(args.out),
            allow_missing=args.allow_missing,
            verify=not args.no_verify,
        )
        return 0

    page_map = None
    if args.page_map_json:
        page_map = json.loads(Path(args.page_map_json).read_text(encoding="utf-8"))

    build_doc(
        out_path=Path(args.out),
        only_check_ch56=args.only_check_ch56,
        allow_missing=args.allow_missing,
        verify=not args.no_verify,
        page_map=page_map,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
