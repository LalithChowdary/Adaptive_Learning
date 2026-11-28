"""Shared styling constants and helpers for the SNU B.Tech CSE thesis report.

Lines 25-214 from 3/build_experiments_docx.py are preserved verbatim with their
exact signatures and defaults, followed by additive helpers for front matter,
equations, code blocks, page geometry, and Roman/Arabic section footers.
"""
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

# Additive styling constants
FONT_BODY = "Times New Roman"
FONT_MONO = "Courier New"
COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_H1 = RGBColor(18, 42, 76)
COLOR_H2 = RGBColor(24, 56, 98)
COLOR_H3 = RGBColor(40, 48, 56)
COLOR_MUTED = RGBColor(100, 108, 116)
COLOR_PLACEHOLDER = RGBColor(110, 118, 128)


# =============================================================================
# VERBATIM HELPERS FROM 3/build_experiments_docx.py (lines 25-214)
# =============================================================================

def fmt_ms(entry, dec=4, pct=False):
    m = entry["mean"]
    sd = entry.get("sd")
    if pct:
        return f"{m * 100:.2f}%" if sd is None else f"{m * 100:.2f}% ± {sd * 100:.2f}%"
    return f"{m:.{dec}f}" if sd is None else f"{m:.{dec}f} ± {sd:.{dec}f}"


def fmt_pp_ci(entry):
    return f"{entry['delta'] * 100:+.2f} pp [{entry['lower_95'] * 100:+.2f}, {entry['upper_95'] * 100:+.2f}]"


def fmt_ll_ci(entry):
    return f"{entry['delta_log_loss']:+.5f} [{entry['lower_95']:+.5f}, {entry['upper_95']:+.5f}]"


def set_run_font(run, size=11, bold=False, italic=False, color=RGBColor(0, 0, 0)):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "Times New Roman")


def add_p(doc, text="", *, size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT,
          before=0, after=6, line_spacing=1.3, color=RGBColor(0, 0, 0), keep_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line_spacing
    pf.keep_with_next = keep_next
    if text:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold, italic=italic, color=color)
    return p


def add_mixed_p(doc, parts, *, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=6, line_spacing=1.3):
    """parts: list of (text, bold, italic)."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line_spacing
    for txt, b, it in parts:
        r = p.add_run(txt)
        set_run_font(r, size=size, bold=b, italic=it)
    return p


def add_bullet(doc, lead, body, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    r1 = p.add_run(lead)
    set_run_font(r1, size=size, bold=True)
    r2 = p.add_run(body)
    set_run_font(r2, size=size, bold=False)
    return p


def add_h1(doc, title, page_break=True):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    set_run_font(r, size=15, bold=True, color=RGBColor(18, 42, 76))
    return p


def add_h2(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    set_run_font(r, size=12.5, bold=True, color=RGBColor(24, 56, 98))
    return p


def add_h3(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    set_run_font(r, size=11, bold=True, italic=False, color=RGBColor(40, 48, 56))
    return p


def add_placeholder_page(doc, title, note="[Placeholder — This section will be written separately.]", page_break=True):
    add_h1(doc, title, page_break=page_break)
    add_p(doc, note, size=11, italic=True, color=RGBColor(110, 118, 128), before=6, after=12)


def style_table(table, col_widths=None, right_align_cols=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    right_align_cols = set(right_align_cols or ())

    # Set clean horizontal borders + light internal grid
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="2C3E50"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="2C3E50"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D5DCE4"/>'
        '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tbl_pr.append(borders)

    for r_idx, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if r_idx == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for c_idx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and c_idx < len(col_widths):
                cell.width = Inches(col_widths[c_idx])
            tc_pr = cell._tc.get_or_add_tcPr()
            # Padding
            mar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                '<w:top w:w="65" w:type="dxa"/><w:bottom w:w="65" w:type="dxa"/>'
                '<w:left w:w="90" w:type="dxa"/><w:right w:w="90" w:type="dxa"/>'
                '</w:tcMar>'
            )
            tc_pr.append(mar)
            if r_idx == 0:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E9EFF6"/>')
                tc_pr.append(shd)
            elif r_idx % 2 == 1:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="FAFBFD"/>')
                tc_pr.append(shd)

            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                if r_idx > 0 and c_idx in right_align_cols:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                elif r_idx == 0 and c_idx in right_align_cols:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for run in p.runs:
                    set_run_font(run, size=8.5 if len(row.cells) >= 7 else 9.5, bold=(r_idx == 0))


def add_table(doc, headers, rows, caption=None, col_widths=None, right_cols=None):
    if caption:
        add_p(doc, caption, size=9.5, bold=True, before=6, after=3, keep_next=True, color=RGBColor(40, 48, 56))
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    for j, h in enumerate(headers):
        tbl.cell(0, j).text = h
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            tbl.cell(i, j).text = str(val)
    style_table(tbl, col_widths=col_widths, right_align_cols=right_cols)
    add_p(doc, "", size=4, before=0, after=4)
    return tbl


def add_figure(doc, img_path, caption, width_in=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Inches(width_in))
    add_p(doc, caption, size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=8)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("Page ")
    set_run_font(r1, size=9, color=RGBColor(100, 108, 116))
    fld = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls("w"))
    p._p.append(fld)


# =============================================================================
# ADDITIVE HELPERS (FOR FRONT MATTER, CH 1-4, CH 7, REFERENCES)
# =============================================================================

def add_centered_p(doc, text="", *, size=11, bold=False, italic=False,
                   before=0, after=6, line_spacing=1.3, color=RGBColor(0, 0, 0), keep_next=False):
    return add_p(
        doc,
        text,
        size=size,
        bold=bold,
        italic=italic,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        before=before,
        after=after,
        line_spacing=line_spacing,
        color=color,
        keep_next=keep_next,
    )


def add_eq_block(doc, equation, label=None, *, size=10.5, before=4, after=6):
    """Render a centered equation paragraph with an optional right-side equation number."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.25
    p_pr = p._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F7F9FC"/>')
    p_pr.append(shd)
    r_eq = p.add_run(equation)
    set_run_font(r_eq, size=size, italic=True, color=RGBColor(18, 42, 76))
    if label:
        r_lbl = p.add_run(f"    ({label})")
        set_run_font(r_lbl, size=size - 0.5, bold=False, italic=False, color=RGBColor(80, 88, 98))
    return p


def add_code_block(doc, code, *, size=9.0, before=4, after=6):
    """Render a shaded monospace code or schema block."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.left_indent = Inches(0.2)
    pf.right_indent = Inches(0.2)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.15
    p_pr = p._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    bdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        '<w:left w:val="single" w:sz="12" w:space="8" w:color="183862"/>'
        '</w:pBdr>'
    )
    p_pr.append(shd)
    p_pr.append(bdr)
    r = p.add_run(code.rstrip("\n"))
    r.font.name = FONT_MONO
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(30, 36, 44)
    rpr = r._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), FONT_MONO)
    rfonts.set(qn("w:hAnsi"), FONT_MONO)
    rfonts.set(qn("w:eastAsia"), FONT_MONO)
    return p


def configure_page_geometry(section):
    """Configure US Letter (8.50 x 11.00 in) with 0.85 in margins on all sides."""
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    return section


def _set_pg_num_type(section, fmt, start=None):
    sect_pr = section._sectPr
    for old in sect_pr.findall(qn("w:pgNumType")):
        sect_pr.remove(old)
    attrs = f'w:fmt="{fmt}"'
    if start is not None:
        attrs += f' w:start="{int(start)}"'
    pg_num_type = parse_xml(f'<w:pgNumType {nsdecls("w")} {attrs}/>')
    sect_pr.append(pg_num_type)


def _clear_footer(footer):
    for p in list(footer.paragraphs)[1:]:
        p._element.getparent().remove(p._element)
    p0 = footer.paragraphs[0]
    for child in list(p0._p):
        if child.tag != qn("w:pPr"):
            p0._p.remove(child)
    return p0


def set_roman_footer(section, *, hide_first_page=True):
    """Set lowercase Roman page numbering (i, ii, ...) in the section footer."""
    configure_page_geometry(section)
    section.different_first_page_header_footer = hide_first_page
    footer = section.footer
    footer.is_linked_to_previous = False
    _clear_footer(footer)
    add_page_number_footer(section)
    _set_pg_num_type(section, "lowerRoman", start=1)
    return section


def set_arabic_footer(section, restart_at_1=True):
    """Set Arabic decimal page numbering (1, 2, ...) in the section footer."""
    configure_page_geometry(section)
    section.different_first_page_header_footer = False
    footer = section.footer
    footer.is_linked_to_previous = False
    _clear_footer(footer)
    add_page_number_footer(section)
    _set_pg_num_type(section, "decimal", start=1 if restart_at_1 else None)
    return section
