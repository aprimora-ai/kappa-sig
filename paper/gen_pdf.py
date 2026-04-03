#!/usr/bin/env python3
"""Generate PDF from the Obsessive Coherence paper markdown."""
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                 Table, TableStyle, KeepTogether)
from reportlab.lib import colors
from pathlib import Path

INPUT = Path(r"C:\Users\ohiod\Projects\kappa-sig\paper\obsessive_coherence.md")
OUTPUT = Path(r"C:\Users\ohiod\Projects\kappa-sig\paper\obsessive_coherence.pdf")

doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter,
                        topMargin=0.9*inch, bottomMargin=0.9*inch,
                        leftMargin=1*inch, rightMargin=1*inch)

# Styles
base = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle("title", parent=base["Title"], fontSize=18,
        spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold"),
    "author": ParagraphStyle("author", parent=base["Normal"], fontSize=11,
        alignment=TA_CENTER, spaceAfter=2, fontName="Helvetica"),
    "abstract_head": ParagraphStyle("abshead", parent=base["Normal"], fontSize=11,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6),
    "abstract": ParagraphStyle("abs", parent=base["Normal"], fontSize=9.5,
        fontName="Helvetica", leading=13, alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, spaceAfter=4),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14,
        fontName="Helvetica-Bold", spaceBefore=24, spaceAfter=10,
        textColor=HexColor("#1a1a2e")),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12,
        fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=8,
        textColor=HexColor("#16213e")),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5,
        fontName="Helvetica-BoldOblique", spaceBefore=12, spaceAfter=6),
    "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10,
        fontName="Helvetica", leading=13.5, alignment=TA_JUSTIFY,
        spaceBefore=3, spaceAfter=3),
    "keywords": ParagraphStyle("kw", parent=base["Normal"], fontSize=9,
        fontName="Helvetica-Oblique", alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, spaceAfter=12),
}

def esc(t):
    """Escape XML special chars for reportlab Paragraphs."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore bold/italic markdown
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'\*(.+?)\*', r'<i>\1</i>', t)
    # Inline code
    t = re.sub(r'`(.+?)`', r'<font face="Courier" size="9">\1</font>', t)
    return t

def parse_table(lines):
    """Parse markdown table lines into reportlab Table."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    # Remove separator row (---|---)
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return None
    # Build table
    tbl_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8edf2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11)
    data = [[Paragraph(esc(c), cell_style) for c in row] for row in rows]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(tbl_style)
    return tbl

def build_story():
    """Parse markdown and build reportlab story."""
    text = INPUT.read_text(encoding="utf-8")
    lines = text.split("\n")
    story = []
    i = 0
    in_abstract = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip horizontal rules
        if stripped == "---":
            i += 1; continue
        
        # Empty line
        if not stripped:
            i += 1; continue
        
        # Title (# )
        if stripped.startswith("# ") and not stripped.startswith("## "):
            story.append(Paragraph(esc(stripped[2:]), styles["title"]))
            i += 1; continue

        # Author line
        if stripped.startswith("**David Ohio**"):
            story.append(Spacer(1, 6))
            story.append(Paragraph("David Ohio", styles["author"]))
            i += 1; continue
        if stripped == "Independent Researcher":
            story.append(Paragraph("Independent Researcher", styles["author"]))
            i += 1; continue
        if stripped == "odavidohio@gmail.com":
            story.append(Paragraph("odavidohio@gmail.com", styles["author"]))
            story.append(Spacer(1, 12))
            i += 1; continue
        
        # Section headers
        if stripped.startswith("### "):
            story.append(Paragraph(esc(stripped[4:]), styles["h3"]))
            i += 1; continue
        if stripped.startswith("## "):
            sec = stripped[3:]
            if sec.strip().startswith("Abstract"):
                story.append(Paragraph("Abstract", styles["abstract_head"]))
                in_abstract = True
                i += 1; continue
            in_abstract = False
            # Page break before major sections (not before section 1)
            if any(sec.startswith(f"{n}.") for n in range(2, 20)):
                story.append(PageBreak())
            story.append(Paragraph(esc(sec), styles["h1"]))
            i += 1; continue

        # Tables
        if stripped.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            tbl = parse_table(tbl_lines)
            if tbl:
                story.append(Spacer(1, 6))
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue
        
        # Blockquotes (formulas)
        if stripped.startswith("> "):
            formula = stripped[2:]
            fstyle = ParagraphStyle("formula", parent=styles["body"],
                fontName="Courier", fontSize=9.5, leftIndent=36,
                spaceBefore=6, spaceAfter=6, textColor=HexColor("#333366"))
            story.append(Paragraph(esc(formula), fstyle))
            i += 1; continue
        
        # Numbered list items
        if re.match(r'^\d+\.\s', stripped):
            lstyle = ParagraphStyle("listitem", parent=styles["body"],
                leftIndent=24, firstLineIndent=-18, spaceBefore=4, spaceAfter=2)
            story.append(Paragraph(esc(stripped), lstyle))
            i += 1; continue

        # Keywords line
        if stripped.startswith("**Keywords:**"):
            kw = stripped.replace("**Keywords:**", "").strip()
            story.append(Paragraph(f"<b>Keywords:</b> {esc(kw)}", styles["keywords"]))
            in_abstract = False
            i += 1; continue
        
        # Submission/license footer lines
        if stripped.startswith("*Corresponding") or stripped.startswith("*Submitted") or stripped.startswith("*License"):
            fstyle = ParagraphStyle("footer", parent=styles["body"],
                fontSize=9, fontName="Helvetica-Oblique", alignment=TA_CENTER)
            story.append(Paragraph(esc(stripped.strip("*")), fstyle))
            i += 1; continue
        
        # Regular paragraph
        if in_abstract:
            story.append(Paragraph(esc(stripped), styles["abstract"]))
        else:
            story.append(Paragraph(esc(stripped), styles["body"]))
        i += 1
    
    return story

def add_page_number(canvas, doc):
    """Add page number and header to each page."""
    canvas.saveState()
    # Page number
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(letter[0]/2, 0.5*inch,
                             f"Page {doc.page}")
    # Header (from page 2)
    if doc.page > 1:
        canvas.setFont("Helvetica-Oblique", 7.5)
        canvas.setFillColor(HexColor("#666666"))
        canvas.drawString(1*inch, letter[1] - 0.55*inch,
                         "Ohio, D. (2026) — Obsessive Coherence: A Domain-Agnostic Structural Signature of Systemic Failure")
        canvas.setStrokeColor(HexColor("#cccccc"))
        canvas.line(1*inch, letter[1] - 0.6*inch, letter[0] - 1*inch, letter[1] - 0.6*inch)
    canvas.restoreState()

# Build
print("Building PDF...")
story = build_story()
print(f"  {len(story)} elements")
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"  PDF saved: {OUTPUT}")
print(f"  Size: {OUTPUT.stat().st_size / 1024:.0f} KB")
