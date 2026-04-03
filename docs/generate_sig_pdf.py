#!/usr/bin/env python3
"""Generate Kappa-SIG paper PDF from markdown using reportlab."""
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

WIDTH, HEIGHT = letter
MARGIN = 0.85 * inch
styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(name='PaperTitle', parent=styles['Title'],
    fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=4,
    textColor=HexColor('#1a1a2e')))
styles.add(ParagraphStyle(name='Author', parent=styles['Normal'],
    fontSize=10, alignment=TA_CENTER, spaceAfter=2,
    textColor=HexColor('#444444')))
styles.add(ParagraphStyle(name='Status', parent=styles['Normal'],
    fontSize=8, alignment=TA_CENTER, spaceAfter=8,
    textColor=HexColor('#666666'), fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='AbstractBody', parent=styles['Normal'],
    fontSize=9, leading=12, alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18,
    spaceAfter=4, fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='SectionH1', parent=styles['Heading1'],
    fontSize=12, spaceBefore=12, spaceAfter=5,
    textColor=HexColor('#1a1a2e')))
styles.add(ParagraphStyle(name='SectionH2', parent=styles['Heading2'],
    fontSize=10.5, spaceBefore=8, spaceAfter=3,
    textColor=HexColor('#2d3436')))

styles.add(ParagraphStyle(name='Body', parent=styles['Normal'],
    fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle(name='BulletItem', parent=styles['Normal'],
    fontSize=9, leading=11.5, leftIndent=20, spaceAfter=2))
styles.add(ParagraphStyle(name='BoldPara', parent=styles['Normal'],
    fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=4))

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(HexColor('#999999'))
    canvas.drawCentredString(WIDTH/2, 0.45*inch,
        f"Ohio, D. (2026) - Kappa-SIG - Page {doc.page}")
    canvas.restoreState()

def safe(text):
    """Escape XML but preserve our tags."""
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Restore allowed tags
    for tag in ['b', '/b', 'i', '/i', 'sub', '/sub', 'super', '/super']:
        text = text.replace(f'&lt;{tag}&gt;', f'<{tag}>')
    return text

def bold_convert(text):
    """Convert **bold** to <b>bold</b>."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

def md_to_story(md_text):
    story = []
    lines = md_text.split('\n')
    i = 0
    in_abstract = False
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # Horizontal rules
        if line.strip() == '---':
            story.append(HRFlowable(width="100%", thickness=0.5,
                         color=HexColor('#cccccc'), spaceAfter=6, spaceBefore=6))
            i += 1; continue

        # H1
        if line.startswith('# ') and not line.startswith('## '):
            story.append(Paragraph(safe(line[2:].strip()), styles['PaperTitle']))
            i += 1; continue
        # H2
        if line.startswith('## '):
            text = line[3:].strip()
            if text == 'Abstract':
                in_abstract = True
                story.append(Spacer(1, 6))
                story.append(Paragraph('Abstract', styles['SectionH1']))
                i += 1; continue
            in_abstract = False
            story.append(Paragraph(safe(text), styles['SectionH1']))
            i += 1; continue
        # H3
        if line.startswith('### '):
            in_abstract = False
            story.append(Paragraph(safe(line[4:].strip()), styles['SectionH2']))
            i += 1; continue

        # Author / status lines
        if line.startswith('**David Ohio**'):
            story.append(Paragraph('David Ohio - Independent Researcher', styles['Author']))
            i += 1; continue
        if line.startswith('odavidohio@'):
            story.append(Paragraph('odavidohio@gmail.com', styles['Author']))
            i += 1; continue
        if any(line.startswith(p) for p in ['**Status:**', '**Extends:**',
               '**Implementation:**', '**License:**', '**Repository:**', '**Contact:**']):
            text = line.replace('**', '')
            story.append(Paragraph(safe(text), styles['Status']))
            i += 1; continue
        # Bullet points
        if line.startswith('- '):
            text = bold_convert(line[2:].strip())
            story.append(Paragraph(safe(text), styles['BulletItem']))
            i += 1; continue

        # Bold paragraphs (**Key:** ...)
        if line.startswith('**') and ':**' in line:
            text = bold_convert(line)
            style = styles['AbstractBody'] if in_abstract else styles['BoldPara']
            story.append(Paragraph(safe(text), style))
            i += 1; continue
        # Tables
        if line.strip().startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if '---' in tl: continue
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                rows.append(cells)
            if rows:
                t = Table(rows, repeatRows=1)
                t.setStyle(TableStyle([
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0,0), (-1,0), HexColor('#f0f0f0')),
                    ('GRID', (0,0), (-1,-1), 0.5, HexColor('#cccccc')),
                    ('TOPPADDING', (0,0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 4))
            continue

        # Numbered items
        m = re.match(r'^(\d+)\.\s+\*\*(.+?)\*\*\s*(.*)', line)
        if m:
            num, bold_part, rest = m.group(1), m.group(2), m.group(3)
            text = f"{num}. <b>{safe(bold_part)}</b> {safe(rest)}"
            story.append(Paragraph(text, styles['BoldPara']))
            i += 1; continue
        # Regular paragraph (collect multi-line)
        para_lines = [line]
        i += 1
        while i < len(lines):
            nl = lines[i].strip()
            if not nl or nl.startswith('#') or nl.startswith('---') or \
               nl.startswith('|') or nl.startswith('- ') or nl.startswith('**'):
                break
            para_lines.append(lines[i].rstrip())
            i += 1
        text = ' '.join(para_lines)
        text = bold_convert(text)
        style = styles['AbstractBody'] if in_abstract else styles['Body']
        story.append(Paragraph(safe(text), style))
    return story


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    md_path = Path(__file__).parent / "KAPPA_SIG.md"
    out_path = Path(__file__).parent / "KAPPA_SIG.pdf"
    print(f"Reading: {md_path}")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = md_to_story(md_text)
    doc.build(story, onFirstPage=add_page_number,
              onLaterPages=add_page_number)
    print(f"PDF generated: {out_path}")
    print(f"Pages: check file")
