import json, re, html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

root=Path(__file__).resolve().parents[2]
session=Path('C:/Users/jaero/.codex/sessions/2026/10/04/rollout-2026-10-04T12-58-56-01a10811-a4f5-77c0-859f-831234a238bf.jsonl')
texts=[]
for line in session.open(encoding='utf-8'):
    item=json.loads(line)
    p=item.get('payload',{})
    if item.get('type')=='response_item' and p.get('role')=='assistant' and p.get('type')=='message':
        t=''.join(c.get('text','') for c in p.get('content',[]))
        if t.startswith('**Conservaría este modelo') and len(t)>30000: texts.append(t)
assert texts, 'No se encontró la auditoría íntegra'
source=texts[-1]
(root/'tmp/pdfs/auditoria_original.md').write_text(source,encoding='utf-8')
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyAudit',fontName='Arial',fontSize=9.1,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name='HeadingAudit',fontName='Arial-Bold',fontSize=13,leading=17,spaceBefore=16,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#163B59')))
styles.add(ParagraphStyle(name='CellAudit',fontName='Arial',fontSize=7.2,leading=9.5,wordWrap='LTR'))
styles.add(ParagraphStyle(name='TitleAudit',fontName='Arial-Bold',fontSize=22,leading=27,spaceAfter=12,textColor=colors.HexColor('#163B59')))
def inline(t):
    t=re.sub(r'\[([^\]]+)\]\((?:<[^>]+>|[^)]+)\)',r'\1',t)
    t=t.replace('–','-').replace('—','-').replace('\u2011','-')
    t=html.escape(t)
    t=re.sub(r'`([^`]+)`',r'<font color="#174C6B">\1</font>',t)
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
    return t
def p(t,sty='BodyAudit'): return Paragraph(inline(t),styles[sty])
story=[p('Auditoría de modelos de Machine Learning','TitleAudit'),p('DATAFEST 2026 | BCP | Informe completo'),p('Fecha: 5 de octubre de 2026. Reproducción íntegra de la auditoría entregada en el chat. Se conserva el contenido; los enlaces locales se muestran mediante su nombre de referencia.'),Spacer(1,10)]
lines=source.splitlines(); i=0; tables=0
while i<len(lines):
    s=lines[i].strip()
    if not s or s=='---': i+=1; continue
    if s.startswith('```'):
        code=[]; i+=1
        while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]); i+=1
        for ln in code: story.append(p(ln))
        i+=1; continue
    if s.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',x) for x in row): rows.append(row)
            i+=1
        n=len(rows[0]); assert all(len(r)==n for r in rows)
        weights=[max(6,min(42,max(len(re.sub(r'[*`]', '',r[c])) for r in rows)))**.65 for c in range(n)]
        widths=[499*w/sum(weights) for w in weights]
        data=[[p(('**'+v+'**') if ri==0 else v,'CellAudit') for v in r] for ri,r in enumerate(rows)]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEAF1')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F9')]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#94AABB')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([tab,Spacer(1,9)]); tables+=1; continue
    if re.match(r'^\*\*\d+\.',s):
        m=re.match(r'^\*\*(.+?)\*\*\s*(.*)',s)
        story.append(p(m[1],'HeadingAudit'))
        if m[2]: story.append(p(m[2]))
    elif s.startswith('- '): story.append(p('• '+s[2:]))
    else: story.append(p(s))
    i+=1
out=root/'output/pdf/auditoria_completa_datafest_2026.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#CDD9E1')); c.line(48,38,547,38)
    c.setFont('Arial',8); c.setFillColor(colors.HexColor('#516775'))
    c.drawString(48,25,'DATAFEST 2026 | Auditoría completa | Desarrollo, no evaluación independiente')
    c.drawRightString(547,25,str(doc.page))
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=52,title='Auditoría completa DATAFEST 2026',author='Auditoría del proyecto DATAFEST')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(out)
text='\n'.join(pg.extract_text() for pg in reader.pages)
assert all(f'{i}.' in text for i in range(1,11))
print(json.dumps({'path':str(out),'pages':len(reader.pages),'source_characters':len(source),'tables':tables,'extracted_characters':len(text)},ensure_ascii=True))
