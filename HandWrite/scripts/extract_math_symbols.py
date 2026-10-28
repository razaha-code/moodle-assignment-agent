"""Extract the 2026-10-27 math sheet, preserving strokes and font-ready outlines.

This is a supplemental collection: it never overwrites the active glyph manifest.
Rows remain pending mapping confirmation until the writer identifies them.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from pathlib import Path

import fitz


# Page-space bands; samples are separated by horizontal gaps, not stroke count.
ROW_BANDS = ((60, 90), (120, 145), (155, 175), (210, 230), (240, 265),
             (290, 305), (320, 340), (355, 375), (385, 405), (413, 435), (455, 485))
EXPECTED_COUNTS = (8, 7, 7, 8, 9, 10, 10, 10, 10, 10, 10)
SUGGESTIONS = ('z or ζ', 'x or χ', 'α or 2', '− (long)', '+', '·', '− (short)', '=', ':', '%', '1 or |')
FLATNESS = 0.015  # points; vector approximation only, no raster tracing


def cubic_points(a, b, c, d, depth=0):
    dx, dy = d.x-a.x, d.y-a.y
    length = math.hypot(dx, dy)
    error = max(abs(dx*(v.y-a.y)-dy*(v.x-a.x))/length if length else abs(v-a) for v in (b,c))
    if error <= FLATNESS or depth >= 14:
        return [a, d]
    ab, bc, cd = (a+b)/2, (b+c)/2, (c+d)/2
    abc, bcd = (ab+bc)/2, (bc+cd)/2
    mid = (abc+bcd)/2
    return cubic_points(a,ab,abc,mid,depth+1)[:-1]+cubic_points(mid,bcd,cd,d,depth+1)


def circle(x, y, r):
    k = r * 0.5522847498
    return (f'M {x+r} {y} C {x+r} {y+k} {x+k} {y+r} {x} {y+r} '
            f'C {x-k} {y+r} {x-r} {y+k} {x-r} {y} '
            f'C {x-r} {y-k} {x-k} {y-r} {x} {y-r} '
            f'C {x+k} {y-r} {x+r} {y-k} {x+r} {y} Z')


def stroke_paths(drawing, offset_x, offset_y):
    """Return exact SVG centerline plus same-winding filled round-stroke paths."""
    exact, outlines, points = [], [], []
    previous = None
    for item in drawing['items']:
        if item[0] not in ('l','c'):
            raise ValueError(f'Unsupported stroke item: {item[0]}')
        vertices = [fitz.Point(p.x-offset_x, p.y-offset_y) for p in item[1:]]
        start, end = vertices[0], vertices[-1]
        if previous is None or abs(previous-start) > 0.001:
            exact.append(f'M {start.x} {start.y}')
        if item[0] == 'l':
            exact.append(f'L {end.x} {end.y}')
            polyline = vertices
        else:
            exact.append('C '+' '.join(f'{p.x} {p.y}' for p in vertices[1:]))
            polyline = cubic_points(*vertices)
        radius = drawing['width']/2
        for a,b in zip(polyline,polyline[1:]):
            length=abs(b-a)
            if not length:
                continue
            nx,ny=-(b.y-a.y)*radius/length,(b.x-a.x)*radius/length
            corners=[(a.x+nx,a.y+ny),(a.x-nx,a.y-ny),(b.x-nx,b.y-ny),(b.x+nx,b.y+ny)]
            outlines.append('M '+' L '.join(f'{x} {y}' for x,y in corners)+' Z')
        points.extend(polyline)
        previous=end
    if drawing.get('closePath'):
        raise ValueError('Unexpected closed centerline')
    for p in points:
        outlines.append(circle(p.x,p.y,drawing['width']/2))
    return ' '.join(exact), outlines


def extract(source: Path, collection: Path):
    source = source.resolve()
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if collection.exists():
        raise ValueError(f'Collection already exists; choose a new output directory: {collection}')
    document=fitz.open(source)
    if len(document)!=1:
        raise ValueError('Expected one page')
    drawings=[d for d in document[0].get_drawings() if d.get('color')==(0,0,0)]
    if any(d['type']!='s' or max(d['lineCap'])!=1 or d['lineJoin']!=1 for d in drawings):
        raise ValueError('Expected round black vector strokes')
    rows=[]
    assigned=set()
    prepared=[]
    for number,(low,high) in enumerate(ROW_BANDS,1):
        strokes=[(i,d) for i,d in enumerate(drawings) if low <= (d['rect'].y0+d['rect'].y1)/2 < high]
        strokes.sort(key=lambda item:item[1]['rect'].x0)
        groups=[]
        right=-math.inf
        for i,d in strokes:
            if d['rect'].x0 > right+4.5:
                groups.append([])
            groups[-1].append((i,d))
            right=max(right,d['rect'].x1)
        if len(groups)!=EXPECTED_COUNTS[number-1]:
            raise ValueError(f'Row {number}: expected {EXPECTED_COUNTS[number-1]} samples, got {len(groups)}')
        samples=[]
        for index,group in enumerate(reversed(groups),1):
            bbox=fitz.Rect(group[0][1]['rect'])
            for i,d in group:
                assigned.add(i)
                bbox.include_rect(d['rect'])
            radius=max(d['width']/2 for _,d in group)
            bbox=fitz.Rect(bbox.x0-radius,bbox.y0-radius,bbox.x1+radius,bbox.y1+radius)
            # Preserve physical dimensions and align ink baseline to the existing builder.
            offset_x=(bbox.x0+bbox.x1)/2-9
            offset_y=bbox.y1-23.5
            raw, filled=[],[]
            for _,d in group:
                exact,outlines=stroke_paths(d,offset_x,offset_y)
                raw.append(f'<path d="{exact}" fill="none" stroke="#000" stroke-width="{d["width"]}" stroke-linecap="round" stroke-linejoin="round"/>')
                filled.extend(f'<path d="{path}"/>' for path in outlines)
            view=(min(0,bbox.x0-offset_x-1),min(0,bbox.y0-offset_y-1),max(18,bbox.width+2),max(36,bbox.height+2))
            header=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view[0]} {view[1]} {view[2]} {view[3]}">'
            stem=f'row_{number:02d}/sample_{index:02d}'
            prepared.append((stem+'.svg',header+'<g fill="#000" fill-rule="nonzero">'+''.join(filled)+'</g></svg>'))
            prepared.append((stem+'.strokes.svg',header+''.join(raw)+'</svg>'))
            samples.append(dict(sample_index=index,file=stem+'.svg',stroke_file=stem+'.strokes.svg',
                                source_drawing_indices=[i for i,_ in group],bbox=list(bbox),
                                source_cell={'center_x':(bbox.x0+bbox.x1)/2,'center_y':(bbox.y0+bbox.y1)/2}))
        rows.append(dict(page=1,row=number,label=None,suggested_label=SUGGESTIONS[number-1],
                         slug=f'math_2026_10_27_row_{number:02d}',mapping_status='needs_user_review',
                         sample_count=len(samples),samples=samples))
    if assigned!=set(range(len(drawings))):
        raise ValueError(f'Unassigned black strokes: {set(range(len(drawings)))-assigned}')
    collection.mkdir(parents=True)
    shutil.copy2(source,collection/'source.pdf')
    for name,content in prepared:
        target=collection/name;target.parent.mkdir(exist_ok=True)
        target.write_text(content,encoding='utf-8')
    payload=dict(schema_version=1,status='extracted_pending_mapping_confirmation',
                 source={'file':'source.pdf','original_path':str(source),'sha256':digest},
                 extraction={'method':'vector_stroke_expansion_round_caps','raster_tracing_used':False,
                             'curve_flatness_points':FLATNESS,'black_stroke_count':len(drawings),
                             'assigned_stroke_count':len(assigned),'sample_count':sum(r['sample_count'] for r in rows)},
                 active_font_modified=False,rows=rows)
    (collection/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    markup=['<html><meta charset="utf-8"><title>Math glyph candidates</title><style>body{font:16px Arial}section{display:flex;align-items:center;gap:12px;margin:16px}img{width:48px;height:86px;border:1px solid #ddd}figure{margin:0;text-align:center}h3{width:160px}</style><body>']
    for row in rows:
        markup.append(f'<section><h3>Row {row["row"]}<br>{html.escape(row["suggested_label"])}</h3>')
        for sample in row['samples']:
            markup.append(f'<figure><img src="{sample["file"]}"><figcaption>{sample["sample_index"]}</figcaption></figure>')
        markup.append('</section>')
    (collection/'preview.html').write_text(''.join(markup)+'</body></html>',encoding='utf-8')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=extract(args.source,args.output)
    print(json.dumps(result['extraction']))
