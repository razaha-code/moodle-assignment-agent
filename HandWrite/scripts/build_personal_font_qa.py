"""Create a deterministic HTML QA page for the personal handwriting fonts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def _font_face(font_path: Path, variant: int) -> str:
    encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return (
        "@font-face {"
        f"font-family:'Raz Personal {variant}';"
        f"src:url(data:font/woff2;base64,{encoded}) format('woff2');"
        "font-weight:400;font-style:normal;font-display:block;}"
    )


def build_qa_html(font_dir: Path) -> str:
    faces = "\n".join(
        _font_face(font_dir / f"RazPersonalHandwriting-{variant}.woff2", variant)
        for variant in range(1, 5)
    )
    alphabet = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
    report_path = font_dir / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    fallback = report.get("fallback_codepoints", [])
    if fallback:
        notice = "האותיות ך ו-ן עדיין מגיעות מפונט ה-fallback. הסוגריים נבדקים כאן בהקשר RTL אמיתי."
    else:
        notice = "עברית, A-Z, a-z, Σ, Π ו-θ מגיעים מכתב היד האישי. הסוגריים נבדקים כאן בהקשר RTL אמיתי."
    rows = "".join(
        f'<section class="variant v{variant}"><h2>וריאציה {variant}</h2>'
        f'<p class="alphabet">{alphabet}</p>'
        '<p>זהו משפט בדיקה שנכתב בכתב היד האישי שלי, עם אותיות חוזרות ושונות טבעית.</p>'
        '<p class="parentheses">בדיקה (ניסיון בתוך סוגריים) ולאחר מכן משפט רגיל.</p>'
        '<p class="finals">מלך קטן הלך לגן, ושם נתן מתנה.</p>'
        '<p class="numbers">0123456789  ! ? , .  (12345)</p></section>'
        f'<p class="latin v{variant}" dir="ltr">ABCDEFGHIJKLMNOPQRSTUVWXYZ</p>'
        f'<p class="latin v{variant}" dir="ltr">abcdefghijklmnopqrstuvwxyz</p>'
        f'<p class="latin v{variant}" dir="ltr">Complexity, automata, NP-complete, graph coloring.</p>'
        f'<p class="latin v{variant}" dir="ltr">Σ Π θ</p>'
        for variant in range(1, 5)
    )
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Personal handwriting font QA</title>
<style>
{faces}
@page {{ size: A4; margin: 13mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #111; background: #fffefb; font-family: Arial,sans-serif; }}
.page {{ min-height: 100vh; padding: 8mm; background: linear-gradient(rgba(92,145,190,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(92,145,190,.13) 1px,transparent 1px),#fffefb; background-size: 7mm 7mm; }}
h1 {{ margin: 0 0 5mm; font: 700 24px Arial,sans-serif; }}
.notice {{ padding: 3mm; border: 1px solid #c7b978; background: #fff9d8; font: 14px Arial,sans-serif; }}
.variant {{ margin: 5mm 0; padding: 3mm 4mm; border-bottom: 1px solid #aaa; break-inside: avoid; font-size: 27px; line-height: 1.55; }}
.variant h2 {{ margin: 0 0 1mm; font: 700 14px Arial,sans-serif; color: #555; }}
.variant p {{ margin: 1mm 0; }}
.alphabet {{ letter-spacing: .03em; }}
.v1 {{ font-family: 'Raz Personal 1',sans-serif; }} .v2 {{ font-family: 'Raz Personal 2',sans-serif; }}
.v3 {{ font-family: 'Raz Personal 3',sans-serif; }} .v4 {{ font-family: 'Raz Personal 4',sans-serif; }}
.mixed {{ font-family: 'Raz Personal 1',sans-serif; font-size: 30px; line-height: 1.7; padding: 4mm; border: 2px solid #555; }}
.gv1 {{ font-family:'Raz Personal 1',sans-serif; }} .gv2 {{ font-family:'Raz Personal 2',sans-serif; }}
.gv3 {{ font-family:'Raz Personal 3',sans-serif; }} .gv4 {{ font-family:'Raz Personal 4',sans-serif; }}
.math {{ direction:ltr; font-family:'Times New Roman',serif; font-size:24px; text-align:center; }}
.latin {{ margin: 1mm 0; font-size: 25px; line-height: 1.35; text-align: left; }}
</style>
<script>
window.__qaReady=false;
window.addEventListener('DOMContentLoaded',async()=>{{
  const families=[1,2,3,4].map(i=>`Raz Personal ${{i}}`);
  await Promise.all(families.map(f=>document.fonts.load(`24px "${{f}}"`,'אבגדה ABC xyz ΣΠθ 0123 ()')));
  const target=document.querySelector('.mixed');
  const original=target.textContent;
  const segmenter=new Intl.Segmenter('he',{{granularity:'grapheme'}});
  const fragment=document.createDocumentFragment(); let position=0;
  for(const part of segmenter.segment(original)){{
    if(/[א-ת]/u.test(part.segment)){{
      const span=document.createElement('span');
      const variant=((position*2654435761)>>>0)%4+1;
      span.className=`gv${{variant}}`; span.dataset.variant=String(variant); span.textContent=part.segment;
      fragment.appendChild(span);
    }}else{{ fragment.appendChild(document.createTextNode(part.segment)); }}
    position+=1;
  }}
  target.replaceChildren(fragment);
  window.__qaOriginalText=original;
  window.__qaReady=families.every(f=>document.fonts.check(`24px "${{f}}"`)) && target.textContent===original;
}});
</script></head><body><main class="page">
<h1>בדיקת הפונט האישי</h1>
<p class="notice">{notice}</p>
{rows}
<h2>ערבוב ארבע וריאציות בתוך אותו משפט</h2>
<p class="mixed">כתב היד שלי נראה טבעי יותר כאשר כל אות חוזרת מקבלת וריאציה מעט שונה.</p>
<p class="math">Σ(i=1…n) i = n(n+1)/2 — mathematical content remains typeset</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_qa_html(args.font_dir), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
