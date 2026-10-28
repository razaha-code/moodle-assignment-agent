"""Source-grounded pedigree data shared by generation and rendering."""
import base64
import html
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any


PEDIGREE_VERSION = 1
PEDIGREE_PROMPT = """
When a requested item asks to draw a genetic pedigree, return a nonempty pedigrees
array in that answer. Never use ASCII art, preformatted trees, or raw SVG.
Each pedigree is {"title":"...","people":[{"id":"p1","label":"...",
"sex":"M|F|U","affected":true|false|null,"carrier":true|false|null,
"father":null,"mother":null,"evidence":{"label":"exact source quote",
"sex":"exact source quote","affected":"exact source quote"}}],
"partnerships":[{"partners":["p1","p2"],"consanguineous":false,
"evidence":"exact source quote"}]}. Parent values are IDs or null.
Evidence must quote the original question, not the instructions or your answer.
Provide evidence for every non-null fact, including parent references and carrier
status. Unknown sex is U; unknown disease/carrier status is null, not false.
Do not infer carrier status or add hypothetical children. A brother is male (M),
not unknown. Represent each person exactly once and partnerships explicitly,
including couples with no children. For ordinary items omit pedigrees.
For instructions mode describe how to solve; structured diagrams are optional.
"""


def requires_pedigree(text: str) -> bool:
    return bool(
        re.search(r"שושלת|שושלות|עץ משפח|pedigree|family tree", text, re.I)
        and re.search(r"צייר|שרטט|ציירו|שרטוט|ציור|draw|construct|diagram|sketch", text, re.I)
    )


@dataclass(frozen=True)
class Pedigree:
    title: str
    people: tuple[dict[str, Any], ...]
    partnerships: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "people": list(self.people),
                "partnerships": list(self.partnerships)}


def parse_pedigrees(raw: Any, source: str | None = None) -> tuple[Pedigree, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("pedigrees must be an array")
    result = []
    for value in raw:
        if not isinstance(value, dict) or not isinstance(value.get("people"), list):
            raise ValueError("pedigree requires people")
        people = value["people"]
        pairs = value.get("partnerships", [])
        if not people or not isinstance(pairs, list):
            raise ValueError("empty pedigree or invalid partnerships")
        ids = [p.get("id") for p in people if isinstance(p, dict)]
        if len(ids) != len(people) or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", i) for i in ids):
            raise ValueError("invalid person ID")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate person ID")
        by_id = dict(zip(ids, people))

        def evidence(quote: Any) -> None:
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("missing source evidence")
            if source is not None and " ".join(quote.split()) not in " ".join(source.split()):
                raise ValueError("evidence is not quoted from the source")

        for p in people:
            if p.get("sex") not in ("M", "F", "U") or not isinstance(p.get("label"), str) or not p["label"].strip():
                raise ValueError("invalid sex or label")
            if any(p.get(k) is not None and type(p[k]) is not bool for k in ("affected", "carrier")):
                raise ValueError("affected/carrier must be boolean or null")
            facts = p.get("evidence", {})
            if not isinstance(facts, dict):
                raise ValueError("invalid evidence map")
            for k in ("label", "sex", "affected", "carrier", "father", "mother"):
                if p.get(k) is not None and not (k == "sex" and p[k] == "U"):
                    evidence(facts.get(k))
            for role, sex in (("father", "M"), ("mother", "F")):
                parent = p.get(role)
                if parent is not None and (parent not in by_id or parent == p["id"] or by_id[parent]["sex"] != sex):
                    raise ValueError(f"invalid {role} for {p['id']}")
        visiting, visited = set(), set()

        def visit(person_id: str) -> None:
            if person_id in visiting:
                raise ValueError("ancestry cycle")
            if person_id in visited:
                return
            visiting.add(person_id)
            for role in ("father", "mother"):
                if by_id[person_id].get(role):
                    visit(by_id[person_id][role])
            visiting.remove(person_id)
            visited.add(person_id)

        for person_id in ids:
            visit(person_id)
        seen_pairs = set()
        for pair in pairs:
            members = pair.get("partners") if isinstance(pair, dict) else None
            if not isinstance(members, list) or len(members) != 2 or any(not isinstance(i, str) or i not in by_id for i in members) or members[0] == members[1]:
                raise ValueError("invalid partnership")
            key = tuple(sorted(members))
            if key in seen_pairs:
                raise ValueError("duplicate partnership")
            seen_pairs.add(key)
            if type(pair.get("consanguineous", False)) is not bool:
                raise ValueError("invalid consanguinity")
            evidence(pair.get("evidence"))
        for p in people:
            if p.get("mother") and p.get("father") and tuple(sorted((p["mother"], p["father"]))) not in seen_pairs:
                raise ValueError("parents must have an explicit partnership")
        result.append(Pedigree(str(value.get("title", "")), tuple(people), tuple(pairs)))
    return tuple(result)


def pedigree_html(pedigree: Pedigree) -> str:
    data = base64.b64encode(json.dumps(pedigree.to_dict(), ensure_ascii=False).encode()).decode()
    return ('<figure class="pedigree-figure" dir="ltr" data-pedigree="' + data + '">'
            '<figcaption dir="rtl">' + html.escape(pedigree.title) + '</figcaption>'
            '<div class="pedigree-canvas"></div>'
            '<p class="pedigree-legend" dir="rtl">ריבוע: זכר; עיגול: נקבה; יהלום: מין לא ידוע. '
            'מלא: חולה; ריק: בריא; ?: מצב מחלה לא ידוע; נקודה: נשאות ידועה. '
            'קו כפול: קרבת משפחה ידועה.</p></figure>')


def pedigree_resources() -> str:
    root = Path(__file__).resolve().parent.parent / "assets" / "vendor" / "pedigree"
    paths = [root / name for name in ("jquery.min.js", "d3.min.js", "pedigreejs.js")]
    paths.append(Path(__file__).with_name("pedigree_render.js"))
    scripts = "\n".join('<script>' + p.read_text(encoding="utf-8").replace("</script", "<\\/script") + '</script>' for p in paths)
    return """<style>
    .pedigree-figure { margin: 1em 0; break-inside: avoid; direction: ltr; }
    .pedigree-canvas { direction: ltr; }
    .pedigree-legend { font-size: 0.85em !important; }
    .pedigree-full-page { break-before: page; break-after: page; }
    .pedigree-canvas svg { overflow: visible; }
    </style>""" + scripts
