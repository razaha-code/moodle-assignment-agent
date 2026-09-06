import html


try:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    rdMolDraw2D = None
    rdBase = None
    RDKIT_AVAILABLE = False


def smiles_to_svg(smiles: str, size: tuple[int, int] = (360, 220)) -> str | None:
    """Return a self-contained molecule SVG, or None for missing/invalid SMILES."""
    value = (smiles or "").strip()
    if not value or not RDKIT_AVAILABLE:
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    options = drawer.drawOptions()
    ink = (0.09, 0.24, 0.45)
    options.setBackgroundColour((1.0, 1.0, 1.0, 0.0))
    options.setSymbolColour(ink)
    options.setAnnotationColour(ink)
    options.setBondNoteColour(ink)
    options.singleColourBonds = True
    options.singleColourWedgeBonds = True
    options.bondLineWidth = 2.2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg"):] if "<svg" in svg else svg


def compound_reference_html(iupac_name: str, smiles: str, role: str = "") -> str:
    label = (iupac_name or "Unnamed compound").strip()
    value = (smiles or "").strip()
    role_text = f" ({role.strip()})" if role and role.strip() else ""
    drawing = smiles_to_svg(value)
    if drawing:
        return (
            '<figure class="compound-card" dir="ltr">'
            f'<div class="molecule-drawing" aria-label="Chemical structure for {html.escape(label)}">'
            f"{drawing}</div>"
            f"<figcaption><strong>{html.escape(label + role_text)}</strong>"
            f"<code>{html.escape(value)}</code></figcaption></figure>"
        )
    status = "SMILES unavailable" if not value else f"Invalid SMILES: {value}"
    return (
        '<div class="compound-card compound-card-invalid" dir="ltr">'
        f"<strong>{html.escape(label + role_text)}</strong>"
        f"<span>{html.escape(status)}</span></div>"
    )
