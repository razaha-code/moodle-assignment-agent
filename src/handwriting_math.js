// Keep MathJax's layout (including fractions, scripts and stretchy operators),
// replacing ordinary glyph outlines with the document's handwriting font.
window.applyHandwritingMath = async () => {
  if (!window.__handwritingFontRequired) return;
  await document.fonts.load('1000px "Handwriting Base"', 'PNxyz0123456789');
  await document.fonts.ready;
  const ns = 'http://www.w3.org/2000/svg';
  const metrics = document.createElement('canvas').getContext('2d');
  metrics.font = '1000px "Handwriting Base"';
  document.querySelectorAll('mjx-container svg use[data-c]').forEach((glyph) => {
    const character = String.fromCodePoint(parseInt(glyph.dataset.c, 16)).normalize('NFKD');
    // Unsupported and stretchy symbols retain their original mathematical paths.
    if (!/^[A-Za-z0-9.,:;!?+\-=()\[\]ΣΠθ]$/u.test(character)) return;
    const box = glyph.getBBox();
    if (!box.width || !box.height) return;
    const group = document.createElementNS(ns, 'g');
    const text = document.createElementNS(ns, 'text');
    text.textContent = character;
    text.setAttribute('font-family', 'Handwriting Base');
    text.setAttribute('font-size', '1000');
    text.setAttribute('font-style', 'normal');
    text.setAttribute('font-weight', '400');
    text.setAttribute('stroke', 'none');
    text.setAttribute('direction', 'ltr');
    group.appendChild(text);
    glyph.parentNode.insertBefore(group, glyph);
    const measured = metrics.measureText(character);
    const ink = {
      x: -measured.actualBoundingBoxLeft,
      y: -measured.actualBoundingBoxAscent,
      width: measured.actualBoundingBoxLeft + measured.actualBoundingBoxRight,
      height: measured.actualBoundingBoxAscent + measured.actualBoundingBoxDescent,
    };
    if (!ink.width || !ink.height) { group.remove(); return; }
    const sx = box.width / ink.width;
    const sy = box.height / ink.height;
    // MathJax uses y-up coordinates; SVG text uses y-down coordinates.
    text.setAttribute('transform', `translate(${box.x - sx * ink.x},${box.y + box.height + sy * ink.y}) scale(${sx},${-sy})`);
    if (glyph.hasAttribute('transform')) group.setAttribute('transform', glyph.getAttribute('transform'));
    group.setAttribute('data-handwriting-math', character);
    glyph.replaceWith(group);
  });
};
