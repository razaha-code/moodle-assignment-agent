window.__pedigreesReady = false;
window.__pedigreeError = null;
window.renderPedigrees = async () => {
  const ns = 'http://www.w3.org/2000/svg';
  const element = (name, attrs, text) => {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  try {
    await document.fonts.ready;
    let index = 0;
    for (const figure of document.querySelectorAll('[data-pedigree]')) {
      const data = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(figure.dataset.pedigree), c => c.charCodeAt(0))));
      const canvas = figure.querySelector('.pedigree-canvas');
      canvas.id = `pedigree-${++index}`;
      const prefix = canvas.id + '-';
      const family = window.__handwritingFontRequired ? 'Handwriting Base' : 'Arial';
      await document.fonts.load(`18px "${family}"`);
      const dataset = data.people.map(p => {
        const person = {name: prefix + p.id, display_name: p.label, sex: p.sex};
        if (p.affected === true) person.disease = true;
        if (p.father && p.mother) {
          person.father = prefix + p.father;
          person.mother = prefix + p.mother;
        } else person.top_level = true;
        return person;
      });
      // Renderer performs topology validation in Python; no editor/UI/storage.
      pedigreejs.pedigreejs.build({targetDiv: canvas.id, btn_target: prefix + 'unused',
        dataset, width: Math.max(760, data.people.length * 100), height: 340,
        symbol_size: 30, font_size: '18px', font_family: family, font_weight: 400,
        max_name_length: 80, labels: [], diseases: [{type: 'disease', colour: '#111'}],
        background: 'transparent', node_background: '#fff', showWidgets: false,
        dragNode: false, zoomSrc: [], edit: false, store_type: 'array', validate: false});
      const svg = canvas.querySelector('svg');
      svg.setAttribute('xmlns', ns);
      const diagram = svg.querySelector('.diagram');
      diagram.removeAttribute('transform');
      svg.querySelector(':scope > rect')?.remove();
      const nodes = new Map();
      diagram.querySelectorAll('g').forEach(g => {
        const d = g.__data__;
        if (d?.data?.name && !d.data.hidden && g.hasAttribute('transform') && 'x' in d) nodes.set(d.data.name.slice(prefix.length), {g, d});
      });
      if (nodes.size !== data.people.length) throw new Error('Pedigree layout omitted or duplicated individuals');
      for (const p of data.people) {
        const {g} = nodes.get(p.id);
        g.dataset.personId = p.id;
        g.dataset.sex = p.sex;
        g.dataset.affected = String(p.affected);
        const outline = g.querySelector(':scope > path');
        outline?.setAttribute('stroke', '#111');
        outline?.setAttribute('stroke-width', '1.6');
        if (outline) g.append(outline); // Draw the complete border over disease fills.
        g.querySelectorAll('text').forEach(t => {
          t.setAttribute('direction', 'rtl'); t.setAttribute('unicode-bidi', 'plaintext');
          t.setAttribute('text-anchor', 'middle'); t.setAttribute('x', '0');
          t.setAttribute('y', '46');
          t.style.fontFamily = family;
        });
        if (p.affected === null || p.affected === undefined) g.append(element('text', {x: 0, y: 6, 'text-anchor': 'middle', 'font-size': 18, fill: '#111'}, '?'));
        if (p.carrier === true && p.affected !== true) g.append(element('circle', {cx: 0, cy: 0, r: 4, fill: '#111'}));
      }
      // Explicit couples without children are absent from pedigreejs's implicit
      // co-parent graph. Add the relationship, never a fictitious child.
      let lane = 0;
      for (const pair of data.partnerships) {
        const [a, b] = pair.partners;
        const parentsOfChild = data.people.some(p => [p.father, p.mother].includes(a) && [p.father, p.mother].includes(b));
        if (parentsOfChild) {
          const line = [...diagram.querySelectorAll(':scope > path')].find(path => {
            const d = path.__data__;
            return d?.mother && d?.father && [d.mother.data.name, d.father.data.name].includes(prefix + a)
              && [d.mother.data.name, d.father.data.name].includes(prefix + b);
          });
          if (!line) throw new Error('Missing co-parent relationship');
          line.dataset.partners = `${a} ${b}`;
          // Different generation depths alone are not evidence of consanguinity.
          const firstStroke = 'M' + line.getAttribute('d').split('M')[1];
          line.setAttribute('d', firstStroke);
          if (pair.consanguineous) {
            const duplicate = line.cloneNode(); duplicate.setAttribute('transform', 'translate(0,-4)'); diagram.insertBefore(duplicate, line);
          }
          continue;
        }
        const first = nodes.get(a).d, second = nodes.get(b).d;
        const left = first.x < second.x ? first : second;
        const right = first.x < second.x ? second : first;
        const blocked = [...nodes.values()].some(({d}) => d !== left && d !== right && d.x > left.x && d.x < right.x && Math.abs(d.y - left.y) < 35);
        const y = Math.max(left.y, right.y) + (blocked || left.y !== right.y ? 85 + lane++ * 14 : 0);
        const path = y === left.y && y === right.y
          ? `M${left.x + 17},${y} H${right.x - 17}`
          : `M${left.x + 17},${left.y} h10 V${y} H${right.x - 27} V${right.y} h10`;
        const line = element('path', {d: path, fill: 'none', stroke: '#111', 'stroke-width': 1.5, 'data-partners': `${a} ${b}`});
        diagram.append(line);
        if (pair.consanguineous) {
          const duplicate = line.cloneNode(); duplicate.setAttribute('transform', 'translate(0,4)'); diagram.append(duplicate);
        }
      }
      // Single known parent: no fabricated second parent. Place the child below
      // that parent and explicitly draw its descent connection.
      for (const p of data.people.filter(p => Boolean(p.father) !== Boolean(p.mother))) {
        const parent = nodes.get(p.father || p.mother).d;
        const child = nodes.get(p.id);
        if (data.people.some(q => q.father === p.id || q.mother === p.id)) throw new Error('Partial-parent multi-generation layout requires review');
        child.d.y = parent.y + 160;
        child.g.setAttribute('transform', `translate(${child.d.x},${child.d.y})`);
        diagram.insertBefore(element('path', {d: `M${parent.x},${parent.y + 17} V${parent.y + 70} H${child.d.x} V${child.d.y - 17}`, fill: 'none', stroke: '#111', 'stroke-width': 1.5}), diagram.firstChild);
      }
      const bounds = diagram.getBBox();
      const width = bounds.width + 60, height = bounds.height + 60;
      const printScale = Math.min(680 / width, 850 / height, 1);
      if (18 * printScale < 11) throw new Error('Pedigree layout is too large to read on A4; split into source-grounded subdiagrams');
      if (height * Math.min(680 / width, 1) > 380) figure.classList.add('pedigree-full-page');
      svg.setAttribute('viewBox', `${bounds.x - 30} ${bounds.y - 30} ${width} ${height}`);
      svg.setAttribute('width', String(width)); svg.setAttribute('height', String(height));
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', data.title);
      svg.style.cssText = 'display:block;width:100%;height:auto;max-height:225mm';
      svg.querySelectorAll('text').forEach(t => {t.style.fontFamily = family; t.style.fill = '#111';});
      const fontCSS = [...document.styleSheets].flatMap(sheet => {
        try { return [...sheet.cssRules].filter(r => r.type === CSSRule.FONT_FACE_RULE).map(r => r.cssText); }
        catch { return []; }
      }).join('\n');
      svg.prepend(element('style', {}, fontCSS));
      // Fail visibly on collisions rather than delivering a misleading diagram.
      const boxes = [...nodes.values()].map(({g}) => ({g, box: g.getBoundingClientRect()}));
      for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i].box, b = boxes[j].box;
        if (a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top) throw new Error('Pedigree labels overlap; layout requires review');
      }
    }
    window.__pedigreesReady = true;
  } catch (error) {
    window.__pedigreeError = String(error);
  }
};
document.addEventListener('DOMContentLoaded', () => window.renderPedigrees());
