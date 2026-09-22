'use strict';

window.GPTJourneyViews = (() => {
  const t = window.t;
  const TOKEN_LABELS = ['Hello', ',', '␣I', '␣am'];

  function escapeHtml(value) {
    return String(value).replace(
      /[&<>"']/g,
      character =>
        ({
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;',
        })[character],
    );
  }

  function flow(items) {
    return `
      <div class="flow">
        ${items
          .map(
            ([title, detail], index) => `
              ${index ? '<span class="arrow">→</span>' : ''}
              <div>
                <strong>${title}</strong>
                <small>${detail}</small>
              </div>
            `,
          )
          .join('')}
      </div>
    `;
  }

  const note = text => `<p class="figure-note">${text}</p>`;
  const status = (id, text = '') => `<p class="selection" id="${id}" role="status">${text}</p>`;

  // Several steps live inside one shared class (GPTModel/TransformerBlock forward).
  // Extract the line(s) matching each operation so "see original code" corresponds
  // to the step instead of repeating the whole class. [src, startNeedle, endNeedle].
  const CODE_EXCERPTS = {
    1: ['GPTModel', 'self.tok_emb(in_idx)', 'self.tok_emb(in_idx)'],
    2: ['GPTModel', 'self.pos_emb(', 'tok_embeds + pos_embeds'],
    5: ['TransformerBlock', '# Shortcut connection for attention', 'x = x + shortcut'],
    7: ['TransformerBlock', '# Shortcut connection for feed-forward', 'x = x + shortcut'],
    8: ['GPTModel', 'self.trf_blocks(x)', 'self.trf_blocks(x)'],
    10: ['GPTModel', 'self.out_head(x)', 'return logits'],
  };

  function excerpt(code, startNeedle, endNeedle) {
    const lines = code.split('\n');
    const start = lines.findIndex(line => line.includes(startNeedle));
    if (start === -1) return code; // ponytail: fall back to whole class if data shifts
    let end = start;
    for (let i = start; i < lines.length; i++) {
      if (lines[i].includes(endNeedle)) {
        end = i;
        break;
      }
    }
    const chosen = lines.slice(start, end + 1);
    const indent = Math.min(
      ...chosen.filter(line => line.trim()).map(line => line.match(/^\s*/)[0].length),
    );
    return chosen.map(line => line.slice(indent)).join('\n');
  }

  function normalizationFigure(isFinal) {
    const scope = t.normScope(isFinal);
    return flow(t.normFlow) + note(t.normNote(scope));
  }

  function residualFigure(isSecond) {
    const input = isSecond ? 'r₁' : 'x';
    const update = t.residualUpdate(isSecond);

    return `
      <div class="residual">
        <div class="shortcut">${t.residualShortcut(input)}</div>
        ${flow(t.residualFlow(input, update))}
      </div>
      ${note(t.residualNote)}
    `;
  }

  function attentionFigure(config) {
    const headOptions = Array.from(
      { length: config.n_heads },
      (_, index) => `<option value="${index}">${t.attOption(index + 1, config.n_heads)}</option>`,
    ).join('');

    const header = TOKEN_LABELS.map(label => `<th>${label}</th>`).join('');
    const rows = TOKEN_LABELS.map(
      (queryLabel, query) => `
        <tr>
          <th>${queryLabel}</th>
          ${TOKEN_LABELS.map((_, key) => {
            const blocked = key > query;
            return `
              <td>
                <button
                  data-cell="${query},${key}"
                  class="${blocked ? 'blocked' : 'allowed'}"
                  aria-label="${t.attCellAria(query, key, blocked)}"
                >${blocked ? '×' : '●'}</button>
              </td>
            `;
          }).join('')}
        </tr>
      `,
    ).join('');

    return `
      <label>${t.attHead} <select id="head">${headOptions}</select></label>
      <div class="matrix-wrap">
        <table class="matrix">
          <caption>${t.attCaption}</caption>
          <thead><tr><th>${t.attQK}</th>${header}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${status('attention-info', t.attStatusDefault)}
      ${note(t.attNote)}
    `;
  }

  function outputFigure() {
    const [seq, pos, logits] = t.outShape;
    return `
      ${flow(t.outFlow)}
      <div class="output-shape">
        1 <span>${seq}</span> × 4 <span>${pos}</span> × ${t.vocabSize} <span>${logits}</span>
      </div>
      ${note(t.outNote)}
      <div class="probability">
        <div class="eyebrow">${t.probEyebrow}</div>
        <div class="formula">pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)</div>
        <p>${t.probText}</p>
        <details>
          <summary>${t.probGreedySummary}</summary>
          <p>${t.probGreedyText}</p>
        </details>
        <details>
          <summary>${t.probTempSummary}</summary>
          <p>${t.probTempText}</p>
        </details>
      </div>
    `;
  }

  function renderFigure(index, state, data) {
    const { token, group, block } = state;
    const config = data.config;

    switch (index) {
      case 0:
        return `
          <div class="sentence">Hello, I am</div>
          <div class="token-row">
            ${data.ids
              .map(
                (id, tokenIndex) => `
                  <button data-token="${tokenIndex}" aria-pressed="${tokenIndex === token}">
                    <span>${TOKEN_LABELS[tokenIndex]}</span>
                    <small>${t.tokIdLabel} ${id}</small>
                    <small>${t.tokPosLabel} ${tokenIndex}</small>
                  </button>
                `,
              )
              .join('')}
          </div>
          ${status('token-info', t.tokInfo(TOKEN_LABELS[token], data.ids[token], token))}
          ${note(t.tokNote)}
        `;
      case 1:
        return `
          <div class="lookup">
            E[<span class="chosen-id">${data.ids[token]}</span>, :]
            <span>→</span><strong>768</strong><small>${t.lookupUnit}</small>
          </div>
          <p>${t.lookupGroupsIntro}</p>
          <div class="groups">
            ${Array.from(
              { length: 12 },
              (_, index) => `
                <button data-group="${index}" aria-pressed="${index === group}">
                  ${index * 64}–${index * 64 + 63}
                </button>
              `,
            ).join('')}
          </div>
          <div id="dimension-grid"></div>
          ${status('dimension-info')}
          ${note(t.lookupNote)}
        `;
      case 2:
        return `
          ${flow(t.posFlow)}
          ${status('position-info', t.posInfo(data.ids[token], token))}
          ${note(t.posNote)}
        `;
      case 3:
      case 9:
        return normalizationFigure(index === 9);
      case 4:
        return attentionFigure(config);
      case 5:
      case 7:
        return residualFigure(index === 7);
      case 6:
        return `
          ${flow(t.ffFlow)}
          <details>
            <summary>${t.ffSummary}</summary>
            <div class="formula">${t.geluFormula}</div>
            <p>${t.ffText}</p>
          </details>
          ${note(t.ffNote)}
        `;
      case 8:
        return `
          <div class="blocks">
            ${Array.from(
              { length: config.n_layers },
              (_, index) => `
                <button data-block="${index}" aria-pressed="${index === block}">
                  <small>${t.blockWord}</small>${String(index + 1).padStart(2, '0')}<span>→</span>
                </button>
              `,
            ).join('')}
          </div>
          ${status('block-info')}
          ${note(t.blocksNote)}
        `;
      case 10:
        return outputFigure();
      case 11:
        return `
          ${flow(t.genFlow)}
          <div class="record-label">${t.genRecordLabel}</div>
          <div id="generated-ids" class="generated-ids"></div>
          <div class="generation-controls">
            <button id="back-round">${t.genBack}</button>
            <button id="next-round" class="primary">${t.genNext}</button>
            <button id="reset-round">${t.genReset}</button>
          </div>
          ${status('generation-info')}
          <details>
            <summary>${t.genFullTextSummary}</summary>
            <blockquote>${escapeHtml(data.text)}</blockquote>
          </details>
          ${note(t.genNote)}
        `;
      default:
        return '';
    }
  }

  function renderDetailedChapter(step, index, state, data) {
    const spec = CODE_EXCERPTS[index];
    let sourceLabel;
    let sourceCode;
    if (index === 0) {
      sourceLabel = t.sourceLabelFirst;
      sourceCode = data.records['15'].code.split('logs =')[0];
    } else if (spec) {
      const [src, startNeedle, endNeedle] = spec;
      sourceLabel = t.sourceLabelForward(src);
      sourceCode = excerpt(data.code[src], startNeedle, endNeedle);
    } else {
      sourceLabel = t.sourceLabel(step.source);
      sourceCode = data.code[step.source];
    }

    return `
      <section class="chapter" id="s${index}" aria-labelledby="title-${index}">
        <div class="chapter-copy">
          <div class="eyebrow">${String(index + 1).padStart(2, '0')} / ${step.label}</div>
          <h2 id="title-${index}">${step.title}</h2>
          <p>${step.description}</p>
          <div class="shape">
            <span>${t.shapeIn}</span><code>${escapeHtml(step.input)}</code>
            <span>${t.shapeOut}</span><code>${escapeHtml(step.output)}</code>
          </div>
          <details class="source-code">
            <summary>${t.seeCode}</summary>
            <p>${sourceLabel}</p>
            <pre><code>${escapeHtml(sourceCode)}</code></pre>
          </details>
        </div>
        <div class="chapter-figure">
          <div class="figure-heading">
            ${[0, 11].includes(index) ? t.figHeadingRecord : t.figHeadingCode}
            <span>${index === 11 ? t.figHeadingNewIds : t.figHeadingSmall}</span>
          </div>
          ${renderFigure(index, state, data)}
        </div>
        <a class="bridge" href="${index < 11 ? `#s${index + 1}` : '#learning'}">
          <span>↓</span> ${step.bridge}
        </a>
      </section>
    `;
  }

  function renderBasicStep(step, index, detailedChapters) {
    const [start, end] = step.range;
    const tokenControl = `
      <label class="advanced-token">
        ${t.trackToken}
        <select id="token">
          <option value="0">Hello · 15496</option>
          <option value="1">, · 11</option>
          <option value="2">␣I · 314</option>
          <option value="3" selected>␣am · 716</option>
        </select>
      </label>
      <p class="shape-guide">${t.shapeGuide}</p>
    `;

    return `
      <section class="basic-step" id="step${index + 1}" aria-labelledby="basic-title-${index}">
        <div class="basic-heading">
          <span class="step-number">0${index + 1}</span>
          <div>
            <div class="eyebrow">${t.stepLabel(index + 1, step.subtitle)}</div>
            <h2 id="basic-title-${index}">${step.title}</h2>
          </div>
        </div>
        <p class="basic-description">${step.description}</p>
        <div class="basic-visual">${flow(step.flow)}</div>
        <p class="takeaway">${step.takeaway}</p>
        <details class="advanced" id="advanced${index + 1}">
          <summary>
            <span>${t.advanced} <small>${step.detail}</small></span>
            <span class="expand-hint" aria-hidden="true">+</span>
          </summary>
          <div class="advanced-body">
            ${index === 0 ? tokenControl : ''}
            ${detailedChapters.slice(start, end).join('')}
          </div>
        </details>
        <a class="basic-next" href="${index < 3 ? `#step${index + 2}` : '#learning'}">
          ${index < 3 ? t.basicNextStep(index + 2) : t.basicNextLast}
          <span>↓</span>
        </a>
      </section>
    `;
  }

  return {
    TOKEN_LABELS,
    escapeHtml,
    flow,
    renderBasicStep,
    renderDetailedChapter,
  };
})();
