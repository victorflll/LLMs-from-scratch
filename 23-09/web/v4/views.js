'use strict';

window.GPTJourneyViews = (() => {
  const TOKEN_LABELS = ['Hello', ',', '␠I', '␠am'];

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

  function normalizationFigure(isFinal) {
    const scope = isFinal
      ? 'Parâmetros da LayerNorm final.'
      : 'Parâmetros da primeira LayerNorm do bloco.';

    return (
      flow([
        ['x', '768 componentes'],
        ['(x − μ) / √(σ² + ε)', 'ε = 0,00001'],
        ['γ × z + β', '768 componentes'],
      ]) +
      note(
        `Média e variância por token · variância populacional (unbiased=False). ${scope} A normalização aproxima média 0 e variância 1 antes de γ e β; após essa escala e deslocamento aprendidos, a saída não precisa manter esses valores.`,
      )
    );
  }

  function residualFigure(isSecond) {
    const input = isSecond ? 'r₁' : 'x';
    const update = isSecond ? 'norm2 → FFN' : 'norm1 → atenção';

    return `
      <div class="residual">
        <div class="shortcut">${input} · cópia preservada</div>
        ${flow([
          [input, 'entrada'],
          [update, 'atualização'],
          ['+', 'soma elemento a elemento'],
        ])}
      </div>
      ${note('O atalho preserva a entrada. Na avaliação, o dropout não altera a atualização.')}
    `;
  }

  function attentionFigure(config) {
    const headOptions = Array.from(
      { length: config.n_heads },
      (_, index) => `<option value="${index}">${index + 1} de ${config.n_heads}</option>`,
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
                  aria-label="Query ${query}, key ${key}: ${blocked ? 'bloqueada' : 'permitida'}"
                >${blocked ? '×' : '●'}</button>
              </td>
            `;
          }).join('')}
        </tr>
      `,
    ).join('');

    return `
      <label>Cabeça <select id="head">${headOptions}</select></label>
      <div class="matrix-wrap">
        <table class="matrix">
          <caption>Máscara causal · query nas linhas, key nas colunas</caption>
          <thead><tr><th>Q ↓ / K →</th>${header}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${status('attention-info', 'Selecione uma conexão para ver o que a máscara permite.')}
      ${note('● conexão permitida · × futuro bloqueado. As cores não representam pesos de atenção. Cada cabeça usa 64 componentes; as projeções Q, K e V leem as 768 dimensões da entrada.')}
    `;
  }

  function outputFigure() {
    return `
      ${flow([
        ['768', 'vetor por posição'],
        ['Wᵀ', '50.257 × 768 pesos'],
        ['50.257', 'scores por posição'],
      ])}
      <div class="output-shape">
        1 <span>sequência</span> × 4 <span>posições</span> × 50.257 <span>logits</span>
      </div>
      ${note('Shape registrado na célula 15. A out_head não tem bias e não compartilha pesos com tok_emb nesta implementação. Não há logits numéricos deste contexto salvos.')}
      <div class="probability">
        <div class="eyebrow">DE SCORES A PROBABILIDADES</div>
        <div class="formula">pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)</div>
        <p>Softmax converte os 50.257 logits de uma posição em probabilidades que somam 1. Cada entrada corresponde a um token do vocabulário, que pode ser apenas parte de uma palavra.</p>
        <details>
          <summary>Softmax é necessária para greedy?</summary>
          <p>Não. Ela preserva a ordem dos scores: <code>argmax(z) = argmax(softmax(z))</code>. Por isso, o código local escolhe diretamente o maior logit. Na atenção, a softmax tem outro papel: normalizar os pesos entre posições permitidas.</p>
        </details>
        <details>
          <summary>E a temperatura?</summary>
          <p>Na amostragem, usa-se <code>softmax(z / τ)</code>, com τ &gt; 0. Temperaturas menores concentram a distribuição; maiores a espalham. Dividir por uma temperatura positiva não muda o argmax. O registro desta aula usa greedy, sem amostragem.</p>
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
                    <small>ID ${id}</small>
                    <small>posição ${tokenIndex}</small>
                  </button>
                `,
              )
              .join('')}
          </div>
          ${status('token-info', `Selecionado: ${TOKEN_LABELS[token]} → ID ${data.ids[token]} → posição ${token}.`)}
          ${note('Tokenização registrada · célula 15 do notebook. ␠ indica um espaço inicial.')}
        `;
      case 1:
        return `
          <div class="lookup">
            E[<span class="chosen-id">${data.ids[token]}</span>, :]
            <span>→</span><strong>768</strong><small>componentes</small>
          </div>
          <p>Abra um dos 12 grupos de 64 dimensões:</p>
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
          ${note('Representação da tabela 50.257 × 768. Os índices são reais; os valores dos pesos não estão salvos.')}
        `;
      case 2:
        return `
          ${flow([
            ['E[token, d]', 'identidade'],
            ['+ P[posição, d]', 'ordem'],
            ['x[posição, d]', 'soma, não concatenação'],
          ])}
          ${status('position-info', `Token ${data.ids[token]}, posição ${token}: E[${data.ids[token]}, d] + P[${token}, d].`)}
          ${note('O vetor de posição é compartilhado entre sequências do batch por broadcasting.')}
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
          ${flow([
            ['768', 'entrada normalizada'],
            ['3.072', 'Linear + GELU'],
            ['768', 'Linear'],
          ])}
          <details>
            <summary>Ver a transformação GELU</summary>
            <div class="formula">½x [1 + tanh(√(2/π) (x + 0,044715x³))]</div>
            <p>Fórmula usada pela classe GELU em gpt.py. A curva é uma função matemática; as ativações desta execução não foram exportadas.</p>
          </details>
          ${note('Duas projeções aprendíveis com bias. A expansão ocorre nas componentes, não na quantidade de tokens.')}
        `;
      case 8:
        return `
          <div class="blocks">
            ${Array.from(
              { length: config.n_layers },
              (_, index) => `
                <button data-block="${index}" aria-pressed="${index === block}">
                  <small>TRANSFORMER</small>${String(index + 1).padStart(2, '0')}<span>→</span>
                </button>
              `,
            ).join('')}
          </div>
          ${status('block-info')}
          ${note('Os 12 registros TransformerBlock do notebook têm saída (1, 4, 768). Selecionar um bloco identifica sua posição, não revela ativações.')}
        `;
      case 10:
        return outputFigure();
      case 11:
        return `
          ${flow([
            ['Última posição', 'logits[:, −1, :]'],
            ['argmax', 'ID de maior score'],
            ['Anexar ID', 'novo contexto → modelo'],
          ])}
          <div class="record-label">REPRODUÇÃO DO REGISTRO · SMALL · SEM TREINAMENTO</div>
          <div id="generated-ids" class="generated-ids"></div>
          <div class="generation-controls">
            <button id="back-round">← ID anterior</button>
            <button id="next-round" class="primary">Revelar próximo ID →</button>
            <button id="reset-round">Reiniciar</button>
          </div>
          ${status('generation-info')}
          <details>
            <summary>Ver texto completo decodificado no notebook</summary>
            <blockquote>${escapeHtml(data.text)}</blockquote>
          </details>
          ${note('A página não executa inferência. Esta é a continuação efetivamente salva na célula 18, não uma frase escolhida para a apresentação. Os IDs são anexados à sequência; o tokenizador decodifica a sequência para exibir texto. O código recalcula o contexto a cada rodada, recortando-o ao limite configurado se necessário.')}
        `;
      default:
        return '';
    }
  }

  function renderDetailedChapter(step, index, state, data) {
    const sourceLabel =
      index === 0
        ? 'chapter4_visual_lesson.ipynb · célula 15'
        : `gpt.py · ${step.source}`;
    const sourceCode =
      index === 0
        ? data.records['15'].code.split('logs =')[0]
        : data.code[step.source];

    return `
      <section class="chapter" id="s${index}" aria-labelledby="title-${index}">
        <div class="chapter-copy">
          <div class="eyebrow">${String(index + 1).padStart(2, '0')} / ${step.label}</div>
          <h2 id="title-${index}">${step.title}</h2>
          <p>${step.description}</p>
          <div class="shape">
            <span>ENTRA</span><code>${escapeHtml(step.input)}</code>
            <span>SAI</span><code>${escapeHtml(step.output)}</code>
          </div>
          <details class="source-code">
            <summary>Ver código original ↗</summary>
            <p>${sourceLabel}</p>
            <pre><code>${escapeHtml(sourceCode)}</code></pre>
          </details>
        </div>
        <div class="chapter-figure">
          <div class="figure-heading">
            ${[0, 11].includes(index) ? 'REGISTRO DO NOTEBOOK' : 'ESTRUTURA DO CÓDIGO'}
            <span>${index === 11 ? '10 NOVOS IDs' : 'GPT-2 SMALL'}</span>
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
        Acompanhar token
        <select id="token">
          <option value="0">Hello · 15496</option>
          <option value="1">, · 11</option>
          <option value="2">␠I · 314</option>
          <option value="3" selected>␠am · 716</option>
        </select>
      </label>
      <p class="shape-guide">
        Como ler as formas: <code>(B, T, D)</code> = sequências, tokens e componentes.
        Neste registro: <code>(1, 4, 768)</code>.
      </p>
    `;

    return `
      <section class="basic-step" id="step${index + 1}" aria-labelledby="basic-title-${index}">
        <div class="basic-heading">
          <span class="step-number">0${index + 1}</span>
          <div>
            <div class="eyebrow">PASSO ${index + 1} / ${step.subtitle}</div>
            <h2 id="basic-title-${index}">${step.title}</h2>
          </div>
        </div>
        <p class="basic-description">${step.description}</p>
        <div class="basic-visual">${flow(step.flow)}</div>
        <p class="takeaway">${step.takeaway}</p>
        <details class="advanced" id="advanced${index + 1}">
          <summary>
            <span>Avançado <small>${step.detail}</small></span>
            <span class="expand-hint" aria-hidden="true">+</span>
          </summary>
          <div class="advanced-body">
            ${index === 0 ? tokenControl : ''}
            ${detailedChapters.slice(start, end).join('')}
          </div>
        </details>
        <a class="basic-next" href="${index < 3 ? `#step${index + 2}` : '#learning'}">
          ${index < 3 ? `Próximo: passo ${index + 2}` : 'Por que o modelo ainda gera texto sem sentido?'}
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
