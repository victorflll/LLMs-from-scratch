'use strict';

const data = window.JOURNEY_DATA;
const content = window.GPTJourneyContent;
const views = window.GPTJourneyViews;

const state = { token: 3, group: 0, head: 0, block: 0, round: 0 };
const byId = id => document.getElementById(id);
const all = selector => [...document.querySelectorAll(selector)];

function renderModelMap() {
  byId('model-map').innerHTML = Array.from(
    { length: data.config.n_layers },
    (_, blockIndex) => `
      <a href="#s8" data-map-block="${blockIndex}"
         aria-label="Explorar bloco ${blockIndex + 1}" style="--i:${blockIndex}">
        <span>${String(blockIndex + 1).padStart(2, '0')}</span>
        ${Array.from({ length: data.config.n_heads }, () => '<i></i>').join('')}
      </a>`,
  ).join('');
}

function renderJourney() {
  const detailedChapters = content.detailedSteps.map((step, index) =>
    views.renderDetailedChapter(step, index, state, data),
  );

  byId('journey').innerHTML = content.basicSteps
    .map((step, index) => views.renderBasicStep(step, index, detailedChapters))
    .join('');

  byId('index').innerHTML = content.basicSteps
    .map(
      (step, index) => `
        <a href="#step${index + 1}">
          <span>0${index + 1}</span>${step.title}
        </a>`,
    )
    .join('');
}

function renderSourceRecords() {
  const recordNames = {
    4: 'Parâmetros',
    8: 'Famílias GPT-2',
    15: 'Tokenização e shapes',
    18: 'Geração registrada',
  };

  byId('source-records').innerHTML = Object.values(data.records)
    .map(
      record => `
        <details>
          <summary>Célula ${record.cell} · ${recordNames[record.cell]}</summary>
          <pre>${views.escapeHtml(record.output)}</pre>
          <details>
            <summary>Código da célula</summary>
            <pre>${views.escapeHtml(record.code)}</pre>
          </details>
        </details>`,
    )
    .join('');
}

function updateToken() {
  all('[data-token]').forEach(button => {
    button.setAttribute('aria-pressed', Number(button.dataset.token) === state.token);
  });

  byId('token').value = state.token;
  byId('token-info').textContent =
    `Selecionado: ${views.TOKEN_LABELS[state.token]} → ` +
    `ID ${data.ids[state.token]} → posição ${state.token}.`;
  document.querySelector('.chosen-id').textContent = data.ids[state.token];
  byId('position-info').textContent =
    `E[${data.ids[state.token]}, d] + P[${state.token}, d] → x[${state.token}, d].`;

  updateDimensions();
}

function updateDimensions() {
  all('[data-group]').forEach(button => {
    button.setAttribute('aria-pressed', Number(button.dataset.group) === state.group);
  });

  const firstDimension = state.group * 64;
  byId('dimension-grid').innerHTML = Array.from({ length: 64 }, (_, offset) => {
    const dimension = firstDimension + offset;
    return `
      <button data-dim="${dimension}" aria-label="Inspecionar dimensão ${dimension}">
        ${dimension}
      </button>`;
  }).join('');

  byId('dimension-info').textContent =
    `E[${data.ids[state.token]}, ${firstDimension}:${firstDimension + 64}] · ` +
    '64 componentes. Valores não exportados.';
}

function updateBlock() {
  all('[data-block]').forEach(button => {
    button.setAttribute('aria-pressed', Number(button.dataset.block) === state.block);
  });

  const input = state.block === 0 ? 'os embeddings somados' : `a saída do bloco ${state.block}`;
  const output =
    state.block === data.config.n_layers - 1
      ? 'à LayerNorm final'
      : `ao bloco ${state.block + 2}`;

  byId('block-info').textContent =
    `Bloco ${state.block + 1}: recebe ${input} e entrega (1, 4, 768) ${output}.`;
}

function updateGeneration() {
  const ids = data.generated.slice(0, data.ids.length + state.round);

  byId('generated-ids').innerHTML = ids
    .map((id, index) => {
      const isNew = index >= data.ids.length;
      const label = isNew ? `+${index - data.ids.length + 1}` : 'CONTEXTO';
      return `<span class="${isNew ? 'new-id' : ''}"><small>${label}</small>${id}</span>`;
    })
    .join('');

  if (state.round === 0) {
    byId('generation-info').textContent =
      '4 IDs iniciais. Revele a primeira escolha salva do argmax.';
  } else {
    const isComplete = state.round === 10;
    byId('generation-info').textContent =
      `Rodada ${state.round} de 10 · ID ${ids.at(-1)} anexado · ` +
      `contexto com ${ids.length} tokens. ` +
      (isComplete ? 'Fim do registro.' : 'A próxima rodada recebe esse contexto ampliado.');
  }

  byId('back-round').disabled = state.round === 0;
  byId('next-round').disabled = state.round === 10;
}

function updateAttention(query, key) {
  all('[data-cell]').forEach(button => {
    button.setAttribute('aria-pressed', button.dataset.cell === `${query},${key}`);
  });

  const result =
    key > query
      ? 'futuro bloqueado; score recebe −∞ e peso após softmax é zero.'
      : 'conexão permitida; score = Q · K / √64. Valor não exportado.';

  byId('attention-info').textContent =
    `Cabeça ${state.head + 1} · query ${query} (${views.TOKEN_LABELS[query]}) → ` +
    `key ${key} (${views.TOKEN_LABELS[key]}): ${result}`;
}

function updateLearning(mode) {
  const selected = content.learningModes[mode];
  all('[data-mode]').forEach(button => {
    button.setAttribute('aria-pressed', button.dataset.mode === mode);
  });
  byId('learning-flow').innerHTML = `${views.flow(selected.items)}<p>${selected.text}</p>`;
}

function revealDetailedOperation(hash) {
  const target = document.getElementById(hash.slice(1));
  if (!target) return;

  const advancedSection = target.closest('details.advanced');
  if (advancedSection) advancedSection.open = true;
  requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
}

function handleButtonClick(button) {
  if (button.dataset.token !== undefined) {
    state.token = Number(button.dataset.token);
    updateToken();
  }
  if (button.dataset.group !== undefined) {
    state.group = Number(button.dataset.group);
    updateDimensions();
  }
  if (button.dataset.dim !== undefined) {
    all('[data-dim]').forEach(item => item.setAttribute('aria-pressed', item === button));
    byId('dimension-info').textContent =
      `E[${data.ids[state.token]}, ${button.dataset.dim}] · ` +
      'um peso da tabela. Valor não exportado.';
  }
  if (button.dataset.cell !== undefined) {
    const [query, key] = button.dataset.cell.split(',').map(Number);
    updateAttention(query, key);
  }
  if (button.dataset.block !== undefined) {
    state.block = Number(button.dataset.block);
    updateBlock();
  }
  if (button.dataset.mode !== undefined) updateLearning(button.dataset.mode);

  if (button.id === 'next-round') {
    state.round = Math.min(10, state.round + 1);
    updateGeneration();
  }
  if (button.id === 'back-round') {
    state.round = Math.max(0, state.round - 1);
    updateGeneration();
  }
  if (button.id === 'reset-round') {
    state.round = 0;
    updateGeneration();
  }
}

function handleDocumentClick(event) {
  const mapBlock = event.target.closest('[data-map-block]');
  if (mapBlock) {
    state.block = Number(mapBlock.dataset.mapBlock);
    updateBlock();
  }

  const operationLink = event.target.closest('a[href^="#s"]');
  if (operationLink) {
    event.preventDefault();
    const hash = operationLink.getAttribute('href');
    history.replaceState(null, '', hash);
    revealDetailedOperation(hash);
  }

  const button = event.target.closest('button');
  if (button) handleButtonClick(button);
  if (event.target.closest('#index a')) document.querySelector('.index').open = false;
}

let scrollUpdatePending = false;

function updateReadingProgress() {
  let currentStep = 0;
  all('.basic-step').forEach((element, index) => {
    if (element.getBoundingClientRect().top < innerHeight * 0.45) currentStep = index;
  });

  const learningIsVisible = byId('learning').getBoundingClientRect().top < innerHeight * 0.45;
  byId('reading-label').textContent = learningIsVisible
    ? 'O papel do treinamento'
    : `Passo ${currentStep + 1} / ${content.basicSteps[currentStep].title}`;

  all('#index a').forEach((link, index) => {
    if (index === currentStep && !learningIsVisible) link.setAttribute('aria-current', 'step');
    else link.removeAttribute('aria-current');
  });

  byId('reading-progress').style.width = `${((currentStep + 1) / 4) * 100}%`;
  scrollUpdatePending = false;
}

function bindEvents() {
  byId('token').addEventListener('change', event => {
    state.token = Number(event.target.value);
    updateToken();
  });

  byId('head').addEventListener('change', event => {
    state.head = Number(event.target.value);
    byId('attention-info').textContent =
      `Cabeça ${state.head + 1}: componentes ${state.head * 64}–${state.head * 64 + 63} ` +
      'das projeções Q/K/V. A máscara é igual nas 12 cabeças; ' +
      'os pesos de atenção não foram exportados.';
  });

  document.addEventListener('click', handleDocumentClick);
  addEventListener('hashchange', () => revealDetailedOperation(location.hash));
  addEventListener(
    'scroll',
    () => {
      if (scrollUpdatePending) return;
      scrollUpdatePending = true;
      requestAnimationFrame(updateReadingProgress);
    },
    { passive: true },
  );

  all('.advanced').forEach(section => {
    section.addEventListener('toggle', updateReadingProgress);
  });
}

function observeChapters() {
  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('seen');
      });
    },
    { threshold: 0.08 },
  );
  all('.chapter').forEach(chapter => observer.observe(chapter));
}

renderModelMap();
renderJourney();
renderSourceRecords();
bindEvents();
observeChapters();

updateToken();
updateBlock();
updateGeneration();
updateLearning('inference');
updateReadingProgress();

byId('parameter-note').textContent =
  'O notebook registra 163.009.536 parâmetros: 38.597.376 na tabela de tokens ' +
  'e outros 38.597.376 na saída. Sem contar a matriz de saída separadamente, ' +
  'o registro chega a 124.412.160 — aproximadamente 124M.';

if (location.hash) requestAnimationFrame(() => revealDetailedOperation(location.hash));
