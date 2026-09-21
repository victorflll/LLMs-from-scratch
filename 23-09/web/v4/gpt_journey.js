'use strict';

const data = window.JOURNEY_DATA;
const content = window.GPTJourneyContent;
const views = window.GPTJourneyViews;
const t = window.t;

const state = { token: 3, group: 0, head: 0, block: 0, round: 0 };
const byId = id => document.getElementById(id);
const all = selector => [...document.querySelectorAll(selector)];

function renderModelMap() {
  byId('model-map').innerHTML = Array.from(
    { length: data.config.n_layers },
    (_, blockIndex) => `
      <a href="#s8" data-map-block="${blockIndex}"
         aria-label="${t.mapBlockAria(blockIndex + 1)}" style="--i:${blockIndex}">
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
  byId('source-records').innerHTML = Object.values(data.records)
    .map(
      record => `
        <details>
          <summary>${t.cellLabel(record.cell, t.recordNames[record.cell])}</summary>
          <pre>${views.escapeHtml(record.output)}</pre>
          <details>
            <summary>${t.cellCode}</summary>
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
  byId('token-info').textContent = t.tokInfo(
    views.TOKEN_LABELS[state.token],
    data.ids[state.token],
    state.token,
  );
  document.querySelector('.chosen-id').textContent = data.ids[state.token];
  byId('position-info').textContent = t.positionInfoArrow(
    data.ids[state.token],
    state.token,
  );

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
      <button data-dim="${dimension}" aria-label="${t.dimInspectAria(dimension)}">
        ${dimension}
      </button>`;
  }).join('');

  byId('dimension-info').textContent = t.dimGroupInfo(
    data.ids[state.token],
    firstDimension,
    firstDimension + 64,
  );
}

function updateBlock() {
  all('[data-block]').forEach(button => {
    button.setAttribute('aria-pressed', Number(button.dataset.block) === state.block);
  });

  const input =
    state.block === 0 ? t.blockInputFirst : t.blockInputPrev(state.block);
  const output =
    state.block === data.config.n_layers - 1
      ? t.blockOutputFinal
      : t.blockOutputNext(state.block + 2);

  byId('block-info').textContent = t.blockInfo(state.block + 1, input, output);
}

function updateGeneration() {
  const ids = data.generated.slice(0, data.ids.length + state.round);

  byId('generated-ids').innerHTML = ids
    .map((id, index) => {
      const isNew = index >= data.ids.length;
      const label = isNew ? `+${index - data.ids.length + 1}` : t.genContext;
      return `<span class="${isNew ? 'new-id' : ''}"><small>${label}</small>${id}</span>`;
    })
    .join('');

  if (state.round === 0) {
    byId('generation-info').textContent = t.genInfoStart;
  } else {
    byId('generation-info').textContent = t.genInfoRound(
      state.round,
      ids.at(-1),
      ids.length,
      state.round === 10,
    );
  }

  byId('back-round').disabled = state.round === 0;
  byId('next-round').disabled = state.round === 10;
}

function updateAttention(query, key) {
  all('[data-cell]').forEach(button => {
    button.setAttribute('aria-pressed', button.dataset.cell === `${query},${key}`);
  });

  const result = key > query ? t.attCellBlocked : t.attCellAllowed;

  byId('attention-info').textContent = t.attCellInfo(
    state.head + 1,
    query,
    views.TOKEN_LABELS[query],
    key,
    views.TOKEN_LABELS[key],
    result,
  );
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
    byId('dimension-info').textContent = t.dimSingleInfo(
      data.ids[state.token],
      button.dataset.dim,
    );
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
  if (operationLink && /^#s\d+$/.test(operationLink.getAttribute('href'))) {
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
    ? t.readingTraining
    : t.readingStep(currentStep + 1, content.basicSteps[currentStep].title);

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
    byId('attention-info').textContent = t.attHeadInfo(
      state.head + 1,
      state.head * 64,
      state.head * 64 + 63,
    );
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

byId('parameter-note').textContent = t.parameterNote;

if (location.hash) requestAnimationFrame(() => revealDetailedOperation(location.hash));
