'use strict';

/*
 * i18n + theme runtime for the GPT journey (v4).
 * Locale is resolved from <html lang> (set pre-paint by the inline head script).
 * Dynamic strings live here as functions; static HTML nodes carry data-i18n.
 * Language switch reloads the page (all content is JS-rendered) — see setLang.
 */

window.I18N = {
  'pt-BR': {
    // <head>
    title: 'GPT por dentro — v4 · do texto à próxima previsão',
    metaDesc:
      'Jornada interativa pela arquitetura GPT-2: o caminho de um token, do texto aos logits, com dados reais do notebook do capítulo 4.',

    // Header / hero
    skip: 'Pular para a jornada',
    brandSub: 'por dentro',
    edition: 'V4 / CAPÍTULO 04',
    sourcesLink: 'Dados e código ↗',
    themeToDark: 'Tema escuro',
    themeToLight: 'Tema claro',
    langLabel: 'EN',
    langAria: 'Switch to English',

    heroEyebrow: 'UMA JORNADA PELA ARQUITETURA GPT-2',
    heroTitle: 'O caminho<br />de um token.',
    heroLead:
      'Um GPT calcula a distribuição do próximo token a partir do contexto. A resposta cresce repetindo esse processo, um token por vez.',
    heroCta: 'Acompanhar a frase <span>↓</span>',
    heroMapAria: 'Visão geral: texto, IDs, vetores, 12 blocos, logits',
    mapLogits: '50.257 LOGITS POR POSIÇÃO',
    mapCaption: '12 linhas = blocos · 12 marcas por linha = cabeças de atenção',
    openingBottomLeft: '01 — 04 <span class="muted">/ role para acompanhar</span>',
    openingBottomRight: 'Registros do notebook · pesos não treinados',
    indexSummary: 'Os 4 passos',

    // Learning epilogue
    learningEyebrow: 'DEPOIS DO PERCURSO / O PAPEL DO TREINAMENTO',
    learningTitle:
      'A arquitetura está pronta.<br />A linguagem ainda precisa ser aprendida.',
    learningLead:
      'A continuação acima vem de um modelo sem treinamento. As matrizes começam com valores aleatórios; os parâmetros de LayerNorm começam com escala 1 e deslocamento 0. Ter os blocos certos não basta para produzir texto coerente.',
    modeGroupAria: 'Comparar geração e treinamento',
    modeInference: 'Durante a geração',
    modeTraining: 'Durante o treinamento',
    trainTargetSummary: 'Como o texto fornece o alvo de treinamento?',
    trainTargetP1:
      'O alvo é o token seguinte na sequência original. Nos IDs já registrados, a posição de <code>Hello</code> tem como próximo alvo <code>,</code>; a vírgula tem <code>␣I</code>; e <code>␣I</code> tem <code>␣am</code>. O alvo após <code>␣am</code> exigiria a continuação do texto de treino.',
    trainTargetP2:
      'É possível calcular a perda de várias posições no mesmo forward. A máscara causal impede que cada posição consulte seu futuro, mesmo com o texto inteiro disponível no batch.',
    paramSummary:
      'Por que “GPT-2 124M” não é a contagem exata desta implementação?',
    paramP2:
      'O nome identifica a configuração Small. Nesta implementação, <code>tok_emb</code> e <code>out_head</code> têm matrizes separadas; compartilhar essas matrizes reduz a contagem. O código também usa <code>qkv_bias=False</code>. As contagens do notebook estão em Dados e código.',
    learningBridge:
      '<span>↓</span> Conferir os registros que sustentam a explicação.',

    // Sources / footer
    sourcesEyebrow: 'RASTREABILIDADE',
    sourcesTitle: 'O que sustenta esta jornada.',
    sourcesLead:
      'IDs, shapes e continuação vêm das saídas já salvas no notebook. Os diagramas explicam as operações de <code>gpt.py</code>. Os pesos, ativações e probabilidades desta execução não foram exportados; por isso, não são apresentados como valores medidos.',
    sourcesChapterLink: 'Código de referência do capítulo ↗',
    sourcesNotebookLink: 'Notebook da aula ↗',
    sourcesImplLink: 'Implementação original ↗',
    footerText:
      'Build a Large Language Model (From Scratch) · Sebastian Raschka',
    footerTop: 'Voltar ao início ↑',
    noscript:
      'Ative JavaScript para explorar a jornada. Os dados também estão em chapter4_visual_lesson.ipynb.',

    // ---- Dynamic (views.js / gpt_journey.js) ----
    comp768: '768 componentes',
    vocabSize: '50.257',
    geluFormula: '½x [1 + tanh(√(2/π) (x + 0,044715x³))]',

    normScope: isFinal =>
      isFinal
        ? 'Parâmetros da LayerNorm final.'
        : 'Parâmetros da primeira LayerNorm do bloco.',
    normFlow: [
      ['x', '768 componentes'],
      ['(x − μ) / √(σ² + ε)', 'ε = 0,00001'],
      ['γ × z + β', '768 componentes'],
    ],
    normNote: scope =>
      `Média e variância por token · variância populacional (unbiased=False). ${scope} A normalização aproxima média 0 e variância 1 antes de γ e β; após essa escala e deslocamento aprendidos, a saída não precisa manter esses valores.`,

    residualUpdate: isSecond => (isSecond ? 'norm2 → FFN' : 'norm1 → atenção'),
    residualFlow: (input, update) => [
      [input, 'entrada'],
      [update, 'atualização'],
      ['+', 'soma elemento a elemento'],
    ],
    residualShortcut: input => `${input} · cópia preservada`,
    residualNote:
      'O atalho preserva a entrada. Na avaliação, o dropout não altera a atualização.',

    attHead: 'Cabeça',
    attOption: (i, n) => `${i} de ${n}`,
    attQK: 'Q ↓ / K →',
    attCaption: 'Máscara causal · query nas linhas, key nas colunas',
    attCellAria: (q, k, blocked) =>
      `Query ${q}, key ${k}: ${blocked ? 'bloqueada' : 'permitida'}`,
    attStatusDefault: 'Selecione uma conexão para ver o que a máscara permite.',
    attNote:
      '● conexão permitida · × futuro bloqueado. As cores não representam pesos de atenção. Cada cabeça usa 64 componentes; as projeções Q, K e V leem as 768 dimensões da entrada.',

    outFlow: [
      ['768', 'vetor por posição'],
      ['Wᵀ', '50.257 × 768 pesos'],
      ['50.257', 'scores por posição'],
    ],
    outShape: ['sequência', 'posições', 'logits'],
    outNote:
      'Shape registrado na célula 15. A out_head não tem bias e não compartilha pesos com tok_emb nesta implementação. Não há logits numéricos deste contexto salvos.',
    probEyebrow: 'DE SCORES A PROBABILIDADES',
    probText:
      'Softmax converte os 50.257 logits de uma posição em probabilidades que somam 1. Cada entrada corresponde a um token do vocabulário, que pode ser apenas parte de uma palavra.',
    probGreedySummary: 'Softmax é necessária para greedy?',
    probGreedyText:
      'Não. Ela preserva a ordem dos scores: <code>argmax(z) = argmax(softmax(z))</code>. Por isso, o código local escolhe diretamente o maior logit. Na atenção, a softmax tem outro papel: normalizar os pesos entre posições permitidas.',
    probTempSummary: 'E a temperatura?',
    probTempText:
      'Na amostragem, usa-se <code>softmax(z / τ)</code>, com τ &gt; 0. Temperaturas menores concentram a distribuição; maiores a espalham. Dividir por uma temperatura positiva não muda o argmax. O registro desta aula usa greedy, sem amostragem.',

    tokIdLabel: 'ID',
    tokPosLabel: 'posição',
    tokInfo: (label, id, pos) =>
      `Selecionado: ${label} → ID ${id} → posição ${pos}.`,
    tokNote:
      'Tokenização registrada · célula 15 do notebook. ␣ indica um espaço inicial — no GPT-2, o espaço faz parte do token (IDs 314 e 716 são “ I” e “ am”).',

    lookupUnit: 'componentes',
    lookupGroupsIntro: 'Abra um dos 12 grupos de 64 dimensões:',
    lookupNote:
      'Representação da tabela 50.257 × 768. Os índices são reais; os valores dos pesos não estão salvos.',

    posFlow: [
      ['E[token, d]', 'identidade'],
      ['+ P[posição, d]', 'ordem'],
      ['x[posição, d]', 'soma, não concatenação'],
    ],
    posInfo: (id, pos) =>
      `Token ${id}, posição ${pos}: E[${id}, d] + P[${pos}, d].`,
    posNote:
      'O vetor de posição é compartilhado entre sequências do batch por broadcasting.',

    ffFlow: [
      ['768', 'entrada normalizada'],
      ['3.072', 'Linear + GELU'],
      ['768', 'Linear'],
    ],
    ffSummary: 'Ver a transformação GELU',
    ffText:
      'Fórmula usada pela classe GELU em gpt.py. A curva é uma função matemática; as ativações desta execução não foram exportadas.',
    ffNote:
      'Duas projeções aprendíveis com bias. A expansão ocorre nas componentes, não na quantidade de tokens.',

    blockWord: 'TRANSFORMER',
    blocksNote:
      'Os 12 registros TransformerBlock do notebook têm saída (1, 4, 768). Selecionar um bloco identifica sua posição, não revela ativações.',

    genFlow: [
      ['Última posição', 'logits[:, −1, :]'],
      ['argmax', 'ID de maior score'],
      ['Anexar ID', 'novo contexto → modelo'],
    ],
    genRecordLabel: 'REPRODUÇÃO DO REGISTRO · SMALL · SEM TREINAMENTO',
    genBack: '← ID anterior',
    genNext: 'Revelar próximo ID →',
    genReset: 'Reiniciar',
    genFullTextSummary: 'Ver texto completo decodificado no notebook',
    genNote:
      'A página não executa inferência. Esta é a continuação efetivamente salva na célula 18, não uma frase escolhida para a apresentação. Os IDs são anexados à sequência; o tokenizador decodifica a sequência para exibir texto. O código recalcula o contexto a cada rodada, recortando-o ao limite configurado se necessário.',
    genContext: 'CONTEXTO',

    sourceLabelFirst: 'chapter4_visual_lesson.ipynb · célula 15',
    sourceLabel: source => `gpt.py · ${source}`,
    figHeadingRecord: 'REGISTRO DO NOTEBOOK',
    figHeadingCode: 'ESTRUTURA DO CÓDIGO',
    figHeadingNewIds: '10 NOVOS IDs',
    figHeadingSmall: 'GPT-2 SMALL',
    shapeIn: 'ENTRA',
    shapeOut: 'SAI',
    seeCode: 'Ver código original ↗',

    trackToken: 'Acompanhar token',
    shapeGuide:
      'Como ler as formas: <code>(B, T, D)</code> = sequências, tokens e componentes. Neste registro: <code>(1, 4, 768)</code>.',
    stepLabel: (n, subtitle) => `PASSO ${n} / ${subtitle}`,
    advanced: 'Avançado',
    basicNextStep: n => `Próximo: passo ${n}`,
    basicNextLast: 'Por que o modelo ainda gera texto sem sentido?',

    mapBlockAria: n => `Explorar bloco ${n}`,
    recordNames: {
      4: 'Parâmetros',
      8: 'Famílias GPT-2',
      15: 'Tokenização e shapes',
      18: 'Geração registrada',
    },
    cellLabel: (cell, name) => `Célula ${cell} · ${name}`,
    cellCode: 'Código da célula',

    positionInfoArrow: (id, pos) =>
      `E[${id}, d] + P[${pos}, d] → x[${pos}, d].`,
    dimInspectAria: dim => `Inspecionar dimensão ${dim}`,
    dimGroupInfo: (id, a, b) =>
      `E[${id}, ${a}:${b}] · 64 componentes. Valores não exportados.`,
    dimSingleInfo: (id, dim) =>
      `E[${id}, ${dim}] · um peso da tabela. Valor não exportado.`,

    blockInputFirst: 'os embeddings somados',
    blockInputPrev: block => `a saída do bloco ${block}`,
    blockOutputFinal: 'à LayerNorm final',
    blockOutputNext: block => `ao bloco ${block}`,
    blockInfo: (n, input, output) =>
      `Bloco ${n}: recebe ${input} e entrega (1, 4, 768) ${output}.`,

    genInfoStart: '4 IDs iniciais. Revele a primeira escolha salva do argmax.',
    genInfoRound: (round, lastId, len, complete) =>
      `Rodada ${round} de 10 · ID ${lastId} anexado · contexto com ${len} tokens. ` +
      (complete
        ? 'Fim do registro.'
        : 'A próxima rodada recebe esse contexto ampliado.'),

    attCellBlocked:
      'futuro bloqueado; score recebe −∞ e peso após softmax é zero.',
    attCellAllowed:
      'conexão permitida; score = Q · K / √64. Valor não exportado.',
    attCellInfo: (head, q, ql, k, kl, result) =>
      `Cabeça ${head} · query ${q} (${ql}) → key ${k} (${kl}): ${result}`,
    attHeadInfo: (head, a, b) =>
      `Cabeça ${head}: componentes ${a}–${b} das projeções Q/K/V. ` +
      'A máscara é igual nas 12 cabeças; os pesos de atenção não foram exportados.',

    readingTraining: 'O papel do treinamento',
    readingStep: (n, title) => `Passo ${n} / ${title}`,

    parameterNote:
      'O notebook registra 163.009.536 parâmetros: 38.597.376 na tabela de tokens ' +
      'e outros 38.597.376 na saída. Sem contar a matriz de saída separadamente, ' +
      'o registro chega a 124.412.160 — aproximadamente 124M.',
  },

  en: {
    // <head>
    title: 'GPT from the inside — v4 · from text to the next prediction',
    metaDesc:
      'Interactive journey through the GPT-2 architecture: the path of a token, from text to logits, with real data from the chapter 4 notebook.',

    // Header / hero
    skip: 'Skip to the journey',
    brandSub: 'from the inside',
    edition: 'V4 / CHAPTER 04',
    sourcesLink: 'Data and code ↗',
    themeToDark: 'Dark theme',
    themeToLight: 'Light theme',
    langLabel: 'PT',
    langAria: 'Mudar para português',

    heroEyebrow: 'A JOURNEY THROUGH THE GPT-2 ARCHITECTURE',
    heroTitle: 'The path<br />of a token.',
    heroLead:
      'A GPT computes the distribution of the next token from the context. The answer grows by repeating this process, one token at a time.',
    heroCta: 'Follow the sentence <span>↓</span>',
    heroMapAria: 'Overview: text, IDs, vectors, 12 blocks, logits',
    mapLogits: '50,257 LOGITS PER POSITION',
    mapCaption: '12 rows = blocks · 12 marks per row = attention heads',
    openingBottomLeft: '01 — 04 <span class="muted">/ scroll to follow along</span>',
    openingBottomRight: 'Notebook records · untrained weights',
    indexSummary: 'The 4 steps',

    // Learning epilogue
    learningEyebrow: 'AFTER THE JOURNEY / THE ROLE OF TRAINING',
    learningTitle:
      'The architecture is ready.<br />The language still has to be learned.',
    learningLead:
      'The continuation above comes from an untrained model. The matrices start with random values; the LayerNorm parameters start with scale 1 and shift 0. Having the right blocks is not enough to produce coherent text.',
    modeGroupAria: 'Compare generation and training',
    modeInference: 'During generation',
    modeTraining: 'During training',
    trainTargetSummary: 'How does the text provide the training target?',
    trainTargetP1:
      'The target is the next token in the original sequence. In the recorded IDs, the position of <code>Hello</code> has <code>,</code> as its next target; the comma has <code>␣I</code>; and <code>␣I</code> has <code>␣am</code>. The target after <code>␣am</code> would require the continuation of the training text.',
    trainTargetP2:
      'The loss can be computed for several positions in the same forward pass. The causal mask prevents each position from looking at its future, even with the whole text available in the batch.',
    paramSummary:
      'Why is “GPT-2 124M” not the exact count of this implementation?',
    paramP2:
      'The name identifies the Small configuration. In this implementation, <code>tok_emb</code> and <code>out_head</code> have separate matrices; sharing those matrices reduces the count. The code also uses <code>qkv_bias=False</code>. The notebook counts are in Data and code.',
    learningBridge:
      '<span>↓</span> Check the records that back the explanation.',

    // Sources / footer
    sourcesEyebrow: 'TRACEABILITY',
    sourcesTitle: 'What backs this journey.',
    sourcesLead:
      'IDs, shapes and continuation come from outputs already saved in the notebook. The diagrams explain the operations in <code>gpt.py</code>. The weights, activations and probabilities of this run were not exported; that is why they are not presented as measured values.',
    sourcesChapterLink: 'Reference code for the chapter ↗',
    sourcesNotebookLink: 'Lesson notebook ↗',
    sourcesImplLink: 'Original implementation ↗',
    footerText:
      'Build a Large Language Model (From Scratch) · Sebastian Raschka',
    footerTop: 'Back to top ↑',
    noscript:
      'Enable JavaScript to explore the journey. The data is also in chapter4_visual_lesson.ipynb.',

    // ---- Dynamic ----
    comp768: '768 components',
    vocabSize: '50,257',
    geluFormula: '½x [1 + tanh(√(2/π) (x + 0.044715x³))]',

    normScope: isFinal =>
      isFinal
        ? 'Parameters of the final LayerNorm.'
        : "Parameters of the block's first LayerNorm.",
    normFlow: [
      ['x', '768 components'],
      ['(x − μ) / √(σ² + ε)', 'ε = 0.00001'],
      ['γ × z + β', '768 components'],
    ],
    normNote: scope =>
      `Mean and variance per token · population variance (unbiased=False). ${scope} Normalization brings the values close to mean 0 and variance 1 before γ and β; after this learned scale and shift, the output need not keep those values.`,

    residualUpdate: isSecond => (isSecond ? 'norm2 → FFN' : 'norm1 → attention'),
    residualFlow: (input, update) => [
      [input, 'input'],
      [update, 'update'],
      ['+', 'element-wise sum'],
    ],
    residualShortcut: input => `${input} · preserved copy`,
    residualNote:
      'The shortcut preserves the input. At evaluation time, dropout does not change the update.',

    attHead: 'Head',
    attOption: (i, n) => `${i} of ${n}`,
    attQK: 'Q ↓ / K →',
    attCaption: 'Causal mask · query in rows, key in columns',
    attCellAria: (q, k, blocked) =>
      `Query ${q}, key ${k}: ${blocked ? 'blocked' : 'allowed'}`,
    attStatusDefault: 'Select a connection to see what the mask allows.',
    attNote:
      '● allowed connection · × blocked future. The colors do not represent attention weights. Each head uses 64 components; the Q, K and V projections read the 768 dimensions of the input.',

    outFlow: [
      ['768', 'vector per position'],
      ['Wᵀ', '50,257 × 768 weights'],
      ['50,257', 'scores per position'],
    ],
    outShape: ['sequence', 'positions', 'logits'],
    outNote:
      'Shape recorded in cell 15. out_head has no bias and does not share weights with tok_emb in this implementation. There are no numeric logits from this context saved.',
    probEyebrow: 'FROM SCORES TO PROBABILITIES',
    probText:
      'Softmax converts the 50,257 logits of one position into probabilities that sum to 1. Each entry corresponds to a vocabulary token, which may be only part of a word.',
    probGreedySummary: 'Is softmax needed for greedy?',
    probGreedyText:
      'No. It preserves the order of the scores: <code>argmax(z) = argmax(softmax(z))</code>. That is why the local code picks the largest logit directly. In attention, softmax has a different role: normalizing the weights among allowed positions.',
    probTempSummary: 'And temperature?',
    probTempText:
      'In sampling, <code>softmax(z / τ)</code> is used, with τ &gt; 0. Lower temperatures concentrate the distribution; higher ones spread it out. Dividing by a positive temperature does not change the argmax. This lesson record uses greedy, without sampling.',

    tokIdLabel: 'ID',
    tokPosLabel: 'position',
    tokInfo: (label, id, pos) =>
      `Selected: ${label} → ID ${id} → position ${pos}.`,
    tokNote:
      'Tokenization recorded · notebook cell 15. ␣ marks a leading space — in GPT-2, the space is part of the token (IDs 314 and 716 are “ I” and “ am”).',

    lookupUnit: 'components',
    lookupGroupsIntro: 'Open one of the 12 groups of 64 dimensions:',
    lookupNote:
      'Representation of the 50,257 × 768 table. The indices are real; the weight values are not saved.',

    posFlow: [
      ['E[token, d]', 'identity'],
      ['+ P[position, d]', 'order'],
      ['x[position, d]', 'sum, not concatenation'],
    ],
    posInfo: (id, pos) =>
      `Token ${id}, position ${pos}: E[${id}, d] + P[${pos}, d].`,
    posNote:
      'The position vector is shared across sequences in the batch by broadcasting.',

    ffFlow: [
      ['768', 'normalized input'],
      ['3,072', 'Linear + GELU'],
      ['768', 'Linear'],
    ],
    ffSummary: 'See the GELU transformation',
    ffText:
      'Formula used by the GELU class in gpt.py. The curve is a mathematical function; the activations of this run were not exported.',
    ffNote:
      'Two learnable projections with bias. The expansion happens in the components, not in the number of tokens.',

    blockWord: 'TRANSFORMER',
    blocksNote:
      'The 12 TransformerBlock records in the notebook have output (1, 4, 768). Selecting a block identifies its position, it does not reveal activations.',

    genFlow: [
      ['Last position', 'logits[:, −1, :]'],
      ['argmax', 'highest-score ID'],
      ['Append ID', 'new context → model'],
    ],
    genRecordLabel: 'RECORD REPLAY · SMALL · UNTRAINED',
    genBack: '← Previous ID',
    genNext: 'Reveal next ID →',
    genReset: 'Restart',
    genFullTextSummary: 'See the full text decoded in the notebook',
    genNote:
      'The page does not run inference. This is the continuation actually saved in cell 18, not a sentence chosen for the presentation. The IDs are appended to the sequence; the tokenizer decodes the sequence to display text. The code recomputes the context each round, trimming it to the configured limit if needed.',
    genContext: 'CONTEXT',

    sourceLabelFirst: 'chapter4_visual_lesson.ipynb · cell 15',
    sourceLabel: source => `gpt.py · ${source}`,
    figHeadingRecord: 'NOTEBOOK RECORD',
    figHeadingCode: 'CODE STRUCTURE',
    figHeadingNewIds: '10 NEW IDs',
    figHeadingSmall: 'GPT-2 SMALL',
    shapeIn: 'IN',
    shapeOut: 'OUT',
    seeCode: 'See original code ↗',

    trackToken: 'Track token',
    shapeGuide:
      'How to read the shapes: <code>(B, T, D)</code> = sequences, tokens and components. In this record: <code>(1, 4, 768)</code>.',
    stepLabel: (n, subtitle) => `STEP ${n} / ${subtitle}`,
    advanced: 'Advanced',
    basicNextStep: n => `Next: step ${n}`,
    basicNextLast: 'Why does the model still generate nonsense?',

    mapBlockAria: n => `Explore block ${n}`,
    recordNames: {
      4: 'Parameters',
      8: 'GPT-2 families',
      15: 'Tokenization and shapes',
      18: 'Recorded generation',
    },
    cellLabel: (cell, name) => `Cell ${cell} · ${name}`,
    cellCode: 'Cell code',

    positionInfoArrow: (id, pos) =>
      `E[${id}, d] + P[${pos}, d] → x[${pos}, d].`,
    dimInspectAria: dim => `Inspect dimension ${dim}`,
    dimGroupInfo: (id, a, b) =>
      `E[${id}, ${a}:${b}] · 64 components. Values not exported.`,
    dimSingleInfo: (id, dim) =>
      `E[${id}, ${dim}] · one weight from the table. Value not exported.`,

    blockInputFirst: 'the summed embeddings',
    blockInputPrev: block => `the output of block ${block}`,
    blockOutputFinal: 'to the final LayerNorm',
    blockOutputNext: block => `to block ${block}`,
    blockInfo: (n, input, output) =>
      `Block ${n}: receives ${input} and delivers (1, 4, 768) ${output}.`,

    genInfoStart: '4 initial IDs. Reveal the first saved argmax choice.',
    genInfoRound: (round, lastId, len, complete) =>
      `Round ${round} of 10 · ID ${lastId} appended · context with ${len} tokens. ` +
      (complete
        ? 'End of the record.'
        : 'The next round receives this extended context.'),

    attCellBlocked:
      'blocked future; the score gets −∞ and the weight after softmax is zero.',
    attCellAllowed:
      'allowed connection; score = Q · K / √64. Value not exported.',
    attCellInfo: (head, q, ql, k, kl, result) =>
      `Head ${head} · query ${q} (${ql}) → key ${k} (${kl}): ${result}`,
    attHeadInfo: (head, a, b) =>
      `Head ${head}: components ${a}–${b} of the Q/K/V projections. ` +
      'The mask is the same across the 12 heads; the attention weights were not exported.',

    readingTraining: 'The role of training',
    readingStep: (n, title) => `Step ${n} / ${title}`,

    parameterNote:
      'The notebook records 163,009,536 parameters: 38,597,376 in the token table ' +
      'and another 38,597,376 in the output. Without counting the output matrix separately, ' +
      'the record reaches 124,412,160 — approximately 124M.',
  },
};

// ---- Runtime ----------------------------------------------------------------

window.LOCALE = document.documentElement.lang === 'en' ? 'en' : 'pt-BR';
window.t = window.I18N[window.LOCALE];

function applyStaticI18n() {
  const t = window.t;
  document.title = t.title;
  const meta = document.querySelector('meta[name="description"]');
  if (meta) meta.setAttribute('content', t.metaDesc);

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const value = t[el.dataset.i18n];
    if (value != null) el.textContent = value;
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const value = t[el.dataset.i18nHtml];
    if (value != null) el.innerHTML = value;
  });
  document.querySelectorAll('[data-i18n-aria]').forEach(el => {
    const value = t[el.dataset.i18nAria];
    if (value != null) el.setAttribute('aria-label', value);
  });
}

function setTheme(theme, persist) {
  document.documentElement.dataset.theme = theme;
  if (persist) {
    try {
      localStorage.setItem('theme', theme);
    } catch (e) {} // ponytail: private mode / blocked storage — theme just won't persist
  }
  const button = document.getElementById('theme-toggle');
  if (button) {
    const toLight = theme === 'dark';
    button.textContent = toLight ? '☀' : '☾';
    button.setAttribute(
      'aria-label',
      toLight ? window.t.themeToLight : window.t.themeToDark,
    );
  }
}

function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

function setLang(locale) {
  try {
    localStorage.setItem('lang', locale);
    sessionStorage.setItem('scrollY', String(window.scrollY));
  } catch (e) {}
  location.reload(); // ponytail: content is JS-rendered; reload beats re-render + rebind
}

function setupControls() {
  const theme = document.getElementById('theme-toggle');
  if (theme) theme.addEventListener('click', () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark', true));

  const lang = document.getElementById('lang-toggle');
  if (lang) lang.addEventListener('click', () => setLang(window.LOCALE === 'en' ? 'pt-BR' : 'en'));

  // restore scroll after a language switch
  try {
    const y = sessionStorage.getItem('scrollY');
    if (y !== null) {
      sessionStorage.removeItem('scrollY');
      if (!location.hash) requestAnimationFrame(() => window.scrollTo(0, Number(y)));
    }
  } catch (e) {}
}

applyStaticI18n();
setTheme(currentTheme(), false);
document.addEventListener('DOMContentLoaded', setupControls);
