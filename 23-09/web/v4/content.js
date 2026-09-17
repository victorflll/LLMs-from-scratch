'use strict';

window.GPTJourneyContent = {
  detailedSteps: [
    {
      label: 'Tokenização',
      title: 'O texto vira IDs.',
      description:
        'Antes de entrar na rede, a frase passa pelo tokenizer GPT-2. Cada fragmento recebe um índice do vocabulário. Um token não é necessariamente uma palavra.',
      input: 'Texto: Hello, I am',
      output: '(1, 4)',
      source: 'GPTModel',
      bridge: 'Agora os IDs podem buscar suas linhas na tabela de embeddings.',
    },
    {
      label: 'Token embedding',
      title: 'Um ID encontra seu vetor.',
      description:
        'O ID não mede significado: ele é um endereço. A tabela tok_emb busca uma linha de 768 componentes para cada um dos quatro tokens.',
      input: '(1, 4)',
      output: '(1, 4, 768)',
      source: 'GPTModel',
      bridge: 'Temos a identidade dos tokens. Ainda falta informar a ordem.',
    },
    {
      label: 'Posição + embedding',
      title: 'A ordem entra na soma.',
      description:
        'Cada posição busca outro vetor de 768 componentes. Ele é somado ao embedding do token, componente por componente. Na avaliação, o dropout é desativado.',
      input: '(1, 4, 768) + (4, 768)',
      output: '(1, 4, 768)',
      source: 'GPTModel',
      bridge: 'Identidade e posição seguem juntas para o primeiro Transformer.',
    },
    {
      label: 'LayerNorm',
      title: 'Preparar antes de relacionar.',
      description:
        'A primeira normalização do bloco calcula média e variância nas 768 componentes de cada token. Isso ajuda a controlar a escala das ativações e estabilizar o treinamento. Tokens diferentes não se misturam nesta operação.',
      input: '(1, 4, 768)',
      output: '(1, 4, 768)',
      source: 'LayerNorm',
      bridge: 'O vetor normalizado alimenta as projeções de query, key e value.',
    },
    {
      label: 'Atenção causal',
      title: 'Cada posição consulta o que já chegou.',
      description:
        'As 12 cabeças relacionam cada posição consigo mesma e com as anteriores. Q e K determinam os pesos; a soma ponderada de V reúne a informação. A máscara impede que uma posição consulte tokens futuros. Depois, as cabeças são concatenadas e projetadas de volta para 768 dimensões.',
      input: '(1, 4, 768)',
      output: '(1, 4, 768)',
      source: 'MultiHeadAttention',
      bridge:
        'A atenção produz uma atualização. O caminho original também chega à próxima soma.',
    },
    {
      label: 'Primeira residual',
      title: 'Somar o contexto à entrada.',
      description:
        'O bloco guardou sua entrada antes da normalização. Agora soma essa cópia à atualização produzida pela atenção, preservando a largura do vetor. O atalho também fornece um caminho direto para o gradiente no treinamento; ele ajuda a otimização, sem garantir que todo problema de gradiente desapareça.',
      input: 'x + atenção(norm1(x))',
      output: '(1, 4, 768)',
      source: 'TransformerBlock',
      bridge: 'A primeira soma segue para a transformação de cada token.',
    },
    {
      label: 'Feed-forward',
      title: 'Expandir. Transformar. Comprimir.',
      description:
        'Após uma segunda LayerNorm, o vetor passa de 768 para 3.072 componentes, atravessa GELU e volta a 768. GELU introduz uma transformação não linear suave, que pode manter valores negativos pequenos. Os mesmos pesos são aplicados separadamente a cada posição.',
      input: '(1, 4, 768)',
      output: '(1, 4, 768)',
      source: 'FeedForward',
      bridge: 'O feed-forward entrega outra atualização, pronta para a segunda residual.',
    },
    {
      label: 'Segunda residual',
      title: 'Completar um bloco.',
      description:
        'A saída da primeira residual é preservada como atalho. A atualização do feed-forward é somada a ela. Os dois atalhos ficam dentro de cada bloco. A forma se mantém, mas o conteúdo do vetor é transformado.',
      input: 'r₁ + FFN(norm2(r₁))',
      output: '(1, 4, 768)',
      source: 'TransformerBlock',
      bridge: 'Um bloco terminou. Sua saída se torna a entrada do próximo.',
    },
    {
      label: 'Pilha de blocos',
      title: 'O mesmo percurso, 12 vezes.',
      description:
        'O modelo empilha 12 Transformers. Cada um repete as mesmas operações com seus próprios parâmetros. O notebook registra a saída de todos os blocos.',
      input: '(1, 4, 768)',
      output: '(1, 4, 768)',
      source: 'GPTModel',
      bridge: 'Depois do último bloco, uma normalização final prepara a projeção de saída.',
    },
    {
      label: 'LayerNorm final',
      title: 'Preparar a leitura final.',
      description:
        'Esta normalização pertence ao modelo, fora da pilha de Transformers. Ela tem seus próprios parâmetros de escala e deslocamento.',
      input: '(1, 4, 768)',
      output: '(1, 4, 768)',
      source: 'LayerNorm',
      bridge: 'Cada vetor final pode agora ser projetado sobre o vocabulário.',
    },
    {
      label: 'Projeção de saída',
      title: '768 componentes. 50.257 logits.',
      description:
        'A camada out_head combina as componentes do vetor para atribuir um score a cada entrada do vocabulário. Um logit é um score bruto, ainda não uma probabilidade.',
      input: '(1, 4, 768)',
      output: '(1, 4, 50257)',
      source: 'GPTModel',
      bridge:
        'Para continuar o texto, generate_text_simple usa apenas os logits da última posição.',
    },
    {
      label: 'Geração',
      title: 'A saída volta a ser entrada.',
      description:
        'O código escolhe o maior logit da última posição com argmax e anexa seu ID. O contexto ampliado atravessa o modelo novamente. Aqui você percorre os IDs já registrados no notebook.',
      input: 'Última posição → argmax',
      output: 'Um novo ID no contexto',
      source: 'generate_text_simple',
      bridge:
        'O ciclo de geração está completo. Falta entender como os pesos aprendem a fazer boas previsões.',
    },
  ],

  basicSteps: [
    {
      title: 'Preparando o texto',
      subtitle: 'Tokenização e embeddings',
      range: [0, 3],
      description:
        'O modelo trabalha com números. Primeiro, o tokenizador transforma o texto em IDs. Cada ID busca um vetor de 768 componentes; depois, o modelo soma a ele um vetor de posição para representar a ordem dos tokens.',
      flow: [
        ['Texto → IDs', 'Hello, I am → 4 tokens'],
        ['IDs → vetores', '768 componentes por token'],
        ['+ posição', 'identidade e ordem juntas'],
      ],
      takeaway:
        'Saída: quatro vetores de 768 componentes, prontos para entrar no primeiro Transformer.',
      detail: 'Operações 01–03 · tokenização, embedding e posição',
    },
    {
      title: 'O bloco Transformer',
      subtitle: 'O coração do modelo',
      range: [3, 9],
      description:
        'Os vetores passam por 12 blocos. Em cada um, a atenção combina informação da própria posição e das anteriores; a rede feed-forward transforma cada vetor. Normalizações ajudam a controlar a escala dos valores, e atalhos somam a entrada às atualizações.',
      flow: [
        ['Atenção causal', 'relaciona posições permitidas'],
        ['Feed-forward + GELU', '768 → 3.072 → 768'],
        ['Repetir ×12', 'mesma estrutura, pesos próprios'],
      ],
      takeaway:
        'A ordem dentro de cada bloco: normalizar → atenção → somar atalho → normalizar → feed-forward → somar atalho. Entram e saem quatro vetores de 768 componentes; seu conteúdo muda.',
      detail: 'Operações 04–09 · normalização, atenção, atalhos, GELU e pilha',
    },
    {
      title: 'A camada final de saída e os logits',
      subtitle: 'Uma pontuação para cada token possível',
      range: [9, 11],
      description:
        'Depois dos 12 blocos, uma última normalização prepara os vetores. A camada de saída projeta cada vetor de 768 componentes em 50.257 pontuações — uma para cada token do vocabulário. Essas pontuações brutas são os logits.',
      flow: [
        ['768', 'componentes por posição'],
        ['Norm final → saída', 'projeção sobre o vocabulário'],
        ['50.257', 'logits por posição'],
      ],
      takeaway:
        'Logits ainda não são probabilidades. O modelo produz essas pontuações para todas as posições; para continuar a frase, usamos apenas a última.',
      detail: 'Operações 10–11 · normalização final e projeção de saída',
    },
    {
      title: 'Escolhendo o próximo token e repetindo',
      subtitle: 'Greedy decoding',
      range: [11, 12],
      description:
        'A softmax transforma os logits da última posição em probabilidades que somam 100%. No modo greedy, escolhemos o token de maior pontuação, anexamos seu ID ao contexto e repetimos o processo. O tokenizador converte os IDs de volta em texto.',
      flow: [
        ['Última posição', 'logits → probabilidades'],
        ['Escolher o maior', 'argmax → ID'],
        ['Anexar e repetir', 'o contexto ganha um token'],
      ],
      takeaway:
        'No código da aula, argmax é aplicado diretamente aos logits: a softmax preserva a ordem e não altera essa escolha. A continuação real salva no notebook está no detalhe avançado abaixo.',
      detail: 'Operação 12 · percorrer a geração registrada, ID por ID',
    },
  ],

  learningModes: {
    inference: {
      items: [
        ['Contexto', 'tokens já disponíveis'],
        ['Forward', 'pesos fixos'],
        ['Próximo ID', 'anexar e repetir'],
      ],
      text: 'Gerar usa os parâmetros existentes. Não há cálculo de gradientes nem atualização dos pesos em generate_text_simple. Esta página reproduz os IDs salvos, sem rodar o modelo.',
    },
    training: {
      items: [
        ['Texto + alvos', 'próximo token real'],
        ['Forward + perda', 'avaliar a previsão'],
        ['Backward + otimizador', 'atualizar parâmetros'],
      ],
      text: 'No pré-treinamento, a entropia cruzada penaliza previsões ruins para o token alvo. Backpropagation calcula os gradientes; o otimizador ajusta os parâmetros. Repetir isso em muitos textos ensina regularidades da linguagem. Este fluxo é conceitual: não há treino executado ou métricas de perda neste registro.',
    },
  },
};
