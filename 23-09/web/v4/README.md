# GPT por dentro — v4

Integra a explicação do funcionamento do GPT à jornada da v3, mantendo os registros reais e os seletores de token, dimensão, cabeça, bloco e rodada.

Cinco momentos organizam a leitura: representar, contextualizar, pontuar, continuar e aprender. As 12 operações continuam acessíveis pelo índice.

As explicações novas ficam junto da operação correspondente: normalização, residual, GELU, softmax, greedy e temperatura. O fechamento compara geração e treinamento. Não há pesos, logits, probabilidades ou inferências inventados.

Abra `index.html` mantendo os arquivos da versão na mesma pasta. Os links de origem pressupõem esta pasta em `23-09/web/v4/`.

## Estrutura do código

- `index.html` contém somente a estrutura permanente da página.
- `content.js` reúne os textos e a configuração dos quatro passos e das 12 operações.
- `views.js` contém os templates das figuras e seções.
- `gpt_journey.js` mantém o estado, os eventos e as atualizações da interface.
- `../shared/gpt_journey_data.js` é o snapshot exportado do notebook e compartilhado com a v3.
- `gpt_journey.css` organiza os estilos por região, com formatação legível e comentários de seção.

Os arquivos usam scripts tradicionais em ordem explícita para continuar funcionando ao abrir a página localmente, sem bundler ou etapa de build.

O exportador lê `lesson/chapter4_visual_lesson.ipynb` e `lesson/gpt.py` e atualiza o snapshot compartilhado. Executar `python3 23-09/scripts/export_journey.py` é suficiente para atualizar v3 e v4; não há cópias por versão.

Referência: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py

## Verificação

Snapshot, hash do notebook, células e código conferidos com as fontes locais. Sintaxe JavaScript validada. No navegador: token, dimensão 767, cabeça 12, máscara causal, bloco 12, dez rodadas de geração, retorno, reinício e modo treinamento. Layout de 390 px sem transbordamento horizontal; nenhuma âncora ausente ou erro JavaScript observado.

## Organização revisada

A leitura principal tem os quatro passos do texto de referência: preparação, Transformer, logits e geração. Cada passo contém um painel Avançado fechado por padrão. Os painéis agrupam as 12 operações originais (01–03, 04–09, 10–11 e 12), mantendo todos os controles e dados. Links diretos a operações abrem o painel correspondente. O treinamento é um complemento após os quatro passos.
