# GPT por dentro — v2

Abra `index.html` no navegador. Funciona offline, mantendo junto os arquivos
`gpt_arch_v2.css`, `gpt_arch_v2_base.js` e `gpt_arch_v2.js`.

A **v1 permanece em `../v1/index.html`**, com seu CSS e JavaScript originais intactos.
O link “Abrir v1” permite comparar as versões.

## Arquitetura variável (modo inicial)

O seletor agora altera o desenho da arquitetura inteira, além dos shapes:

| Família | Dimensões | Cabeças | Blocos | Neurônios ocultos do FFN |
|---|---:|---:|---:|---:|
| Small | 768 | 12 | 12 | 3.072 |
| Medium | 1.024 | 16 | 24 | 4.096 |
| Large | 1.280 | 20 | 36 | 5.120 |
| XL | 1.600 | 25 | 48 | 6.400 |

- **Embeddings:** todos os componentes são cobertos por grupos de 64 dimensões;
  clique num grupo para abrir suas 64 dimensões individuais.
- **Atenção:** cada cabeça é desenhada e selecionável. A seleção identifica sua
  faixa de 64 dimensões. A matriz mostra conexões permitidas/bloqueadas, não pesos
  inventados. A máscara pode ser removida para comparar a estrutura.
- **Feed-forward:** cada nó representa 64 neurônios. Selecione um grupo oculto para
  destacar suas conexões com todos os grupos de entrada e saída. Cada ligação
  entre grupos equivale a 4.096 pesos, não a uma única sinapse.
- **Blocos:** todos os 12/24/36/48 blocos são desenhados e selecionáveis; conectores
  seguem a ordem numérica. Cada bloco tem pesos independentes.
- **Saída:** 50.257 logits são cobertos por 49 grupos de 1.024 e um grupo de 81.
- **Geração:** cinco cliques completam “Hello, I am a GPT built from scratch”.
  É um roteiro guiado, com segmentação local, e não texto previsto por GPT-2.
  Volte às etapas anteriores para explorar a sequência ampliada.

A animação representa **fluxo esquemático**. A arquitetura respeita as dimensões
GPT-2 do capítulo, com QKV sem bias, mas não carrega pesos nem executa um GPT-2
completo no navegador. Escalas de agrupamento são explícitas. Escolher uma cabeça
ou um bloco não produz valores de ativações nesse modo estrutural.

## Laboratório numérico

O seletor “Exploração” abre o mini-GPT da v1: D=8, H=2, L=2, vocabulário local
com 12 entradas, pesos determinísticos e não treinados. Nesse modo os números
são calculados e GPT-2 serve apenas como referência. O cabeçalho identifica essa
mudança. Frase guiada e argmax continuam disponíveis no laboratório.

Ao voltar do laboratório para a arquitetura, uma sequência fora do roteiro é
restaurada para “Hello, I am”. Uma sequência do roteiro é preservada.

## Controles

Setas: navegar; espaço fora dos controles: reproduzir/pausar; “Ampliar diagrama”:
mais espaço para a rede. Diagramas longos usam a rolagem da página. O seletor de
família preserva a etapa e limita a seleção de grupos/cabeças/blocos ao novo modelo.

## Validação realizada

- 48 combinações (4 famílias × 12 etapas) renderizadas no navegador sem erros.
- Contagens DOM: dimensões agrupadas 12/16/20/25; cabeças 12/16/20/25;
  grupos ocultos 48/64/80/100; blocos 12/24/36/48.
- Seleção do bloco 48 do XL e dimensões 1536–1599; troca de volta para Small.
- Frase guiada completa; alternância entre modo estrutural e laboratório numérico.
- Layout móvel com viewport de 390 px sem transbordamento horizontal da página.
- SHA-256 dos três arquivos da v1 conferidos antes/depois: idênticos.

## Interações da arquitetura

A v2 inclui também `gpt_arch_v2_interactions.js`; mantenha esse arquivo junto dos
outros. A camada de interação acrescenta:

- **Animar operação**: percorre uma vez os subpassos da etapa. Cada subpasso pode
  ser escolhido manualmente; a reprodução pode ser pausada. Mudar de etapa,
  modelo ou modo interrompe o percurso. A animação é explicativa, não inferência.
- **Embedding**: clique numa ocorrência do contexto para buscar sua linha; clique
  numa célula para abrir um grupo; clique num ponto para inspecionar a dimensão.
  O painel identifica `E[token, d]` e `P[posição, d]` sem inventar pesos.
- **LayerNorm**: grupos e dimensões selecionáveis mostram qual componente recebe
  gamma e beta, e que média/variância ainda usam todas as dimensões do token.
- **Atenção**: cada célula identifica query, key, bloqueio causal e value consultado.
- **Residuais**: “Seguir o atalho”, “Seguir a atualização” e “Encontrar na soma”
  destacam os caminhos correspondentes.
- **Feed-forward**: explorador GELU numérico independente; o controle não modifica
  nem presume ativações do GPT-2.
- **Blocos**: avanço e retorno entre blocos, além da seleção direta.
- **Saída**: grupos de logits selecionáveis; informe um índice e use “Inspecionar
  coluna” para rastrear seus D pesos na projeção. O logit não é calculado.

O inspetor acompanha a rolagem no desktop para mostrar o resultado de cada clique.
