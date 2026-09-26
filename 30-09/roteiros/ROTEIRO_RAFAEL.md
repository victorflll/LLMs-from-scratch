# RAFAEL — Material de estudo + roteiro de apresentação

Rafael apresenta **dois notebooks** da aula 30-09 (ver `30-09/README.md`):

| Ordem na aula | Notebook | O que sai dele |
|---|---|---|
| 2º bloco (logo depois do Elmo) | `02_atencao.ipynb` | `MultiHeadAttention` → vira peça do `TransformerBlock` no notebook 03 (Victor) |
| 6º bloco (logo depois do classificador do Elmo) | `06_assistente.ipynb` | GPT-2 que segue instruções |

> **Como usar este material:** cada ETAPA corresponde a uma célula de código do notebook, **na mesma ordem**.
> As células de texto (markdown) aparecem como "títulos de seção" — elas são o momento de fazer a transição.
> No final de cada notebook há o **roteiro contínuo**: só as falas, em ordem, para ensaiar.

> **Tudo que está em "Resultado" foi tirado da saída salva no próprio notebook.** Quando o resultado pode mudar ao rodar ao vivo, isso está avisado.

---

# NOTEBOOK 02 — `02_atencao.ipynb`

## Mapa do notebook

```text
x (6 tokens × 3 dims)  ← embeddings de brinquedo: "Your journey starts with one step"
   │
   ├─ Seção 1: Q, K, V e o score        → Q, K, V (6,2) → scores (6,6) → pesos (6,6) → contexto (6,2)
   ├─ Seção 2: outras formas de score   → produto escalar / escalado / aditivo / cosseno
   │           por que dividir por √d   → experimento com d = 2, 64, 768
   ├─ Seção 3: máscara causal           → não olhar o futuro
   ├─ Seção 4: multi-head               → pesos_atencao(); nº de heads (1, 2, 4, 12, 24)
   ├─ Seção 5: não divisível            → num_heads=5 dá erro; solução: head_dim explícito
   └─ Seção 6: GPT-2 de verdade         → heatmaps das 12 heads da camada 5; para onde " it" olha
                                                     │
                                                     ▼
                         notebook 03 (Victor): MultiHeadAttention vira parte do TransformerBlock
```

Cadeia de variáveis da Seção 1 (a mais importante):

```text
x (6,3)
 ├─ W_q ─► Q (6,2) ─┐
 ├─ W_k ─► K (6,2) ─┴─► scores = Q @ Kᵀ (6,6) ─► ÷√d_k ─► softmax ─► pesos (6,6) ─┐
 └─ W_v ─► V (6,2) ──────────────────────────────────────────────────────────────┴─► contexto = pesos @ V (6,2)

scores → usado de novo na Seção 2 (variantes) e na Seção 3 (máscara)
Q, K   → usados de novo na Seção 2 (aditivo e cosseno)
```

---

## Abertura (markdown 0)

**Entra:** embeddings do notebook 01. **Sai:** `MultiHeadAttention`.
Perguntas: 1) outras formas de gerar o score? 2) visualizar os scores de 1 head? 3) mudar a quantidade de heads? 4) e se a dimensão não for divisível pelo número de heads?

**Fala:**
"O Elmo terminou com cada token virando um vetor. Só que esse vetor é isolado: o vetor de 'it' é o mesmo vetor em qualquer frase, não sabe se 'it' é um animal ou uma rua. A atenção é o mecanismo que deixa cada token olhar para os outros tokens da frase e misturar a informação deles no seu próprio vetor. Eu vou montar isso passo a passo e responder quatro perguntas: se existem outras formas de calcular a atenção, o que muda com o número de heads, o que acontece se a dimensão não for divisível pelas heads, e no final a gente vai olhar a atenção de verdade dentro do GPT-2."

---

## ETAPA 1 — Imports, embeddings de exemplo e função de desenho (célula 1)

### Código
```python
import matplotlib.pyplot as plt
import tiktoken
import torch

from aula import load_gpt2
from llms_from_scratch.ch03 import MultiHeadAttention

torch.manual_seed(123)
tokens = ["Your", "journey", "starts", "with", "one", "step"]
# embeddings de 3 dimensões (exemplo do livro) para caber na tela
x = torch.tensor([[0.43, 0.15, 0.89], [0.55, 0.87, 0.66], [0.57, 0.85, 0.64],
                  [0.22, 0.58, 0.33], [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]])


def mapa(ax, pesos, titulo, rotulos=tokens):
    ax.imshow(pesos.detach(), cmap="Blues", vmin=0)
    ax.set_xticks(range(len(rotulos)), rotulos, rotation=60)
    ax.set_yticks(range(len(rotulos)), rotulos)
    ax.set_title(titulo, fontsize=9)
```

### O que está acontecendo?
- `MultiHeadAttention` vem do capítulo 3 do livro (`pkg/llms_from_scratch/ch03.py`). É a classe que o GPT usa de verdade; aparece a partir da Seção 4.
- `load_gpt2` (de `aula.py`) só é usado na Seção 6.
- `tokens`: as 6 palavras da frase de exemplo do livro.
- `x`: um **embedding de brinquedo**, digitado à mão: 6 linhas (uma por token) × 3 colunas (3 dimensões, em vez de 768, para caber na tela). No GPT real isso seria o tensor que o Elmo produziu.
- `mapa(ax, pesos, titulo, rotulos)`: função auxiliar que desenha uma matriz de pesos como heatmap azul (mais escuro = mais atenção). **Linha = quem está olhando, coluna = para quem está olhando.** Vai ser usada nas Seções 2, 3 e 6.

### Variáveis
| Variável | O que é | Tipo / shape | Usada depois em |
|---|---|---|---|
| `tokens` | rótulos dos 6 tokens | `list[str]` | rótulos dos heatmaps (ETAPAS 3 e 5) |
| **`x`** | embeddings de brinquedo | `torch.Tensor`, **(6, 3)** = (tokens, dimensões) | **ETAPA 2** (gera Q, K, V) |
| `mapa` | função de desenho | função | ETAPAS 3, 5, 9 |

### Fala
"Para dar para enxergar as contas, eu não vou usar os vetores de 768 números do Elmo. Vou usar o exemplo do livro: a frase 'Your journey starts with one step', com 6 tokens, e cada token com um embedding de só 3 números. Então esse `x` é uma matriz 6 por 3: uma linha por token. Essa função `mapa` só desenha as matrizes de atenção. Na hora de ler os gráficos, lembrem: a linha é quem está olhando, a coluna é para quem ele está olhando."

---

## Seção 1 (markdown 2): "Q, K, V e o score"

Cada token vira uma **query** (o que procuro), uma **key** (o que ofereço) e um **value** (o que entrego). Score = quão bem a query de um token "casa" com a key de outro.

**Transição:** "A ideia da atenção usa uma analogia de busca. Cada token vai gerar três vetores: uma query, que é 'o que eu estou procurando'; uma key, que é 'o que eu tenho para oferecer'; e um value, que é 'o conteúdo que eu entrego se alguém me escolher'. O score mede o quanto a query de um token combina com a key de outro."

---

## ETAPA 2 — Q, K, V → scores → softmax → pesos → contexto (célula 3)

### Código
```python
d_k = 2
W_q, W_k, W_v = (torch.nn.Linear(3, d_k, bias=False) for _ in range(3))
Q, K, V = W_q(x), W_k(x), W_v(x)
scores = Q @ K.T
pesos = torch.softmax(scores / d_k**0.5, dim=-1)  # cada linha soma 1
contexto = pesos @ V
print("scores", tuple(scores.shape), "| pesos (linha soma):", pesos.sum(-1)[:2], "| contexto", tuple(contexto.shape))
```

### O que está acontecendo? (linha por linha)

**1. `d_k = 2`** — tamanho dos vetores de query e key (e aqui também de value). Escolhido pequeno para o exemplo.

**2. Criar as projeções `W_q`, `W_k`, `W_v`**
`torch.nn.Linear(3, d_k, bias=False)` é uma camada que multiplica o vetor de entrada (3 números) por uma matriz de pesos treináveis, gerando um vetor de 2 números. São **três camadas diferentes**, cada uma com seus próprios pesos (aleatórios, porque não houve treino — o `manual_seed(123)` fixa esses valores).

**3. Criar Q, K, V**
`W_q(x)` aplica a projeção em **cada uma das 6 linhas** de `x` de uma vez:
```text
x (6, 3)  ──W_q──►  Q (6, 2)     uma query por token
x (6, 3)  ──W_k──►  K (6, 2)     uma key por token
x (6, 3)  ──W_v──►  V (6, 2)     um value por token
```
O **mesmo** embedding gera três vetores diferentes, porque as matrizes são diferentes. É isso que permite o token "procurar" uma coisa e "oferecer" outra.

**4. Scores: `scores = Q @ K.T`**
- `K.T` é a transposta de K: (6, 2) → (2, 6).
- `@` é multiplicação de matrizes: (6, 2) @ (2, 6) = **(6, 6)**.
- Cada posição `scores[i, j]` é o **produto escalar** (dot product) entre a query do token `i` e a key do token `j`: multiplica número a número e soma. Quanto maior, mais os dois vetores "apontam na mesma direção" → mais o token `i` se interessa pelo token `j`.
```text
scores[i, j] = Q[i] · K[j] = Q[i,0]·K[j,0] + Q[i,1]·K[j,1]

             key de: Your journey starts with one step
query de Your      [  s     s       s     s    s    s  ]
query de journey   [  s     s       s     s    s    s  ]
...                                                        (6 × 6)
```

**5. Escala + softmax: `pesos = torch.softmax(scores / d_k**0.5, dim=-1)`**
- `scores / d_k**0.5` divide por √2. É a **escala** — o motivo aparece na ETAPA 4.
- `softmax(..., dim=-1)` age em **cada linha** separadamente: transforma os 6 scores de um token em 6 números positivos que **somam 1**. Score maior → peso maior. Os scores são "notas cruas"; os pesos são "porcentagens de atenção".

**6. Soma ponderada: `contexto = pesos @ V`**
- (6, 6) @ (6, 2) = **(6, 2)**.
- A linha `i` do resultado é `pesos[i,0]·V[0] + pesos[i,1]·V[1] + ... + pesos[i,5]·V[5]`: uma **média ponderada dos values** de todos os tokens, usando os pesos de atenção do token `i`.
- Esse é o **vetor de contexto**: a nova representação do token `i`, que agora contém informação dos outros tokens.

O fluxo completo:
```text
Q (6,2)   K (6,2)   V (6,2)
   └── Q @ Kᵀ ──┘       │
        ↓               │
Attention Scores (6,6)  │   notas cruas: query i × key j
        ↓ ÷ √d_k        │
        ↓ softmax (por linha)
Attention Weights (6,6) │   cada linha soma 1
        └──── @ V ──────┘
        ↓
Weighted Sum = contexto (6,2)   um vetor novo por token
```

### Variáveis
| Variável | O que é | Shape | Usada depois em |
|---|---|---|---|
| `d_k` | dimensão de query/key | `int` = 2 | ETAPAS 3 e 5 (escala) |
| `W_q`, `W_k`, `W_v` | projeções lineares 3 → 2 | pesos (2, 3) cada | aqui |
| **`Q`, `K`** | queries e keys | **(6, 2)** | **ETAPA 3** (score aditivo e cosseno) |
| `V` | values | (6, 2) | aqui |
| **`scores`** | attention scores | **(6, 6)** | **ETAPA 3** e **ETAPA 5** (máscara) |
| `pesos` | attention weights | (6, 6) | aqui |
| `contexto` | vetores de contexto | (6, 2) | aqui |

### Resultado
```
scores (6, 6) | pesos (linha soma): tensor([1.0000, 1.0000], grad_fn=<SliceBackward0>) | contexto (6, 2)
```
- `scores` é 6×6: um score para cada par (token olhando, token olhado).
- As duas primeiras linhas de `pesos` somam exatamente 1: o softmax virou distribuição de probabilidade.
- `contexto` tem o mesmo número de tokens (6) e a dimensão do value (2).
- O `grad_fn=...` só indica que o PyTorch está rastreando essas contas para poder treinar os pesos de `W_q`, `W_k`, `W_v` — não é um erro.

### Fala
"Essa célula é a atenção inteira em cinco linhas, então vou devagar.
Primeiro eu crio três camadas lineares: `W_q`, `W_k` e `W_v`. Cada uma é uma matriz de pesos que transforma o embedding de 3 números num vetor de 2. Aplicando no `x`, cada token ganha três vetores: sua query, sua key e seu value. É o mesmo embedding, mas três matrizes diferentes, então três papéis diferentes.
Depois vem o score: `Q @ K.T`. Isso é uma multiplicação de matrizes que calcula, de uma vez só, o produto escalar da query de cada token com a key de cada token. Sai uma matriz 6 por 6: na linha 'journey', coluna 'step', está o quanto 'journey' se interessa por 'step'.
Esses scores são notas cruas, podem ser qualquer número. O softmax transforma cada linha em porcentagens: tudo positivo e somando 1. Olhem aqui: as linhas somam 1. Antes do softmax eu divido por raiz de `d_k`; já já eu mostro por quê.
E por último, `pesos @ V`: para cada token eu faço uma média dos values de todos os tokens, usando essas porcentagens. Se 'journey' dá 50% de atenção para 'starts', metade do novo vetor de 'journey' vem do value de 'starts'. Esse é o vetor de contexto: a nova versão do token, agora misturada com a informação da frase."

---

## Seção 2 (markdown 4): "Outras formas de calcular o score"

**Transição:** "Aqui entra a primeira pergunta. Esse produto escalar escalado é o que o GPT usa, mas não é a única forma de medir o quanto uma query combina com uma key. Vou comparar quatro."

---

## ETAPA 3 — Quatro formas de calcular o score (célula 5)

### Código
```python
torch.manual_seed(0)
W_a = torch.nn.Linear(2 * d_k, 4)
v_a = torch.nn.Linear(4, 1, bias=False)
aditivo = v_a(torch.tanh(W_a(torch.cat(torch.broadcast_tensors(Q[:, None], K[None, :]), dim=-1)))).squeeze(-1)

variantes = {
    "produto escalar\nQ·K": scores,
    "escalado (GPT)\nQ·K/√d": scores / d_k**0.5,
    "aditivo (Bahdanau 2014)\nvᵀ tanh(W[q;k])": aditivo,
    "cosseno\ncos(q, k)/0.1": torch.nn.functional.cosine_similarity(Q[:, None], K[None, :], dim=-1) / 0.1,
}
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, (nome, s) in zip(axs, variantes.items()):
    mapa(ax, torch.softmax(s, dim=-1), nome)
plt.show()
```

### O que está acontecendo?
São 4 maneiras de gerar a matriz de scores (6×6). Depois, para cada uma, aplica o softmax por linha e desenha.

**1) Produto escalar `Q·K`** — é o `scores` da ETAPA 2, sem dividir.

**2) Produto escalar escalado `Q·K/√d`** — o que o GPT usa (o mesmo cálculo da ETAPA 2).

**3) Aditivo (Bahdanau, 2014)** — a atenção original, anterior ao Transformer. Em vez de multiplicar q e k, **concatena** os dois e passa por uma pequena rede neural:
```text
Q[:, None]   (6, 1, 2)
K[None, :]   (1, 6, 2)
   ↓ broadcast_tensors → ambos viram (6, 6, 2)   ← todos os pares (query i, key j)
   ↓ torch.cat(dim=-1)  → (6, 6, 4)              ← [q_i ; k_j] lado a lado
   ↓ W_a  Linear(4, 4)  → (6, 6, 4)
   ↓ tanh
   ↓ v_a  Linear(4, 1)  → (6, 6, 1)
   ↓ squeeze(-1)        → (6, 6)                 ← um score por par
```
`W_a` e `v_a` são pesos novos, aleatórios (seed 0). É mais flexível, mas mais caro que um produto escalar.

**4) Cosseno `cos(q, k)/0.1`** — similaridade cosseno (a mesma que o Elmo usou nas palavras): só compara a **direção** dos vetores, ignora o tamanho. Fica entre −1 e 1; dividir por 0,1 (uma "temperatura") estica o intervalo para −10 a 10, senão o softmax ficaria quase uniforme.

### Variáveis
| Variável | O que é | Shape |
|---|---|---|
| `W_a`, `v_a` | pesos da atenção aditiva | Linear(4,4) e Linear(4,1) |
| `aditivo` | scores aditivos | (6, 6) |
| `variantes` | nome → matriz de scores | dict com 4 matrizes (6, 6) |

### Resultado (gráfico salvo: 4 heatmaps 6×6)
- **Produto escalar** e **escalado**: cores muito parecidas e quase uniformes. O escalado é ainda um pouco mais "liso" (dividir por √2 aproxima os scores).
- **Aditivo**: praticamente uniforme — todos os tokens recebem quase a mesma atenção.
- **Cosseno /0.1**: bem mais contrastado — algumas células escuras, outras quase brancas.
- ⚠️ Interpretação honesta: **todos os pesos aqui são aleatórios (não treinados)** e os vetores são minúsculos (2 dimensões). O gráfico mostra que cada fórmula gera uma distribuição diferente, **não** qual é "melhor". Quem decide isso é o treino.

### Fala
"Aqui eu calculo o score de quatro formas diferentes. As duas primeiras vocês já conhecem: o produto escalar puro e o escalado, que é o do GPT. A terceira é a atenção aditiva, de 2014, que é anterior ao Transformer: em vez de multiplicar a query pela key, ela coloca as duas lado a lado e passa por uma mini rede neural. Esse monte de `broadcast`, `cat` e `squeeze` é só para montar todos os pares query-key de uma vez. A quarta é a similaridade cosseno, que só olha a direção dos vetores; eu divido por 0,1 para esticar o intervalo, senão o softmax fica tudo igual.
O resultado: cada fórmula distribui a atenção de um jeito. Mas atenção: aqui todos os pesos são aleatórios, ninguém treinou nada. Então o gráfico não diz qual é melhor; ele mostra que o score é uma escolha de projeto. O Transformer ficou com o produto escalar porque é só uma multiplicação de matrizes, e isso é extremamente rápido em GPU."

### Possível pergunta do professor/aluno
**"Existem outras abordagens para gerar os attention scores?"**
Resposta: "Sim, essa célula mostra três alternativas: produto escalar sem escala, aditivo e cosseno. O produto escalar escalado venceu por ser barato e funcionar bem. E, como diz o próximo markdown, os modelos modernos quase não mexem na fórmula do score; eles mexem em como Q, K e V são calculados."

---

## Markdown 6: "Por que dividir por √d?"

**Transição:** "Agora a pergunta que ficou pendente: por que dividir por raiz de d? Com vetores grandes, o produto escalar fica grande, e o softmax exagera."

---

## ETAPA 4 — Por que a escala √d importa (célula 7)

### Código
```python
for d in [2, 64, 768]:
    q, k = torch.randn(d), torch.randn(10, d)
    s = k @ q
    print(f"d={d:>3}: maior peso sem escala = {torch.softmax(s, 0).max():.2f} | com escala = {torch.softmax(s / d**0.5, 0).max():.2f}")
```

### O que está acontecendo?
- O `for` testa 3 tamanhos de vetor: 2 (nosso exemplo), 64 (o tamanho de cada head no GPT-2) e 768.
- `q`: **uma** query aleatória, shape (d,). `k`: **10** keys aleatórias, shape (10, d).
- `s = k @ q` → shape (10,): o score da query com cada uma das 10 keys.
- Imprime o **maior peso** depois do softmax, sem escala e com escala.
- Por quê: um produto escalar é uma soma de `d` termos; quanto maior `d`, maiores (em módulo) ficam os scores — o desvio padrão cresce como √d. Dividir por √d traz os scores de volta para uma escala parecida, qualquer que seja o `d`.

### Resultado
```
d=  2: maior peso sem escala = 0.45 | com escala = 0.33
d= 64: maior peso sem escala = 0.95 | com escala = 0.26
d=768: maior peso sem escala = 1.00 | com escala = 0.24
```
- Sem escala, com d=768, **um único token leva 100% da atenção** (softmax virou quase *one-hot*: um 1 e o resto 0).
- Com escala, o maior peso fica em torno de 0,25 em todos os tamanhos: a atenção continua distribuída.
- Por que isso é ruim no treino: quando o softmax satura em 0 e 1, o gradiente fica quase zero e o modelo praticamente para de aprender a ajustar a atenção.

### Fala
"Aqui eu pego uma query e 10 keys aleatórias, calculo os scores e vejo qual é o maior peso depois do softmax. Com vetores de 2 dimensões, tanto faz. Mas com 64, que é o tamanho de cada head no GPT-2, sem escala um único token já leva 95% da atenção. Com 768 dimensões, leva 100%: o modelo só olha para uma palavra. Isso acontece porque o produto escalar soma `d` termos, então quanto mais dimensões, maior o número. Com a divisão por raiz de d, o maior peso fica em uns 25% para qualquer tamanho. E isso importa no treino: quando o softmax trava em 0 e 1, o gradiente praticamente some e o modelo não consegue aprender para onde olhar."

---

## Markdown 8 — variações modernas + Seção 3 "Máscara causal"

O markdown lista: GQA (Llama), MLA (DeepSeek), janela deslizante (Mistral/Gemma), atenção linear / DeltaNet. Código no repo: `ch04/04_gqa`, `05_mla`, `06_swa`, `08_deltanet`.

**Fala (curta, complementa a pergunta 1):**
"Só para completar a primeira pergunta: os modelos modernos mexem mais em quais Q, K e V são calculados do que na fórmula do score. O Llama, por exemplo, faz várias queries compartilharem a mesma key e value, o que economiza memória; o Mistral deixa cada token olhar só para uma janela de tokens próximos. Tem implementação de tudo isso no repositório, se alguém quiser ver depois."

**Transição para a máscara:** "Agora, um problema. Até aqui, todo token olha para todos os outros, inclusive os que vêm depois. Só que o GPT gera texto da esquerda para a direita: quando ele está prevendo a próxima palavra, ela ainda não existe. Se no treino ele pudesse olhar o futuro, ele simplesmente copiaria a resposta. Por isso existe a máscara causal."

---

## ETAPA 5 — Máscara causal (célula 9)

### Código
```python
n = len(tokens)
mascara = torch.triu(torch.ones(n, n), diagonal=1).bool()
s_mascarado = (scores / d_k**0.5).masked_fill(mascara, -torch.inf)  # -inf vira 0 depois do softmax

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
mapa(axs[0], torch.softmax(scores / d_k**0.5, -1), "sem máscara (vê o futuro)")
mapa(axs[1], torch.softmax(s_mascarado, -1), "com máscara causal (GPT)")
plt.show()
```

### O que está acontecendo?
- `n = 6` tokens.
- `torch.ones(n, n)` → matriz 6×6 de 1s. `torch.triu(..., diagonal=1)` mantém só o que está **acima da diagonal principal** (o resto vira 0). `.bool()` → `True` acima da diagonal = "posições do futuro".
```text
mascara (True = proibido)
           Your journey starts with one step
Your      [ F    T      T     T    T    T ]
journey   [ F    F      T     T    T    T ]
starts    [ F    F      F     T    T    T ]
with      [ F    F      F     F    T    T ]
one       [ F    F      F     F    F    T ]
step      [ F    F      F     F    F    F ]
```
- `masked_fill(mascara, -torch.inf)`: onde a máscara é `True`, troca o score (já escalado) por **−∞**.
- No softmax, e^(−∞) = 0 → esses tokens recebem **peso exatamente 0**, e o resto da linha é redistribuído para continuar somando 1.
- Os dois gráficos: softmax sem máscara (esquerda) e com máscara (direita).

### Variáveis
| Variável | O que é | Shape |
|---|---|---|
| `n` | nº de tokens | `int` = 6 |
| `mascara` | onde está o futuro | (6, 6) bool |
| `s_mascarado` | scores escalados com −∞ no futuro | (6, 6) |

### Resultado (gráfico salvo)
- **Esquerda:** matriz toda preenchida — todo token vê todos.
- **Direita:** triângulo inferior. "Your" (1º token) só vê a si mesmo → peso 1 (azul-escuro). "journey" divide entre 2 tokens, "starts" entre 3, ... "step" entre os 6. Tudo acima da diagonal é branco (peso 0).
- Repare que as linhas de baixo ficam mais claras: o mesmo 100% é dividido entre mais tokens.

### Fala
"A máscara é uma matriz 6 por 6 com `True` acima da diagonal: são as posições do futuro. Nessas posições eu troco o score por menos infinito. E aí vem o truque: no softmax, menos infinito vira exatamente zero. Então esses tokens recebem atenção zero, e o resto da linha se reajusta para continuar somando 1.
Comparem os dois gráficos. Sem máscara, todo mundo vê todo mundo. Com máscara, vira um triângulo: 'Your', que é o primeiro, só pode olhar para ele mesmo, então leva 100%. 'journey' divide entre 'Your' e ele mesmo. E 'step', o último, é o único que vê a frase inteira. Isso, aliás, é o motivo de o Elmo usar o último token no classificador de spam, mais tarde na aula."

---

## Seção 4 (markdown 10): "Multi-head: várias atenções em paralelo"

A `MultiHeadAttention` do livro **fatia** `d_out` em `num_heads` pedaços de `head_dim = d_out / num_heads`. Ela não devolve os pesos, então o notebook reproduz seus passos até o softmax.

**Transição:** "Até agora foi uma atenção só. Mas uma frase tem várias relações ao mesmo tempo: sujeito e verbo, pronome e substantivo, uma palavra e a vizinha. Em vez de uma atenção grande, o GPT faz várias atenções menores em paralelo, cada uma chamada de head. E o jeito de fazer isso é fatiar o vetor."

---

## ETAPA 6 — `pesos_atencao`: multi-head por dentro (célula 11)

### Código
```python
def pesos_atencao(mha, x):
    """Mesmo cálculo de MultiHeadAttention.forward, parando nos pesos: (batch, heads, tokens, tokens)."""
    b, t, _ = x.shape
    q = mha.W_query(x).view(b, t, mha.num_heads, mha.head_dim).transpose(1, 2)
    k = mha.W_key(x).view(b, t, mha.num_heads, mha.head_dim).transpose(1, 2)
    s = (q @ k.transpose(2, 3)).masked_fill(mha.mask.bool()[:t, :t], -torch.inf)
    return torch.softmax(s / mha.head_dim**0.5, dim=-1)
```

### O que a função recebe e devolve
- **Recebe:** `mha` (um objeto `MultiHeadAttention` já criado, com seus pesos) e `x` com shape `(batch, tokens, d_in)`.
- **Devolve:** os **attention weights de todas as heads**, shape `(batch, heads, tokens, tokens)`.
- **Por que existe:** a classe do livro só devolve o vetor de contexto final, não os pesos. Para poder **desenhar** a atenção (Seção 6), essa função repete as mesmas contas da classe, parando no softmax.

### O que acontece dentro (é aqui que a divisão em heads aparece)
Exemplo com o GPT-2: `b=1`, `t=6`, `d_out=768`, `num_heads=12`, `head_dim=64`.

```text
x                               (1, 6, 768)
  ↓ mha.W_query(x)              (1, 6, 768)       mesma projeção da ETAPA 2, só que 768 → 768
  ↓ .view(b, t, 12, 64)         (1, 6, 12, 64)    FATIA o vetor de 768 em 12 pedaços de 64
  ↓ .transpose(1, 2)            (1, 12, 6, 64)    coloca as heads "na frente": 12 atenções independentes
q                               (1, 12, 6, 64)
k  (mesmo processo)             (1, 12, 6, 64)

k.transpose(2, 3)               (1, 12, 64, 6)
q @ kᵀ                          (1, 12, 6, 6)     um scores 6×6 POR HEAD (a matmul age nas 2 últimas dims)
  ↓ masked_fill(mask, -inf)                       máscara causal (mha.mask é 1024×1024; [:t, :t] recorta 6×6)
  ↓ ÷ √64  e  softmax(dim=-1)
pesos                           (1, 12, 6, 6)     12 matrizes de atenção, cada linha soma 1
```

- **`.view`** não copia nem muda números: só reinterpreta os 768 números de cada token como 12 grupos de 64. Os números 0–63 vão para a head 0, 64–127 para a head 1, etc.
- **`.transpose(1, 2)`** troca as dimensões "tokens" e "heads" de lugar, para que a multiplicação de matrizes aconteça **separadamente para cada head**.
- É exatamente o que a ETAPA 2 fazia, só que 12 vezes em paralelo e com vetores de 64.

### O resto da `MultiHeadAttention.forward` (código do livro, `ch03.py`, não está no notebook)
A função acima para no softmax. A classe continua assim — e é aqui que ocorre a **concatenação**:
```text
values (mesmo view/transpose)    (1, 12, 6, 64)
pesos @ values                   (1, 12, 6, 64)   soma ponderada, dentro de cada head
  ↓ .transpose(1, 2)             (1, 6, 12, 64)   volta tokens para a frente
  ↓ .reshape(b, t, 768)          (1, 6, 768)      CONCATENA as 12 heads: 12 × 64 = 768
  ↓ out_proj  Linear(768, 768)   (1, 6, 768)      mistura a informação das heads
saída                            (1, 6, 768)
```
A classe também aplica dropout nos pesos (aqui usamos `dropout=0.0`).

### Fala
"A classe do livro só devolve o resultado final, não os pesos. Então essa função refaz as mesmas contas e para no softmax, para a gente conseguir desenhar.
E aqui aparece o que é multi-head. Eu faço a projeção normal, que gera um vetor de 768 por token. Aí o `.view` fatia esse vetor: 12 pedaços de 64. Os primeiros 64 números são da head 0, os próximos 64 da head 1, e assim por diante. O `transpose` só reorganiza, colocando as heads na frente, para que cada head faça sua própria atenção separada: seu próprio `q @ kᵀ`, sua própria máscara, seu próprio softmax. O resultado são 12 matrizes de atenção, uma por head.
Dentro da classe, depois disso, cada head faz a soma ponderada com os seus values, e as 12 saídas de 64 são concatenadas de volta num vetor de 768. Uma última camada linear, a `out_proj`, mistura o que as heads encontraram."

---

## ETAPA 7 — 🔧 Mudando o número de heads (célula 12)

### Código
```python
# 🔧 Mudando o número de heads: parâmetros e formato de saída NÃO mudam, só como d_out é fatiado
X = torch.randn(1, 6, 768)
for h in [1, 2, 4, 12, 24]:
    mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=h)
    print(f"heads={h:>2}  head_dim={mha.head_dim:>3}  params={sum(p.numel() for p in mha.parameters()):,}"
          f"  saída={tuple(mha(X).shape)}  pesos={tuple(pesos_atencao(mha, X).shape)}")
```

### O que está acontecendo?
- `X`: entrada aleatória com o formato do GPT-2: 1 frase, 6 tokens, 768 dimensões — o mesmo formato `(batch, tokens, emb_dim)` que o Elmo produziu.
- O `for` cria uma `MultiHeadAttention` com 1, 2, 4, 12 e 24 heads. Parâmetros:
  - `d_in=768`: tamanho do vetor que entra;
  - `d_out=768`: tamanho do vetor que sai;
  - `context_length=1024`: tamanho máximo da máscara causal;
  - `dropout=0.0`: sem dropout;
  - `num_heads=h`.
- Para cada uma, imprime `head_dim`, nº de parâmetros, shape da saída (`mha(X)`) e shape dos pesos (`pesos_atencao`).

### Resultado
```
heads= 1  head_dim=768  params=2,360,064  saída=(1, 6, 768)  pesos=(1, 1, 6, 6)
heads= 2  head_dim=384  params=2,360,064  saída=(1, 6, 768)  pesos=(1, 2, 6, 6)
heads= 4  head_dim=192  params=2,360,064  saída=(1, 6, 768)  pesos=(1, 4, 6, 6)
heads=12  head_dim= 64  params=2,360,064  saída=(1, 6, 768)  pesos=(1, 12, 6, 6)
heads=24  head_dim= 32  params=2,360,064  saída=(1, 6, 768)  pesos=(1, 24, 6, 6)
```
- `head_dim = 768 / heads`.
- **Parâmetros iguais** em todos: 2.360.064 = W_query + W_key + W_value (3 × 768 × 768, sem bias) + out_proj (768 × 768 + 768 de bias). As matrizes são as mesmas; só muda como o vetor é fatiado.
- **Saída igual**: (1, 6, 768) — por isso dá para trocar o número de heads sem mexer no resto do modelo.
- **Pesos**: uma matriz 6×6 por head.

### Fala
"Aqui respondo a terceira pergunta: o que acontece se eu mudar o número de heads? Crio a atenção com 1, 2, 4, 12 e 24 heads. Olhem o que não muda: o número de parâmetros é sempre 2,36 milhões e a saída é sempre 1 por 6 por 768. As matrizes são as mesmas; a única coisa que muda é em quantos pedaços eu fatio o vetor. Com 1 head, é uma atenção de 768 dimensões; com 12, que é o GPT-2, são 12 atenções de 64; com 24, são 24 de 32. E os pesos agora têm uma matriz 6 por 6 para cada head."

---

## Markdown 13: trade-off do nº de heads

Mais heads = mais "pontos de vista" independentes, cada um com menos dimensões. GPT-2 mantém **head_dim = 64** em todos os tamanhos (768/12, 1024/16, 1280/20, 1600/25).

**Fala:**
"E qual é melhor? É um equilíbrio. Mais heads significa mais padrões diferentes sendo procurados ao mesmo tempo, mas cada head fica com menos dimensões para trabalhar; se ficar pequeno demais, ela não consegue representar muita coisa. O GPT-2 resolveu isso fixando 64 dimensões por head: quando o modelo cresce, aumentam as heads, não o tamanho de cada uma."

### Possível pergunta do professor/aluno
**"O que acontece se mudarmos a quantidade de heads?"**
Resposta: "Pela célula: não muda o custo em parâmetros nem o formato da saída, só a divisão do vetor. O efeito aparece na qualidade depois do treino: poucas heads = poucos padrões, mas cada um mais rico; muitas = muitos padrões, cada um com poucas dimensões. O notebook não treina com números diferentes de heads; isso seria um experimento de treino, como os que a Lilian faz no notebook 04."

---

## Seção 5 (markdown 13 → "E se não for divisível?")

**Transição:** "E isso leva à quarta pergunta: se eu fatio 768 em pedaços iguais, o que acontece se eu pedir 5 heads?"

---

## ETAPA 8 — `num_heads=5`: não divisível (célula 14)

### Código
```python
try:
    MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=5)
except AssertionError as e:
    print("AssertionError:", e)
print("768 / 5 =", 768 / 5, "→ não dá para fazer .view(b, t, 5, 153.6)")
```

### O que está acontecendo?
- Tenta criar uma atenção com 768 dimensões e 5 heads.
- O `try/except` captura o erro para mostrar a mensagem sem quebrar o notebook.
- O erro vem desta linha no `__init__` da classe (`ch03.py`):
  `assert d_out % num_heads == 0, "d_out must be divisible by n_heads"`

### Resultado
```
AssertionError: d_out must be divisible by n_heads
768 / 5 = 153.6 → não dá para fazer .view(b, t, 5, 153.6)
```
- O `.view` da ETAPA 6 precisa dividir o vetor em pedaços **inteiros e iguais**. Não existe head com 153,6 dimensões. A classe verifica isso logo na criação, em vez de deixar quebrar lá dentro.

### Fala
"Com 5 heads, a classe nem deixa criar: dá esse erro, 'd_out must be divisible by n_heads'. E faz sentido: 768 dividido por 5 dá 153,6. O `.view` que a gente viu precisa cortar o vetor em pedaços inteiros e iguais, e não existe head com 153 dimensões e meia."

---

## Markdown 15 — como os modelos reais resolvem

Escolher `head_dim` explicitamente e fazer `d_out = num_heads × head_dim` (Llama/Gemma/Qwen definem `head_dim` no config), e projetar de volta para `emb_dim` na `out_proj`.

---

## ETAPA 9 — Solução: 5 heads de 64 (célula 16)

### Código
```python
# Exemplo: 5 heads de 64 → d_out=320, e a out_proj devolve 320 → 768 seria o próximo passo
mha5 = MultiHeadAttention(d_in=768, d_out=5 * 64, context_length=1024, dropout=0.0, num_heads=5)
print(tuple(mha5(X).shape))
```

### O que está acontecendo?
- Em vez de fixar `d_out = 768` e dividir, fixa **`head_dim = 64`** e calcula `d_out = 5 × 64 = 320`.
- Aqui `d_in` (768) e `d_out` (320) são diferentes: as projeções W_query/W_key/W_value vão de 768 para 320.

### Resultado
```
(1, 6, 320)
```
- Funciona, mas a saída tem **320** dimensões, não 768.
- ⚠️ Detalhe importante (explica o comentário do código): na classe do livro, a `out_proj` é `Linear(d_out, d_out)`, ou seja, 320 → 320. Num GPT, a saída da atenção é **somada** de volta ao vetor de entrada (conexão residual, notebook 03), então ela precisa voltar para 768. Os modelos que fazem isso (Llama, Gemma, Qwen) usam uma projeção de saída `num_heads × head_dim → emb_dim`. A classe do livro **não** faz isso; seria "o próximo passo", como diz o comentário.

### Fala
"A solução usada na prática é inverter a lógica: em vez de escolher a dimensão e dividir, eu escolho o tamanho de cada head, por exemplo 64, e multiplico: 5 heads de 64 dão 320. Funciona, a saída sai com 320. Só que o GPT precisa que a saída da atenção volte para 768, porque ela é somada de volta à entrada, e o Victor vai mostrar isso no próximo notebook. Modelos como Llama e Qwen fazem exatamente isso: definem o `head_dim` na configuração e usam a projeção de saída para voltar ao tamanho do modelo. A classe do livro não faz esse último passo, por isso o comentário diz que 'seria o próximo passo'."

### Possível pergunta do professor/aluno
**"O que acontece se a dimensão dos vetores não for divisível pelo número de heads?"**
Resposta: "Na implementação do livro, dá erro na hora de criar a camada, porque o `.view` precisa de pedaços inteiros. Na prática, os modelos definem o `head_dim` explicitamente, calculam `num_heads × head_dim`, que pode ser diferente do tamanho do modelo, e usam a projeção de saída para voltar ao tamanho original."

---

## Seção 6 (markdown 17): "Scores de verdade: as 12 heads de uma camada do GPT-2 treinado"

**Transição:** "Até agora, tudo foi com pesos aleatórios. Agora a segunda pergunta: dá para ver a atenção de verdade? Vou carregar o GPT-2 treinado e olhar as 12 heads de uma camada."

---

## ETAPA 10 — Heatmaps das 12 heads da camada 5 (célula 18)

### Código
```python
gpt2, _ = load_gpt2("124M")
bpe = tiktoken.get_encoding("gpt2")
frase = "The animal didn't cross the street because it was too tired"
ids = torch.tensor([bpe.encode(frase)])
rotulos = [bpe.decode([i]) for i in ids[0]]

CAMADA = 5  # 🔧 troque a camada (0–11) e veja os padrões mudarem
with torch.no_grad():
    h = gpt2.tok_emb(ids) + gpt2.pos_emb(torch.arange(ids.shape[1]))
    for bloco in gpt2.trf_blocks[:CAMADA]:
        h = bloco(h)
    bloco = gpt2.trf_blocks[CAMADA]
    P = pesos_atencao(bloco.att, bloco.norm1(h))[0]

fig, axs = plt.subplots(3, 4, figsize=(16, 12))
for head, ax in enumerate(axs.flat):
    mapa(ax, P[head], f"camada {CAMADA} · head {head}", rotulos)
plt.tight_layout()
plt.show()
```

### O que está acontecendo?
1. Carrega o GPT-2 124M com pesos oficiais e o tokenizador BPE.
2. `frase`: a frase clássica para atenção — "The animal didn't cross the street because it was too tired". A pergunta interessante: a quem " it" se refere? (ao animal).
3. `ids = torch.tensor([bpe.encode(frase)])` → IDs com shape **(1, 12)**. Os colchetes externos criam a dimensão de batch.
4. `rotulos`: os 12 tokens como texto (`'The', ' animal', ' didn', "'t", ' cross', ' the', ' street', ' because', ' it', ' was', ' too', ' tired'`), para os eixos.
5. `CAMADA = 5`: vamos olhar a 6ª camada (conta a partir de 0), de 12.
6. Dentro de `torch.no_grad()` (sem treino):
   - `h = gpt2.tok_emb(ids) + gpt2.pos_emb(torch.arange(12))` → **exatamente a linha que o Elmo mostrou no notebook 01**, agora com a tabela treinada. Shape (1, 12, 768).
   - O `for` passa `h` pelos blocos 0 a 4 (`trf_blocks[:5]`) — cada bloco devolve o mesmo shape (1, 12, 768).
   - `bloco = gpt2.trf_blocks[5]`.
   - `bloco.norm1(h)`: dentro de um bloco do GPT, a atenção recebe a entrada **normalizada** (LayerNorm), não `h` cru. Para reproduzir fielmente, aplicamos `norm1` antes (o Victor explica o bloco no notebook 03).
   - `pesos_atencao(bloco.att, ...)` → (1, 12, 12, 12) = (batch, heads, tokens, tokens). `[0]` tira o batch → **`P` com shape (12, 12, 12)**: 12 heads, cada uma com uma matriz 12×12.
7. Grade 3×4: o `for` percorre as 12 heads e desenha `P[head]`.

### Variáveis
| Variável | O que é | Shape | Usada depois em |
|---|---|---|---|
| `gpt2` | GPT-2 treinado | — | aqui |
| `bpe`, `frase` | tokenizador e frase | — | aqui |
| `ids` | IDs da frase | (1, 12) | aqui |
| **`rotulos`** | tokens como texto | 12 strings | **ETAPA 11** (achar " it") |
| **`CAMADA`** | camada escolhida | `int` = 5 | **ETAPA 11** (título) |
| `h` | vetores entrando na camada 5 | (1, 12, 768) | aqui |
| **`P`** | pesos de atenção das 12 heads | **(12, 12, 12)** | **ETAPA 11** |

### Resultado (gráfico salvo: 12 heatmaps triangulares)
- Todos são **triangulares**: a máscara causal está ativa (ninguém olha o futuro).
- Em **quase todas as heads, a primeira coluna ("The") é bem escura**: muitos tokens jogam grande parte da atenção no primeiro token. A **head 1** é o caso extremo (quase toda a atenção em "The"). Isso é um fenômeno conhecido em modelos treinados (às vezes chamado de *attention sink*): o primeiro token funciona como um "lugar neutro" para depositar atenção quando a head não tem nada de útil para olhar.
- Algumas heads mostram padrões diferentes (observações a partir do gráfico salvo):
  - **head 10:** a linha " it" tem peso visível em **" animal"** (e " was" e " tired" também olham para " animal");
  - **head 11:** uma diagonal clara — vários tokens olham para **si mesmos**;
  - **head 4:** " tired" olha bastante para o token anterior " too";
  - **heads 5 e 8:** a linha " it" tem peso em " cross".
- Mensagem principal: **cada head aprendeu um padrão diferente**, sem ninguém programar isso.

### Fala
"Agora a atenção de verdade. Carrego o GPT-2 treinado e uso uma frase clássica: 'The animal didn't cross the street because it was too tired'. A pergunta é: quando o modelo lê 'it', ele sabe que é o animal?
Reparem nessa linha: `tok_emb` mais `pos_emb`. É exatamente o que o Elmo mostrou, agora com a tabela treinada. Passo esses vetores pelas 5 primeiras camadas, e na sexta eu uso a nossa função `pesos_atencao` para pegar os pesos. O `norm1` é porque, dentro do bloco, a atenção recebe a entrada normalizada; o Victor vai mostrar isso. Saem 12 matrizes, uma por head.
O que a gente vê: todas são triangulares, é a máscara causal. Quase todas têm a primeira coluna escura: muitas heads jogam atenção no primeiro token, o 'The', que funciona como um lugar neutro quando a head não precisa olhar para nada específico. A head 1 faz só isso. Mas as outras têm padrões próprios: na head 11 cada token olha para si mesmo; na head 10, olhem a linha do 'it': tem peso em 'animal'. Ninguém programou isso: cada head aprendeu sozinha um tipo de relação."

---

## ETAPA 11 — Para onde " it" olha em uma head (célula 19)

### Código
```python
# Uma head só, e para onde " it" olha
HEAD = 0  # 🔧
i_it = rotulos.index(" it")
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.bar(rotulos[: i_it + 1], P[HEAD, i_it, : i_it + 1])
ax.set_title(f'camada {CAMADA}, head {HEAD}: atenção do token " it"')
plt.xticks(rotation=45)
plt.show()
```

### O que está acontecendo?
- `HEAD = 0`: qual head olhar (🔧 = pode ser trocado ao vivo).
- `i_it = rotulos.index(" it")`: posição do token " it" na frase (índice 8, o 9º token). Note o espaço antes: é assim que o BPE representa "it" no meio da frase.
- `P[HEAD, i_it, : i_it + 1]`: da head escolhida, pega a **linha** de " it" (para quem ele olha) e só as colunas até ele mesmo (as seguintes são 0 por causa da máscara).
- Desenha um gráfico de barras: altura = peso de atenção. As barras somam 1.

### Resultado (gráfico salvo, HEAD = 0)
- **"The" ≈ 0,7**: a maior parte da atenção de " it", na head 0, vai para o primeiro token (o padrão "lugar neutro" da etapa anterior).
- " cross" ≈ 0,1 e " because" ≈ 0,1; " animal" praticamente 0.
- ⚠️ Ou seja: **a head 0 da camada 5 NÃO liga " it" a " animal"**. Não apresente isso como "o modelo entendeu que it é o animal" com a head 0.
- **Recomendação para a apresentação:** pelo heatmap da ETAPA 10, a **head 10** é a candidata mais promissora (a linha " it" tem peso em " animal"). Troque `HEAD = 10` e **rode antes da aula** para confirmar. Também vale testar outras camadas (`CAMADA`).

### Fala (versão com HEAD = 0, como está no notebook)
"Agora eu isolo uma head só e pergunto: para onde o 'it' olha? Com a head 0, a resposta é meio decepcionante: 70% vai para o 'The', o primeiro token, aquele lugar neutro. Essa head não está resolvendo a quem o 'it' se refere. E isso é uma lição importante: não é toda head que faz algo interpretável. Vamos trocar para a head 10, que no mapa parecia olhar para 'animal'…"

*(Se confirmar antes da aula que a head 10 destaca " animal", continue:)*
"…e aqui está: na head 10, o 'it' dá um peso bem maior para 'animal'. Uma head específica, numa camada específica, aprendeu a ligar o pronome ao substantivo. E dá para explorar: é só trocar `CAMADA` e `HEAD` e ver os padrões mudarem."

### Possível pergunta do professor/aluno
**"Conseguimos visualizar os scores em pelo menos uma head?"**
Resposta: "Sim, essas duas últimas células fazem isso: o heatmap mostra os pesos das 12 heads da camada 5, e o gráfico de barras mostra uma head só, para um token só. Tecnicamente, o que desenhamos são os pesos (depois do softmax), porque são mais fáceis de ler: cada linha soma 1. Para ver os scores crus, bastaria tirar o softmax na função `pesos_atencao`."

---

## Markdown 20 — ✅ Construto (fechamento)

`MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)` — no notebook 03 ela vira uma peça do `TransformerBlock`.

**Fala de fechamento / passagem para o Victor:**
"Recapitulando: cada token gera query, key e value; o produto escalar da query com as keys dá os scores; dividimos por raiz de d para o softmax não saturar; a máscara impede olhar o futuro; o softmax vira porcentagens; e a soma ponderada dos values gera o novo vetor de cada token. Multi-head é fazer isso várias vezes em paralelo, fatiando o vetor, e concatenar no final. A entrada é lote por tokens por 768, e a saída também. Por isso essa peça pode ser empilhada: o Victor vai mostrar como ela entra no bloco do Transformer, junto com normalização e uma rede feed-forward, e como 12 desses blocos viram o GPT."

---

## Roteiro contínuo do Rafael — Notebook 02

**[Título / markdown 0]**
"O Elmo terminou com cada token virando um vetor. Só que esse vetor é isolado: o vetor de 'it' é o mesmo em qualquer frase, não sabe se 'it' é um animal ou uma rua. A atenção é o mecanismo que deixa cada token olhar para os outros e misturar a informação deles no seu próprio vetor. Vou montar isso passo a passo e responder quatro perguntas: se existem outras formas de calcular a atenção, o que muda com o número de heads, o que acontece se a dimensão não for divisível, e no final vamos olhar a atenção de verdade dentro do GPT-2."

**[Célula 1 — setup]**
"Para dar para enxergar as contas, vou usar o exemplo do livro: 'Your journey starts with one step', 6 tokens, cada um com um embedding de só 3 números. Esse `x` é uma matriz 6 por 3. A função `mapa` só desenha as matrizes de atenção: a linha é quem está olhando, a coluna é para quem."

**[Markdown 2]**
"A atenção usa uma analogia de busca. Cada token gera uma query, 'o que eu procuro'; uma key, 'o que eu ofereço'; e um value, 'o conteúdo que eu entrego'. O score mede o quanto a query de um combina com a key de outro."

**[Célula 3 — Q, K, V, scores, softmax, contexto]**
"Essa célula é a atenção inteira em cinco linhas. Crio três camadas lineares, `W_q`, `W_k` e `W_v`, cada uma transformando o embedding de 3 números num vetor de 2. Cada token ganha sua query, sua key e seu value: o mesmo embedding, três papéis.
O score é `Q @ K.T`: uma multiplicação de matrizes que calcula o produto escalar da query de cada token com a key de cada token. Sai uma matriz 6 por 6: linha 'journey', coluna 'step' é o quanto 'journey' se interessa por 'step'.
Esses scores são notas cruas. O softmax transforma cada linha em porcentagens que somam 1, e dá para ver aqui que as linhas somam 1. Antes eu divido por raiz de `d_k`, já mostro por quê.
Por último, `pesos @ V`: para cada token, uma média dos values de todos, usando essas porcentagens. Esse é o vetor de contexto: a nova versão do token, misturada com a informação da frase."

**[Markdown 4]**
"Primeira pergunta: o produto escalar escalado é o que o GPT usa, mas não é a única forma de medir o quanto uma query combina com uma key. Vou comparar quatro."

**[Célula 5 — variantes de score]**
"As duas primeiras vocês já conhecem: produto escalar puro e escalado. A terceira é a atenção aditiva, de 2014, anterior ao Transformer: em vez de multiplicar, coloca query e key lado a lado e passa por uma mini rede neural. A quarta é o cosseno, que só olha a direção dos vetores; divido por 0,1 para esticar o intervalo.
Cada fórmula distribui a atenção de um jeito. Mas aqui os pesos são aleatórios, ninguém treinou nada: o gráfico não diz qual é melhor, mostra que o score é uma escolha de projeto. O Transformer ficou com o produto escalar porque é só multiplicação de matrizes, extremamente rápido em GPU."

**[Markdown 6]**
"Agora a pergunta pendente: por que dividir por raiz de d? Com vetores grandes, o produto escalar fica grande, e o softmax exagera."

**[Célula 7 — escala]**
"Pego uma query e 10 keys aleatórias e vejo o maior peso depois do softmax. Com 2 dimensões, tanto faz. Com 64, que é o tamanho de cada head no GPT-2, sem escala um único token leva 95% da atenção. Com 768, leva 100%. O produto escalar soma `d` termos, então cresce com as dimensões. Dividindo por raiz de d, o maior peso fica em uns 25% para qualquer tamanho. E quando o softmax trava em 0 e 1, o gradiente some e o modelo não aprende para onde olhar."

**[Markdown 8 — variações modernas]**
"Completando a primeira pergunta: os modelos modernos mexem mais em quais Q, K e V são calculados do que na fórmula. O Llama faz várias queries compartilharem a mesma key e value; o Mistral deixa cada token olhar só para uma janela próxima. Tem implementação disso no repositório.
Agora, um problema: até aqui todo token olha para todos, inclusive os que vêm depois. Mas o GPT gera da esquerda para a direita; se no treino ele pudesse olhar o futuro, copiaria a resposta. Por isso existe a máscara causal."

**[Célula 9 — máscara]**
"A máscara é uma matriz 6 por 6 com `True` acima da diagonal: as posições do futuro. Nelas eu troco o score por menos infinito, e no softmax menos infinito vira exatamente zero. O resto da linha se reajusta para somar 1.
Sem máscara, todo mundo vê todo mundo. Com máscara, vira um triângulo: 'Your' só olha para si mesmo e leva 100%; 'step', o último, é o único que vê a frase inteira. É por isso que o Elmo vai usar o último token no classificador de spam."

**[Markdown 10]**
"Até agora foi uma atenção só. Mas uma frase tem várias relações ao mesmo tempo. Em vez de uma atenção grande, o GPT faz várias menores em paralelo, as heads. E o jeito de fazer isso é fatiar o vetor."

**[Célula 11 — pesos_atencao]**
"A classe do livro só devolve o resultado final, então essa função refaz as contas e para no softmax, para a gente poder desenhar. A projeção gera um vetor de 768 por token; o `.view` fatia em 12 pedaços de 64: os primeiros 64 números são da head 0, os próximos da head 1, e assim por diante. O `transpose` coloca as heads na frente, para cada uma fazer sua própria atenção: seu `q @ kᵀ`, sua máscara, seu softmax. Saem 12 matrizes de atenção. Dentro da classe, cada head faz a soma com seus values, as 12 saídas de 64 são concatenadas de volta em 768, e a `out_proj` mistura o que as heads encontraram."

**[Célula 12 — nº de heads]**
"Terceira pergunta: e se eu mudar o número de heads? Com 1, 2, 4, 12 e 24 heads, o número de parâmetros é sempre 2,36 milhões e a saída sempre 1 por 6 por 768. As matrizes são as mesmas; só muda em quantos pedaços eu fatio. Com 12, que é o GPT-2, são 12 atenções de 64."

**[Markdown 13]**
"Qual é melhor? É um equilíbrio: mais heads, mais padrões ao mesmo tempo, mas cada uma com menos dimensões. O GPT-2 fixou 64 por head: quando o modelo cresce, aumentam as heads, não o tamanho de cada uma. E isso leva à quarta pergunta: e se eu pedir 5 heads?"

**[Célula 14 — não divisível]**
"Com 5 heads, a classe nem deixa criar: 'd_out must be divisible by n_heads'. 768 dividido por 5 dá 153,6, e o `.view` precisa de pedaços inteiros e iguais."

**[Célula 16 — 5 × 64]**
"A solução na prática é inverter: escolho o tamanho de cada head, 64, e multiplico: 5 heads dão 320. Funciona, a saída sai com 320. Mas o GPT precisa voltar para 768, porque a saída da atenção é somada de volta à entrada, como o Victor vai mostrar. Llama e Qwen fazem isso: definem o `head_dim` e usam a projeção de saída para voltar ao tamanho do modelo. A classe do livro não faz esse último passo."

**[Markdown 17]**
"Até agora tudo foi com pesos aleatórios. Segunda pergunta: dá para ver a atenção de verdade? Vou carregar o GPT-2 treinado e olhar as 12 heads de uma camada."

**[Célula 18 — 12 heads]**
"A frase é 'The animal didn't cross the street because it was too tired'. Quando o modelo lê 'it', ele sabe que é o animal? Essa linha, `tok_emb` mais `pos_emb`, é exatamente o que o Elmo mostrou, agora com a tabela treinada. Passo pelas 5 primeiras camadas e, na sexta, uso a `pesos_atencao`. Saem 12 matrizes.
Todas são triangulares: é a máscara. Quase todas têm a primeira coluna escura: muitas heads jogam atenção no 'The', um lugar neutro quando não precisam olhar nada específico; a head 1 faz só isso. Mas outras têm padrões próprios: na head 11 cada token olha para si mesmo; na head 10, a linha do 'it' tem peso em 'animal'. Ninguém programou isso."

**[Célula 19 — barras do " it"]**
"Agora isolo uma head e pergunto: para onde o 'it' olha? Na head 0, 70% vai para o 'The'. Essa head não resolve a quem o 'it' se refere, e é uma lição: nem toda head faz algo interpretável. Trocando para a head 10… *(se confirmado antes da aula)* o 'it' dá um peso bem maior para 'animal'. Uma head específica aprendeu a ligar o pronome ao substantivo. E dá para explorar trocando `CAMADA` e `HEAD`."

**[Markdown 20 — fechamento / passagem]**
"Recapitulando: query, key e value; produto escalar dá os scores; divide por raiz de d; máscara para não olhar o futuro; softmax vira porcentagens; soma ponderada dos values gera o novo vetor. Multi-head é fazer isso em paralelo, fatiando o vetor, e concatenar. Entra lote por tokens por 768 e sai o mesmo formato, então dá para empilhar. O Victor vai mostrar como isso entra no bloco do Transformer e como 12 blocos viram o GPT."

---
---

# NOTEBOOK 06 — `06_assistente.ipynb`

> ⚠️ **Antes de apresentar:**
> 1. A célula 13 tem `CARREGAR_CHECKPOINT = True` e o arquivo `checkpoints/assistente_124M.pth` já existe. **Ao vivo, ela vai imprimir `checkpoint carregado: ...`** em vez do log de treino salvo no notebook.
> 2. O checkpoint foi salvo em **bfloat16** (metade da precisão) e as respostas salvas nas células 15 e 16 vieram do modelo recém-treinado em float32. **Ao carregar o checkpoint, as respostas podem sair um pouco diferentes** das que estão salvas. Rode o notebook antes da aula e ajuste as falas se alguma resposta mudar.

## Mapa do notebook

```text
instruction-data.json (1.100 pares instrução → resposta)
   │ format_input (estilo Alpaca)
   │ divisão 85/10/5 → treino (cortado para 400) / teste 110 / validação 55
   ▼
InstructionDataset (texto completo tokenizado)
   │ custom_collate_fn: padding com 50256 + alvo deslocado + -100 no padding extra
   ▼
train_loader / val_loader
   │
GPT-2 124M ── antes: respostas sem sentido
   │ train_model_simple (o MESMO loop do pré-treino)  — ou checkpoint
   ▼
depois: segue o formato, para no <|endoftext|>, mas erra fatos
   │
meus_dados → mesmo formato → pronto para o mesmo DataLoader
```

---

## Abertura (markdown 0)

**Entra:** GPT-2 124M pré-treinado. **Sai:** um modelo que responde instruções.
Perguntas: 1) como um modelo que só "continua texto" vira um assistente? 2) onde conseguir dados de fine-tuning? Como construir do zero?

**Fala:**
"O Elmo acabou de mostrar que, quando a gente pergunta alguma coisa para o GPT-2 base, ele não responde: só continua o texto. Mas o ChatGPT responde. Qual é a diferença? Neste notebook eu vou mostrar que a diferença, no fundo, são os dados. A gente pega o mesmo GPT-2, o mesmo loop de treino do pré-treino, e treina com exemplos de instrução e resposta. E no final eu mostro onde conseguir esses dados e como montar os seus."

---

## ETAPA 1 — Imports e parâmetros (célula 1)

### Código
```python
import json
import os
import time
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

from aula import REPO_DIR, CHECKPOINTS, device, load_gpt2
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text, train_model_simple
from llms_from_scratch.ch07 import format_input, InstructionDataset, custom_collate_fn

CARREGAR_CHECKPOINT = True  # 🔧 True = usa o assistente salvo (se existir)
N_EXEMPLOS_TREINO = 400      # 🔧 subconjunto para caber ao vivo (o livro usa ~935 e o GPT-2 355M)
bpe = tiktoken.get_encoding("gpt2")
```

### O que está acontecendo?
- `partial` (do Python) cria uma versão de uma função com alguns argumentos já preenchidos — usado na ETAPA 5.
- Do capítulo 5: `generate`, conversão texto ↔ IDs e **`train_model_simple`**, que é o loop de treino do **pré-treino**. Esse detalhe é o ponto do notebook.
- Do capítulo 7: `format_input` (monta o prompt), `InstructionDataset` (tokeniza os exemplos), `custom_collate_fn` (monta os lotes).
- `N_EXEMPLOS_TREINO = 400`: usa só 400 exemplos para o treino caber ao vivo. O livro usa ~935 e o modelo maior, 355M.

### Fala
"Os imports. Reparem numa coisa: a função de treino que eu importo é a `train_model_simple`, do capítulo 5, a mesma que a Lilian usou no pré-treino. Não tem um algoritmo novo aqui. Do capítulo 7 vêm três funções para preparar os dados. E eu uso só 400 exemplos, para dar para treinar ao vivo; o livro usa uns 900 e um modelo maior."

---

## Seção 1 (markdown 2): "Os dados: pares instrução → resposta"

---

## ETAPA 2 — Carregar o dataset (célula 3)

### Código
```python
with open(os.path.join(REPO_DIR, "ch07", "01_main-chapter-code", "instruction-data.json")) as f:
    dados = json.load(f)
print(len(dados), "exemplos\n")
print(json.dumps(dados[50], indent=2, ensure_ascii=False))
```

### O que está acontecendo?
- Lê o JSON do capítulo 7 do livro. `dados` vira uma **lista de dicionários**, cada um com três campos: `instruction` (o que fazer), `input` (dado extra, pode ser vazio) e `output` (a resposta esperada).
- Imprime o total e o exemplo de índice 50, formatado.

### Variáveis
| Variável | O que é | Tipo | Usada depois em |
|---|---|---|---|
| **`dados`** | todos os exemplos | `list[dict]`, 1.100 itens | ETAPAS 3 e 4 |

### Resultado
```
1100 exemplos

{
  "instruction": "Identify the correct spelling of the following word.",
  "input": "Ocassion",
  "output": "The correct spelling is 'Occasion.'"
}
```

### Fala
"O dataset são 1.100 exemplos, e cada exemplo é só um dicionário com três campos: a instrução, que é o que a pessoa pede; o input, que é um dado extra e pode ficar vazio; e o output, a resposta que queremos. Por exemplo: 'identifique a grafia correta', com o input 'Ocassion', e a resposta 'A grafia correta é Occasion'."

---

## Markdown 4: "Formatamos no estilo Alpaca"

**Transição:** "Só que o modelo não lê dicionário, lê texto. Então a gente transforma cada exemplo num texto com um molde fixo, o estilo Alpaca."

---

## ETAPA 3 — `format_input`: o molde Alpaca (célula 5)

### Código
```python
print(format_input(dados[50]) + f"\n\n### Response:\n{dados[50]['output']}")
```

### O que está acontecendo?
`format_input(entry)` (cap. 7) monta:
1. um cabeçalho fixo: "Below is an instruction that describes a task. Write a response that appropriately completes the request.";
2. `### Instruction:` + a instrução;
3. `### Input:` + o input — **só se o input não for vazio**.

`format_input` **não** inclui a resposta: ela é o prompt. A célula acrescenta à mão `### Response:` + o output para mostrar como fica o exemplo completo de treino (o `InstructionDataset` faz a mesma coisa na ETAPA 6).

### Resultado
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

### Fala
"É esse o molde. Um cabeçalho fixo, a instrução, o input se tiver, e o marcador '### Response:', depois do qual vem a resposta. O modelo vai ver centenas de textos exatamente nesse formato. E o que ele aprende é: 'depois de ### Response: vem uma resposta para a instrução, e depois a resposta termina'. Na hora de usar, eu dou só a parte de cima, e ele completa."

---

## ETAPA 4 — Treino, teste e validação (célula 6)

### Código
```python
n_tr, n_te = int(len(dados) * 0.85), int(len(dados) * 0.10)
treino, teste, val = dados[:n_tr], dados[n_tr:n_tr + n_te], dados[n_tr + n_te:]
treino = treino[:N_EXEMPLOS_TREINO]
print(len(treino), "treino |", len(val), "validação |", len(teste), "teste")
```

### O que está acontecendo?
- `n_tr = 935` (85%), `n_te = 110` (10%).
- Fatia a lista **na ordem**: os primeiros 935 para treino, os 110 seguintes para teste e o que sobra (55, ~5%) para validação. Repare na ordem da atribuição: `treino, teste, val`.
- Depois corta o treino para os primeiros 400.

### Variáveis
| Variável | O que é | Tamanho | Usada depois em |
|---|---|---|---|
| **`treino`** | exemplos de treino | 400 | ETAPA 6 (train_loader) |
| **`teste`** | exemplos para comparar antes/depois | 110 | ETAPAS 7 e 9 |
| **`val`** | exemplos de validação | 55 | ETAPA 6 (val_loader) e ETAPA 8 (`val[0]` como exemplo durante o treino) |

### Resultado
```
400 treino | 55 validação | 110 teste
```

### Fala
"Divido os dados: 85% para treino, 10% para teste e o resto, uns 5%, para validação. E do treino eu uso só os primeiros 400. O teste é o que eu vou usar para comparar o modelo antes e depois do fine-tuning."

---

## Seção 2 (markdown 7): "Batches: padding e o `-100`"

Frases têm tamanhos diferentes → completamos com `<|endoftext|>` (50256). No alvo, o padding extra vira **−100**, que o `cross_entropy` ignora.

**Transição:** "Agora um detalhe técnico que é importante. Cada exemplo tem um tamanho, mas um lote precisa ser um bloco retangular. O Elmo resolveu isso no classificador completando com o token 50256. Aqui a gente faz o mesmo, mas com um cuidado extra no alvo."

---

## ETAPA 5 — `custom_collate_fn` com um exemplo de brinquedo (célula 8)

### Código
```python
collate = partial(custom_collate_fn, device=device, allowed_max_length=1024)
x, y = collate([[1, 2, 3, 4, 5], [6, 7], [8, 9, 10]])
print("entradas:\n", x.cpu(), "\nalvos:\n", y.cpu())
```

### O que está acontecendo?
- `partial(...)` fixa dois argumentos de `custom_collate_fn`: `device` (onde colocar os tensores) e `allowed_max_length=1024` (corta sequências maiores que o contexto do GPT-2). `collate` é essa versão pronta — vai ser usada no DataLoader (ETAPA 6).
- Teste com um "lote" de 3 sequências falsas, de tamanhos 5, 2 e 3.

O que `custom_collate_fn` faz (cap. 7), para cada sequência:
1. calcula `batch_max_length` = maior tamanho + 1 = 6;
2. adiciona **um** `50256` (`<|endoftext|>`) no final — é o sinal de "a resposta acabou";
3. completa com mais `50256` até 6;
4. **entrada** = tudo menos o último; **alvo** = tudo menos o primeiro → o alvo é a entrada deslocada 1 posição (igual ao dataloader do Elmo no notebook 01);
5. no alvo, troca **todos os `50256` menos o primeiro** por **−100**.

Exemplo com `[6, 7]`:
```text
[6, 7] + [50256] + padding     → [6, 7, 50256, 50256, 50256, 50256]
entrada = [:-1]                → [6, 7, 50256, 50256, 50256]
alvo    = [1:]                 → [7, 50256, 50256, 50256, 50256]
alvo, só o 1º 50256 mantido    → [7, 50256, -100, -100, -100]
```

### Variáveis
| Variável | O que é | Shape | Usada depois em |
|---|---|---|---|
| **`collate`** | função que monta lotes | função | **ETAPA 6** (DataLoader) |
| `x`, `y` | entrada e alvo do exemplo | (3, 5) cada | só aqui |

### Resultado
```
entradas:
 tensor([[    1,     2,     3,     4,     5],
        [    6,     7, 50256, 50256, 50256],
        [    8,     9,    10, 50256, 50256]])
alvos:
 tensor([[    2,     3,     4,     5, 50256],
        [    7, 50256,  -100,  -100,  -100],
        [    9,    10, 50256,  -100,  -100]])
```
- Todas as linhas ficaram com 5 posições.
- Em cada alvo, **o primeiro `50256` fica**: o modelo aprende "depois da resposta, gere `<|endoftext|>`". É isso que, depois, faz ele **parar de falar**.
- Os demais viram **−100**: `torch.nn.functional.cross_entropy` ignora por padrão posições com alvo −100 (`ignore_index=-100`). O modelo não é avaliado nem punido por "prever padding".
- As **entradas** continuam com `50256`, não −100: elas passam pelo embedding, e −100 não é um ID válido.

### Fala
"Aqui eu testo a função que monta os lotes com três sequências de mentira: de tamanho 5, 2 e 3. Ela completa todas com o token 50256 até o mesmo tamanho e cria o alvo deslocado uma posição, igual ao que o Elmo mostrou no começo da aula.
Mas olhem o alvo. O primeiro 50256 de cada linha fica. Ele é o `<|endoftext|>`, e é ele que ensina o modelo a terminar a resposta, a parar de falar. Os outros 50256 viraram -100. O -100 é um código que a função de erro, a `cross_entropy`, ignora. Então o modelo não perde ponto por errar o preenchimento: ele só é cobrado pelo texto de verdade e pelo sinal de fim."

### Possível pergunta
**"Por que não colocar −100 na entrada também?"**
"Porque a entrada vai para a tabela de embeddings, que só aceita IDs de 0 a 50.256. O −100 só faz sentido no alvo, que é onde a loss é calculada."

---

## ETAPA 6 — DataLoaders de instrução (célula 9)

### Código
```python
torch.manual_seed(123)
train_loader = DataLoader(InstructionDataset(treino, bpe), batch_size=8, collate_fn=collate, shuffle=True, drop_last=True)
val_loader = DataLoader(InstructionDataset(val, bpe), batch_size=8, collate_fn=collate)
```

### O que está acontecendo?
- `InstructionDataset(dados, tokenizer)` (cap. 7): para cada exemplo, monta o texto completo `format_input(exemplo) + "\n\n### Response:\n" + output` (o mesmo da ETAPA 3) e tokeniza. Cada item é uma lista de IDs, com tamanhos diferentes.
- `DataLoader(..., collate_fn=collate)`: agrupa de 8 em 8 e usa **a nossa `collate`** (ETAPA 5) para fazer o padding e criar os alvos com −100.
- Treino: `shuffle=True`, `drop_last=True` → 400 / 8 = 50 lotes por época.
- Diferente do classificador do Elmo, aqui o padding é **por lote** (até a maior sequência daquele lote), e não até um tamanho fixo.

### Variáveis
| Variável | O que é | Shape de um lote | Usada depois em |
|---|---|---|---|
| **`train_loader`** | lotes de treino | x, y: (8, maior sequência do lote) | ETAPA 8 |
| **`val_loader`** | lotes de validação | idem | ETAPA 8 |

### Fala
"Aqui junto tudo. O `InstructionDataset` monta cada exemplo no molde Alpaca, com a resposta, e tokeniza. O DataLoader agrupa de 8 em 8 e usa a nossa função de collate para completar e criar os alvos com -100. São 50 lotes por época."

---

## Seção 3 (markdown 10): "Antes do fine-tuning"

**Transição:** "Antes de treinar, vamos ver como o GPT-2 base se sai com esse formato."

---

## ETAPA 7 — Função `responder` e o modelo ANTES do fine-tuning (célula 11)

### Código
```python
gpt2, cfg = load_gpt2("124M")
gpt2.to(device)

def responder(model, entrada, max_new_tokens=60):
    prompt = format_input(entrada)
    ids = generate(model, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=max_new_tokens,
                   context_size=cfg["context_length"], eos_id=50256)
    return token_ids_to_text(ids, bpe)[len(prompt):].replace("### Response:", "").strip()

for e in teste[:3]:
    print(f"▶ {e['instruction']} {e['input']}\n  modelo: {responder(gpt2, e)!r}\n")
```

### O que está acontecendo?
- Carrega o GPT-2 124M pré-treinado (sem nenhum fine-tuning).
- **`responder(model, entrada, max_new_tokens=60)`:**
  - **recebe** o modelo e um exemplo (dicionário com `instruction`/`input`);
  - monta o prompt com `format_input` — **sem** o `### Response:`; o modelo precisa escrever esse marcador sozinho;
  - `generate(...)` gera até 60 tokens, sempre o mais provável; **`eos_id=50256`** faz a geração **parar** quando o modelo gerar `<|endoftext|>` (é aqui que o que a ETAPA 5 ensinou vai ser usado);
  - converte para texto, corta o prompt (`[len(prompt):]`), remove o marcador `### Response:` e os espaços;
  - **devolve** só a resposta, como texto.
- O `for` roda nos 3 primeiros exemplos de teste. `!r` mostra a string com aspas e `\n` visíveis.

### Variáveis
| Variável | O que é | Usada depois em |
|---|---|---|
| **`gpt2`** | o modelo — vai ser treinado na ETAPA 8 | ETAPAS 8, 9, 10 |
| **`cfg`** | config (`context_length=1024`) | dentro de `responder` |
| **`responder`** | função pergunta → resposta | ETAPAS 9 e 10 |

### Resultado
```
▶ Rewrite the sentence using a simile. The car is very fast.
  modelo: '### Output:\n\nThe car is very fast.\n\n### Error:\n\nThe car is very fast.\n\n### Error: ...'

▶ What type of cloud is typically associated with thunderstorms?
  modelo: 'Thunderstorms are the most common type of thunderstorm. They are usually caused by lightning, thunderstorms, ...'

▶ Name the author of 'Pride and Prejudice'.
  modelo: "### Description:\n\nThe author of 'Pride and Prejudice' is a young man who has been a member of the Church ..."
```
- Ele **imita o formato de markdown** (`### Output:`, `### Error:`, `### Description:`), mas inventa seções.
- **Não para**: usa os 60 tokens inteiros, repetindo frases.
- O conteúdo não responde a instrução (nuvem de tempestade: texto circular; autor: inventado).

### Fala
"Essa função `responder` monta o prompt no molde Alpaca, gera até 60 tokens e para se o modelo gerar o `<|endoftext|>`, e devolve só a resposta.
Olhem o GPT-2 base. Ele percebe que é um texto com cabeçalhos em markdown e inventa cabeçalhos: '### Output', '### Error', '### Description'. Não para de falar, usa os 60 tokens e fica se repetindo. E o conteúdo não responde nada: 'tempestades são o tipo mais comum de tempestade'. Ele é bom em continuar texto com cara de texto, mas não sabe que deveria responder."

---

## Seção 4 (markdown 12): "Fine-tuning (ou checkpoint)"

É o **mesmo loop do pré-treino** (`train_model_simple`) — só mudaram os dados.

**Transição:** "E agora o ponto principal do notebook: o fine-tuning de instrução é o mesmo loop do pré-treino. A única coisa que muda são os dados."

---

## ETAPA 8 — Treinar ou carregar checkpoint (célula 13)

### Código
```python
ckpt = os.path.join(CHECKPOINTS, "assistente_124M.pth")
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print("checkpoint carregado:", ckpt)
else:
    torch.manual_seed(123)
    t0 = time.time()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)
    train_model_simple(gpt2, train_loader, val_loader, optimizer, device, num_epochs=2,
                       eval_freq=10, eval_iter=5, start_context=format_input(val[0]), tokenizer=bpe)
    print(f"{(time.time() - t0) / 60:.1f} min")
    # bfloat16 = metade do tamanho (~330 MB); load_state_dict converte de volta para float32
    torch.save({k: v.to(torch.bfloat16) for k, v in gpt2.state_dict().items()}, ckpt + ".part")
    os.replace(ckpt + ".part", ckpt)
```

### O que está acontecendo?
**Caminho 1 — checkpoint (o que vai acontecer ao vivo):** carrega **todos** os pesos do modelo ajustado. Aqui não precisa de `strict=False` (diferente do Elmo), porque foi salvo o modelo inteiro — **todos os 124M de parâmetros foram treinados**, nada foi congelado.

**Caminho 2 — treinar:**
- `AdamW` com `lr=5e-5` e `weight_decay=0.1` (mesmos valores do classificador do Elmo), sobre **todos** os parâmetros (`gpt2.parameters()`).
- `train_model_simple` (cap. 5), para cada época e cada lote:
  1. zera os gradientes;
  2. roda o modelo → logits `(8, tokens, 50257)`: para cada posição, uma nota para cada token do vocabulário;
  3. `cross_entropy` entre os logits e o alvo em **todas as posições** (as com −100 são ignoradas) — exatamente a loss do pré-treino, "prever o próximo token";
  4. `backward()` + `optimizer.step()`;
  5. a cada 10 passos imprime a loss de treino e de validação (em 5 lotes);
  6. no fim de cada época, gera 50 tokens a partir de `format_input(val[0])` e imprime (com `\n` trocado por espaço) — para acompanhar a evolução.
- Salva em bfloat16 (~330 MB em vez de ~660 MB).

```text
Diferença para o classificador do Elmo:
  classificador:  cabeça nova (768→2), só o último bloco treina, loss só na ÚLTIMA posição
  assistente:     MESMA cabeça (768→50257), TUDO treina, loss em TODAS as posições  ← igual ao pré-treino
```

### Resultado (log salvo — de uma execução com treino)
```
Ep 1 (Step 000000): Train loss 3.022, Val loss 3.057
Ep 1 (Step 000010): Train loss 1.163, Val loss 1.216
...
Ep 1 (Step 000040): Train loss 0.771, Val loss 0.943
Below is an instruction ... ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The chef cooks the meal every day.<|endoftext|>The following is an instruction ...
Ep 2 (Step 000050): Train loss 0.684, Val loss 0.902
...
Ep 2 (Step 000090): Train loss 0.527, Val loss 0.861
Below is an instruction ... ### Response: The chef cooks the meal every day.<|endoftext|>The following is ...
1.1 min
```
- A loss cai muito rápido: 3,0 → 1,2 em só 10 passos. Aprender o **formato** é fácil.
- A loss de validação para de cair perto de 0,86 enquanto a de treino continua caindo (0,53): com só 400 exemplos, o modelo começa a se ajustar demais ao treino.
- O exemplo de validação é "passe para a voz passiva: 'The chef cooks the meal every day.'" (resposta esperada no dataset: "The meal is cooked by the chef every day."). O modelo **responde no lugar certo e gera `<|endoftext|>`**, mas só **repete a frase** — não fez a transformação.
- O texto continua depois do `<|endoftext|>` porque essa função de amostra do livro (`generate_and_print_sample`) sempre gera 50 tokens, sem parar no fim. A nossa `responder` para.
- 50 lotes × 2 épocas = 100 passos (log até o passo 90). 1,1 minuto.

**Ao vivo, com o checkpoint, a saída será apenas:** `checkpoint carregado: .../checkpoints/assistente_124M.pth`.

### Fala
"Como deixei o modelo salvo, ele só carrega. Mas vale olhar o log do treino. O otimizador e a loss são os mesmos do pré-treino: prever o próximo token em todas as posições, com o -100 ignorando o padding. E agora eu treino o modelo inteiro, não só a última camada como no classificador do Elmo.
A loss cai de 3 para 1,2 em só dez passos. Aprender o formato é fácil. No fim de cada época ele mostra um exemplo: 'passe para a voz passiva: The chef cooks the meal every day'. E olhem: ele responde depois do 'Response', no lugar certo, e gera o `<|endoftext|>`, ou seja, aprendeu a parar. Mas a resposta é a mesma frase, não passou para a passiva. Aprendeu o formato, não necessariamente a tarefa. E a loss de validação para de cair perto de 0,86: com 400 exemplos, ele já começa a decorar o treino. Um minuto de treino."

---

## Seção 5 (markdown 14): "Depois do fine-tuning"

---

## ETAPA 9 — Comparando com as respostas esperadas (célula 15)

### Código
```python
gpt2.eval()
torch.manual_seed(123)
for e in teste[:6]:
    print(f"▶ {e['instruction']} {e['input']}\n  esperado: {e['output']}\n  modelo:   {responder(gpt2, e)}\n")
```

### O que está acontecendo?
- `gpt2.eval()`: modo de avaliação (desliga o dropout).
- Para os 6 primeiros exemplos de teste (os 3 primeiros são os mesmos da ETAPA 7), mostra a resposta esperada e a do modelo, usando a mesma `responder`.

### Resultado (salvo — pode variar um pouco com o checkpoint em bfloat16)
```
▶ Rewrite the sentence using a simile. The car is very fast.
  esperado: The car is as fast as lightning.
  modelo:   The car is very fast.

▶ What type of cloud is typically associated with thunderstorms?
  esperado: The type of cloud typically associated with thunderstorms is cumulonimbus.
  modelo:   A type of cloud is typically associated with thunderstorms.

▶ Name the author of 'Pride and Prejudice'.
  esperado: Jane Austen.
  modelo:   The author of 'Pride and Prejudice' is William Shakespeare.

▶ What is the periodic symbol for chlorine?
  esperado: The periodic symbol for chlorine is Cl.
  modelo:   The periodic symbol for chlorine is CH3.

▶ Correct the punctuation in the sentence. Its time to go home.
  esperado: The corrected sentence should be: 'It's time to go home.'
  modelo:   The time to go home is 3 hours.

▶ Rewrite the sentence. The lecture was delivered in a clear manner.
  esperado: The lecture was delivered clearly.
  modelo:   The lecture was delivered in a clear manner.
```
Como interpretar:
- ✅ **Formato:** respostas curtas, diretas, sem cabeçalhos inventados, e **param** (efeito do `<|endoftext|>` + `eos_id`). Compare com a ETAPA 7.
- ✅ **Estilo:** responde "no estilo" da pergunta ("The author of ... is ...", "The periodic symbol for chlorine is ...").
- ❌ **Fatos errados:** Shakespeare em vez de Jane Austen; CH3 em vez de Cl.
- ❌ **Transformações não feitas:** a símile, a pontuação e a reescrita saem iguais à entrada ou erradas.

### Fala
"Agora o mesmo teste depois do fine-tuning, com a resposta esperada do lado. A primeira diferença salta aos olhos: as respostas são curtas, diretas, sem cabeçalho inventado, e ele para. Ele aprendeu a se comportar como assistente.
Mas olhem o conteúdo. 'Quem escreveu Orgulho e Preconceito?' William Shakespeare. 'Símbolo do cloro?' CH3. Ele responde com toda a confiança e no formato certo, só que errado. E quando a tarefa é transformar a frase, como fazer uma comparação ou corrigir a pontuação, ele devolve a frase igual ou inventa."

---

## ETAPA 10 — ✍️ Instrução da turma (célula 16)

### Código
```python
# ✍️ Instrução da turma
print(responder(gpt2, {"instruction": "Rewrite the sentence in passive voice.", "input": "The students built a GPT."}))
```

### O que está acontecendo?
Monta um exemplo novo na hora (um dicionário com `instruction` e `input`) e passa para `responder`. Dá para trocar pelo que a turma sugerir.

### Resultado
```
The students built a GPT.
```
Não passou para a passiva ("A GPT was built by the students"): repetiu a frase — o mesmo comportamento do exemplo de validação durante o treino.

> Dica: instruções em inglês funcionam melhor; o dataset é todo em inglês.

### Fala
"Alguém quer dar uma instrução? Enquanto isso, eu testei uma: 'reescreva na voz passiva: The students built a GPT'. E ele devolve a mesma frase. Consistente com o que vimos: formato certo, tarefa não cumprida."

---

## Markdown 17 — por que ele erra + Seção 6 "Onde conseguir dados?"

Com 124M e poucos exemplos ele aprende o **formato**, mas erra fatos. **Conhecimento vem do tamanho/pré-treino, não do fine-tuning**; o livro usa 355M e o dataset inteiro (e avalia as respostas com outro LLM — `ch07/03_model-evaluation`).

Dados prontos: Alpaca (52k, gerado com LLM), Dolly 15k (escrito por humanos, licença comercial), OpenAssistant (conversas), FLAN, Hugging Face Hub. Em português: traduções do Alpaca/Dolly e datasets da comunidade.

**Fala:**
"Isso responde a primeira pergunta do notebook. O fine-tuning de instrução ensina o comportamento: o formato, onde começar, quando parar. Mas ele não coloca conhecimento novo. Se o modelo não sabe quem escreveu Orgulho e Preconceito, 400 exemplos não vão ensinar isso. O conhecimento vem do pré-treino e do tamanho do modelo; o livro usa o GPT-2 de 355 milhões e o dataset inteiro, e os resultados melhoram bastante.
Segunda pergunta: onde conseguir dados? Tem muita coisa pronta: o Alpaca, com 52 mil exemplos gerados por um LLM; o Dolly, com 15 mil escritos por pessoas e liberado para uso comercial; o OpenAssistant, com conversas; e milhares de datasets no Hugging Face. Em português tem traduções do Alpaca e do Dolly. E para montar do zero, é mais simples do que parece."

---

## ETAPA 11 — Montando o seu próprio dataset (célula 18)

### Código
```python
meus_dados = [
    {"instruction": "Traduza para o inglês.", "input": "Bom dia, turma!", "output": "Good morning, class!"},
    {"instruction": "Classifique o sentimento.", "input": "Adorei a aula de hoje.", "output": "Positivo"},
    {"instruction": "O que é um token?", "input": "", "output": "Um pedaço de texto (palavra, subpalavra ou byte) que o modelo processa."},
]
with open(os.path.join(CHECKPOINTS, "meus_dados.json"), "w", encoding="utf-8") as f:
    json.dump(meus_dados, f, ensure_ascii=False, indent=2)
print(format_input(meus_dados[2]))
print(len(InstructionDataset(meus_dados, bpe)), "exemplos prontos para o mesmo DataLoader")
```

### O que está acontecendo?
- `meus_dados`: 3 exemplos escritos à mão, **no mesmo formato** do dataset do livro (`instruction`/`input`/`output`). O terceiro tem `input` vazio.
- Salva em `checkpoints/meus_dados.json` (`ensure_ascii=False` mantém os acentos).
- `format_input(meus_dados[2])`: mostra o prompt do exemplo com input vazio — a seção `### Input:` **não aparece**.
- `InstructionDataset(meus_dados, bpe)`: prova que esses dados entram no mesmo pipeline das ETAPAS 6 e 8, sem mudar nada.

### Resultado
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
O que é um token?
3 exemplos prontos para o mesmo DataLoader
```

### Fala
"Um dataset de instrução é só uma lista de dicionários com três campos. Aqui eu escrevi três exemplos em português: uma tradução, uma classificação de sentimento e uma pergunta sem input. Salvo em JSON e pronto. Olhem que, quando o input está vazio, o molde simplesmente não coloca a seção de Input. E esses três exemplos já entram no mesmo `InstructionDataset`, no mesmo DataLoader e no mesmo loop de treino que usamos agora há pouco. Só precisaria de muito mais exemplos."

---

## Markdown 19 — fontes e ✅ Construto (fechamento)

Fontes: logs de atendimento/FAQ anonimizados, documentação interna, respostas de especialistas, ou **dados sintéticos** gerados por um LLM maior e revisados (`ch07/05_dataset-generation`). **Qualidade > quantidade:** o paper LIMA mostrou bons resultados com ~1.000 exemplos bem curados. Depois vem o alinhamento por preferência (RLHF/DPO — `ch07/04_preference-tuning-with-dpo`).

**Fala de fechamento:**
"De onde tirar exemplos para o seu? FAQ e atendimento da empresa, anonimizados; documentação interna; respostas escritas por especialistas; ou gerar com um LLM maior e revisar à mão. E o mais importante: qualidade vale mais que quantidade. Um trabalho chamado LIMA mostrou bons resultados com só mil exemplos bem escolhidos.
Então, resumindo: GPT-2 pré-treinado, mais dados de instrução no formato certo, mais o mesmo loop de treino do pré-treino, dá um mini assistente. O ChatGPT ainda tem mais uma etapa depois disso, o alinhamento por preferência, com RLHF ou DPO, em que o modelo aprende qual de duas respostas as pessoas preferem. Tem código disso no repositório também. E, para fechar, a Cristiane vai explicar de onde vêm todos esses números que usamos, como 768 dimensões e 12 camadas."

---

## Roteiro contínuo do Rafael — Notebook 06

**[Título / markdown 0]**
"O Elmo acabou de mostrar que, quando a gente pergunta alguma coisa para o GPT-2 base, ele não responde: só continua o texto. Mas o ChatGPT responde. Qual é a diferença? Vou mostrar que, no fundo, são os dados. Mesmo GPT-2, mesmo loop de treino do pré-treino, treinado com exemplos de instrução e resposta. E no final mostro onde conseguir esses dados e como montar os seus."

**[Célula 1 — imports]**
"Reparem: a função de treino que eu importo é a `train_model_simple`, do capítulo 5, a mesma do pré-treino. Não tem algoritmo novo. Do capítulo 7 vêm três funções para preparar os dados. E uso só 400 exemplos para treinar ao vivo; o livro usa uns 900 e um modelo maior."

**[Célula 3 — dados]**
"São 1.100 exemplos, e cada um é um dicionário com três campos: a instrução, o input, que pode ser vazio, e o output, a resposta que queremos. Por exemplo: 'identifique a grafia correta', input 'Ocassion', resposta 'Occasion'."

**[Markdown 4 → célula 5 — Alpaca]**
"O modelo não lê dicionário, lê texto. Então cada exemplo vira um texto com um molde fixo, o estilo Alpaca: cabeçalho, instrução, input se tiver, e o marcador '### Response:', depois do qual vem a resposta. O modelo aprende: 'depois de Response vem a resposta, e depois ela termina'. Na hora de usar, eu dou só a parte de cima e ele completa."

**[Célula 6 — divisão]**
"Divido: 85% treino, 10% teste e o resto, uns 5%, validação. Do treino uso os primeiros 400. O teste é o que vou usar para comparar antes e depois."

**[Markdown 7]**
"Um detalhe técnico importante: cada exemplo tem um tamanho, mas um lote precisa ser retangular. O Elmo resolveu isso completando com o token 50256. Aqui fazemos o mesmo, com um cuidado extra no alvo."

**[Célula 8 — collate]**
"Testo a função com três sequências de mentira. Ela completa com 50256 e cria o alvo deslocado uma posição, igual ao começo da aula. Mas no alvo, o primeiro 50256 de cada linha fica: é o `<|endoftext|>`, que ensina o modelo a parar de falar. Os outros viram -100, um código que a `cross_entropy` ignora. O modelo não perde ponto pelo preenchimento, só é cobrado pelo texto de verdade e pelo sinal de fim."

**[Célula 9 — loaders]**
"O `InstructionDataset` monta cada exemplo no molde Alpaca, com a resposta, e tokeniza. O DataLoader agrupa de 8 em 8 com a nossa função de collate. São 50 lotes por época."

**[Markdown 10 → célula 11 — antes]**
"Antes de treinar, vamos ver o GPT-2 base. A função `responder` monta o prompt, gera até 60 tokens, para se aparecer o `<|endoftext|>` e devolve só a resposta.
Ele percebe os cabeçalhos em markdown e inventa outros: '### Output', '### Error'. Não para de falar, se repete, e não responde nada: 'tempestades são o tipo mais comum de tempestade'. Bom em continuar texto, mas não sabe que deveria responder."

**[Markdown 12 → célula 13 — treino/checkpoint]**
"O ponto principal: o fine-tuning de instrução é o mesmo loop do pré-treino, só mudam os dados. Como o modelo está salvo, ele só carrega. Mas no log: mesma loss do pré-treino, prever o próximo token, e agora treino o modelo inteiro. A loss cai de 3 para 1,2 em dez passos: aprender o formato é fácil. No exemplo de voz passiva, ele responde no lugar certo e gera o `<|endoftext|>`, aprendeu a parar, mas devolve a mesma frase. Aprendeu o formato, não necessariamente a tarefa. E a validação para de cair perto de 0,86: com 400 exemplos, começa a decorar. Um minuto de treino."

**[Markdown 14 → célula 15 — depois]**
"Agora o mesmo teste depois do fine-tuning. As respostas são curtas, diretas, sem cabeçalho inventado, e ele para. Aprendeu a se comportar como assistente. Mas o conteúdo: 'Quem escreveu Orgulho e Preconceito?' Shakespeare. 'Símbolo do cloro?' CH3. Confiante, no formato certo, e errado. E quando é para transformar a frase, devolve igual ou inventa."

**[Célula 16 — instrução da turma]**
"Alguém quer dar uma instrução? Eu testei 'reescreva na voz passiva: The students built a GPT', e ele devolve a mesma frase. Formato certo, tarefa não cumprida."

**[Markdown 17]**
"Isso responde a primeira pergunta: o fine-tuning de instrução ensina comportamento, formato, onde começar, quando parar. Mas não coloca conhecimento novo. O conhecimento vem do pré-treino e do tamanho do modelo; o livro usa 355 milhões e o dataset inteiro, e melhora bastante.
Onde conseguir dados? Alpaca, 52 mil exemplos gerados por LLM; Dolly, 15 mil escritos por pessoas e liberado para uso comercial; OpenAssistant, com conversas; e milhares no Hugging Face. Em português, traduções do Alpaca e do Dolly. E montar do zero é mais simples do que parece."

**[Célula 18 — meus dados]**
"Um dataset de instrução é só uma lista de dicionários com três campos. Escrevi três em português: uma tradução, uma classificação de sentimento e uma pergunta sem input. Salvo em JSON. Quando o input está vazio, o molde não coloca a seção de Input. E esses três exemplos já entram no mesmo dataset, no mesmo DataLoader e no mesmo loop de treino. Só precisaria de muito mais exemplos."

**[Markdown 19 — fechamento]**
"De onde tirar exemplos? FAQ e atendimento anonimizados, documentação interna, respostas de especialistas, ou gerar com um LLM maior e revisar. Qualidade vale mais que quantidade: o trabalho LIMA mostrou bons resultados com mil exemplos bem escolhidos.
Resumindo: GPT-2 pré-treinado, mais dados de instrução no formato certo, mais o mesmo loop do pré-treino, dá um mini assistente. O ChatGPT ainda tem o alinhamento por preferência, RLHF ou DPO, em que o modelo aprende qual resposta as pessoas preferem. E para fechar, a Cristiane vai explicar de onde vêm todos esses números que usamos."
