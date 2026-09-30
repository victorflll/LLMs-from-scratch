# NOTEBOOK 02 — Atenção

**O que o notebook constrói:** a peça de atenção (`MultiHeadAttention`) que faz cada token "olhar" para os outros e misturar a informação deles. O Victor a encaixa no GPT no notebook 03.

## Preparação (primeiro bloco, antes da seção 1)

*Bloco que começa com* `import matplotlib.pyplot as plt`

**Objetivo:** preparar a entrada da atenção: 6 tokens, cada um representado por um vetor.

**Cálculo:** nenhum. `x` é uma tabela 6 × 3 digitada (exemplo do livro); a saída mostra cada palavra com seus 3 números. A função `mapa` só é **definida** (desenha a partir da seção 1).

**Conceitos:**

- **Token:** pedaço de texto (palavra, parte de palavra ou símbolo) que o modelo processa.
- **Embedding:** vetor de números que representa um token (3 números aqui, 768 no GPT-2).
- **Tensor / shape:** tabela de números / o seu formato; ex.: (6, 3) = 6 linhas × 3 colunas.
- **Seed (semente):** ponto de partida do sorteio: mesma seed = mesmos números "aleatórios".

**Perguntas possíveis:**

- *Por que os números do `x` estão digitados?* É o exemplo do livro (cap. 3): 3 números inventados por palavra, para caber na tela. No GPT real seriam 768 números aprendidos no treino (notebook 01).
- *O que é `torch.manual_seed(123)`?* Fixa o "sorteio": os pesos iniciais de `W_q`, `W_k`, `W_v` são aleatórios, mas saem sempre iguais, e os números da tela são sempre os mesmos. O 123 é só um número escolhido.
- *Por que o bloco do `mapa` não mostra nada?* Ela só define a função (a "receita"); o desenho aparece quando a função é usada, a partir da seção 1.
- *Onde está o `llms_from_scratch.ch03`?* Em `pkg/llms_from_scratch/ch03.py` (código do cap. 3 do livro; a `MultiHeadAttention` está na linha 98). Funciona porque o `import aula` coloca a pasta `pkg/` no caminho do Python; por isso ele vem antes.

**Fala:** "Para as contas caberem na tela, uso o exemplo do livro: 6 palavras, cada uma com só 3 números, em vez de 768. Olhem: cada palavra virou 3 números, e é com eles que a atenção vai trabalhar. A função `mapa` vai desenhar a atenção; por enquanto eu só defino, o primeiro gráfico vem já na próxima célula."

## 1. Q, K, V e o score

*Bloco que começa com* `d_k = 2`

**Objetivo:** calcular, para cada token, **quanto ele deve prestar atenção em cada um dos outros** e produzir um **novo vetor** que mistura a informação da frase.

**Cálculo (5 passos):**

1. **Criar as 3 transformações** (1ª e 2ª linhas): `d_k = 2` define o tamanho dos novos vetores. `torch.nn.Linear(3, d_k, bias=False)` cria uma *camada linear*: uma matriz de pesos 2 × 3 que transforma um vetor de 3 números em um de 2 (`bias=False` = só multiplica, sem somar nada no fim). O `for _ in range(3)` cria **três** camadas independentes, cada uma com seus próprios números sorteados (entre −0,58 e 0,58), e elas são guardadas em `W_q`, `W_k` e `W_v`.
2. `W_q`, `W_k`, `W_v` transformam `x` em três vetores por token: **Q** (o que procura), **K** (o que oferece), **V** (o que entrega) → cada um 6 × 2.
3. `Q @ K.T` → *score* de cada par (6 × 6): *produto escalar* da query de um com a key do outro.
4. `softmax(scores / √d_k)` → *pesos*: porcentagens que somam 1 em cada linha.
5. `pesos @ V` → *contexto*: cada token vira a média dos values, ponderada pelos pesos (6 × 2).

No heatmap, todos os pesos ficam entre **0,16 e 0,18** (≈ 1/6): os pesos de `W_q`/`W_k` são aleatórios.

**E se trocar?**

- **Outra semente** (`manual_seed`): os números mudam um pouco (ex.: seed 42 → 0,13–0,19), mas continua quase uniforme.
- **`d_k` maior** (16, 64, 256): continua quase uniforme (0,14–0,20). O tamanho sozinho não cria padrão; falta treino.
- **`x` × 10** (vetores maiores): a atenção se concentra (pesos de 0,00 a 0,54). Vetores maiores dão scores maiores, e o softmax "escolhe" mais.

**Conceitos:**

- **Camada linear (`nn.Linear`):** uma matriz de pesos que multiplica o vetor de entrada e devolve um vetor de outro tamanho (aqui, 3 → 2). Os pesos começam sorteados e são o que o treino ajusta.
- **Query / Key / Value:** três versões de cada token: o que procura / o que oferece / o que entrega.
- **Transposta (`.T`):** a tabela com linhas e colunas trocadas.
- **Multiplicação de matrizes (`@`):** combina cada linha de uma tabela com cada coluna da outra (multiplica e soma).
- **Produto escalar:** multiplica os números correspondentes de dois vetores e soma: mede o quanto "apontam para o mesmo lado".
- **Score:** resultado de query × key: o quanto um token combina com outro (antes do softmax).
- **Softmax:** transforma scores em porcentagens positivas que somam 1.
- **Pesos de atenção:** os scores depois do softmax: quanto cada token olha para cada outro.
- **Contexto:** o novo vetor de cada token: média dos values ponderada pelos pesos.
- **Escala √d:** divisão dos scores pela raiz do tamanho do vetor (aqui √2). Um produto escalar soma `d` termos, então cresce com o tamanho do vetor; dividir por √d mantém os scores numa escala parecida.

**Perguntas possíveis:**

- *O que significam o `@` e o `.T` em `Q @ K.T`?* `.T` troca linhas por colunas (K tem 1 linha por token, 6×2; K.T tem 1 coluna por token, 2×6). Precisa porque o `@` combina linha de uma tabela com coluna da outra; sem o `.T`, (6×2) @ (6×2) não encaixa e dá erro. O `@` multiplica e soma para cada par (ex.: journey × one = (−0,30 × −0,47) + (−0,03 × 0,17) ≈ 0,14). Resultado: score 6×6, um por par.
- *Por que três matrizes, e não uma só?* Porque o mesmo token precisa de três papéis diferentes. Com uma matriz só, Q, K e V seriam iguais, e o token não conseguiria "procurar" uma coisa e "oferecer" outra.
- *O que é o `bias=False`?* Uma camada linear pode multiplicar e depois somar um vetor fixo (o bias). Com `False`, ela só multiplica; é o que o livro usa para Q, K e V.
- *Por que dividir por √d_k?* Sem a divisão, com vetores grandes os scores ficam enormes e o softmax dá quase 100% para um único token. O modelo "trava" e para de aprender para onde olhar. No GPT-2 (vetores de 64 por head), um token chegaria a levar ~95% da atenção sem escala, contra ~25% com escala.
- *Por que o gráfico é todo azul?* Sem treino, os scores são parecidos, e o softmax dá ~17% para cada token.

**Fala:** "Esse bloco é a atenção inteira. Primeiro eu crio três camadas lineares, `W_q`, `W_k` e `W_v`: são três matrizes de números sorteados que transformam os 3 números de cada palavra em 2. Com elas, cada token ganha três vetores: a query, 'o que eu procuro'; a key, 'o que eu ofereço'; e o value, 'o que eu entrego'. `Q @ K.T` dá um score para cada par de tokens: o quanto um combina com o outro. O softmax transforma os scores de cada linha em porcentagens que somam 100%. E `pesos @ V` mistura os values usando essas porcentagens: se 'journey' dá 50% de atenção para 'starts', metade do novo 'journey' vem de 'starts'.
E esse é o nosso primeiro mapa de atenção: a linha é quem olha, a coluna é para quem. Tudo azul-claro, e os números confirmam: uns 17% para cada. Ninguém treinou nada ainda. Ou seja: sem treino, a atenção não sabe para onde olhar. No final, no GPT-2 treinado, vamos ver padrões de verdade."

## 2. Outras formas de calcular o score *(pergunta 1)*

*Bloco que começa com* `torch.manual_seed(0)`

**Objetivo:** mostrar que o produto escalar escalado (o do GPT) é **uma escolha**, não a única forma de medir o quanto dois tokens combinam.

**Cálculo:** 4 fórmulas para o score, depois softmax:

- produto escalar: multiplica e soma;
- escalado: o mesmo ÷ √d (o GPT usa este);
- aditivo (Bahdanau, 2014): junta query e key e passa por uma mini rede neural;
- cosseno: compara só a direção dos vetores (÷ 0,1 para ampliar).

As três primeiras ficam em **0,15–0,18**; o cosseno vai de **0,01 a 0,47**.

**E se trocar?** O divisor do cosseno controla o quanto a atenção se concentra:

- ÷ 1 → 0,13–0,20 (quase igual);
- ÷ 0,1 → 0,01–0,47;
- ÷ 0,01 → 0,00–0,99 (quase tudo num token só).

**Perguntas possíveis:**

- *Existem outras formas de calcular a atenção?* Sim: a aditiva e o cosseno, que estão neste bloco. O GPT usa o produto escalar escalado porque é só multiplicação de matrizes, muito rápido.
- *Por que três gráficos são quase iguais?* Sem treino, as três fórmulas dão scores muito parecidos (0,15–0,18); só o cosseno, multiplicado por 10, concentra mais.

**Fala:** "Existem outras formas de calcular o score. A aditiva, anterior ao Transformer, usa uma mini rede neural; o cosseno só compara a direção dos vetores. Sem treino, as três primeiras espalham a atenção quase igual; o cosseno, que aqui foi multiplicado por 10, concentra mais. O gráfico não diz qual é melhor, porque nada foi treinado. O GPT usa o produto escalar porque é só multiplicação de matrizes, muito rápido."

## 3. Máscara causal: o GPT não pode olhar o futuro

*Bloco que começa com* `n = len(tokens)`

**Objetivo:** fazer cada token olhar **só para si mesmo e para os tokens anteriores**. O GPT escreve uma palavra de cada vez, da esquerda para a direita; quando ele está escrevendo a palavra 3, as palavras 4, 5 e 6 ainda não existem. Se no treino ele pudesse vê-las, ele copiaria a resposta em vez de aprender a prever.

**Cálculo, linha a linha:**

1. `mascara = torch.triu(torch.ones(n, n), diagonal=1).bool()` monta uma tabela 6 × 6 que marca com `True` tudo o que está **acima da diagonal**. Cada `True` é um par "token olhando para um token que vem depois dele":

```text
            Your journey starts with  one  step
Your      [  .     X      X     X    X    X  ]
journey   [  .     .      X     X    X    X  ]
starts    [  .     .      .     X    X    X  ]
with      [  .     .      .     .    X    X  ]
one       [  .     .      .     .    .    X  ]
step      [  .     .      .     .    .    .  ]
                       X = futuro (proibido)
```

2. `masked_fill(mascara, -torch.inf)` troca o score dessas posições por **menos infinito**.
3. O softmax transforma −∞ em **exatamente 0%** e divide os 100% da linha só entre os tokens permitidos.

**O que os dois gráficos mostram:**

- **Sem máscara (esquerda):** todos veem todos: ~0,17 em cada quadrado, como na seção 1.
- **Com máscara (direita):** um triângulo, onde cada linha divide 100% só entre os tokens que já apareceram:
    - "Your" é o 1º token, só pode olhar para si → **1,00**;
    - "journey" divide entre 2 tokens → **0,48 / 0,52**;
    - "starts" divide entre 3 → **0,32 / 0,34 / 0,34**;
    - … "step", o último, divide entre os 6 → ~0,17 cada. É o único que vê a frase inteira.

Os valores de cada linha somam 1, mas quanto mais para baixo, mais tokens para dividir, e por isso as linhas vão clareando.

**E se trocar?**

- **Sem a máscara:** no treino, o modelo veria a palavra que deveria prever e só a copiaria; aprenderia a "colar", não a prever.
- **Com 0 no lugar de −∞:** não funcionaria. O softmax de 0 não é 0 (vira e⁰ = 1, um peso normal), então o futuro continuaria recebendo atenção. Só −∞ vira exatamente 0.

**Conceitos:**

- **Máscara causal:** a tabela que bloqueia a atenção para os tokens seguintes (o futuro). "Causal" porque só o que veio antes pode influenciar o que vem depois.

**Perguntas possíveis:**

- *Por que −∞ e não 0?* Porque o softmax calcula e^score: e⁰ = 1 (ainda conta), e e^(−∞) = 0 (some de verdade).
- *Por que a primeira linha tem 1,00?* O primeiro token não tem ninguém antes dele; toda a atenção vai para ele mesmo.
- *Isso tem a ver com o classificador do Elmo?* Sim: por causa da máscara, só o último token enxerga a mensagem inteira, e por isso o classificador de spam usa a saída do último token.

**Fala:** "O GPT escreve uma palavra de cada vez, da esquerda para a direita. Então, quando ele está escrevendo uma palavra, as seguintes ainda não existem, e no treino ele não pode espiar, senão só copiaria a resposta. A máscara é uma tabela que marca tudo acima da diagonal: são os pares 'token olhando para o futuro'. Nesses lugares eu coloco menos infinito, e o softmax transforma isso em exatamente zero.
Comparem os dois gráficos. Sem máscara, todo mundo olha para todo mundo, uns 17% em cada. Com máscara vira um triângulo, e cada linha divide os 100% só entre quem já apareceu: 'Your', o primeiro, só pode olhar para si mesmo, 100%; 'journey' divide entre dois, quase meio a meio; 'starts' entre três; e o último token é o único que vê a frase inteira. É por isso que o Elmo usa o último token no classificador de spam."

## 4. Multi-head: várias atenções em paralelo

**Ideia da seção:** em vez de uma atenção só, o GPT faz **várias atenções menores ao mesmo tempo** (as *heads*). Cada head pega um pedaço do vetor de cada token e procura um tipo de relação diferente (ex.: uma olha a palavra anterior, outra liga pronome e substantivo, como veremos na seção 6).

**De onde vem o 768:** até a seção 3, cada palavra tinha 3 números (exemplo de brinquedo). A partir daqui o notebook usa o tamanho real do GPT-2: **cada token é representado por 768 números** (o tamanho do embedding, `"emb_dim": 768` em `30-09/aula.py`; é a tabela de 50.257 × 768 que o Elmo mostra no notebook 01). Esses 768 números é que são cortados entre as heads: 12 heads × 64 = 768.

### 4.1 `pesos_atencao`: multi-head por dentro

*Bloco que começa com* `# a MultiHeadAttention do livro só devolve o resultado final`

**Objetivo:** conseguir **ver** os pesos de atenção de cada head. A classe `MultiHeadAttention` do livro faz toda a conta, mas só devolve o resultado final (o novo vetor de cada token); os pesos ficam escondidos lá dentro. Esta função repete os mesmos passos da classe e para no softmax, para podermos desenhar os pesos nas seções 4.2 e 6.

**Resultado:** nenhum. O bloco só **define** a função; ela é usada em 4.2 e na seção 6.

**O que a função recebe e devolve:**

- recebe `mha`: uma atenção já criada (um objeto `MultiHeadAttention`, com suas matrizes `W_query`, `W_key`, o `num_heads`, o `head_dim` e a `mask`);
- recebe `x`: os vetores dos tokens, no formato (frases, tokens, números), ex.: (1, 6, 768);
- devolve os pesos de todas as heads, no formato (frases, heads, tokens, tokens), ex.: (1, 12, 6, 6).

**Cálculo, linha a linha** (exemplo com 1 frase, 6 tokens, 768 números e 12 heads):

1. `b, t, _ = x.shape`: o `.shape` diz o **tamanho de cada dimensão** do tensor: (1, 6, 768). A linha guarda o 1 em `b` (frases) e o 6 em `t` (tokens); o `_` descarta o 768, que não é usado.
2. `mha.W_query(x)`: aplica a matriz de queries da própria classe (igual ao `W_q` da seção 1, só que 768 → 768) → (1, 6, 768).
3. `.view(b, t, num_heads, head_dim)`: **reorganiza** os números sem mudar nenhum: os 768 números de cada token são vistos como 12 pedaços de 64 → (1, 6, 12, 64). Os números 0–63 vão para a head 0, os 64–127 para a head 1, e assim por diante.
4. `.transpose(1, 2)`: **troca de lugar** as dimensões 1 (tokens) e 2 (heads) → (1, 12, 6, 64). Assim cada head fica com a sua própria tabela 6 × 64 e faz a conta separada das outras.
5. O mesmo para as keys (`k`).
6. `q @ k.transpose(2, 3)`: é o `Q @ K.T` da seção 1, feito para as 12 heads de uma vez. O `transpose(2, 3)` "deita" as duas últimas dimensões de `k`: (1, 12, 64, 6). Resultado: (1, 12, 6, 6), uma tabela de scores 6 × 6 por head.
7. `.masked_fill(mha.mask.bool()[:t, :t], -torch.inf)`: a máscara causal da seção 3. O `mha.mask` é uma máscara 1024 × 1024 guardada dentro da classe; o `[:t, :t]` recorta só o pedaço 6 × 6 que precisamos; o `.bool()` transforma 1/0 em True/False.
8. `torch.softmax(s / mha.head_dim**0.5, dim=-1)`: divide por √64 = 8 e transforma os scores de cada linha em porcentagens, como na seção 1.

```text
x                     (1, 6, 768)     1 frase, 6 tokens, 768 números
W_query(x)            (1, 6, 768)
.view(1, 6, 12, 64)   (1, 6, 12, 64)  768 = 12 pedaços de 64
.transpose(1, 2)      (1, 12, 6, 64)  heads à frente: 12 tabelas 6 × 64
q @ kᵀ                (1, 12, 6, 6)   12 tabelas de scores 6 × 6
máscara + softmax     (1, 12, 6, 6)   12 tabelas de pesos
```

**O que a classe faz depois** (código do livro, não está no notebook): multiplica os pesos pelos values de cada head, **junta** (concatena) as 12 saídas de 64 de volta em 768 e passa por mais uma camada linear, a `out_proj`. Por isso a saída final continua com 768 números por token.

**Conceitos:**

- **Head / multi-head:** uma atenção independente / várias em paralelo, cada uma com um pedaço do vetor.
- **head_dim:** tamanho do pedaço de cada head (768 ÷ 12 = 64 no GPT-2).
- **`.shape`:** o formato de um tensor, o tamanho de cada dimensão, ex.: (1, 6, 768).
- **`.view`:** reorganiza os números num novo formato, sem mudar nenhum valor (o total de números tem que continuar o mesmo: 6 × 768 = 6 × 12 × 64).
- **`.transpose(a, b)`:** troca de lugar duas dimensões (o `.T` da seção 1 é o caso com duas dimensões).
- **Concatenar:** juntar pedaços lado a lado (12 pedaços de 64 → 768).

**Perguntas possíveis:**

- *O que é esse 768?* É quantos números o GPT-2 usa para representar cada token (o tamanho do embedding). Nas seções 1–3 eram 3, para caber na tela; aqui é o tamanho real. Foi uma escolha de projeto da OpenAI: mais números guardam mais "significado", mas o modelo fica maior (o GPT-2 medium usa 1024; o XL, 1600).
- *O que o `.shape` faz?* Só informa o formato do tensor: quantos elementos há em cada dimensão. Não muda nada.
- *Qual a diferença entre `.view` e `.transpose`?* O `.view` muda como os números são agrupados (768 → 12 × 64); o `.transpose` muda a ordem das dimensões (tokens ↔ heads). Nenhum dos dois altera valores.
- *Por que refazer a conta, se a classe já faz?* Porque a classe não devolve os pesos, e sem eles não dá para desenhar a atenção.

**Fala:** "A partir de agora eu saio do exemplo de brinquedo: no GPT-2 de verdade, cada palavra não tem 3 números, tem 768. Multi-head é fazer várias atenções em paralelo, e o que se divide entre elas são justamente esses 768 números. A classe do livro faz tudo, mas só devolve o resultado final; por isso essa função refaz a conta e para nos pesos, para a gente conseguir desenhar. O `.shape` só informa o tamanho: 1 frase, 6 tokens, 768 números. Aí o `.view` corta os 768 números de cada token em 12 pedaços de 64, e o `.transpose` coloca as heads na frente, para cada uma fazer a sua própria atenção. Daí para frente é a mesma conta da seção 1, com a máscara da seção 3, só que 12 vezes ao mesmo tempo. No final, a classe junta os 12 pedaços de volta em 768."

### 4.2 Mudando o número de heads *(pergunta 3)*

*Bloco que começa com* `# Mudando o número de heads`

**Objetivo:** responder "o que acontece se eu mudar a quantidade de heads?", medindo o que muda e o que não muda.

**Cálculo, linha a linha:**

1. `X = torch.randn(1, 6, 768)`: cria uma entrada de mentira, com números aleatórios, no formato do GPT-2: 1 frase, 6 tokens, 768 números por token. Só serve para testar os formatos.
2. `for h in [1, 2, 4, 12, 24]`: repete o teste para cada número de heads.
3. `MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=h)` cria a atenção:
    - `d_in` / `d_out`: tamanho do vetor que entra e que sai (768);
    - `context_length`: máximo de tokens que ela aceita (1024, como no GPT-2); define o tamanho da máscara;
    - `dropout=0.0`: desliga o "apagar aleatório" usado no treino;
    - `num_heads`: quantas heads.
4. O `print` mede quatro coisas:
    - `mha.head_dim`: tamanho de cada pedaço (768 ÷ heads);
    - `sum(p.numel() for p in mha.parameters())`: o total de parâmetros. O `.parameters()` lista as matrizes de pesos da atenção, o `.numel()` conta quantos números cada uma tem, e o `sum` soma tudo;
    - `mha(X).shape`: roda a atenção de verdade e mostra o formato da saída;
    - `pesos_atencao(mha, X).shape`: o formato dos pesos (a função de 4.1).

**Resultado:**

| heads | head_dim | parâmetros | saída | pesos |
|---|---|---|---|---|
| 1 | 768 | 2.360.064 | (1, 6, 768) | (1, 1, 6, 6) |
| 12 | 64 | 2.360.064 | (1, 6, 768) | (1, 12, 6, 6) |
| 24 | 32 | 2.360.064 | (1, 6, 768) | (1, 24, 6, 6) |

- **Parâmetros iguais**: as matrizes são as mesmas (W_query, W_key, W_value e out_proj, cada uma 768 × 768, mais 768 de bias = 2.360.064). Mudar as heads só muda **como os 768 números são cortados**.
- **Saída igual**: sempre 768 números por token, por isso dá para trocar o número de heads sem mexer no resto do GPT.
- **Pesos**: uma tabela 6 × 6 **por head**.

**E se trocar?**

- Mais heads → mais tipos de relação procurados ao mesmo tempo, mas cada head com menos números para trabalhar.
- Menos heads → cada head mais "rica", mas menos padrões.
- O custo é o mesmo; o efeito na qualidade só aparece treinando (não mostrado aqui). O GPT-2 mantém 64 por head em todos os tamanhos (texto depois do bloco).

**Conceitos:**

- **Parâmetros (pesos):** os números ajustáveis do modelo, ex.: as matrizes `W_query`, `W_key`, `W_value` e `out_proj`.
- **Dropout:** no treino, desliga aleatoriamente parte dos valores para o modelo não "decorar"; aqui fica em 0.

**Perguntas possíveis:**

- *E se mudar o número de heads?* Parâmetros e saída ficam iguais; só muda o tamanho de cada pedaço. O efeito na qualidade só aparece treinando.
- *O que são o `.parameters()` e o `.numel()`?* O `.parameters()` devolve as tabelas de pesos da camada; o `.numel()` conta quantos números uma tabela tem (ex.: 768 × 768 = 589.824).
- *Por que usar números aleatórios (`randn`) no `X`?* Porque aqui só interessa o formato da saída, não o conteúdo.

**Fala:** "Aqui eu testo a pergunta: o que acontece se eu mudar o número de heads? Crio uma frase de mentira no formato do GPT-2 e monto a atenção com 1, 2, 4, 12 e 24 heads. Olhem o que não muda: o número de parâmetros é sempre 2,36 milhões e a saída é sempre 768 números por token, porque as matrizes são as mesmas; só muda em quantos pedaços o vetor é cortado. Com 12, que é o GPT-2, são 12 pedaços de 64. Mais heads procuram mais tipos de relação, mas cada uma com menos números. O GPT-2 usa sempre 64 por head."

## 5. E se não for divisível? *(pergunta 4)*

### 5.1 O erro

*Bloco que começa com* `try:`

**Objetivo:** mostrar a restrição do corte em heads.

**Cálculo:** 5 heads → `AssertionError`, porque 768 ÷ 5 = 153,6, e não existe head com 153,6 números.

**E se trocar?** Qualquer número que divida 768 funciona (1, 2, 3, 4, 6, 8, 12, 16, 24…).

**Perguntas possíveis:**

- *E se a dimensão não for divisível pelo número de heads?* Dá erro na criação da camada. A solução (5.2) é fixar o tamanho de cada head e multiplicar.

**Fala:** "Com 5 heads dá erro: 768 dividido por 5 dá 153,6, e não dá para cortar em pedaços iguais."

### 5.2 A solução

*Bloco que começa com* `# 5 heads de 64`

**Objetivo:** mostrar como os modelos reais fazem quando querem um número de heads "que não encaixa".

**Cálculo:** fixa 64 números por head e multiplica: 5 × 64 = **320** → saída (1, 6, 320). Em modelos reais, a `out_proj` depois volta ao tamanho do modelo. Exemplo no repositório: o Qwen3 tem 16 × 128 = 2048 → 1024.

**Fala:** "A solução é inverter: escolho o tamanho de cada head e multiplico, 5 vezes 64 dá 320. Modelos como o Qwen fazem assim e depois voltam para o tamanho do modelo."

## 6. Scores de verdade: as 12 heads de uma camada do GPT-2 treinado *(pergunta 2)*

### 6.1 As 12 heads da camada 4

*Bloco que começa com* `gpt2, _ = load_gpt2("124M")`

**Objetivo:** ver a atenção de um modelo **treinado** e comparar com o mapa azul da seção 1.

**Cálculo:**

- A frase vira IDs, depois *embedding* de token + posição (igual ao notebook 01).
- Passa pelas camadas 0 a 3; na **camada 4**, calcula os pesos das 12 heads (`norm1` = normalização que o GPT aplica antes da atenção).
- Nos mapas grandes, só aparecem os valores a partir de 0,10.

Padrões:

- **head 11:** cada token olha para o **anterior** (1,00);
- **head 7:** cada token olha para **si mesmo**;
- **head 1:** " it", " was" e " too" olham para **" because"**;
- várias heads jogam atenção no **primeiro token** ("lugar neutro");
- **head 3:** " it" olha para **" animal"**.

**E se trocar a `CAMADA`?** Melhor head para " it" → " animal" em cada camada:

| Camada | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7–11 |
|---|---|---|---|---|---|---|---|---|
| " it" → " animal" | 0,21 | 0,28 | 0,09 | 0,29 | **0,85** | 0,41 | 0,40 | ≤ 0,36 |

Nas camadas finais, as heads jogam cada vez mais atenção no primeiro token (média de 0,50 na camada 4, 0,79 na 10).

**Conceitos:**

- **Camada / bloco:** uma etapa do GPT (atenção + rede feed-forward); o GPT-2 small tem 12.
- **LayerNorm (`norm1`):** normalização dos vetores antes da atenção, para manter os valores numa escala estável.

**Perguntas possíveis:**

- *Dá para ver a atenção de uma head?* Sim: em 6.1 (as 12 heads da camada 4) e em 6.2 (camada 4, head 3: " it" → " animal" = 0,85). Tecnicamente, o que aparece são os pesos (depois do softmax).

**Fala:** "Agora a atenção de verdade. A pergunta é: quando o modelo lê 'it', ele sabe que é o animal? Lembram do mapa todo azul-claro do começo? Aquilo era atenção sem treino; aqui, depois do treino, cada head aprendeu para onde olhar. A head 11 olha sempre para a palavra anterior, com 100%; a head 7 para a própria palavra; várias jogam atenção no primeiro token, como um lugar neutro. E na head 3, o 'it' olha para 'animal'. Ninguém programou isso."

### 6.2 Para onde o " it" olha

*Bloco que começa com* `# Uma head só`

**Objetivo:** isolar uma head e mostrar, em barras, para onde vai a atenção do " it".

**Cálculo:** pega a linha do " it" na head 3 da camada 4: **" animal" = 0,85**, "The" = 0,08, " street" ≈ 0. É a melhor das 144 combinações (camada × head).

**E se trocar o `HEAD`?** Na camada 4, o " it" olha mais para:

- head 0 → "The" (0,79);
- head 1 → " because" (0,73);
- head 7 → ele mesmo (0,53);
- head 11 → " because" (1,00; a palavra anterior).

**Fala:** "Isolando essa head: 85% da atenção do 'it' vai para 'animal', e quase nada para 'street'. Ela resolveu a quem o pronome se refere. Mas é uma head específica: trocando o `HEAD`, a maioria olha para outras coisas."

## Construto (fechamento)

**Perguntas possíveis:**

- *O que é o "construto"?* O nome que a aula dá à peça que cada notebook entrega para o próximo: no 02 é a `MultiHeadAttention`, que o Victor usa no GPT.

**Fala:** "Resumindo: query com key dá os scores, divido por raiz de d, bloqueio o futuro, o softmax vira porcentagens e misturo os values. Multi-head é fazer isso em paralelo e juntar. Entra 768 e sai 768, então dá para empilhar. É essa peça que o Victor vai encaixar no GPT."

---

# NOTEBOOK 06 — Assistente

**O que o notebook constrói:** um GPT-2 que responde instruções. O caminho é: pegar o GPT-2 pré-treinado, mostrar exemplos de pergunta → resposta e rodar o **mesmo treino do pré-treino** com esses exemplos (*fine-tuning*).

## Preparação (primeiro bloco, antes da seção 1)

*Bloco que começa com* `import json`

**Objetivo:** carregar as ferramentas e definir as duas opções da aula.

**Linha a linha:**

1. `import json`, `os`, `time`: ler arquivos JSON (onde estão os exemplos), montar caminhos de pastas e medir tempo.
2. `from functools import partial` e `DataLoader`: ferramentas para montar os lotes de treino (usadas na seção 3, só quando o treino roda ao vivo).
3. `import tiktoken` e `torch`: o tokenizador do GPT-2 (texto ↔ números) e o PyTorch.
4. `from aula import REPO_DIR, CHECKPOINTS, device, load_gpt2`: a pasta do repositório, a pasta dos modelos salvos, onde rodar (GPU/MPS/CPU) e a função que carrega o GPT-2.
5. `from llms_from_scratch.ch05 import ...`: do cap. 5 do livro vêm `generate` (escrever texto), `text_to_token_ids` / `token_ids_to_text` (texto ↔ IDs) e `train_model_simple`, o **mesmo loop de treino do pré-treino**.
6. `from llms_from_scratch.ch07 import ...`: do cap. 7 vêm `format_input` (monta o prompt), `InstructionDataset` (prepara os exemplos para o treino) e `custom_collate_fn` (monta os lotes de treino).
7. `CARREGAR_CHECKPOINT = True`: usa o modelo já treinado e salvo, em vez de treinar ao vivo.
8. `N_EXEMPLOS_TREINO = 400`: quantos exemplos usar no treino (poucos, para caber na aula).
9. `bpe = tiktoken.get_encoding("gpt2")`: o tokenizador do GPT-2.

**Conceitos:**

- **Pré-treino:** treino inicial em muito texto para prever o próximo token (de onde vem o conhecimento).
- **Fine-tuning:** treino extra, com poucos exemplos, para ensinar uma tarefa ou um comportamento.
- **Checkpoint:** arquivo com os pesos salvos de um modelo.

**Fala:** "O Elmo mostrou que o GPT-2 base não responde pergunta, só continua o texto. Aqui eu vou mostrar como ele vira um assistente. Reparem na função de treino que eu importo: é a `train_model_simple`, a mesma do pré-treino. Não tem algoritmo novo; o que muda são os dados. E como o treino leva um minuto, eu deixei o modelo treinado salvo, e essa flag faz ele só carregar."

## 1. Os dados: pares instrução → resposta

### 1.1 Carregando os exemplos

*Bloco que começa com* `with open(`

**Objetivo:** ver como é um exemplo de treino de um assistente.

**Linha a linha:**

1. `os.path.join(REPO_DIR, "ch07", "01_main-chapter-code", "instruction-data.json")`: monta o caminho do arquivo de exemplos do livro.
2. `with open(...) as f:` abre o arquivo; `json.load(f)` lê o conteúdo e transforma numa **lista de dicionários**, `dados`.
3. `len(dados)`: quantos exemplos há → **1100**.
4. `json.dumps(dados[50], indent=2, ensure_ascii=False)`: mostra o exemplo de número 50 "bonito": `indent=2` quebra em linhas, e `ensure_ascii=False` mantém os acentos.

**Resultado:** cada exemplo tem três campos:

- `instruction`: o pedido ("Identify the correct spelling of the following word.");
- `input`: um dado extra, que pode ficar vazio ("Ocassion");
- `output`: a resposta certa ("The correct spelling is 'Occasion.'").

**Conceitos:**

- **Dicionário:** estrutura do Python com pares nome → valor, ex.: `{"instruction": "...", "output": "..."}`.

**Fala:** "Esses são os dados: 1.100 exemplos. Cada exemplo é só um dicionário com três campos: a instrução, que é o pedido; um input opcional; e o output, a resposta que queremos que o modelo aprenda a dar."

### 1.2 Molde Alpaca

*Bloco que começa com* `print(format_input(dados[50])`

**Objetivo:** transformar o exemplo num **texto** que o modelo consiga ler, porque o modelo não lê dicionário.

**Linha a linha:**

1. `format_input(dados[50])` monta o prompt:
    - um cabeçalho fixo ("Below is an instruction that describes a task...");
    - `### Instruction:` + a instrução;
    - `### Input:` + o input (só se ele não estiver vazio).
2. `+ f"\n\n### Response:\n{dados[50]['output']}"` acrescenta o marcador `### Response:` e a resposta. Os `\n` são quebras de linha.

**Resultado:** o exemplo completo, do jeito que o modelo vê no treino (cabeçalho → Instruction → Input → Response).

**Por que isso funciona:** vendo centenas de textos nesse molde, o modelo aprende que depois de `### Response:` vem uma resposta, e que depois dela o texto acaba. Na hora de usar, damos só a parte de cima, e ele completa.

**Conceitos:**

- **Prompt:** o texto de entrada que damos ao modelo.
- **Molde Alpaca:** o formato fixo Instruction → Input → Response (vem do dataset Alpaca).

**Fala:** "O modelo só lê texto, então cada exemplo vira um texto com esse molde: um cabeçalho, a instrução, o input e o marcador 'Response' com a resposta. Ele vai ver centenas de textos assim e aprender que depois de 'Response' vem a resposta, e que depois ela acaba."

### 1.3 Divisão treino / validação / teste

*Bloco que começa com* `n_tr, n_te =`

**Objetivo:** separar os exemplos em três grupos: um para **aprender**, um para **acompanhar** o treino e um para **testar** no final.

**Linha a linha:**

1. `int(len(dados) * 0.85)` e `int(len(dados) * 0.10)`: 85% de 1100 = **935** (treino) e 10% = **110** (teste).
2. `dados[:n_tr]`, `dados[n_tr:n_tr + n_te]`, `dados[n_tr + n_te:]`: corta a lista em três pedaços, na ordem: os primeiros 935, os 110 seguintes e o resto (**55**, a validação).
3. `treino = treino[:N_EXEMPLOS_TREINO]`: fica só com os primeiros **400** para o treino ser rápido.

**Resultado:** `400 treino | 55 validação | 110 teste`.

**E se trocar?** `N_EXEMPLOS_TREINO` maior deixa o treino mais lento; o livro usa todos os 935 (e o GPT-2 de 355M), com resultados melhores.

**Perguntas possíveis:**

- *Por que separar um grupo de teste?* Para medir o modelo com exemplos que ele **nunca viu** no treino; senão ele poderia só ter decorado.

**Fala:** "Separo os exemplos: a maior parte para treinar, um pedaço para acompanhar o treino e outro para testar no final, com perguntas que o modelo nunca viu. E do treino eu uso só 400, para caber na aula."

## 2. Antes do fine-tuning

*Bloco que começa com* `gpt2, cfg = load_gpt2("124M")`

**Objetivo:** ver como o GPT-2 **original** responde, para comparar depois do treino.

**Linha a linha:**

1. `gpt2, cfg = load_gpt2("124M")`: carrega o GPT-2 pré-treinado (`gpt2`) e as configurações dele (`cfg`, ex.: o contexto máximo de 1024 tokens).
2. `gpt2.to(device)`: coloca o modelo na GPU/MPS (ou CPU), onde as contas vão rodar.
3. `def responder(model, entrada, max_new_tokens=60):` define a função que faz uma pergunta e devolve só a resposta:
    - `format_input(entrada)`: monta o prompt no molde Alpaca, **sem** a resposta (é o modelo que tem que escrever);
    - `text_to_token_ids(prompt, bpe)`: transforma o texto em IDs de tokens;
    - `generate(...)`: o modelo escreve um token de cada vez, sempre o mais provável, até `max_new_tokens=60` tokens; `eos_id=50256` faz ele **parar** se escrever `<|endoftext|>`;
    - `token_ids_to_text(ids, bpe)`: transforma os IDs de volta em texto;
    - `[len(prompt):]`: corta o prompt do começo, sobrando só o que o modelo escreveu;
    - `.replace("### Response:", "").strip()`: tira o marcador e os espaços das pontas.
4. `for e in teste[:3]:` faz as 3 primeiras perguntas do teste; o `!r` mostra o texto com os `\n`, para vermos as quebras de linha.

**Resultado:** o GPT-2 original não responde:

- imita o formato e inventa seções ("### Output:", "### Error:", "### Description:");
- repete a mesma frase várias vezes;
- não para: usa os 60 tokens inteiros.

**E se trocar?** (testado no modelo já treinado)

- `eos_id=None` → ele não para: depois de "Paris." inventa outra pergunta ("capital of Canada?") e responde ("Ottawa").
- `max_new_tokens=5` → a resposta sai vazia, porque os primeiros tokens que ele escreve são o próprio "### Response:".

**Conceitos:**

- **`<|endoftext|>` (50256):** token de "fim de texto"; quando o modelo o escreve, a resposta acabou.

**Perguntas possíveis:**

- *Por que o prompt não tem a resposta?* Porque é exatamente o que o modelo tem que escrever.
- *Por que `max_new_tokens=60`?* É um limite de segurança: o máximo que o modelo pode escrever. As respostas do dataset são curtas (típica de 11 tokens, maior de 50; com os 6 tokens de "### Response:", todas cabem em 60). O modelo treinado para antes, no `<|endoftext|>`; o limite só é atingido pelo GPT-2 original, que não sabe parar. Menor demais (5) → resposta vazia; maior → só mais texto repetido e mais tempo.
- *O que é o `[len(prompt):]`?* O `generate` devolve o prompt + o que foi escrito; cortando o tamanho do prompt, sobra só a resposta.

**Fala:** "Essa função `responder` monta o pedido no molde, deixa o modelo escrever até 60 tokens, para se ele escrever o código de fim, e devolve só a resposta. Olhem o GPT-2 original: ele percebe o formato e inventa cabeçalhos, 'Output', 'Error', 'Description'; repete a mesma frase e não para de falar. Ele é bom em continuar texto, mas não sabe que deveria responder."

## 3. Fine-tuning (ou checkpoint)

*Bloco que começa com* `ckpt = os.path.join(`

**Objetivo:** ajustar **todos** os pesos do GPT-2 com os exemplos de instrução. Ao vivo, só carrega o modelo já ajustado.

**Linha a linha:**

1. `ckpt = os.path.join(CHECKPOINTS, "assistente_124M.pth")`: o caminho do modelo já treinado.
2. `if CARREGAR_CHECKPOINT and os.path.exists(ckpt):` se escolhemos usar o salvo e o arquivo existe…
    - `torch.load(ckpt, map_location=device, weights_only=True)` lê os pesos do arquivo (`map_location` = para qual dispositivo; `weights_only` = só números, por segurança);
    - `gpt2.load_state_dict(...)` coloca esses pesos dentro do modelo. **É o que roda ao vivo.**
3. `else:` treina agora (~1 min):
    - `torch.manual_seed(123)` e `t0 = time.time()`: resultado reproduzível e hora de início;
    - `collate = partial(custom_collate_fn, ...)`: a regra de montar lotes. Os exemplos têm tamanhos diferentes, então ela completa os mais curtos com 50256 (`<|endoftext|>`) até todos ficarem iguais, e cria o alvo (a entrada deslocada 1 posição). No alvo, o **primeiro** 50256 fica (ensina o modelo a parar) e os outros viram **−100** (não contam como erro);
    - `train_loader = DataLoader(InstructionDataset(treino, bpe), batch_size=8, ...)`: transforma os 400 exemplos em texto no molde Alpaca, depois em IDs, e agrupa em **50 lotes de 8**, embaralhados (`shuffle=True`); `drop_last=True` descarta um lote final incompleto;
    - `val_loader = DataLoader(InstructionDataset(val, bpe), ...)`: o mesmo com os 55 de validação (7 lotes), usados só para medir o erro durante o treino;
    - `torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)`: o *otimizador*, que ajusta **todos** os pesos a cada erro; `lr` é o tamanho de cada ajuste (bem pequeno, para não estragar o que o modelo já sabe); `weight_decay` segura os pesos para não crescerem demais;
    - `train_model_simple(...)`: o loop do pré-treino: 2 passadas pelos exemplos (`num_epochs=2`), medindo o erro a cada 10 passos (`eval_freq=10`) e mostrando uma resposta de exemplo (`start_context`) no fim de cada passada;
    - `torch.save(... bfloat16 ...)` e `os.replace(...)`: salva o modelo em meia precisão (~330 MB) num arquivo temporário e só renomeia no fim, para não deixar arquivo quebrado.

**Resultado:** `checkpoint carregado: .../assistente_124M.pth`. Os números do treino estão no texto da seção: **loss 3,02 → 0,53 em 100 passos (~1 min)**.

**E se trocar?** `CARREGAR_CHECKPOINT = False` treina ao vivo (~1 min no M1 Pro) e **sobrescreve** o `assistente_124M.pth` com o modelo novo.

**Conceitos:**

- **Lote (batch):** pacote de exemplos processados juntos (aqui, 8).
- **Padding / −100:** preenchimento para igualar o tamanho dos exemplos / código que manda a loss ignorar aquela posição.
- **Loss:** a medida do erro do modelo; o treino tenta diminuí-la (aqui: errar menos o próximo token das respostas).
- **Época:** uma passada por todos os exemplos de treino.
- **Otimizador (AdamW) / lr:** o algoritmo que ajusta os pesos / o tamanho de cada ajuste (learning rate).
- **bfloat16:** formato de número com metade da precisão, usado para salvar o arquivo menor.

**Perguntas possíveis:**

- *O que muda em relação ao pré-treino?* Só os dados. O loop, a loss (prever o próximo token) e o modelo são os mesmos.
- *O que significa "loss 3,02 → 0,53 em 100 passos"?* A loss é o erro médio ao prever o próximo token da resposta; e^(−loss) dá, mais ou menos, a chance que o modelo dá para o token certo: loss 3,02 ≈ 5% (antes) e 0,53 ≈ 59% (depois). Um **passo** é uma rodada de ajuste com 1 lote de 8 exemplos: 400 ÷ 8 = 50 lotes por época × 2 épocas = 100 passos. A loss de validação (exemplos que ele não treina) termina maior, em 0,86: com poucos exemplos, ele começa a decorar um pouco.
- *Por que o primeiro 50256 fica no alvo e os outros viram −100?* O primeiro ensina o modelo a escrever "fim de texto" depois da resposta, ou seja, a **parar**. Os outros são só preenchimento; se contassem, o modelo seria cobrado por prever enchimento.
- *Por que o `lr` é tão pequeno?* Para ajustar o modelo sem apagar o que ele aprendeu no pré-treino.

**Fala:** "Aqui é o treino. Como deixei o modelo salvo, ele só carrega os pesos já ajustados. Mas se treinasse, seria o mesmo loop do pré-treino: os exemplos são agrupados em lotes de 8, completando os mais curtos com o código de fim de texto; o otimizador ajusta todos os pesos, com passos bem pequenos para não estragar o que o modelo já sabe, em duas passadas pelos exemplos. O erro caiu de 3 para 0,5 em um minuto. Não tem algoritmo novo; só mudaram os dados."

## 4. Depois do fine-tuning

### 4.1 Respostas do teste

*Bloco que começa com* `gpt2.eval()`

**Objetivo:** comparar com a seção 2 (antes) e com a resposta certa.

**Linha a linha:**

1. `gpt2.eval()`: coloca o modelo em modo de uso (desliga partes que só servem no treino, como o dropout).
2. `torch.manual_seed(123)`: reprodutibilidade (aqui a escrita já é sempre igual, porque ele escolhe o token mais provável).
3. `for e in teste[:6]:` faz 6 perguntas do teste e mostra o `esperado` (a resposta certa) ao lado do `modelo`.

**Resultado:**

- ✅ **Formato:** respostas curtas, diretas, sem cabeçalhos inventados, e ele para.
- ❌ **Fatos:** "Pride and Prejudice" → **William Shakespeare** (certo: Jane Austen); cloro → **CH3** (certo: Cl).
- ❌ **Transformar frases:** símile, pontuação e reescrita saem iguais à entrada ou erradas.

**Perguntas possíveis:**

- *Por que ele erra fatos?* O fine-tuning ensina o formato; o conhecimento vem do pré-treino e do tamanho do modelo (ver 4.2).

**Fala:** "Agora as mesmas perguntas depois do treino, com a resposta certa ao lado. A diferença salta aos olhos: ele responde curto, direto, sem inventar cabeçalho, e para. Aprendeu a se comportar como assistente. Mas olhem o conteúdo: diz que Shakespeare escreveu Orgulho e Preconceito, e que o símbolo do cloro é CH3. Responde com confiança, no formato certo, e errado."

### 4.2 Por que ele erra (texto, antes do título da seção 5) *(pergunta 1)*

**Fala:** "Isso responde a primeira pergunta do notebook: o fine-tuning ensina **comportamento** (o formato, onde começar, quando parar), não **conhecimento**. O conhecimento vem do pré-treino e do tamanho do modelo; o livro usa o GPT-2 de 355 milhões e todos os exemplos, e melhora bastante."

## 5. Onde conseguir dados? Como construir do zero? *(pergunta 2)*

### 5.1 Dados prontos (texto)

**Fala:** "E onde conseguir dados? Tem prontos: Alpaca, Dolly, OpenAssistant e milhares no Hugging Face, inclusive traduções para o português."

### 5.2 Do zero: nosso próprio dataset

*Bloco que começa com* `meus_dados = [`

**Objetivo:** mostrar que criar dados de fine-tuning é só escrever exemplos no mesmo formato.

**Linha a linha:**

1. `meus_dados = [ {...}, {...}, {...} ]`: 3 exemplos em português, com os mesmos campos `instruction` / `input` / `output`. O terceiro tem `input` vazio.
2. `open(..., "w", encoding="utf-8")` + `json.dump(meus_dados, f, ensure_ascii=False, indent=2)`: salva os exemplos num arquivo JSON, mantendo os acentos.
3. `format_input(meus_dados[2])`: mostra o prompt do exemplo sem input; a parte `### Input:` **some**.
4. `len(InstructionDataset(meus_dados, bpe))`: prova que os exemplos já entram no preparo de dados do treino → **3 exemplos prontos**.

**Fala:** "Para montar o seu, é só uma lista com instrução, input e resposta. Escrevi três exemplos em português e salvei num arquivo. Quando o input está vazio, o molde simplesmente não coloca essa parte. E esses exemplos já entram no mesmo preparo do treino, sem mudar nada; só precisaria de muito mais deles."

### 5.3 Fontes e construto (fechamento)

**Fala:** "De onde tirar exemplos? FAQ e atendimento da empresa, documentação, respostas de especialistas, ou gerar com um LLM maior e revisar. E qualidade vale mais que quantidade: o trabalho LIMA mostrou bons resultados com mil exemplos bem feitos, mas num modelo de 65 bilhões de parâmetros, que já sabia muita coisa. Resumindo: modelo pré-treinado, mais exemplos de instrução, mais o mesmo treino, dá um mini assistente. O ChatGPT ainda tem uma etapa de preferência, com RLHF ou DPO."

---

## Referências do código

**Arquivos da aula e do livro (neste repositório)**

| Referência no código | Onde fica | O que é |
|---|---|---|
| `from aula import ...` | `30-09/aula.py` | Utilidades da aula: `load_gpt2` (baixa/carrega os pesos do GPT-2), `device` (mps/cuda/cpu), caminhos `REPO_DIR` e `CHECKPOINTS`. Também coloca `pkg/` no caminho do Python (linha 13) |
| `llms_from_scratch.ch03.MultiHeadAttention` | `pkg/llms_from_scratch/ch03.py`, linha 98 | A atenção multi-head do livro (cap. 3) |
| `gpt2.trf_blocks`, `bloco.att`, `bloco.norm1` | `pkg/llms_from_scratch/ch04.py`, linhas 49 (`TransformerBlock`) e 82 (`GPTModel`) | O GPT do livro (cap. 4): blocos, atenção e normalização |
| `generate`, `text_to_token_ids`, `token_ids_to_text` | `pkg/llms_from_scratch/ch05.py`, linhas 19, 188 e 194 | Gerar texto e converter texto ↔ IDs (cap. 5) |
| `train_model_simple` | `pkg/llms_from_scratch/ch05.py`, linha 62 | O loop de treino do pré-treino (cap. 5) |
| `format_input`, `InstructionDataset`, `custom_collate_fn` | `pkg/llms_from_scratch/ch07.py`, linhas 57, 69 e 154 | Molde Alpaca, dataset de instruções e montagem dos lotes (cap. 7) |
| `instruction-data.json` | `ch07/01_main-chapter-code/` | Os 1.100 exemplos de instrução do livro |
| Qwen3 (02, seção 5.2) | `pkg/llms_from_scratch/qwen3.py`, linha 288 | Exemplo real de `out_proj` voltando ao tamanho do modelo (2048 → 1024) |
| Avaliação com outro LLM, DPO, geração de dados (textos do 06) | `ch07/03_model-evaluation`, `ch07/04_preference-tuning-with-dpo`, `ch07/05_dataset-generation` | Extras do cap. 7 |
| Pesos salvos | `30-09/checkpoints/` | `gpt2-small-124M.pth` (GPT-2 original) e `assistente_124M.pth` (após o fine-tuning) |

**Trabalhos e datasets citados**

| Nome | O que é |
|---|---|
| Bahdanau (2014) | Primeiro artigo a usar atenção (aditiva) em tradução automática |
| GPT-2 124M (OpenAI) | O modelo cujos pesos oficiais são carregados |
| Alpaca | 52 mil exemplos de instrução gerados por um LLM; dá nome ao molde do prompt |
| Dolly 15k | 15 mil exemplos escritos por pessoas, com licença comercial |
| OpenAssistant / FLAN | Conversas feitas por voluntários / grande coleção de tarefas com instruções |
| LIMA | Artigo que mostrou bons resultados com ~1.000 exemplos bem escolhidos (num modelo de 65B) |
| RLHF / DPO | Etapas de alinhamento por preferência, depois do fine-tuning de instruções |
