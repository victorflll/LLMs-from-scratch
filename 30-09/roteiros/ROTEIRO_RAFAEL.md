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

**Cálculo (4 passos):**

1. `W_q`, `W_k`, `W_v` transformam `x` em três vetores por token: **Q** (o que procura), **K** (o que oferece), **V** (o que entrega) → cada um 6 × 2.
2. `Q @ K.T` → *score* de cada par (6 × 6): *produto escalar* da query de um com a key do outro.
3. `softmax(scores / √d_k)` → *pesos*: porcentagens que somam 1 em cada linha.
4. `pesos @ V` → *contexto*: cada token vira a média dos values, ponderada pelos pesos (6 × 2).

No heatmap, todos os pesos ficam entre **0,16 e 0,18** (≈ 1/6): os pesos de `W_q`/`W_k` são aleatórios.

**E se trocar?**

- **Outra semente** (`manual_seed`): os números mudam um pouco (ex.: seed 42 → 0,13–0,19), mas continua quase uniforme.
- **`d_k` maior** (16, 64, 256): continua quase uniforme (0,14–0,20). O tamanho sozinho não cria padrão; falta treino.
- **`x` × 10** (vetores maiores): a atenção se concentra (pesos de 0,00 a 0,54). Vetores maiores dão scores maiores, e o softmax "escolhe" mais.

**Conceitos:**

- **Query / Key / Value:** três versões de cada token: o que procura / o que oferece / o que entrega.
- **Transposta (`.T`):** a tabela com linhas e colunas trocadas.
- **Multiplicação de matrizes (`@`):** combina cada linha de uma tabela com cada coluna da outra (multiplica e soma).
- **Produto escalar:** multiplica os números correspondentes de dois vetores e soma: mede o quanto "apontam para o mesmo lado".
- **Score:** resultado de query × key: o quanto um token combina com outro (antes do softmax).
- **Softmax:** transforma scores em porcentagens positivas que somam 1.
- **Pesos de atenção:** os scores depois do softmax: quanto cada token olha para cada outro.
- **Contexto:** o novo vetor de cada token: média dos values ponderada pelos pesos.

**Perguntas possíveis:**

- *O que significam o `@` e o `.T` em `Q @ K.T`?* `.T` troca linhas por colunas (K tem 1 linha por token, 6×2; K.T tem 1 coluna por token, 2×6). Precisa porque o `@` combina linha de uma tabela com coluna da outra; sem o `.T`, (6×2) @ (6×2) não encaixa e dá erro. O `@` multiplica e soma para cada par (ex.: journey × one = (−0,30 × −0,47) + (−0,03 × 0,17) ≈ 0,14). Resultado: score 6×6, um por par.
- *Por que o gráfico é todo azul?* Sem treino, os scores são parecidos, e o softmax dá ~17% para cada token.

**Fala:** "Esse bloco é a atenção inteira. Cada token ganha três vetores: a query, 'o que eu procuro'; a key, 'o que eu ofereço'; e o value, 'o que eu entrego'. `Q @ K.T` dá um score para cada par de tokens: o quanto um combina com o outro. O softmax transforma os scores de cada linha em porcentagens que somam 100%. E `pesos @ V` mistura os values usando essas porcentagens: se 'journey' dá 50% de atenção para 'starts', metade do novo 'journey' vem de 'starts'.
E esse é o nosso primeiro mapa de atenção: a linha é quem olha, a coluna é para quem. Tudo azul-claro, e os números confirmam: uns 17% para cada. Ninguém treinou nada ainda. Ou seja: sem treino, a atenção não sabe para onde olhar. No final, no GPT-2 treinado, vamos ver padrões de verdade."

## 2. Outras formas de calcular o score *(pergunta 1)*

### 2.1 As quatro fórmulas

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

- *Existem outras formas de calcular a atenção?* Sim: aditiva e cosseno (2.1). Os modelos novos mudam principalmente como Q, K e V são calculados (2.3).
- *Por que três gráficos são quase iguais?* Sem treino, as três fórmulas dão scores muito parecidos (0,15–0,18); só o cosseno, multiplicado por 10, concentra mais.

**Fala:** "Existem outras formas de calcular o score. A aditiva, anterior ao Transformer, usa uma mini rede neural; o cosseno só compara a direção dos vetores. Sem treino, as três primeiras espalham a atenção quase igual; o cosseno, que aqui foi multiplicado por 10, concentra mais. O gráfico não diz qual é melhor, porque nada foi treinado. O GPT usa o produto escalar porque é só multiplicação de matrizes, muito rápido."

### 2.2 Por que dividir por √d?

*Bloco que começa com* `for d in [2, 64, 768]:`

**Objetivo:** justificar a divisão por √d da seção 1.

**Cálculo:** 1 query e 10 keys aleatórias; compara o **maior peso** com e sem a divisão, para vetores de tamanho 2, 64 e 768.

- Sem escala: **0,45 → 0,95 → 1,00**.
- Com escala: **~0,25** em todos.

Um produto escalar soma `d` termos, então cresce com o tamanho do vetor; dividir por √d devolve os scores a uma escala parecida.

**E se trocar?** Sem a divisão, num vetor do tamanho do GPT-2 (768) um único token leva 100% da atenção. O softmax "trava" em 0 e 1 e o *gradiente* some: o modelo para de aprender para onde olhar.

**Conceitos:**

- **Escala √d:** divisão dos scores pela raiz do tamanho do vetor, para o softmax não "travar".
- **Gradiente:** indica como ajustar cada peso para reduzir o erro; se some, o modelo para de aprender.

**Fala:** "Por que dividir por raiz de d? Com vetores grandes, os scores ficam enormes, e um único token leva 100% da atenção. Aí o modelo trava e não aprende. Dividindo por raiz de d, a atenção continua distribuída, em uns 25%, qualquer que seja o tamanho."

### 2.3 Variações modernas (texto, antes do título da seção 3)

**Objetivo:** mostrar o que os modelos atuais mudam na atenção.

**Fala:** "Os modelos novos mexem mais em como Q, K e V são calculados do que na fórmula: o Llama compartilha K e V entre queries, o Mistral olha só uma janela de tokens próximos."

## 3. Máscara causal: o GPT não pode olhar o futuro

*Bloco que começa com* `n = len(tokens)`

**Objetivo:** impedir que um token olhe para os tokens **seguintes**, porque o GPT gera texto da esquerda para a direita.

**Cálculo:** `triu(..., diagonal=1)` marca o que está acima da diagonal (o futuro). `masked_fill(-inf)` coloca −∞ nesses scores, e o softmax transforma −∞ em **0**; o resto da linha é redistribuído para somar 1.

- "Your" = **1,00**;
- "journey" = **0,48 / 0,52**;
- "starts" = **0,32 / 0,34 / 0,34**;
- … "step" = ~0,17 em cada.

**E se trocar?** Sem a máscara, no treino o modelo veria a próxima palavra e só a copiaria: aprenderia a "colar", não a prever.

**Conceitos:**

- **Máscara causal:** bloqueia a atenção para os tokens seguintes (o futuro).

**Fala:** "O GPT escreve da esquerda para a direita, então não pode olhar o futuro, senão copiaria a resposta. Nas posições do futuro eu coloco menos infinito, e o softmax zera. Sem máscara, todo mundo vê todo mundo, uns 0,17 em cada. Com máscara vira um triângulo: 'Your' só vê a si mesmo, 100%; 'journey' divide quase meio a meio; e o último token é o único que vê a frase inteira. É por isso que o Elmo usa o último token no classificador de spam."

## 4. Multi-head: várias atenções em paralelo

### 4.1 `pesos_atencao`: multi-head por dentro

*Bloco que começa com* `# a MultiHeadAttention do livro só devolve o resultado final`

**Objetivo:** fazer **várias atenções em paralelo** (*heads*) e conseguir desenhar os pesos de cada uma.

**Cálculo:** a classe do livro não devolve os pesos, então `pesos_atencao` refaz a conta até o softmax.

- O `.view` corta o vetor de 768 números em **12 pedaços de 64** (*head_dim*), um por head.
- Cada head calcula seus próprios scores, máscara e softmax.
- Dentro da classe, as 12 saídas são **concatenadas** de volta em 768 e passam pela `out_proj`.

**Conceitos:**

- **Head / multi-head:** uma atenção independente / várias em paralelo, cada uma com um pedaço do vetor.
- **head_dim:** tamanho do pedaço de cada head (768 ÷ 12 = 64 no GPT-2).

**Fala:** "Multi-head é fazer várias atenções em paralelo. O `.view` corta o vetor de 768 números em 12 pedaços de 64, e cada pedaço faz sua própria atenção. No final, as 12 saídas são juntadas de novo em 768. Essa função só refaz a conta para a gente conseguir desenhar."

### 4.2 Mudando o número de heads *(pergunta 3)*

*Bloco que começa com* `# Mudando o número de heads`

**Objetivo:** ver o que muda (e o que não muda) com mais ou menos heads.

**Cálculo:** cria a atenção com 1, 2, 4, 12 e 24 heads e mede:

- ***parâmetros***: sempre **2.360.064**;
- **saída**: sempre **(1, 6, 768)**;
- **tamanho de cada head**: 768 ÷ heads (768, 384, 192, 64, 32).

**E se trocar?**

- Mais heads → mais padrões procurados ao mesmo tempo, mas cada head com menos números.
- Menos heads → cada head mais "rica", mas menos padrões.
- O custo em parâmetros é o mesmo; o efeito na qualidade só aparece treinando (não mostrado aqui).

**Conceitos:**

- **Parâmetros (pesos):** os números ajustáveis do modelo, ex.: as matrizes `W_query`, `W_key`, `W_value` e `out_proj`.

**Perguntas possíveis:**

- *E se mudar o número de heads?* Parâmetros e saída ficam iguais; só muda o tamanho de cada pedaço. O efeito na qualidade só aparece treinando.

**Fala:** "Mudar o número de heads não muda o número de parâmetros nem o tamanho da saída; muda só em quantos pedaços o vetor é cortado. Mais heads procuram mais padrões, mas cada uma com menos números. O GPT-2 usa sempre 64 por head."

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

**O que o notebook constrói:** um GPT-2 que responde instruções, usando o **mesmo treino do pré-treino** com outros dados (*fine-tuning*).

## Preparação (primeiro bloco, antes da seção 1)

*Bloco que começa com* `import json`

**Objetivo:** carregar as ferramentas. O treino é o `train_model_simple`, **o mesmo do pré-treino**.

**E se trocar?**

- `CARREGAR_CHECKPOINT = False` → treina ao vivo (~1 min no M1 Pro).
- `N_EXEMPLOS_TREINO` maior → treino mais lento. O livro usa ~935 exemplos e o modelo de 355M, com resultados melhores.

**Conceitos:**

- **Pré-treino:** treino inicial em muito texto para prever o próximo token (de onde vem o conhecimento).
- **Fine-tuning:** treino extra, com poucos exemplos, para uma tarefa ou comportamento.
- **Checkpoint:** arquivo com os pesos salvos de um modelo.

**Fala:** "A função de treino é a mesma do pré-treino. Não tem algoritmo novo. Uso só 400 exemplos para caber na aula."

## 1. Os dados: pares instrução → resposta

### 1.1 Carregando os exemplos

*Bloco que começa com* `with open(`

**Objetivo:** ver o formato dos exemplos.

**Cálculo:** 1.100 exemplos, cada um com instrução, input (opcional) e a resposta certa.

**Fala:** "Cada exemplo tem uma instrução, um input opcional e a resposta esperada."

### 1.2 Molde Alpaca

*Bloco que começa com* `print(format_input(dados[50])`

**Objetivo:** transformar cada exemplo num **texto** que o modelo consiga ler e imitar.

**Cálculo:** cabeçalho + `### Instruction:` + `### Input:` (se houver) + `### Response:` + resposta.

**Conceitos:**

- **Prompt / molde Alpaca:** texto de entrada / formato fixo Instruction–Input–Response.

**Fala:** "O modelo só lê texto, então cada exemplo vira um texto com esse molde. Ele aprende que depois de 'Response' vem a resposta, e que depois ela acaba."

### 1.3 Divisão treino / validação / teste

*Bloco que começa com* `n_tr, n_te =`

**Objetivo:** separar os dados para aprender (treino), acompanhar (validação) e comparar antes × depois (teste).

**Cálculo:** 85% / 10% / 5% → 935 / 110 / 55; o treino é cortado para 400.

## 2. Batches: padding e o `-100`

### 2.1 Preenchimento e −100

*Bloco que começa com* `# completa os exemplos de um lote`

**Objetivo:** montar lotes retangulares **sem** ensinar coisa errada ao modelo.

**Cálculo:**

- Completa os exemplos com 50256 (`<|endoftext|>`) até o mesmo tamanho.
- O alvo é a entrada deslocada 1 posição (prever o próximo token).
- No alvo, o **primeiro** 50256 fica (ensina a parar); os outros viram **−100**, que a *loss* ignora.

**E se trocar?** Sem o −100, o modelo seria cobrado por prever o preenchimento: gastaria esforço aprendendo a repetir 50256 em vez do texto útil.

**Conceitos:**

- **Padding:** preenchimento para igualar o tamanho dos exemplos de um lote.
- **`<|endoftext|>` (50256):** token de "fim de texto": sinaliza onde a resposta acaba.
- **Loss (cross-entropy):** medida do erro do modelo; o treino tenta diminuí-la.
- **−100:** código que manda a loss ignorar aquela posição.

**Fala:** "Os exemplos têm tamanhos diferentes, então completo com o código de fim de texto. No alvo, o primeiro código de fim fica: é ele que ensina o modelo a parar de falar. O resto vira -100, que quer dizer 'ignore, não conta como erro'."

### 2.2 Lotes

*Bloco que começa com* `torch.manual_seed(123)`

**Objetivo:** agrupar os exemplos em lotes de 8 para o treino: 400 ÷ 8 = 50 lotes por *época*.

**Conceitos:**

- **Lote (batch) / época:** pacote de exemplos processados juntos / uma passada por todos os exemplos.

## 3. Antes do fine-tuning

*Bloco que começa com* `gpt2, cfg = load_gpt2("124M")`

**Objetivo:** ter uma referência de **como o modelo responde antes** do treino.

**Cálculo:**

- `responder` monta o prompt, gera até 60 tokens e para no `<|endoftext|>` (`eos_id=50256`).
- O GPT-2 base inventa cabeçalhos ("### Error:"), se repete e não para.

**E se trocar?** (testado no modelo treinado)

- `eos_id=None` → ele não para: depois de "Paris." inventa outra pergunta ("capital of Canada?") e responde ("Ottawa").
- `max_new_tokens=5` → a resposta sai vazia, porque os primeiros tokens gerados são o próprio "### Response:".

**Fala:** "Antes do treino, ele imita o formato, inventa seções, não para de falar e não responde nada."

## 4. Fine-tuning (ou checkpoint)

*Bloco que começa com* `ckpt = os.path.join(`

**Objetivo:** ajustar **todos** os pesos do GPT-2 com os exemplos de instrução.

**Cálculo:**

- AdamW (*otimizador*) com passo pequeno (`lr = 5e-5`, para não estragar o pré-treino), 2 épocas.
- A loss é a mesma do pré-treino: prever o próximo token.
- Números do treino (no texto da seção 4): **loss 3,02 → 0,53 em ~1 min**. Ao vivo, só carrega o *checkpoint*.

**Conceitos:**

- **Otimizador (AdamW) / lr:** algoritmo que ajusta os pesos / tamanho de cada ajuste (learning rate).
- **bfloat16:** formato de número com metade da precisão, usado para salvar o checkpoint menor.

**Fala:** "Carrego o modelo salvo, mas os números do treino estão aqui: o erro caiu de 3 para 0,5 em um minuto. É o mesmo treino do pré-treino; só mudaram os dados."

## 5. Depois do fine-tuning

### 5.1 Respostas do teste

*Bloco que começa com* `gpt2.eval()`

**Objetivo:** comparar com a seção 3 (antes do fine-tuning) e com a resposta esperada.

**Cálculo:** mesmo `responder`, 6 exemplos de teste.

- Agora responde curto, no formato, e para.
- Mas erra fatos: "Pride and Prejudice" → **Shakespeare**; cloro → **CH3**.

**Perguntas possíveis:**

- *Por que ele erra fatos?* O fine-tuning ensina o formato; o conhecimento vem do pré-treino e do tamanho do modelo (ver 5.2 e 5.3).

**Fala:** "Agora ele se comporta como assistente: responde curto e para. Mas diz que Shakespeare escreveu Orgulho e Preconceito, e que o símbolo do cloro é CH3. Aprendeu o formato, não os fatos."

### 5.2 Instruções da turma

*Bloco que começa com* `# Instrução da turma`

**Objetivo:** separar o que vem do pré-treino do que vem do fine-tuning.

**Cálculo:** "capital da França?" → **Paris** ✅ · voz passiva → repete a frase ❌.

**E se trocar?** Testado com o checkpoint: tradução para espanhol, sinônimo de "happy" e conversão km → m também falham. Fatos muito comuns tendem a sair certos.

**Fala:** "Paris ele acerta, porque é um fato muito comum no texto do pré-treino. A voz passiva ele não faz. Alguém quer testar uma?"

### 5.3 Por que ele erra (texto, antes do título da seção 6) *(pergunta 1)*

**Fala:** "Isso responde a primeira pergunta: o fine-tuning ensina comportamento, não conhecimento. O conhecimento vem do pré-treino e do tamanho do modelo."

## 6. Onde conseguir dados? Como construir do zero? *(pergunta 2)*

### 6.1 Dados prontos (texto)

**Fala:** "E onde conseguir dados? Tem prontos: Alpaca, Dolly, OpenAssistant e milhares no Hugging Face, inclusive traduções para o português."

### 6.2 Do zero: nosso próprio dataset

*Bloco que começa com* `meus_dados = [`

**Objetivo:** mostrar que criar dados é só escrever no mesmo formato.

**Cálculo:**

- 3 exemplos em português, salvos em JSON.
- Com input vazio, a seção "### Input:" some.
- Os exemplos já entram no `InstructionDataset` do treino.

**Fala:** "Para montar o seu, é só uma lista com instrução, input e resposta. Já entra no mesmo pipeline, sem mudar nada."

### 6.3 Fontes e construto (fechamento)

**Fala:** "Qualidade vale mais que quantidade: o trabalho LIMA mostrou bons resultados com mil exemplos bem feitos, mas num modelo de 65 bilhões de parâmetros, que já sabia muita coisa. Resumindo: modelo pré-treinado, mais exemplos de instrução, mais o mesmo treino, dá um mini assistente. O ChatGPT ainda tem uma etapa de preferência, com RLHF ou DPO."

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
| GQA, MLA, SWA, DeltaNet (02, seção 2.3) | `ch04/04_gqa`, `ch04/05_mla`, `ch04/06_swa`, `ch04/08_deltanet` | Implementações das variações modernas de atenção |
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
