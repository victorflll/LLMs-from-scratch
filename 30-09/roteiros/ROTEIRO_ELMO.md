# ELMO — Material de estudo + roteiro de apresentação

Elmo apresenta **dois notebooks** da aula 30-09 (ver `30-09/README.md`):

| Ordem na aula | Notebook | O que sai dele |
|---|---|---|
| 1º bloco da aula | `01_tokenizacao.ipynb` | tokenizer + dataloader + embeddings → vão para o notebook 02 (Rafael) |
| 5º bloco da aula | `05_classificador.ipynb` | GPT-2 adaptado para classificar spam / não-spam |

> **Como usar este material:** cada ETAPA corresponde a uma célula de código do notebook, **na mesma ordem**.
> As células de texto (markdown) do notebook aparecem como "títulos de seção" — elas são o momento de fazer a transição.
> No final de cada notebook há o **roteiro contínuo**: só as falas, em ordem, para ensaiar.

> **Tudo que está em "Resultado" foi tirado da saída salva no próprio notebook.** Quando o resultado pode mudar ao rodar ao vivo, isso está avisado.

---

# NOTEBOOK 01 — `01_tokenizacao.ipynb`

## Mapa do notebook (para ter na cabeça)

```text
texto (the-verdict, 20.479 caracteres)
   │
   ├─ Seção 1: 3 jeitos de quebrar texto  → caractere / palavra / BPE (bpe)
   │            + tokenizadores novos      → gpt2 / cl100k_base / o200k_base
   ├─ Seção 2: meu próprio BPE do zero     → treinar_bpe → merges, meu_vocab, tamanhos
   ├─ Seção 3: pares (entrada, alvo)       → loader → x, y           shape (2, 4)
   ├─ Seção 4: embeddings                  → tok_emb, pos_emb → entrada  shape (2, 4, 768)
   └─ Seção 5: visualizar os pesos         → aleatório vs GPT-2 treinado, similaridade cosseno
                                                     │
                                                     ▼
                                   notebook 02 (Rafael): esses vetores entram na atenção
```

Cadeia de variáveis principal (a que conecta com o Rafael):

```text
texto ──bpe──► token ids ──create_dataloader_v1──► x (2,4) , y (2,4)
                                                    │
                                   tok_emb(x) + pos_emb(0..3)
                                                    ▼
                                            entrada (2, 4, 768)  ──► notebook 02
```

---

## Abertura (célula markdown 0)

O notebook abre com o título e 4 perguntas:
1. Como seria se trocássemos o BPE por outro tokenizador?
2. E se eu quiser construir **meu** vocabulário?
3. O que acontece se mudarmos a dimensão dos embeddings?
4. Conseguimos visualizar os pesos dentro dos vetores?

**Fala:**
"Oi, pessoal. Eu vou abrir a aula com a primeira peça de um GPT: como o texto vira número. Um modelo de linguagem não lê letras, ele só faz contas com números. Então antes de qualquer atenção, qualquer Transformer, a gente precisa responder: como eu pego uma frase e transformo em algo que o modelo consegue processar? Ao longo do notebook eu vou responder quatro perguntas rodando código: o que muda se a gente trocar o BPE, como construir nosso próprio vocabulário, o que acontece se mudarmos o tamanho do embedding, e se dá para enxergar os números dentro desses vetores."

---

## ETAPA 1 — Imports e preparação (célula 1)

### Código
```python
import re
from collections import Counter

import matplotlib.pyplot as plt
import tiktoken
import torch

from aula import the_verdict, load_gpt2
from llms_from_scratch.ch02 import create_dataloader_v1

torch.manual_seed(123)
texto = the_verdict()
frase = "Hello, class! Tokenização do GPT-2 é unbelievably interessante."
print(len(texto), "caracteres no livro de treino (the-verdict)")
```

### O que está acontecendo?
- `re` = expressões regulares, vai ser usada para quebrar texto em palavras.
- `Counter` = conta quantas vezes cada coisa aparece; vai ser usado no BPE feito do zero.
- `tiktoken` = biblioteca da OpenAI que tem os tokenizadores reais do GPT-2, GPT-4 e GPT-4o.
- `torch` = PyTorch, onde estão os embeddings.
- `the_verdict` e `load_gpt2` vêm do arquivo `aula.py` da pasta: o primeiro lê o conto *The Verdict* (texto usado no livro), o segundo baixa/carrega os pesos oficiais do GPT-2.
- `create_dataloader_v1` vem do código do capítulo 2 do livro (`pkg/llms_from_scratch/ch02.py`).
- `torch.manual_seed(123)` fixa a aleatoriedade: os embeddings aleatórios criados mais à frente vão sair sempre iguais.

### Variáveis
| Variável | O que é | Tipo | Usada depois em |
|---|---|---|---|
| `texto` | o conto *The Verdict* inteiro | `str` com 20.479 caracteres | ETAPA 2 (vocabulários), ETAPA 4 (treinar nosso BPE), ETAPA 6/7 (dataloader) |
| `frase` | frase de teste, misturando inglês e português, com acentos, "GPT-2" e uma palavra longa ("unbelievably") | `str` | ETAPA 2 e ETAPA 5 — é a frase que vamos tokenizar de vários jeitos |

### Por que fazemos isso?
Precisamos de duas coisas: um **corpus** (o texto de onde os vocabulários são aprendidos) e uma **frase de teste** propositalmente difícil, para ver onde cada tokenizador quebra.

### Resultado
```
20479 caracteres no livro de treino (the-verdict)
```
O conto tem ~20 mil caracteres. É um texto pequeno e **em inglês, sem acentos** — isso vai ser importante mais à frente.

### Fala
"Aqui é só a preparação. Eu carrego o conto *The Verdict*, que é o texto que o livro usa para treinar, com uns 20 mil caracteres, e crio uma frase de teste de propósito difícil: tem inglês, português com acento, 'GPT-2' e uma palavra comprida, 'unbelievably'. Vou usar essa mesma frase em todos os tokenizadores para a gente comparar. O `manual_seed` é só para os números aleatórios saírem sempre iguais, aí o resultado que eu mostro é o mesmo que vocês vão ver se rodarem."

---

## Seção 1 (markdown 2): "Três jeitos de quebrar texto"

**Transição:** "Primeira pergunta: existem vários jeitos de quebrar texto em pedaços, que a gente chama de tokens. Vou montar três e comparar."

---

## ETAPA 2 — Caractere vs Palavra vs BPE (célula 3)

### Código
```python
# Nível caractere: vocabulário = todo caractere que apareceu no texto
vocab_char = sorted(set(texto))
fora_char = [c for c in frase if c not in vocab_char]
print("vocab caractere:", len(vocab_char), "| tokens da frase:", len(frase), "| desconhecidos:", fora_char)

# Nível palavra (igual ao SimpleTokenizer do cap. 2)
def palavras(t):
    return [p for p in re.split(r'([,.:;?_!"()\']|--|\s)', t) if p.strip()]

vocab_pal = set(palavras(texto))
tokens_pal = palavras(frase)
print("vocab palavra:", len(vocab_pal), "| tokens da frase:", len(tokens_pal),
      "| desconhecidos:", [p for p in tokens_pal if p not in vocab_pal])

# BPE do GPT-2: nunca tem token desconhecido, quebra em pedaços (subpalavras / bytes)
bpe = tiktoken.get_encoding("gpt2")
ids = bpe.encode(frase)
print("vocab BPE:", bpe.n_vocab, "| tokens da frase:", len(ids))
print([bpe.decode([i]) for i in ids])
```

### O que está acontecendo?
São três tokenizadores, um depois do outro.

**1) Nível caractere**
- `set(texto)` pega cada caractere diferente que aparece no conto; `sorted` ordena. Isso **é** o vocabulário: cada caractere é um token.
- `fora_char` percorre (`for c in frase`) cada caractere da frase de teste e guarda os que **não** estão no vocabulário.
- Nesse nível, cada caractere da frase vira 1 token, por isso "tokens da frase" = `len(frase)`.

**2) Nível palavra**
- `palavras(t)` quebra o texto com `re.split` usando como separadores pontuação (`, . : ; ? _ ! " ( ) '`), `--` e espaços. Os parênteses na regex fazem a pontuação também virar token. O `if p.strip()` joga fora pedaços vazios/só espaço.
- `vocab_pal` = conjunto de palavras distintas do conto.
- `tokens_pal` = a frase de teste quebrada em palavras.
- A lista impressa no final mostra quais palavras da frase **não existem** no vocabulário.

**3) BPE do GPT-2**
- `tiktoken.get_encoding("gpt2")` carrega o tokenizador **real** do GPT-2.
- `bpe.encode(frase)` transforma a frase em uma lista de **IDs** (inteiros).
- `bpe.decode([i])` faz o caminho inverso para cada ID, para a gente ver qual pedaço de texto cada ID representa.

### Variáveis
| Variável | O que é | Tipo | Usada depois em |
|---|---|---|---|
| `vocab_char` | lista ordenada dos caracteres do conto | `list[str]`, 62 itens | só aqui |
| `fora_char` | caracteres da frase que o vocabulário não conhece | `list[str]` | só aqui |
| `palavras` | função: texto → lista de palavras/pontuação | função | só aqui |
| `vocab_pal` | palavras distintas do conto | `set[str]`, 1.130 itens | só aqui |
| `tokens_pal` | a frase quebrada em palavras | `list[str]`, 11 itens | só aqui |
| **`bpe`** | tokenizador do GPT-2 | objeto `tiktoken.Encoding` | **ETAPA 7** (decodificar entrada/alvo) e **ETAPA 11/12** (IDs das palavras visualizadas) |
| `ids` | IDs da frase no BPE | `list[int]`, 20 itens | só aqui |

### Por que fazemos isso?
Para mostrar o **trade-off** de cada abordagem com a mesma frase: tamanho do vocabulário × quantidade de tokens × palavras desconhecidas.

### Resultado
```
vocab caractere: 62 | tokens da frase: 63 | desconhecidos: ['ç', 'ã', '2', 'é']
vocab palavra: 1130 | tokens da frase: 11 | desconhecidos: ['Hello', 'class', 'Tokenização', 'GPT-2', 'é', 'unbelievably', 'interessante']
vocab BPE: 50257 | tokens da frase: 20
['Hello', ',', ' class', '!', ' Token', 'iza', 'ç', 'ão', ' do', ' G', 'PT', '-', '2', ' é', ' unbelievably', ' int', 'e', 'ress', 'ante', '.']
```
Como ler:
- **Caractere:** vocabulário minúsculo (62), mas a frase vira 63 tokens (1 por caractere). E mesmo assim tem desconhecidos: `ç`, `ã`, `é` (o conto não tem acento) e até o `2` (o conto não tem esse dígito).
- **Palavra:** a frase fica curtinha (11 tokens), mas **7 das 11 palavras** são desconhecidas — o modelo simplesmente não teria como representá-las (no livro, elas virariam `<|unk|>`).
- **BPE:** vocabulário fixo de 50.257, 20 tokens e **nenhum desconhecido**. Palavras comuns ficam inteiras (`' class'`, `' unbelievably'`), palavras que ele não conhece bem são quebradas em pedaços (`' Token' 'iza' 'ç' 'ão'`, `' int' 'e' 'ress' 'ante'`).
- Repare que vários tokens começam com espaço (`' class'`): no GPT-2 o espaço faz parte do token.

### Fala
"Aqui eu faço a mesma frase passar por três tokenizadores. No primeiro, cada caractere é um token: o vocabulário fica pequenininho, 62 símbolos, mas a frase vira 63 tokens, um por caractere. E olha que curioso: mesmo assim tem caractere desconhecido, porque o conto está em inglês e não tem 'ç', 'ã', 'é', nem o número 2.
No segundo, cada palavra é um token. A frase fica com só 11 tokens, ótimo, mas 7 dessas 11 palavras o vocabulário nunca viu. Qualquer palavra nova vira 'desconhecido' e a informação se perde.
O terceiro é o BPE, o tokenizador de verdade do GPT-2. Ele tem 50.257 tokens, a frase vira 20 tokens e nenhum é desconhecido. Palavra comum, tipo ' class' ou ' unbelievably', fica inteira. Palavra que ele não conhece, tipo 'Tokenização', ele quebra em pedaços: ' Token', 'iza', 'ç', 'ão'. Esses números que o `encode` devolve são os IDs: cada ID só identifica um pedaço dentro do vocabulário. Ainda não é o vetor que o modelo usa, isso vem mais à frente."

### Possível pergunta do professor/aluno
**"E se mudássemos o BPE para outra abordagem?"**
Resposta: "Essa célula mostra justamente isso. Se fosse por caractere, nunca teria palavra nova, mas a sequência fica muito longa e o contexto do modelo enche rápido. Se fosse por palavra, a sequência fica curta, mas qualquer palavra fora do vocabulário se perde. O BPE fica no meio: vocabulário fixo e, no pior caso, ele cai para bytes, então nunca tem token desconhecido. Existem outros tokenizadores de subpalavra, como WordPiece (usado no BERT) e Unigram/SentencePiece (usado em modelos como T5 e Llama 2), mas eles não estão neste notebook — a ideia geral é a mesma: quebrar em pedaços menores que palavras."

---

## Markdown 4 — Leitura + "E tokenizadores mais novos?"

**Transição:** "Resumindo: caractere tem sequência longa, palavra tem desconhecido, BPE é o meio-termo. Mas o BPE do GPT-2 é de 2019. Os modelos mais novos usam o mesmo algoritmo com vocabulários maiores. Vamos ver o que muda."

---

## ETAPA 3 — Comparando gpt2, cl100k_base e o200k_base (célula 5)

### Código
```python
frases = {
    "inglês": "The quick brown fox jumps over the lazy dog.",
    "português": "A rápida raposa marrom pula sobre o cachorro preguiçoso.",
    "código": "def soma(a, b):\n    return a + b",
}
for nome in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(nome)
    contagem = {k: len(enc.encode(v)) for k, v in frases.items()}
    print(f"{nome:12s} vocab={enc.n_vocab:>7,}  tokens={contagem}")
```

### O que está acontecendo?
- `frases` é um dicionário com 3 textos: inglês, português e código Python (com `\n` e 4 espaços de indentação).
- O `for` percorre os **nomes de 3 tokenizadores reais**: `gpt2` (GPT-2), `cl100k_base` (GPT-4 / GPT-3.5) e `o200k_base` (GPT-4o).
- Para cada tokenizador, `contagem` conta quantos tokens cada frase gerou.

### Variáveis
| Variável | O que é | Tipo |
|---|---|---|
| `frases` | nome → texto | `dict[str, str]` |
| `nome` | nome do tokenizador da vez no `for` | `str` |
| `enc` | tokenizador da vez | `tiktoken.Encoding` |
| `contagem` | nome da frase → nº de tokens | `dict[str, int]` |

Nada daqui é usado depois; é uma célula de comparação.

### Resultado
```
gpt2         vocab= 50,257  tokens={'inglês': 10, 'português': 23, 'código': 16}
cl100k_base  vocab=100,277  tokens={'inglês': 10, 'português': 18, 'código': 11}
o200k_base   vocab=200,019  tokens={'inglês': 10, 'português': 15, 'código': 11}
```
- Inglês: 10 tokens nos três — o inglês já era bem coberto no GPT-2.
- Português: 23 → 18 → 15. Com vocabulário maior e treinado com mais idiomas, o português precisa de menos pedaços.
- Código: 16 → 11. O GPT-2 gasta tokens com os espaços da indentação; os mais novos têm tokens para isso.

### Fala
"Aqui eu comparo três tokenizadores de verdade: o do GPT-2, com 50 mil tokens, o do GPT-4, com 100 mil, e o do GPT-4o, com 200 mil. Em inglês não muda nada: 10 tokens nos três. Mas olhem o português: 23 no GPT-2, 18 no GPT-4, 15 no GPT-4o. E o código cai de 16 para 11. Ou seja, vocabulário maior, e treinado com mais línguas e código, significa menos tokens para dizer a mesma coisa. Na prática, cabe mais texto na janela de contexto e a geração fica mais barata."

---

## Markdown 6 — o custo de um vocabulário maior + Seção 2 "BPE do zero"

O markdown diz: vocabulário maior ⇒ menos tokens, **mas** a camada de embedding e a de saída ficam maiores (`vocab × emb_dim` parâmetros cada).

**Transição:** "Só que isso tem um preço: cada token do vocabulário vai ter uma linha na tabela de embeddings, então vocabulário maior é modelo maior — a gente vai ver esses números daqui a pouco. Agora, segunda pergunta: e se eu quiser construir o **meu** vocabulário? Dá para fazer um BPE do zero em umas 20 linhas. A ideia é: começo com os 256 bytes possíveis e, a cada passo, junto o par de vizinhos mais frequente num token novo."

---

## ETAPA 4 — BPE do zero: `treinar_bpe` e `codificar` (célula 7)

### Código
```python
def treinar_bpe(texto, n_merges):
    ids = list(texto.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges, tamanhos = {}, [len(ids)]
    for novo in range(256, 256 + n_merges):
        par = Counter(zip(ids, ids[1:])).most_common(1)[0][0]
        merges[par] = novo
        vocab[novo] = vocab[par[0]] + vocab[par[1]]
        saida, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == par:
                saida.append(novo)
                i += 2
            else:
                saida.append(ids[i])
                i += 1
        ids = saida
        tamanhos.append(len(ids))
    return merges, vocab, tamanhos


def codificar(texto, merges):
    ids = list(texto.encode("utf-8"))
    for par, novo in merges.items():  # aplica os merges na ordem em que foram aprendidos
        saida, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == par:
                saida.append(novo)
                i += 2
            else:
                saida.append(ids[i])
                i += 1
        ids = saida
    return ids


merges, meu_vocab, tamanhos = treinar_bpe(texto, n_merges=300)
print("primeiros tokens aprendidos:", [meu_vocab[i].decode("utf-8", "replace") for i in range(256, 276)])
print("últimos tokens aprendidos:  ", [meu_vocab[i].decode("utf-8", "replace") for i in range(536, 556)])
```

### O que está acontecendo?

**`treinar_bpe(texto, n_merges)`** — recebe o texto e quantas fusões fazer; devolve as regras aprendidas.

1. `texto.encode("utf-8")` transforma o texto em **bytes** (números de 0 a 255). `ids` começa como essa lista de bytes. Letra sem acento = 1 byte; `ç`, `ã`, `é` = 2 bytes cada.
2. `vocab` começa com 256 entradas: o ID `i` representa o byte `i`. É por isso que BPE **nunca** tem token desconhecido: qualquer texto pode ser escrito com esses 256 bytes.
3. `tamanhos` guarda quantos tokens o texto inteiro tem; começa com o tamanho original.
4. O `for novo in range(256, 256 + n_merges)` roda uma vez por fusão; `novo` é o ID do token que vai ser criado (256, 257, 258, ...).
   - `zip(ids, ids[1:])` gera todos os **pares de vizinhos** (1º com 2º, 2º com 3º, ...).
   - `Counter(...).most_common(1)[0][0]` pega o **par mais frequente**.
   - `merges[par] = novo` guarda a regra: "esse par vira o token `novo`".
   - `vocab[novo] = vocab[par[0]] + vocab[par[1]]` define que o token novo é a concatenação dos bytes dos dois.
   - O `while` percorre `ids` da esquerda para a direita: sempre que acha o par, coloca `novo` no lugar dos dois e pula 2 posições; senão copia o ID e anda 1.
   - `ids = saida` — o texto agora está mais curto; o tamanho vai para `tamanhos`.

**`codificar(texto, merges)`** — usa as regras aprendidas para tokenizar um texto novo: converte para bytes e aplica **cada merge, na ordem em que foi aprendido**, com o mesmo `while` de substituição.

### Variáveis
| Variável | O que é | Tipo | Usada depois em |
|---|---|---|---|
| `merges` | regras de fusão: `(id_a, id_b) → id_novo`, na ordem aprendida | `dict[tuple[int,int], int]`, 300 regras | ETAPA 5 (`codificar(frase, merges)`) |
| `meu_vocab` | ID → bytes que ele representa | `dict[int, bytes]`, 256 + 300 = 556 entradas | ETAPA 5 (mostrar tokens) |
| `tamanhos` | nº de tokens do conto depois de cada merge | `list[int]`, 301 valores | ETAPA 5 (gráfico) |

### Por que fazemos isso?
Para mostrar que o BPE não é mágica: é só **contar pares e juntar o mais frequente**, repetidas vezes. E para responder "e se quisermos construir nosso próprio vocabulário?" — é exatamente isto.

### Resultado
```
primeiros tokens aprendidos: ['e ', ' t', 'd ', 't ', 'in', 's ', 'he ', 'ha', ', ', 'ou', 'er', 'an', 'on', 'en', ' the ', 'y ', '. ', 'o ', 'ing', 'hi']
últimos tokens aprendidos:   ['room', 'when ', 'Stroud ', 'ough', 'Th', 'ed it', 'las', 'picture ', 'is ', 's, ', 'And ', 'say', 'ear', 'I f', 'al ', 'ous', 'fa', 'pre', 'il', 'lif']
```
- Primeiros merges: pedaços curtíssimos e muito comuns em inglês (`'e '`, `' t'`, `'in'`, `'ing'`) e já `' the '`.
- Últimos merges: pedaços maiores e específicos do conto, como `'Stroud '` (nome de um personagem) e `'picture '` (o conto é sobre um pintor).
- Repare que tokens como `'ed it'` e `'I f'` **atravessam o espaço**: nosso BPE simples não separa palavras antes. O GPT-2 real faz uma pré-tokenização com regex para evitar isso (o próprio notebook cita a versão completa em `ch02/05_bpe-from-scratch/`).

### Fala
"Agora a parte divertida: construir o nosso próprio vocabulário. O BPE começa com os 256 bytes possíveis: qualquer texto do mundo pode ser escrito com eles, por isso ele nunca tem token desconhecido. Aí, a cada rodada, ele olha todos os pares de vizinhos no texto, pega o par que mais aparece e cria um token novo para ele. Esse `Counter(zip(ids, ids[1:]))` é exatamente isso: conta os pares de vizinhos. E esse `while` só percorre o texto trocando o par pelo ID novo. Faço isso 300 vezes, então meu vocabulário fica com 256 mais 300, ou seja, 556 tokens.
Olhem o que ele aprendeu primeiro: 'e ', ' t', 'in', 'ing', ' the '. São os pedaços mais comuns do inglês. E os últimos já são coisas do conto: 'Stroud', que é um personagem, e 'picture', porque o conto é sobre um pintor. Ou seja: o vocabulário é um retrato do texto em que ele foi treinado."

### Possível pergunta do professor/aluno
**"E se quisermos construir nosso próprio vocabulário para usar num GPT?"**
Resposta: "Seria esse processo, com um corpus grande — por exemplo em português — e muito mais merges (o GPT-2 fez 50 mil). Mas tem um detalhe: os pesos do GPT-2 estão amarrados aos IDs do vocabulário dele. Se eu trocar o vocabulário, o ID 1000 passa a significar outra coisa, então a tabela de embeddings e a camada de saída mudam de tamanho (`vocab_size`) e o modelo precisaria ser treinado de novo."

---

## ETAPA 5 — Usando o nosso BPE na frase + gráfico (célula 8)

### Código
```python
meus_ids = codificar(frase, merges)
print(len(meus_ids), "tokens:", [meu_vocab[i].decode("utf-8", "replace") for i in meus_ids])
assert b"".join(meu_vocab[i] for i in meus_ids).decode("utf-8") == frase  # sem perda

plt.figure(figsize=(5, 3))
plt.plot(tamanhos)
plt.xlabel("nº de merges (tamanho do vocab − 256)")
plt.ylabel("tokens para o livro inteiro")
plt.title("Mais vocabulário ⇒ menos tokens")
plt.show()
```

### O que está acontecendo?
- `codificar(frase, merges)` tokeniza a **mesma frase de teste** com o vocabulário que acabamos de treinar.
- O `print` mostra cada token decodificado. `decode("utf-8", "replace")` troca bytes que sozinhos não formam um caractere válido por `�`.
- O `assert` junta os bytes de todos os tokens e confere que dá **exatamente** a frase original: a tokenização não perde informação.
- O gráfico plota `tamanhos`: quantos tokens o conto inteiro tem depois de cada merge.

### Variáveis
| Variável | O que é | Tipo |
|---|---|---|
| `meus_ids` | a frase tokenizada pelo nosso BPE | `list[int]`, 45 itens |

### Resultado
```
45 tokens: ['H', 'ell', 'o', ', ', 'c', 'las', 's', '! ', 'T', 'o', 'k', 'en', 'i', 'z', 'a', '�', '�', '�', '�', 'o ', 'd', 'o ', 'G', 'P', 'T', '-', '2', ' ', '�', '�', ' ', 'un', 'b', 'el', 'i', 'ev', 'ab', 'ly ', 'in', 'ter', 'es', 's', 'ant', 'e', '.']
```
+ gráfico: começa em ~20.500 tokens (0 merges = 1 token por byte) e cai rápido nos primeiros merges, depois mais devagar, chegando perto de ~9.000 tokens com 300 merges.

Como ler:
- **45 tokens** contra **20 do GPT-2**. Nosso vocabulário é pequeno (556) e treinado num único conto.
- Os `�` são os bytes soltos de `ç`, `ã` (2 bytes cada → 4 `�` em "Tokenização") e `é` (2 `�`). Como o conto não tem acentos, nosso BPE **nunca aprendeu a juntar esses bytes**. Mas a informação não se perdeu: o `assert` passou.
- O gráfico mostra a regra geral: **mais vocabulário ⇒ menos tokens**, com retorno decrescente.

### Fala
"Agora eu uso o meu BPE na mesma frase de antes. O GPT-2 tinha precisado de 20 tokens; o meu precisa de 45. E esses pontos de interrogação são os bytes do 'ç', do 'ã' e do 'é'. Como o conto não tem nenhum acento, meu BPE nunca aprendeu a juntar esses bytes. Mas reparem no `assert`: se eu juntar todos os bytes de volta, sai exatamente a frase original. Então funciona, só fica mais longo.
O gráfico mostra o texto inteiro: sem nenhum merge são uns 20 mil tokens, um por byte, e com 300 merges cai para uns 9 mil. Quanto mais vocabulário, menos tokens, só que cada merge novo ajuda menos que o anterior.
E isso é exatamente o que a gente viu antes com o GPT-2 comparado com o GPT-4o no português: o vocabulário reflete o corpus de treino."

---

## Markdown 9 — conclusão do BPE + Seção 3 "pares (entrada, alvo)"

**Transição:** "Bom, agora que o texto virou IDs, precisamos montar os exemplos de treino. O GPT aprende uma coisa só: prever o próximo token. Então o alvo é simplesmente a entrada deslocada uma posição para a frente. Daqui para frente eu volto a usar o BPE do GPT-2."

---

## ETAPA 6 — DataLoader: entrada `x` e alvo `y` (célula 10)

### Código
```python
loader = create_dataloader_v1(texto, batch_size=2, max_length=4, stride=4, shuffle=False)
x, y = next(iter(loader))
print("entrada:\n", x, "\nalvo:\n", y)
for a, b in zip(x[0], y[0]):
    print(f"{bpe.decode([a.item()])!r:>12} → {bpe.decode([b.item()])!r}")
```

### O que está acontecendo?
`create_dataloader_v1` é a função do capítulo 2 do livro (`pkg/llms_from_scratch/ch02.py`). Internamente ela:
1. tokeniza o `texto` inteiro com o BPE do GPT-2;
2. passa uma **janela deslizante** de tamanho `max_length` pelos IDs, andando `stride` posições por vez;
3. para cada janela guarda `entrada = ids[i : i+max_length]` e `alvo = ids[i+1 : i+max_length+1]` (a mesma janela, uma posição à frente);
4. agrupa os exemplos em lotes (batches) de `batch_size`.

Parâmetros usados aqui:
- `batch_size=2` → 2 exemplos por lote;
- `max_length=4` → cada exemplo tem 4 tokens (pequeno de propósito, para caber na tela);
- `stride=4` → a janela anda 4 posições, então as janelas não se sobrepõem;
- `shuffle=False` → não embaralha, para o primeiro lote ser o começo do conto.

`next(iter(loader))` pega **só o primeiro lote**. O `for` final percorre, lado a lado, os 4 tokens da primeira linha de `x` e de `y` e mostra o par "entrada → alvo" já decodificado.

### Variáveis
| Variável | O que é | Tipo / shape | Usada depois em |
|---|---|---|---|
| `loader` | DataLoader com todos os pares do conto | `torch.utils.data.DataLoader` | só aqui |
| **`x`** | IDs de entrada do primeiro lote | `torch.Tensor` inteiro, shape **(2, 4)** | **ETAPA 8** (`tok_emb(x)`) |
| `y` | IDs de alvo (o próximo token de cada posição) | `torch.Tensor`, shape (2, 4) | só aqui (no treino, é com ele que se calcula a loss) |

```text
x shape (2, 4)
         │  └── 4 = max_length: tokens por exemplo
         └───── 2 = batch_size: exemplos no lote
```

### Resultado
```
entrada:
 tensor([[  40,  367, 2885, 1464],
        [1807, 3619,  402,  271]])
alvo:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899]])
         'I' → ' H'
        ' H' → 'AD'
        'AD' → ' always'
   ' always' → ' thought'
```
- A linha 1 da entrada é o começo do conto ("I HAD always…"). `HAD` virou dois tokens: `' H'` + `'AD'`.
- **O alvo é a entrada deslocada 1 posição:** o alvo começa em 367, que é o 2º token da entrada.
- O último alvo da linha 1 (1807, `' thought'`) é o **primeiro token da linha 2** — porque `stride = max_length = 4`, as janelas ficam encostadas.
- Cada par mostrado é uma tarefa de previsão: vendo "I", preveja " H"; vendo "I H", preveja "AD"; e assim por diante. Com 4 posições, **um exemplo gera 4 previsões** (por causa da máscara causal que o Rafael vai mostrar, cada posição só enxerga as anteriores).

### Fala
"Aqui eu transformo o livro em exemplos de treino. O dataloader corta os IDs em janelas de 4 tokens e, para cada janela, o alvo é a mesma janela deslocada uma posição. Olhem a primeira linha: a entrada começa com 40, 367, 2885, 1464, e o alvo começa com 367. Ou seja, para cada posição o alvo é o token seguinte. Decodificando fica mais claro: vendo 'I', o modelo tem que prever ' H'; vendo ' H', prever 'AD'; e assim por diante. Isso é tudo que um GPT aprende no pré-treino: prever o próximo token.
Esse `x` tem shape 2 por 4: são 2 exemplos, cada um com 4 tokens. Guardem esse `x`, porque daqui a duas células ele vai virar os vetores de embedding."

---

## ETAPA 7 — 🔧 Experimento com `stride` (célula 11)

### Código
```python
# 🔧 stride: 1 = janelas sobrepostas (mais amostras, mais repetição); = max_length = sem sobreposição
for stride in [1, 2, 4]:
    print(f"stride={stride}: {len(create_dataloader_v1(texto, batch_size=1, max_length=4, stride=stride).dataset):>5} amostras")
```

### O que está acontecendo?
O `for` testa 3 valores de `stride`. Para cada um cria um dataloader e conta quantos exemplos (janelas) existem (`len(... .dataset)`).

### Resultado
```
stride=1:  5141 amostras
stride=2:  2571 amostras
stride=4:  1286 amostras
```
- `stride=1`: a janela anda 1 token por vez → uma janela começando em cada posição → ~5 mil exemplos, muito repetidos.
- `stride=4`: janelas encostadas, sem repetir tokens → ~4 vezes menos exemplos.
- Os números batem: 5141 / 2 ≈ 2571 e 5141 / 4 ≈ 1286.

### Fala
"Esse é um experimento rápido com o `stride`, que é o quanto a janela anda. Com stride 1, a janela anda um token por vez: tenho umas 5 mil amostras, mas elas se repetem muito. Com stride igual ao tamanho da janela, que é 4, não tem sobreposição nenhuma e fico com mais ou menos um quarto disso. Mais sobreposição dá mais exemplos, mas exemplos quase iguais, o que pode levar o modelo a decorar."

---

## Seção 4 (markdown 12): "Embeddings: cada id vira um vetor treinável"

**Transição:** "Até agora, cada token é só um número, um ID. Só que um ID não carrega significado nenhum: o 367 não é 'maior' nem 'mais parecido' com o 368. O modelo precisa de um vetor para cada token, algo que ele consiga ajustar durante o treino. É isso que o embedding faz."

---

## ETAPA 8 — Embedding de token + embedding de posição (célula 13)

### Código
```python
emb_dim, contexto = 768, 4
tok_emb = torch.nn.Embedding(bpe.n_vocab, emb_dim)
pos_emb = torch.nn.Embedding(contexto, emb_dim)
entrada = tok_emb(x) + pos_emb(torch.arange(contexto))
print("ids:", tuple(x.shape), "→ embeddings:", tuple(entrada.shape))
```

### O que está acontecendo?

**`torch.nn.Embedding(num_embeddings, embedding_dim)`** é uma **tabela** (uma matriz) de pesos treináveis:
- `num_embeddings` = quantas linhas → quantos itens diferentes existem;
- `embedding_dim` = quantas colunas → o tamanho do vetor de cada item.

Quando você passa um ID, ela devolve **a linha daquele ID**. É uma consulta (lookup), não uma conta. Os valores começam **aleatórios** e são ajustados no treino.

Aqui são duas tabelas:
1. `tok_emb = nn.Embedding(bpe.n_vocab, emb_dim)` → **50.257 linhas × 768 colunas**. Uma linha por token do vocabulário do GPT-2. `768` é o tamanho do vetor do GPT-2 small.
2. `pos_emb = nn.Embedding(contexto, emb_dim)` → **4 linhas × 768 colunas**. Uma linha por **posição** (0, 1, 2, 3). Necessária porque a atenção, sozinha, não sabe a ordem das palavras: sem isso, "o cachorro mordeu o homem" e "o homem mordeu o cachorro" teriam os mesmos vetores.

A linha principal:
```python
entrada = tok_emb(x) + pos_emb(torch.arange(contexto))
```
- `tok_emb(x)`: para cada um dos 2×4 IDs em `x`, busca a linha correspondente → shape (2, 4, 768).
- `torch.arange(contexto)` = `tensor([0, 1, 2, 3])` → as posições.
- `pos_emb(...)` → shape (4, 768): um vetor para cada posição.
- A soma usa *broadcasting*: o mesmo (4, 768) de posições é somado aos 2 exemplos do lote.

```text
x                      (2, 4)            IDs
   │ tok_emb (lookup na tabela 50257×768)
   ▼
tok_emb(x)             (2, 4, 768)       vetor de "qual token é"
                  +
pos_emb([0,1,2,3])        (4, 768)       vetor de "em que posição está"   (broadcast nos 2 exemplos)
   ▼
entrada                (2, 4, 768)
                        │  │  └── 768 = emb_dim: tamanho de cada vetor
                        │  └───── 4   = tokens por exemplo
                        └──────── 2   = exemplos no lote
```

### Variáveis
| Variável | O que é | Tipo / shape | Usada depois em |
|---|---|---|---|
| `emb_dim` | tamanho do vetor de cada token | `int` = 768 | aqui |
| `contexto` | nº de posições (igual ao `max_length` do dataloader) | `int` = 4 | aqui |
| **`tok_emb`** | tabela de embeddings de token, **aleatória** | `nn.Embedding`, pesos (50257, 768) | **ETAPA 10** (heatmap "aleatório") |
| `pos_emb` | tabela de embeddings de posição | `nn.Embedding`, pesos (4, 768) | só aqui |
| **`entrada`** | vetores que entram no modelo | `torch.Tensor` float, **(2, 4, 768)** | é exatamente o formato que o **notebook 02** recebe na atenção |

### Por que fazemos isso?
Porque o modelo precisa de **números contínuos e ajustáveis** para cada token, e precisa saber **a posição** de cada um. Esse tensor `(batch, tokens, emb_dim)` é o formato que passa por todo o GPT.

### Resultado
```
ids: (2, 4) → embeddings: (2, 4, 768)
```
Cada um dos 8 IDs virou um vetor de 768 números. Nada mais mudou de formato: continuam 2 exemplos de 4 tokens.

### Fala
"Aqui é o coração deste notebook. O `nn.Embedding` é basicamente uma tabela. Essa primeira tem 50.257 linhas, uma para cada token do vocabulário do GPT-2, e 768 colunas, que é o tamanho do vetor de cada token no GPT-2 small. Quando eu passo um ID, ela devolve a linha daquele ID. Não tem conta nenhuma, é só consulta. E esses números começam aleatórios: é o treino que vai ajustar.
Só que tem um problema: a atenção, que o Rafael vai mostrar, não sabe a ordem das palavras. Por isso existe uma segunda tabela, a de posição, com uma linha para cada posição. Aqui tenho 4 posições, então 4 linhas. Eu somo o vetor do token com o vetor da posição dele, e aí cada vetor carrega duas informações: qual é o token e onde ele está.
O resultado: meu `x`, que era 2 por 4, virou 2 por 4 por 768. Continuam 2 exemplos de 4 tokens, só que agora cada token é um vetor de 768 números. E é exatamente esse tensor que o Rafael vai receber para calcular a atenção."

---

## ETAPA 9 — 🔧 E se mudarmos `emb_dim`? (célula 14)

### Código
```python
# 🔧 E se mudarmos emb_dim? A tabela cresce linearmente (e o modelo inteiro ~quadraticamente, ver notebook 07)
for d in [4, 64, 768, 1024, 1600, 12288]:
    p = bpe.n_vocab * d
    print(f"emb_dim={d:>6}: {p:>13,} parâmetros só no tok_emb  ≈ {p * 4 / 1e6:8.1f} MB em float32")
```

### O que está acontecendo?
O `for` testa vários tamanhos de embedding. Para cada `d`:
- `p = bpe.n_vocab * d` → nº de parâmetros da tabela `tok_emb` (linhas × colunas = 50.257 × d);
- `p * 4 / 1e6` → memória em MB, porque cada número em `float32` ocupa 4 bytes.

### Resultado
```
emb_dim=     4:       201,028 parâmetros só no tok_emb  ≈      0.8 MB em float32
emb_dim=    64:     3,216,448 parâmetros só no tok_emb  ≈     12.9 MB em float32
emb_dim=   768:    38,597,376 parâmetros só no tok_emb  ≈    154.4 MB em float32
emb_dim=  1024:    51,463,168 parâmetros só no tok_emb  ≈    205.9 MB em float32
emb_dim=  1600:    80,411,200 parâmetros só no tok_emb  ≈    321.6 MB em float32
emb_dim= 12288:   617,558,016 parâmetros só no tok_emb  ≈   2470.2 MB em float32
```
- 768 = GPT-2 small: só a tabela de tokens já tem **38,6 milhões** de parâmetros (quase um terço dos 124M do modelo).
- 1024 = GPT-2 medium, 1600 = GPT-2 XL, 12288 = GPT-3 (dito no markdown seguinte).
- A tabela cresce **linearmente** com `d` (dobrou `d`, dobrou o tamanho). O comentário do código lembra que o modelo inteiro cresce ~quadraticamente, mas isso é assunto do notebook 07 (Cristiane).

### Fala
"Terceira pergunta: e se eu mudar o tamanho do embedding? Aqui eu só faço a conta: a tabela tem 50.257 linhas vezes `d` colunas. Com 768, que é o GPT-2 small, já são 38 milhões e meio de parâmetros só nessa tabela, uns 150 MB. No GPT-2 XL, com 1600, são 80 milhões. No GPT-3, com 12.288, são 617 milhões, quase 2,5 GB só para a tabela de tokens.
Vetor maior dá mais espaço para codificar significado, mas custa memória e precisa de mais dados para treinar. E, voltando ao que falei antes: se o vocabulário fosse de 200 mil, como no GPT-4o, essa tabela seria quatro vezes maior."

### Possível pergunta do professor/aluno
**"O que acontece se alterarmos a dimensão dos embeddings?"**
Resposta: "Muda o tamanho do vetor de cada token. Aumentar dá mais capacidade de representar nuances, mas a tabela cresce na mesma proporção, e como todas as camadas seguintes trabalham com vetores desse tamanho, o modelo inteiro fica maior e mais lento. Diminuir muito — tipo 4 — é barato, mas 4 números não conseguem separar 50 mil tokens de um jeito útil. E tem uma restrição que o Rafael vai mostrar: o `emb_dim` precisa ser divisível pelo número de heads da atenção."

---

## Markdown 15 + Seção 5 "Visualizando os pesos dentro dos vetores"

**Transição:** "Quarta pergunta: dá para ver o que tem dentro desses vetores? Vou comparar a nossa tabela aleatória, que acabou de ser criada, com a tabela do GPT-2 de verdade, já treinado pela OpenAI."

---

## ETAPA 10 — Heatmap: embedding aleatório vs GPT-2 treinado (célula 16)

### Código
```python
gpt2, _ = load_gpt2("124M")
palavras_viz = [" king", " queen", " man", " woman", " Paris", " France", " dog", " cat", " one", " two"]
ids_viz = [bpe.encode(p)[0] for p in palavras_viz]

fig, axs = plt.subplots(1, 2, figsize=(12, 3))
for ax, (nome, tabela) in zip(axs, [("aleatório", tok_emb.weight), ("GPT-2 treinado", gpt2.tok_emb.weight)]):
    im = ax.imshow(tabela[ids_viz, :64].detach(), aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(palavras_viz)), palavras_viz)
    ax.set_title(f"{nome}: 64 primeiras dimensões")
    fig.colorbar(im, ax=ax)
plt.show()
```

### O que está acontecendo?
- `load_gpt2("124M")` (de `aula.py`) baixa os pesos oficiais do GPT-2 small na primeira vez (~500 MB, salvos em `checkpoints/`) e devolve o modelo pronto. O `_` descarta a config, que não é usada aqui.
- `palavras_viz` são 10 palavras escolhidas **em pares relacionados**: rei/rainha, homem/mulher, Paris/França, cachorro/gato, um/dois. O espaço antes de cada palavra é proposital: é assim que elas aparecem no meio de uma frase no BPE do GPT-2.
- `ids_viz`: o ID de cada palavra. `[0]` pega o primeiro token (todas elas são 1 token só).
- O `for` percorre as 2 tabelas: `tok_emb.weight` (nossa tabela aleatória da ETAPA 8) e `gpt2.tok_emb.weight` (a tabela treinada).
- `tabela[ids_viz, :64]` → seleciona as 10 linhas dessas palavras e só as **64 primeiras colunas** (das 768), para caber na tela. Shape (10, 64).
- `.detach()` desliga o rastreamento de gradiente, necessário para o matplotlib conseguir desenhar.
- `imshow` com `coolwarm`: vermelho = valor positivo, azul = negativo.

### Variáveis
| Variável | O que é | Tipo / shape | Usada depois em |
|---|---|---|---|
| **`gpt2`** | GPT-2 124M com pesos oficiais | `GPTModel` (código do cap. 4) | ETAPA 11 |
| **`palavras_viz`** | 10 palavras em pares | `list[str]` | ETAPA 11 |
| **`ids_viz`** | IDs dessas palavras | `list[int]`, 10 itens | ETAPA 11 |
| `gpt2.tok_emb.weight` | a tabela de embeddings treinada | (50257, 768) | ETAPA 11 |

### Resultado (gráfico salvo)
- **Esquerda (aleatório):** padrão de ruído, sem estrutura, valores de ~−3 a +3 (a inicialização do `nn.Embedding` sorteia de uma normal padrão).
- **Direita (GPT-2 treinado):** valores bem menores (~−0,3 a +0,3) e aparecem **faixas verticais**: algumas dimensões (colunas) têm valor parecido para quase todas as palavras — por exemplo, colunas perto da 6 e da 36 ficam azul-escuro em todas as linhas.
- Olhando linha por linha, é difícil dizer "essa coluna é gênero" ou "essa coluna é país". Isso é o ponto da próxima célula.

### Fala
"Agora eu carrego o GPT-2 de verdade, com os pesos treinados pela OpenAI, e pego 10 palavras em pares: rei e rainha, homem e mulher, Paris e França, cachorro e gato, um e dois. Para cada palavra, eu pego a linha dela na tabela de embeddings e mostro só as 64 primeiras dimensões, das 768, porque senão não cabe.
Na esquerda é a nossa tabela aleatória: é só ruído. Na direita é o GPT-2 treinado: os valores são bem menores e aparecem umas faixas verticais, dimensões que são parecidas para quase todas as palavras. Mas tentem olhar para uma coluna e dizer 'essa aqui é gênero' ou 'essa é país'... não dá. Uma dimensão sozinha não tem um significado que a gente consiga ler. O significado aparece quando comparamos os vetores inteiros, e é isso que eu faço agora."

---

## ETAPA 11 — Similaridade cosseno entre as palavras (célula 17)

### Código
```python
# Cada dimensão isolada não "significa" nada — o significado aparece na geometria: vetores parecidos = palavras parecidas
E = gpt2.tok_emb.weight[ids_viz].detach()
sim = torch.nn.functional.cosine_similarity(E[:, None], E[None, :], dim=-1)
plt.figure(figsize=(6, 5))
plt.imshow(sim, cmap="viridis")
plt.xticks(range(len(palavras_viz)), palavras_viz, rotation=45)
plt.yticks(range(len(palavras_viz)), palavras_viz)
plt.colorbar(label="similaridade cosseno")
plt.title("GPT-2: pares relacionados ficam próximos")
plt.show()
```

### O que está acontecendo?
- `E` = os vetores **completos** (768 dimensões) das 10 palavras, tirados da tabela treinada → shape (10, 768).
- **Similaridade cosseno** mede o quanto dois vetores **apontam para a mesma direção**: 1 = mesma direção, 0 = sem relação (perpendiculares), −1 = opostos. Ela ignora o tamanho do vetor.
- O truque de shapes para comparar **todos contra todos** de uma vez:
```text
E[:, None]   (10, 1, 768)
E[None, :]   (1, 10, 768)
      ↓ broadcasting
     (10, 10, 768)   ← todos os pares (palavra i, palavra j)
      ↓ cosine_similarity(..., dim=-1)   compara ao longo das 768 dimensões
sim  (10, 10)        ← sim[i, j] = parecença entre palavra i e palavra j
```
- O `imshow` desenha essa matriz 10×10.

### Variáveis
| Variável | O que é | Shape |
|---|---|---|
| `E` | embeddings treinados das 10 palavras | (10, 768) |
| `sim` | matriz de similaridade | (10, 10) |

### Resultado (gráfico salvo)
- Diagonal amarela (= 1): cada palavra comparada com ela mesma.
- **Blocos 2×2 mais claros** ao lado da diagonal: king–queen, man–woman, Paris–France, dog–cat e one–two têm similaridade em torno de 0,55–0,65, enquanto pares não relacionados (ex.: Paris–dog) ficam por volta de 0,2–0,3.
- Também aparece uma similaridade intermediária entre {king, queen} e {man, woman} (~0,3–0,4).

### Por que fazemos isso?
Para mostrar **onde** está o significado: não em uma dimensão, mas na **geometria** — palavras usadas em contextos parecidos ganham vetores que apontam em direções parecidas. Ninguém programou isso: surgiu do treino de "prever o próximo token".

### Fala
"Aqui eu comparo os vetores inteiros, as 768 dimensões. A similaridade cosseno mede se dois vetores apontam para a mesma direção: 1 é a mesma direção, 0 é sem relação nenhuma. Esse `E[:, None]` com `E[None, :]` é só um truque para comparar todas as palavras com todas de uma vez, e sai uma matriz 10 por 10.
A diagonal é amarela porque é cada palavra com ela mesma. Mas olhem esses quadradinhos claros do lado da diagonal: rei com rainha, homem com mulher, Paris com França, cachorro com gato, um com dois. Os pares relacionados ficam perto; os que não têm nada a ver, tipo Paris com cachorro, ficam longe. Ninguém ensinou isso ao GPT-2: ele só aprendeu a prever a próxima palavra, e palavras que aparecem em contextos parecidos acabaram com vetores parecidos. Então, respondendo à pergunta: dá para ver os pesos, mas o significado não está numa coluna, está na geometria."

### Possível pergunta do professor/aluno
**"Conseguimos visualizar os pesos dentro dos vetores?"**
Resposta: "Sim, as duas últimas células fazem isso. A primeira mostra os números crus, e dá para ver a diferença entre aleatório e treinado, mas não dá para ler cada dimensão. A segunda mostra que a informação está nas relações entre vetores. Outro jeito comum, que não está no notebook, é reduzir os 768 números para 2 com PCA ou t-SNE e plotar as palavras num plano."

---

## Markdown 18 — ✅ Construto (fechamento)

`bpe` (tokenizer) → `create_dataloader_v1` (pares entrada/alvo) → `tok_emb + pos_emb` (tensor `batch × tokens × emb_dim`). No notebook 02 esses vetores entram na atenção.

**Fala de fechamento / passagem para o Rafael:**
"Então, recapitulando o que construímos: o BPE transforma texto em IDs, o dataloader monta os pares de entrada e alvo, onde o alvo é o próximo token, e os embeddings de token mais posição transformam cada ID num vetor de 768 números. Saímos daqui com um tensor no formato lote, por tokens, por 768. Só que, até agora, cada vetor só sabe dele mesmo: o vetor de 'it' não sabe se 'it' é o animal ou a rua. Fazer os tokens trocarem informação entre si é o trabalho da atenção, e é aí que o Rafael entra."

---

## Roteiro contínuo do Elmo — Notebook 01

> Só as falas, na ordem. Entre colchetes: em que célula você deve estar.

**[Título / célula 0]**
"Oi, pessoal. Eu vou abrir a aula com a primeira peça de um GPT: como o texto vira número. Um modelo de linguagem não lê letras, ele só faz contas com números. Então antes de qualquer atenção, qualquer Transformer, a gente precisa responder: como eu pego uma frase e transformo em algo que o modelo consegue processar? Ao longo do notebook eu vou responder quatro perguntas rodando código: o que muda se a gente trocar o BPE, como construir nosso próprio vocabulário, o que acontece se mudarmos o tamanho do embedding, e se dá para enxergar os números dentro desses vetores."

**[Célula 1 — imports]**
"Aqui é só a preparação. Eu carrego o conto *The Verdict*, que é o texto que o livro usa para treinar, com uns 20 mil caracteres, e crio uma frase de teste de propósito difícil: tem inglês, português com acento, 'GPT-2' e uma palavra comprida, 'unbelievably'. Vou usar essa mesma frase em todos os tokenizadores para a gente comparar. O `manual_seed` é só para os números aleatórios saírem sempre iguais."

**[Markdown 2]**
"Primeira pergunta: existem vários jeitos de quebrar texto em pedaços, que a gente chama de tokens. Vou montar três e comparar."

**[Célula 3 — caractere / palavra / BPE]**
"Aqui eu faço a mesma frase passar por três tokenizadores. No primeiro, cada caractere é um token: o vocabulário fica pequenininho, 62 símbolos, mas a frase vira 63 tokens, um por caractere. E mesmo assim tem caractere desconhecido, porque o conto está em inglês e não tem 'ç', 'ã', 'é', nem o número 2.
No segundo, cada palavra é um token. A frase fica com só 11 tokens, mas 7 dessas 11 palavras o vocabulário nunca viu. Qualquer palavra nova vira 'desconhecido' e a informação se perde.
O terceiro é o BPE, o tokenizador de verdade do GPT-2. Ele tem 50.257 tokens, a frase vira 20 tokens e nenhum é desconhecido. Palavra comum, tipo ' class' ou ' unbelievably', fica inteira. Palavra que ele não conhece, tipo 'Tokenização', ele quebra em pedaços. Esses números que o `encode` devolve são os IDs: cada ID só identifica um pedaço dentro do vocabulário. Ainda não é o vetor que o modelo usa, isso vem mais à frente."

**[Markdown 4]**
"Resumindo: caractere tem sequência longa, palavra tem desconhecido, BPE é o meio-termo. Mas o BPE do GPT-2 é de 2019. Os modelos mais novos usam o mesmo algoritmo com vocabulários maiores. Vamos ver o que muda."

**[Célula 5 — gpt2 / cl100k / o200k]**
"Aqui eu comparo três tokenizadores de verdade: o do GPT-2, com 50 mil tokens, o do GPT-4, com 100 mil, e o do GPT-4o, com 200 mil. Em inglês não muda nada: 10 tokens nos três. Mas olhem o português: 23, 18, 15. E o código cai de 16 para 11. Vocabulário maior, e treinado com mais línguas e código, significa menos tokens para dizer a mesma coisa: cabe mais texto no contexto e fica mais barato."

**[Markdown 6]**
"Só que isso tem um preço: cada token do vocabulário vai ter uma linha na tabela de embeddings, então vocabulário maior é modelo maior. Agora, segunda pergunta: e se eu quiser construir o meu vocabulário? Dá para fazer um BPE do zero em umas 20 linhas: começo com os 256 bytes e, a cada passo, junto o par de vizinhos mais frequente num token novo."

**[Célula 7 — treinar_bpe]**
"O BPE começa com os 256 bytes possíveis: qualquer texto do mundo pode ser escrito com eles, por isso ele nunca tem token desconhecido. A cada rodada, ele olha todos os pares de vizinhos, pega o que mais aparece e cria um token novo para ele. Esse `Counter(zip(ids, ids[1:]))` é exatamente isso: conta os pares de vizinhos. E esse `while` percorre o texto trocando o par pelo ID novo. Faço isso 300 vezes, então meu vocabulário fica com 556 tokens.
Olhem o que ele aprendeu primeiro: 'e ', ' t', 'in', 'ing', ' the '. São os pedaços mais comuns do inglês. E os últimos já são coisas do conto: 'Stroud', que é um personagem, e 'picture', porque o conto é sobre um pintor. O vocabulário é um retrato do texto em que ele foi treinado."

**[Célula 8 — meu BPE na frase + gráfico]**
"Agora eu uso o meu BPE na mesma frase. O GPT-2 tinha precisado de 20 tokens; o meu precisa de 45. Esses pontos de interrogação são os bytes do 'ç', do 'ã' e do 'é': o conto não tem acento, então meu BPE nunca aprendeu a juntar esses bytes. Mas o `assert` mostra que, se eu juntar tudo de volta, sai exatamente a frase original.
O gráfico mostra o texto inteiro: sem merge são uns 20 mil tokens, com 300 merges cai para uns 9 mil. Mais vocabulário, menos tokens, mas cada merge novo ajuda menos. É o mesmo que vimos com o GPT-2 contra o GPT-4o no português: o vocabulário reflete o corpus de treino."

**[Markdown 9]**
"Agora que o texto virou IDs, precisamos montar os exemplos de treino. O GPT aprende uma coisa só: prever o próximo token. Então o alvo é a entrada deslocada uma posição. Daqui para frente eu volto a usar o BPE do GPT-2."

**[Célula 10 — dataloader]**
"O dataloader corta os IDs em janelas de 4 tokens e, para cada janela, o alvo é a mesma janela deslocada uma posição. A entrada começa com 40, 367, 2885, 1464, e o alvo começa com 367. Decodificando: vendo 'I', prever ' H'; vendo ' H', prever 'AD'; e assim por diante. Isso é tudo que um GPT aprende no pré-treino.
Esse `x` tem shape 2 por 4: 2 exemplos, 4 tokens cada. Guardem esse `x`, porque daqui a duas células ele vai virar embedding."

**[Célula 11 — stride]**
"Um experimento rápido com o `stride`, que é o quanto a janela anda. Com stride 1, umas 5 mil amostras, mas muito repetidas. Com stride 4, sem sobreposição, fico com mais ou menos um quarto disso. Mais sobreposição dá mais exemplos, mas quase iguais."

**[Markdown 12]**
"Até agora cada token é só um ID. Um ID não carrega significado: o 367 não é 'mais parecido' com o 368. O modelo precisa de um vetor para cada token, que ele consiga ajustar no treino. É isso que o embedding faz."

**[Célula 13 — tok_emb + pos_emb]**
"O `nn.Embedding` é uma tabela. Essa primeira tem 50.257 linhas, uma para cada token do vocabulário, e 768 colunas, que é o tamanho do vetor no GPT-2 small. Eu passo um ID e ela devolve a linha daquele ID. É só consulta, e os números começam aleatórios; o treino ajusta.
Só que a atenção não sabe a ordem das palavras. Por isso existe uma segunda tabela, de posição, com uma linha por posição; aqui são 4. Eu somo o vetor do token com o vetor da posição, e cada vetor passa a carregar qual é o token e onde ele está.
Meu `x`, que era 2 por 4, virou 2 por 4 por 768: os mesmos 2 exemplos de 4 tokens, mas cada token agora é um vetor de 768 números. É esse tensor que o Rafael vai receber."

**[Célula 14 — emb_dim]**
"Terceira pergunta: e se eu mudar o tamanho do embedding? A tabela tem 50.257 linhas vezes `d` colunas. Com 768 já são 38 milhões e meio de parâmetros, uns 150 MB. No GPT-2 XL, com 1600, 80 milhões. No GPT-3, com 12.288, 617 milhões, quase 2,5 GB só nessa tabela. Vetor maior dá mais espaço para significado, mas custa memória e precisa de mais dados."

**[Markdown 15]**
"Quarta pergunta: dá para ver o que tem dentro desses vetores? Vou comparar a nossa tabela aleatória com a do GPT-2 treinado."

**[Célula 16 — heatmaps]**
"Carrego o GPT-2 de verdade e pego 10 palavras em pares: rei e rainha, homem e mulher, Paris e França, cachorro e gato, um e dois. Mostro a linha de cada uma, só as 64 primeiras dimensões. Na esquerda, a tabela aleatória: só ruído. Na direita, o GPT-2 treinado: valores menores e umas faixas verticais. Mas tentem dizer 'essa coluna é gênero'... não dá. Uma dimensão sozinha não tem significado legível. O significado aparece quando comparamos os vetores inteiros."

**[Célula 17 — similaridade cosseno]**
"A similaridade cosseno mede se dois vetores apontam para a mesma direção: 1 é a mesma direção, 0 é sem relação. Esse `E[:, None]` com `E[None, :]` é um truque para comparar todas com todas de uma vez, e sai uma matriz 10 por 10. A diagonal é cada palavra com ela mesma. E olhem os quadradinhos claros: rei com rainha, homem com mulher, Paris com França, cachorro com gato, um com dois. Ninguém ensinou isso: o GPT-2 só aprendeu a prever a próxima palavra, e palavras em contextos parecidos acabaram com vetores parecidos. O significado está na geometria."

**[Markdown 18 — construto / passagem]**
"Recapitulando: o BPE transforma texto em IDs, o dataloader monta os pares de entrada e alvo, e os embeddings de token mais posição transformam cada ID num vetor de 768 números. Saímos com um tensor lote, por tokens, por 768. Só que cada vetor ainda só sabe dele mesmo: o vetor de 'it' não sabe se 'it' é o animal ou a rua. Fazer os tokens trocarem informação é o trabalho da atenção, e é aí que o Rafael entra."

---
---

# NOTEBOOK 05 — `05_classificador.ipynb`

> ⚠️ **Antes de apresentar:** a célula 13 tem `CARREGAR_CHECKPOINT = True` e o arquivo `checkpoints/classificador_spam.pth` já existe na pasta. Então, **ao vivo, ela vai imprimir `checkpoint carregado: ...`** em vez do log de treino que está salvo no notebook (o log salvo é de quando alguém treinou; o caminho `/Users/victor/...` na célula 3 também vem dessa execução). O material abaixo explica o log salvo, porque é bom mostrá-lo ou comentá-lo.

## Mapa do notebook

```text
SMS Spam Collection (5.572 mensagens)
   │ balancear + mapear rótulos + dividir 70/10/20
   ▼
train.csv / validation.csv / test.csv
   │ SpamDataset (tokeniza + padding até 120)
   ▼
train_loader / val_loader / test_loader        lotes: x (8, 120), rótulo (8,)
   │
GPT-2 124M (gpt2) ── tentar via prompt? não funciona
   │ congelar tudo · out_head → Linear(768, 2) · liberar último bloco + final_norm
   ▼
saída (8, 120, 2) → usa só [:, -1, :] (último token)
   │ treinar (ou carregar checkpoint)
   ▼
~96% no teste → classify_review(mensagens da turma)
```

---

## Abertura (markdown 0)

**Entra:** GPT-2 124M pré-treinado. **Sai:** classificador spam / não-spam.
Ideia central: trocar a cabeça de saída (50.257 tokens → **2 classes**) e treinar pouca coisa.

**Fala:**
"Agora a gente já tem um GPT-2 pré-treinado, que a Lilian mostrou no notebook anterior. Ele sabe muito sobre a língua, mas só sabe fazer uma coisa: prever o próximo token. Neste notebook eu vou mostrar como reaproveitar tudo isso para uma tarefa diferente: dizer se um SMS é spam ou não. A ideia central é simples: em vez de a última camada escolher entre 50 mil tokens, ela vai escolher entre 2 classes. E a gente treina só um pedacinho do modelo."

---

## ETAPA 1 — Imports e parâmetros (célula 1)

### Código
```python
import os
import time
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from aula import DATA, CHECKPOINTS, device, load_gpt2, n_params
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.ch06 import (download_and_unzip_spam_data, create_balanced_dataset, random_split,
                                    SpamDataset, calc_accuracy_loader, train_classifier_simple, classify_review)

# 🔧 True = usa o classificador salvo (se existir) em vez de treinar ao vivo
CARREGAR_CHECKPOINT = True
N_EPOCAS = 5
bpe = tiktoken.get_encoding("gpt2")
```

### O que está acontecendo?
- De `aula.py`: `DATA` e `CHECKPOINTS` (caminhos das pastas `data/` e `checkpoints/`), `device` (usa `mps` no Mac, `cuda` numa GPU NVIDIA, senão `cpu`), `load_gpt2` e `n_params` (conta parâmetros; com `treinaveis=True` conta só os que vão ser treinados).
- Do capítulo 5 do livro: `generate` (gera texto token a token), `text_to_token_ids` / `token_ids_to_text` (texto ↔ tensor de IDs).
- Do capítulo 6 do livro: todas as funções de dados, treino e avaliação do classificador (cada uma explicada quando aparecer).
- `CARREGAR_CHECKPOINT = True`: se já existe um classificador salvo, carrega em vez de treinar (treinar leva ~2 min no M1 Pro).
- `N_EPOCAS = 5`: quantas vezes o treino passa pelo conjunto inteiro.
- `bpe`: o mesmo tokenizador do GPT-2 do notebook 01.

### Fala
"Aqui são os imports. Quase tudo vem do código do livro: do capítulo 5 vêm as funções de gerar texto, do capítulo 6 as funções do classificador. Tem essa flag `CARREGAR_CHECKPOINT`: como o treino leva uns dois minutos, deixei um modelo já treinado salvo; com `True` ele só carrega. E o tokenizador é o mesmo BPE do GPT-2 que vimos no começo da aula."

---

## Seção 1 (markdown 2): "Dados: SMS Spam Collection (UCI)"

**Transição:** "Primeiro, os dados. Vou usar um dataset público de SMS rotulados como spam ou não."

---

## ETAPA 2 — Baixar, balancear e dividir os dados (célula 3)

### Código
```python
tsv = Path(DATA) / "SMSSpamCollection.tsv"
download_and_unzip_spam_data("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                             Path(DATA) / "sms_spam_collection.zip", DATA, tsv)
df = pd.read_csv(tsv, sep="\t", header=None, names=["Label", "Text"])
print(df["Label"].value_counts(), "\n")

df = create_balanced_dataset(df)  # mesmo número de spam e ham
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
treino, val, teste = random_split(df, 0.7, 0.1)
for nome, parte in [("train", treino), ("validation", val), ("test", teste)]:
    parte.to_csv(Path(DATA) / f"{nome}.csv", index=None)
df.sample(5, random_state=1)
```

### O que está acontecendo?
1. `download_and_unzip_spam_data(...)`: se o `.tsv` já existe, não baixa de novo (é o que a saída mostra); senão baixa o zip da UCI e extrai.
2. `pd.read_csv(..., sep="\t")`: lê o arquivo separado por TAB em uma tabela (DataFrame) com colunas `Label` e `Text`.
3. `value_counts()`: conta quantas mensagens de cada classe.
4. `create_balanced_dataset(df)`: pega **todos** os spams e sorteia a mesma quantidade de "ham" (não-spam). Jogamos fora mensagens normais para as classes ficarem iguais.
5. `.map({"ham": 0, "spam": 1})`: troca os rótulos texto por números, porque o modelo vai prever a classe 0 ou 1.
6. `random_split(df, 0.7, 0.1)`: embaralha e divide em 70% treino, 10% validação e o resto (20%) teste.
7. O `for` salva cada parte em CSV (`train.csv`, `validation.csv`, `test.csv`) — a próxima célula lê esses arquivos.
8. `df.sample(5, random_state=1)`: mostra 5 linhas aleatórias só para visualizar.

### Variáveis
| Variável | O que é | Usada depois em |
|---|---|---|
| `tsv` | caminho do arquivo de dados | aqui |
| `df` | tabela com `Label` (0/1) e `Text` | aqui |
| `treino`, `val`, `teste` | as três partes da tabela | salvas em CSV → ETAPA 3 |

### Resultado
```
Label
ham     4825
spam     747
```
e uma amostra com rótulos já numéricos (ex.: "Save money on wedding lingerie at www.bridal.p..." → 1; "No pic. Please re-send." → 0).
- O dataset original é **desbalanceado**: 4.825 normais contra 747 spams (~13% spam). Um modelo que sempre dissesse "não é spam" já acertaria ~87%, sem aprender nada. Por isso balanceamos: ficam 747 + 747 = 1.494 mensagens.

### Fala
"Os dados são 5.572 SMS reais, rotulados. Olhem o problema: 4.825 mensagens normais, que o dataset chama de 'ham', e só 747 spams. Se eu treinar assim, o modelo pode aprender a dizer 'não é spam' sempre e já acerta 87%. Então eu balanceio: pego os 747 spams e sorteio 747 normais. Depois troco os rótulos por números, ham vira 0 e spam vira 1, e divido em 70% para treinar, 10% para validar durante o treino e 20% para o teste final, que o modelo nunca vê."

### Possível pergunta
**"Por que jogar fora mensagens em vez de usar todas?"**
"É a solução mais simples do livro para não enviesar o modelo. Alternativas seriam dar peso maior ao erro na classe rara na loss, ou medir com métricas como F1 em vez de só acurácia — mas não estão neste notebook."

---

## ETAPA 3 — Datasets e DataLoaders (célula 4)

### Código
```python
train_ds = SpamDataset(Path(DATA) / "train.csv", bpe)  # padding até a maior mensagem
val_ds = SpamDataset(Path(DATA) / "validation.csv", bpe, max_length=train_ds.max_length)
test_ds = SpamDataset(Path(DATA) / "test.csv", bpe, max_length=train_ds.max_length)
torch.manual_seed(123)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8)
test_loader = DataLoader(test_ds, batch_size=8)
print(len(train_ds), "treino |", len(val_ds), "validação |", len(test_ds), "teste | max_length =", train_ds.max_length)
```

### O que está acontecendo?
`SpamDataset` (do capítulo 6 do livro) recebe um CSV e o tokenizador e:
1. tokeniza cada mensagem com o BPE;
2. se `max_length` não foi passado (caso do treino), usa o tamanho da **maior** mensagem; se foi passado (validação e teste), **corta** mensagens maiores que isso;
3. completa (padding) todas as mensagens com o token `50256` (`<|endoftext|>`) até ficarem com o mesmo tamanho;
4. cada item devolvido é `(tensor de IDs, rótulo)`.

Validação e teste usam o **mesmo** `max_length` do treino, para todos terem o mesmo formato.

`DataLoader` agrupa em lotes de 8. No treino: `shuffle=True` (embaralha a cada época) e `drop_last=True` (descarta o último lote se ficar incompleto).

### Variáveis
| Variável | O que é | Shape de um lote | Usada depois em |
|---|---|---|---|
| **`train_ds`** | dataset de treino; guarda `max_length` | — | ETAPA 12 (`max_length=train_ds.max_length`) |
| `val_ds`, `test_ds` | datasets de validação e teste | — | — |
| **`train_loader`** | lotes de treino | x: **(8, 120)**, rótulos: **(8,)** | ETAPAS 7, 9, 10 |
| **`val_loader`**, **`test_loader`** | lotes de validação / teste | idem | ETAPAS 7, 9, 10 |

```text
um lote do train_loader
   x        (8, 120)   ← 8 mensagens, cada uma com 120 IDs (texto + padding 50256)
   rótulo   (8,)       ← 0 ou 1 para cada mensagem
```

### Resultado
```
1045 treino | 149 validação | 300 teste | max_length = 120
```
- 1.494 × 0,7 ≈ 1.045; × 0,1 ≈ 149; resto = 300.
- A maior mensagem de treino tem 120 tokens, então **toda mensagem vira uma sequência de 120 IDs**.

### Fala
"O `SpamDataset` tokeniza cada mensagem e resolve um problema: as mensagens têm tamanhos diferentes, e um lote precisa ser um bloco retangular. Então ele completa todas com o token 50256, o `<|endoftext|>` do GPT-2, até o tamanho da maior mensagem, que dá 120 tokens. Validação e teste usam esse mesmo tamanho. Aí o DataLoader junta de 8 em 8: cada lote é 8 mensagens por 120 tokens, mais 8 rótulos. Ficamos com 1.045 mensagens de treino, 149 de validação e 300 de teste."

---

## Seção 2 (markdown 5): "Dá para classificar só com o GPT-2 pré-treinado, via prompt?"

**Transição:** "Antes de treinar qualquer coisa, a pergunta óbvia: o GPT-2 já sabe muita coisa. Será que não dá para só perguntar para ele?"

---

## ETAPA 4 — Tentando classificar via prompt (célula 6)

### Código
```python
gpt2, cfg = load_gpt2("124M")
gpt2.to(device)
prompt = ("Is the following text 'spam'? Answer with 'yes' or 'no': "
          "'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'")
out = generate(gpt2, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=20, context_size=cfg["context_length"])
print(token_ids_to_text(out, bpe)[len(prompt):])
```

### O que está acontecendo?
- `load_gpt2("124M")` devolve o modelo com pesos oficiais (`gpt2`) e sua config (`cfg`, com `emb_dim=768`, `context_length=1024`...). `.to(device)` manda para a GPU/MPS.
- `prompt`: pergunta se uma mensagem claramente spam é spam, pedindo "yes" ou "no".
- `text_to_token_ids(prompt, bpe)`: texto → IDs, com shape (1, n_tokens) (o 1 é o lote).
- `generate(...)`: gera **20 tokens novos**, um por vez, sempre escolhendo o mais provável (sem temperatura).
- `token_ids_to_text(out, bpe)[len(prompt):]`: converte de volta para texto e corta a parte do prompt, sobrando só o que o modelo gerou.

### Variáveis
| Variável | O que é | Usada depois em |
|---|---|---|
| **`gpt2`** | o GPT-2 124M — **esse é o modelo que vamos adaptar** | ETAPAS 5 a 12 |
| **`cfg`** | config do modelo | ETAPA 5 (`cfg["emb_dim"]`) |

### Resultado
```


The following text 'spam'? Answer with 'yes' or 'no': 'You
```
Ele **não respondeu** "yes" nem "no": repetiu o padrão da pergunta, continuando o texto.

### Fala
"Então eu carrego o GPT-2 e pergunto: 'Esse texto é spam? Responda yes ou no', com uma mensagem que é spam óbvio, dinheiro grátis. E olhem a resposta: ele não respondeu nada, só repetiu o formato da pergunta. Isso é porque o modelo base só sabe continuar texto. Ele nunca foi ensinado a obedecer uma instrução; isso é o que o Rafael vai fazer no notebook 06. Então aqui a gente vai por outro caminho: mudar a arquitetura."

---

## Seção 3 (markdown 7): "Adaptando a arquitetura"

Plano: 1. congela tudo 2. troca `out_head` por `Linear(768, 2)` 3. libera o último bloco e o LayerNorm final.

**Transição:** "O plano tem três passos: congelar o modelo inteiro, trocar a camada de saída por uma com 2 saídas, e liberar para treino só o último bloco e a normalização final."

---

## ETAPA 5 — Congelar, trocar a cabeça, liberar o final (célula 8)

### Código
```python
for p in gpt2.parameters():
    p.requires_grad = False
torch.manual_seed(123)
gpt2.out_head = torch.nn.Linear(cfg["emb_dim"], 2).to(device)
for p in list(gpt2.trf_blocks[-1].parameters()) + list(gpt2.final_norm.parameters()):
    p.requires_grad = True
print(f"treináveis: {n_params(gpt2, treinaveis=True):,} de {n_params(gpt2):,}")
```

### O que está acontecendo?
1. **Congelar:** o primeiro `for` percorre **todos** os tensores de pesos do modelo e coloca `requires_grad = False`. Isso diz ao PyTorch "não calcule gradiente, não mude esses pesos no treino".
2. **Trocar a cabeça:** `gpt2.out_head` era `Linear(768, 50257)`: transformava o vetor final de cada token em 50.257 notas, uma por token do vocabulário. Agora vira `Linear(768, 2)`: 2 notas, uma para "não spam" (classe 0) e outra para "spam" (classe 1). Uma camada nova já nasce com `requires_grad=True` e pesos aleatórios (por isso o `manual_seed` antes).
3. **Liberar o final:** o segundo `for` percorre os parâmetros do **último bloco Transformer** (`trf_blocks[-1]`, o 12º) e do **LayerNorm final** (`final_norm`), e volta a permitir o treino deles.
4. `n_params` conta os parâmetros treináveis e o total.

```text
ANTES:  ... → bloco 12 → final_norm → out_head Linear(768 → 50257)  → "qual o próximo token?"
DEPOIS: ... → bloco 12 → final_norm → out_head Linear(768 → 2)      → "spam ou não?"
         ❄️ congelado    🔥 treina    🔥 treina                   🔥 nova
         (blocos 1–11, embeddings)
```

### Resultado
```
treináveis: 7,090,946 de 124,441,346
```
- Vamos treinar ~7 milhões de parâmetros, **cerca de 5,7%** do modelo.
- O total é 124,4 milhões porque a cabeça antiga (768 × 50.257 ≈ 38,6 milhões) foi substituída por uma de 768 × 2 + 2 = 1.538.

### Fala
"Primeiro eu congelo o modelo inteiro: `requires_grad = False` quer dizer 'não mexa nesses pesos'. Tudo que o GPT-2 aprendeu sobre a língua fica preservado. Depois eu troco a última camada. Ela recebia o vetor de 768 números e devolvia 50.257 notas, uma por token; agora devolve só 2: uma nota para 'não spam' e outra para 'spam'. E por fim eu descongelo o último bloco Transformer e a normalização final, para o modelo poder se ajustar um pouquinho à tarefa nova. Resultado: treino 7 milhões de parâmetros de 124 milhões, menos de 6%."

### Possível pergunta
**"Por que não treinar só a cabeça, ou o modelo inteiro?"** → a próxima célula responde com números.

---

## ETAPA 6 — 🔧 Quanto treinar? (célula 9)

### Código
```python
# 🔧 Quanto treinar? Contagem de parâmetros treináveis em cada estratégia
blocos = gpt2.trf_blocks
por_bloco = sum(p.numel() for p in blocos[0].parameters())
for nome, n in [("só a cabeça", 768 * 2 + 2), ("cabeça + último bloco", 768 * 2 + 2 + por_bloco),
                ("tudo (full fine-tuning)", n_params(gpt2))]:
    print(f"{nome:25s} {n:>12,}")
```

### O que está acontecendo?
- `por_bloco`: soma o número de elementos (`numel()`) de todos os pesos de **um** bloco Transformer (todos têm o mesmo tamanho).
- `768 * 2 + 2`: pesos (768×2) + bias (2) da cabeça nova.
- O `for` imprime as três estratégias. É só conta, não treina nada.

### Resultado
```
só a cabeça                      1,538
cabeça + último bloco        7,089,410
tudo (full fine-tuning)    124,441,346
```
- A diferença entre 7.089.410 (aqui) e 7.090.946 (célula anterior) é **1.536** = os parâmetros do `final_norm` (768 de escala + 768 de deslocamento), que a célula anterior também liberou e esta conta não inclui.

### Fala
"Aqui só comparo as opções. Treinar só a cabeça são 1.538 parâmetros: rapidíssimo, mas pouco flexível. Treinar tudo são 124 milhões: mais lento, mais memória, e com só mil exemplos o risco de estragar o que o modelo já sabia é maior. A gente ficou no meio: cabeça mais último bloco, uns 7 milhões. Essa diferença de 1.536 para a célula anterior é a normalização final, que eu também liberei."

---

## Markdown 10: "Usamos só o último token"

**Transição:** "Tem um detalhe importante. O GPT devolve uma saída para cada token da mensagem, mas eu só quero uma resposta por mensagem. Qual posição eu uso? A última, porque, pela máscara causal que o Rafael mostrou, é o único token que enxergou a mensagem inteira."

---

## ETAPA 7 — Shape da saída e acurácia antes do treino (célula 11)

### Código
```python
with torch.no_grad():
    print("saída do modelo:", tuple(gpt2(next(iter(train_loader))[0].to(device)).shape), "→ usamos [:, -1, :]")
torch.manual_seed(123)
print(f"acurácia ANTES do treino — val: {calc_accuracy_loader(val_loader, gpt2, device, num_batches=10):.0%}")
```

### O que está acontecendo?
- `next(iter(train_loader))[0]` pega o primeiro lote de treino e só as entradas (`[0]`), shape (8, 120).
- `gpt2(...)` roda o modelo. `torch.no_grad()` desliga o gradiente (só estamos olhando, não treinando).
- `calc_accuracy_loader` (cap. 6): para cada lote, roda o modelo, pega `[:, -1, :]` (a saída do **último token** de cada mensagem), escolhe a classe com maior nota (`argmax`) e compara com o rótulo. `num_batches=10` → só 10 lotes = 80 mensagens.

```text
entrada        (8, 120)          8 mensagens × 120 tokens
   ↓ gpt2
saída          (8, 120, 2)       uma nota por classe, para CADA token
   ↓ [:, -1, :]
último token   (8, 2)            uma nota por classe, por mensagem
   ↓ argmax
previsão       (8,)              0 ou 1
```

### Resultado
```
saída do modelo: (8, 120, 2) → usamos [:, -1, :]
acurácia ANTES do treino — val: 45%
```
- A última dimensão agora é **2** (antes seria 50.257).
- 45% é praticamente **chute** (50% seria aleatório com 2 classes balanceadas): a cabeça nova tem pesos aleatórios.

> Observação: com padding, o "último token" é quase sempre um `<|endoftext|>` de padding, não a última palavra real. Funciona porque o token de padding, pela máscara causal, enxerga a mensagem inteira. Isso é o que o livro faz; o notebook não discute isso explicitamente.

### Fala
"Olhem o formato da saída: 8 mensagens, 120 posições, e agora 2 números em vez de 50 mil. O modelo dá uma nota de spam para cada posição, mas eu uso só a última, com esse `[:, -1, :]`. Por quê? Por causa da máscara causal: cada token só enxerga os anteriores, então só o último enxergou a mensagem inteira.
E antes do treino, a acurácia na validação é 45%. Com duas classes balanceadas, isso é chute; faz sentido, a cabeça nova ainda está aleatória. Agora vamos treinar."

---

## Seção 4 (markdown 12): "Treino (ou checkpoint)"

---

## ETAPA 8 — Treinar ou carregar checkpoint (célula 13)

### Código
```python
ckpt = os.path.join(CHECKPOINTS, "classificador_spam.pth")
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    print("checkpoint carregado:", ckpt)
else:
    torch.manual_seed(123)
    t0 = time.time()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)
    train_classifier_simple(gpt2, train_loader, val_loader, optimizer, device,
                            num_epochs=N_EPOCAS, eval_freq=50, eval_iter=5)
    print(f"{(time.time() - t0) / 60:.1f} min")
    # salva só o que foi treinado (~28 MB); o resto é o GPT-2 original
    torch.save({n: p.detach() for n, p in gpt2.named_parameters() if p.requires_grad}, ckpt + ".part")
    os.replace(ckpt + ".part", ckpt)
```

### O que está acontecendo?
**Caminho 1 — checkpoint (o que vai acontecer ao vivo):**
- `torch.load` lê os pesos salvos; `load_state_dict(..., strict=False)` coloca esses pesos no modelo.
- `strict=False` é necessário porque o arquivo só tem os **pesos treinados** (último bloco, final_norm, cabeça nova); o resto continua sendo o GPT-2 original que já está carregado.

**Caminho 2 — treinar ao vivo:**
- `AdamW` é o otimizador: a regra que ajusta os pesos. `lr=5e-5` = passo pequeno (não queremos destruir o pré-treino); `weight_decay=0.1` = penaliza pesos grandes (ajuda a não decorar). Os pesos congelados não mudam porque não têm gradiente.
- `train_classifier_simple` (cap. 6), resumidamente, para cada época e para cada lote:
  1. zera os gradientes;
  2. calcula a **loss**: roda o modelo, pega `[:, -1, :]` e aplica `cross_entropy` com o rótulo (0/1);
  3. `loss.backward()` calcula os gradientes;
  4. `optimizer.step()` atualiza os pesos;
  5. a cada 50 passos (`eval_freq=50`) imprime a loss de treino e validação medida em 5 lotes (`eval_iter=5`);
  6. no fim de cada época imprime a acurácia (também em 5 lotes).
- `torch.save` salva **só** os parâmetros treináveis (~28 MB); `.part` + `os.replace` evita deixar um arquivo corrompido se o processo for interrompido.

### Resultado (log salvo no notebook — vem de uma execução com treino)
```
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
...
Training accuracy: 70.00% | Validation accuracy: 72.50%        ← fim da época 1
...
Training accuracy: 82.50% | Validation accuracy: 85.00%        ← época 2
Training accuracy: 90.00% | Validation accuracy: 90.00%        ← época 3
Training accuracy: 100.00% | Validation accuracy: 97.50%       ← época 4
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%       ← época 5
2.0 min
```
- A loss cai de ~2,2 para ~0,08: o modelo está errando cada vez menos.
- A loss de validação acompanha a de treino → não está decorando.
- 1.045 / 8 = 130 lotes por época × 5 épocas = 650 passos (o log vai até o passo 600).
- Essas acurácias por época usam só 5 lotes (40 mensagens), por isso são números "redondos" e oscilam. A medida de verdade é a próxima célula.

**Ao vivo, com o checkpoint, a saída será apenas:** `checkpoint carregado: .../checkpoints/classificador_spam.pth`.

### Fala
"Aqui tem duas opções. Como eu deixei o modelo treinado salvo, ele só carrega; o `strict=False` é porque eu salvei só a parte que foi treinada, uns 28 MB, e o resto é o GPT-2 original.
Mas vale mostrar o que acontece no treino, que está salvo aqui no log. O otimizador AdamW, com uma taxa de aprendizado bem pequena, para não estragar o que o modelo já sabia. O loop é o padrão: passa um lote, calcula o erro comparando a nota da última posição com o rótulo, calcula os gradientes e ajusta os pesos. A loss começa em 2,1 e termina em 0,08, e a de validação acompanha, então ele não está decorando. A acurácia sai de uns 70% na primeira época para 97% na quarta. Tudo isso em dois minutos num notebook."

---

## ETAPA 9 — Acurácia final em treino, validação e teste (célula 14)

### Código
```python
for nome, loader in [("treino", train_loader), ("validação", val_loader), ("teste", test_loader)]:
    print(f"acurácia {nome:10s}: {calc_accuracy_loader(loader, gpt2, device):.1%}")
```

### O que está acontecendo?
O `for` percorre os três loaders e mede a acurácia em **todos os lotes** (sem `num_batches`), com a mesma lógica: último token → `argmax` → compara com o rótulo.

### Resultado
```
acurácia treino    : 97.2%
acurácia validação : 97.3%
acurácia teste     : 95.7%
```
- 95,7% no **teste**, mensagens que o modelo nunca viu.
- Treino e teste muito próximos → **generaliza bem**, não decorou.
- Ao vivo, com o checkpoint, os números devem ser os mesmos ou muito próximos (é o mesmo modelo salvo).

### Fala
"Agora a medida de verdade, em todas as mensagens: 97% no treino, 97% na validação e quase 96% no teste, que são mensagens que o modelo nunca viu. E o treino e o teste estão bem próximos, então ele aprendeu o padrão de spam e não decorou os exemplos. Saímos de 45%, que era chute, para 96%, treinando menos de 6% do modelo."

---

## Seção 5 (markdown 15): "Testando com mensagens da turma ✍️"

**Transição:** "Agora a melhor parte: testar com mensagens que não estão no dataset. Se alguém quiser sugerir uma, eu coloco na lista."

---

## ETAPA 10 — `classify_review` com mensagens novas (célula 16)

### Código
```python
mensagens = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    "URGENT! Your account has been suspended. Click here to verify your password now.",
    "Can you send me the slides from class before Friday?",
]
for m in mensagens:
    print(f"{classify_review(m, gpt2, bpe, device, max_length=train_ds.max_length):9s} ← {m}")
```

### O que está acontecendo?
`classify_review` (cap. 6) faz, para uma mensagem só, o mesmo caminho do treino:
1. tokeniza com o BPE;
2. corta em `max_length` (120, vindo de `train_ds` da ETAPA 3) e completa com padding `50256` até 120;
3. transforma em tensor com shape (1, 120);
4. roda o modelo, pega `[:, -1, :]` → (1, 2), faz `argmax`;
5. devolve `"spam"` se a classe for 1, senão `"not spam"`.

O `for` percorre as 4 mensagens.

### Resultado
```
spam      ← You are a winner you have been specially selected to receive $1000 cash or a $2000 award.
not spam  ← Hey, just wanted to check if we're still on for dinner tonight? Let me know!
spam      ← URGENT! Your account has been suspended. Click here to verify your password now.
not spam  ← Can you send me the slides from class before Friday?
```
Acertou as 4. Repare que a primeira é **exatamente** a mensagem que o GPT-2 não conseguiu classificar via prompt na ETAPA 4.

> Dica para mensagens da turma: o modelo foi treinado com SMS **em inglês**. Mensagens em português podem dar resultados estranhos — isso não foi testado no notebook.

### Fala
"O `classify_review` faz para uma mensagem o mesmo que fizemos no treino: tokeniza, completa com padding até 120, roda o modelo, olha a última posição e escolhe a classe com a maior nota. Olhem a primeira: é exatamente a mensagem que o GPT-2 não conseguiu responder via prompt. Agora ele diz spam. O convite para jantar: não é spam. O 'URGENTE, sua conta foi suspensa, clique aqui': spam, e ela nem estava no dataset. E o pedido dos slides: não é spam. Alguém quer testar uma? Só lembrando que ele foi treinado com SMS em inglês."

---

## Markdown 17 — ✅ Construto (fechamento)

**Fala de fechamento:**
"Então, o que construímos: um GPT-2 com uma cabeça de 2 classes. O truque é reaproveitável: para classificar sentimento, tópico ou intenção, é só trocar o número de saídas da última camada. O modelo já sabe a língua; a gente só ensina a decisão final. Mas reparem que isso não transforma o GPT num assistente: ele não conversa, só classifica. Fazer ele seguir instruções é outro tipo de fine-tuning, e é o que o Rafael mostra agora."

---

## Roteiro contínuo do Elmo — Notebook 05

**[Título / markdown 0]**
"Agora a gente já tem um GPT-2 pré-treinado, que a Lilian mostrou. Ele sabe muito sobre a língua, mas só sabe prever o próximo token. Neste notebook vou reaproveitar tudo isso para outra tarefa: dizer se um SMS é spam ou não. Em vez de a última camada escolher entre 50 mil tokens, ela vai escolher entre 2 classes. E a gente treina só um pedacinho do modelo."

**[Célula 1 — imports]**
"Quase tudo vem do código do livro: do capítulo 5 as funções de gerar texto, do capítulo 6 as do classificador. Essa flag `CARREGAR_CHECKPOINT`: como o treino leva uns dois minutos, deixei um modelo já treinado salvo e ele só carrega. E o tokenizador é o mesmo BPE do começo da aula."

**[Markdown 2]**
"Primeiro, os dados: um dataset público de SMS rotulados como spam ou não."

**[Célula 3 — dados]**
"São 5.572 SMS reais. O problema: 4.825 normais e só 747 spams. Se eu treinar assim, o modelo pode dizer sempre 'não é spam' e já acerta 87%. Então eu balanceio: os 747 spams e 747 normais sorteados. Troco os rótulos por números, ham vira 0 e spam vira 1, e divido em 70% treino, 10% validação e 20% teste, que o modelo nunca vê."

**[Célula 4 — datasets]**
"As mensagens têm tamanhos diferentes, e um lote precisa ser retangular. Então o `SpamDataset` completa todas com o token 50256, o `<|endoftext|>`, até o tamanho da maior, que dá 120 tokens. O DataLoader junta de 8 em 8: cada lote é 8 mensagens por 120 tokens, mais 8 rótulos. Ficamos com 1.045 de treino, 149 de validação e 300 de teste."

**[Markdown 5]**
"Antes de treinar: o GPT-2 já sabe muita coisa. Será que não dá para só perguntar para ele?"

**[Célula 6 — prompt]**
"Pergunto: 'Esse texto é spam? Responda yes ou no', com um spam óbvio. E ele não responde: só repete o formato da pergunta. O modelo base só continua texto; nunca foi ensinado a obedecer instrução, isso é o notebook do Rafael. Então vamos por outro caminho: mudar a arquitetura."

**[Markdown 7]**
"O plano: congelar o modelo, trocar a camada de saída por uma com 2 saídas e liberar só o último bloco e a normalização final."

**[Célula 8 — congelar/trocar/liberar]**
"Congelo tudo: `requires_grad = False` quer dizer 'não mexa nesses pesos', e o que o GPT-2 aprendeu fica preservado. Troco a última camada: ela recebia 768 números e devolvia 50.257 notas; agora devolve 2, 'não spam' e 'spam'. E descongelo o último bloco e a normalização final para o modelo se ajustar à tarefa. Treino 7 milhões de 124 milhões, menos de 6%."

**[Célula 9 — quanto treinar]**
"Comparando as opções: só a cabeça são 1.538 parâmetros, rápido mas pouco flexível. Tudo são 124 milhões, mais caro e com risco de estragar o que ele sabia. Ficamos no meio, uns 7 milhões. A diferença de 1.536 para a célula anterior é a normalização final."

**[Markdown 10]**
"O GPT devolve uma saída por token, mas eu quero uma resposta por mensagem. Uso a última posição, porque, pela máscara causal, é o único token que enxergou a mensagem inteira."

**[Célula 11 — shape e acurácia antes]**
"A saída é 8 mensagens, 120 posições e agora 2 números em vez de 50 mil. Uso só a última com `[:, -1, :]`. Antes do treino, 45% na validação: com duas classes balanceadas, é chute, porque a cabeça nova está aleatória. Vamos treinar."

**[Célula 13 — treino/checkpoint]**
"Como o modelo está salvo, ele só carrega; o `strict=False` é porque salvei só a parte treinada, e o resto é o GPT-2 original. Mas o log do treino está aqui: AdamW com taxa de aprendizado bem pequena para não estragar o pré-treino; o loop passa um lote, compara a nota da última posição com o rótulo, calcula gradientes e ajusta. A loss vai de 2,1 para 0,08 e a validação acompanha. A acurácia sai de 70% na primeira época para 97% na quarta. Dois minutos num notebook."

**[Célula 14 — acurácia final]**
"A medida de verdade: 97% no treino, 97% na validação e quase 96% no teste, mensagens que ele nunca viu. Treino e teste próximos: aprendeu o padrão, não decorou. De 45% para 96% treinando menos de 6% do modelo."

**[Markdown 15]**
"Agora, testar com mensagens que não estão no dataset. Se alguém quiser sugerir uma, eu coloco."

**[Célula 16 — mensagens]**
"O `classify_review` faz o mesmo caminho: tokeniza, completa até 120, roda, olha a última posição e escolhe a maior nota. A primeira é exatamente a que o GPT-2 não conseguiu responder via prompt: agora é spam. O jantar: não é spam. O 'URGENTE, sua conta foi suspensa': spam, e nem estava no dataset. Os slides: não é spam. Alguém quer testar? Lembrando que ele foi treinado com SMS em inglês."

**[Markdown 17 — fechamento / passagem]**
"Construímos um GPT-2 com cabeça de 2 classes. O truque serve para sentimento, tópico, intenção: é só trocar o número de saídas. O modelo já sabe a língua; a gente só ensina a decisão final. Mas isso não transforma o GPT num assistente: ele não conversa, só classifica. Fazer ele seguir instruções é outro tipo de fine-tuning, e é o que o Rafael mostra agora."
