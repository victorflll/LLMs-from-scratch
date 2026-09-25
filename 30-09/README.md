# Aula 30-09 — Construindo o GPT-2 na prática

Aula 100% em notebooks. Cada notebook constrói uma peça (o **construto**) usada no seguinte,
e responde às perguntas do grupo rodando código.

| # | Notebook | Quem | Construto | Perguntas respondidas | Tempo* |
|---|---|---|---|---|---|
| 1 | [01_tokenizacao](01_tokenizacao.ipynb) | Elmo | tokenizer + dataloader + embeddings | BPE vs caractere/palavra/cl100k/o200k · BPE próprio do zero · tamanho do embedding · visualizar pesos | 10 s |
| 2 | [02_atencao](02_atencao.ipynb) | Rafael | `MultiHeadAttention` | outros scores (aditivo, cosseno) · nº de heads · não divisível · heatmaps das 12 heads do GPT-2 | 5 s |
| 3 | [03_gpt_e_geracao](03_gpt_e_geracao.ipynb) | Victor | `GPTModel` + geração | bloco passo a passo · texto sem pré-treino · sem LayerNorm/FFN, ReLU · 12 vs 24 camadas | 15 s |
| 4 | [04_pre_treino](04_pre_treino.ipynb) | Lilian | modelo treinado + pesos GPT-2 | loss · temperatura/top-k · ablações **com treino** · alternativas ao Transformer | 2 min |
| 5 | [05_classificador](05_classificador.ipynb) | Elmo | classificador de spam | fine-tuning de classificação (≈96% no teste) | 2,5 min / 5 s com checkpoint |
| 6 | [06_assistente](06_assistente.ipynb) | Rafael | GPT que segue instruções | fine-tuning de instruções · onde achar / como criar dados | 2 min / 15 s com checkpoint |
| 7 | [07_parametros](07_parametros.ipynb) | Cristiane | — | por que esses valores · small→XL · ×3 · GPT-3 · viés e gradiente · Llama/GPT-4/Claude | 15 s |

\* MacBook M1 Pro (MPS). Células marcadas com 🔧 são experimentos de parâmetros — a Cristiane pode puxá-las durante qualquer bloco.

## Setup

```sh
cd 30-09
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

O modelo **não é reescrito aqui**: tudo vem de `../pkg/llms_from_scratch` (código do livro). `aula.py` só tem device,
configs, download dos pesos do GPT-2 (Hugging Face, sem TensorFlow) e `variante()` para as ablações.

**Antes da aula:** rode os notebooks em ordem uma vez. Isso baixa os pesos (`checkpoints/`, ~700 MB) e o dataset de spam
(`data/`), e salva os checkpoints dos fine-tunings. Com `CARREGAR_CHECKPOINT = True` (padrão) os notebooks 05 e 06 carregam
esses checkpoints; mude para `False` para treinar ao vivo.
