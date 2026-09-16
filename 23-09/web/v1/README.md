# GPT por dentro — v1


Abra **`index.html` no navegador**. Mantenha `gpt_arch.css` e `gpt_arch.js`
na mesma pasta. Funciona offline, sem instalar bibliotecas ou iniciar um servidor.
A versão anterior está preservada em `../original/index.html`.

A apresentação percorre 12 etapas: tokenização, embeddings, posição e dropout,
LayerNorm, atenção causal, primeira residual, LayerNorm + feed-forward/GELU,
segunda residual, empilhamento, LayerNorm final, logits e geração autoregressiva.

- **Setas ← / →:** etapa anterior e seguinte. **Espaço:** reproduzir/pausar (fora dos controles).
- **Reproduzir:** avanço automático a cada 10, 18 ou 30 segundos; para ao chegar ao final.
- **Tokens no topo:** selecionam a posição cujo vetor será inspecionado.
- **Atenção:** alterne cabeças; clique nas células para ver score e peso; retire a
  máscara para comparar com atenção não causal. Essa alteração recalcula todo o
  mini-GPT e permanece indicada no cabeçalho até ser revertida.
- **Feed-forward:** selecione um dos 32 neurônios e explore a GELU no controle deslizante.
- **Gerar e repetir:** por padrão, completa a frase guiada token a token; o modo
  argmax acrescenta o token de maior logit. Ambos executam novamente a rede;
  limite didático de 12 posições. **Reiniciar** restaura os quatro tokens e a máscara.
- **Ampliar diagrama:** abre espaço para mostrar as conexões em maior escala.
- **Tela cheia:** usa a tela cheia do navegador. Em telas baixas, diagramas longos
  e explicações têm rolagem própria. Em celulares, as explicações ficam abaixo.
- **Ver a implementação em PyTorch:** revela o código correspondente à etapa.

### O que os números representam

O JavaScript executa um **mini-GPT não treinado** com 8 dimensões, 2 cabeças,
2 blocos, vocabulário local de 12 tokens e 2.000 parâmetros. Embeddings, projeções,
atenção, LayerNorm, GELU, residuais, logits e softmax são calculados numericamente.
Pesos são pseudoaleatórios determinísticos (seed 123); dropout está desativado e
as LayerNorms usam gamma = 1 e beta = 0. Não são pesos ou previsões de um GPT-2 treinado.
Os IDs locais não são IDs BPE; a correspondência BPE da frase inicial é mostrada
apenas como referência. Não há tokenizer para texto livre nesta demonstração.

O seletor **GPT-2 Small → XL** muda somente os shapes e a contagem de parâmetros
na área de referência, conforme a arquitetura de `gpt.py` (QKV sem bias).
O final abre em **Frase guiada**: cinco cliques acrescentam “ a”, “ GPT”, “ built”, “ from” e “ scratch”. A frase é escolhida pelo roteiro, não prevista pelo modelo; as probabilidades continuam sendo calculadas pela rede sem treino. O seletor de modo reinicia o contexto e permite comparar com o argmax do capítulo 4. O controle GELU explora a função sem
modificar a execução da rede. As cores codificam sinal e magnitude, não semântica.

Validação numérica independente de bibliotecas do navegador:

```bash
node 23-09/tests/test_gpt_arch.cjs  # a partir da raiz do repositório
```

Verifica invariância causal ao mudar ou anexar tokens futuros, efeito da remoção
da máscara, shapes de 1 a 12 tokens, soma das probabilidades, LayerNorm, GELU e
contagem dos parâmetros efetivamente armazenados.
