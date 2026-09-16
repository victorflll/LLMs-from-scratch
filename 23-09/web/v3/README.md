# GPT por dentro — jornada contínua

Abra `index.html`. Funciona offline com `gpt_journey.css`,
`gpt_journey.js` e `gpt_journey_data.js` na mesma pasta.

A leitura agora percorre as 12 operações em uma página contínua. O índice fixo
permite saltar para qualquer operação; o token selecionado acompanha a consulta
ao embedding e à posição. A paleta usa preto, branco e azul.

## Dados e limites

- IDs GPT-2, shapes e geração são extraídos das saídas salvas de
  `chapter4_visual_lesson.ipynb`, sem executar nem simular um modelo.
- A jornada acompanha Small: 768 dimensões, 12 cabeças, 12 blocos.
- As famílias e contagens de parâmetros continuam disponíveis em Dados e código,
  como registros do notebook; não são apresentadas como execuções adicionais.
- Os diagramas são estruturais. Não há valores inventados de pesos, ativações,
  logits ou probabilidades. A máscara causal mostra permissões, não intensidades.
- Geração permite revelar, voltar e reiniciar os 10 IDs realmente registrados.
  O texto completo decodificado também vem do notebook. Não ocorre inferência no navegador.
- Cada operação permite abrir o código original, extraído de `gpt.py`; a
  tokenização mostra a célula correspondente do notebook.
- Células são numeradas a partir de 1, incluindo Markdown, na ordem do arquivo.

## Atualizar os registros

Após salvar novas saídas compatíveis no notebook:

```sh
python3 23-09/scripts/export_journey.py
python3 23-09/tests/test_journey.py
node --check 23-09/web/v2/gpt_journey.js
```

O exportador usa somente a biblioteca padrão do Python, inclui o SHA-256 do
notebook e copia literalmente as saídas e os trechos de código. Alterações na
estrutura/configuração do notebook exigem revisar também os textos da jornada.

## Validação

- Origem de todas as células exportadas conferida automaticamente contra o notebook.
- 12 seções, 12 registros TransformerBlock e shape final conferidos.
- Navegador: troca de token, grupo final e dimensão 767, cabeça 12, máscara
  bloqueada, bloco 12 e percurso dos 10 IDs, incluindo retorno e reinício.
- Sem erros JavaScript durante as interações verificadas.
- Layout de 390 px sem transbordamento horizontal da página.
- Movimento reduzido respeitado; controles nativos acessíveis por teclado.

## Versões anteriores

- [V2 anterior](../v2-anterior/index.html) · [documentação](../v2-anterior/README.md).
- [V1](../v1/index.html).
- [Diagrama original](../original/index.html).
