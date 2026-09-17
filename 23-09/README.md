# Aula 23-09 — Implementando um GPT do zero

**Interface atual:** [abrir a jornada v4](web/v4/index.html).

A v4 integra a explicação conceitual à jornada e aos dados da v3, preservada em `web/v3/`.
Funciona offline; abra o HTML no navegador mantendo a estrutura de pastas.

## Organização

```text
23-09/
├── index.html              # Entrada para a interface atual
├── README.md
├── lesson/                # Notebooks, módulos Python e dependências
│   ├── chapter4_visual_lesson.ipynb
│   ├── ch04.ipynb
│   ├── exercise-solutions.ipynb
│   ├── gpt.py
│   ├── previous_chapters.py
│   ├── requirements.txt
│   └── tests.py             # Teste original do modelo
├── web/
│   ├── v4/                 # Jornada integrada atual
│   ├── v3/                 # Jornada contínua anterior
│   ├── v2/                 # Versão anterior
│   ├── v1/                 # Mini-GPT didático
│   ├── original/           # Primeiro diagrama
│   └── shared/             # Snapshot do notebook usado por v3 e v4
├── scripts/
│   └── export_journey.py
└── tests/
    ├── test_journey.py      # Fidelidade dos dados ao notebook
    └── test_gpt_arch.cjs    # Cálculos do mini-GPT da v1
```

Cada versão web tem seu próprio `index.html` e seus arquivos de suporte.
A documentação específica fica no `README.md` da versão.
Os notebooks ficam junto de `gpt.py` e `previous_chapters.py` para preservar
suas importações locais. Seus conteúdos e saídas salvas não foram alterados.

## Trabalhar com os notebooks

A partir da raiz do repositório, usando um Python compatível com as dependências:

```sh
cd 23-09/lesson
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Para executar o modelo: `python gpt.py`, dentro de `lesson/`.
O teste original pode ser executado com `pytest tests.py` nesse diretório,
com PyTorch, tiktoken e pytest instalados.

## Atualizar e verificar a interface

Comandos a partir da raiz do repositório:

```sh
python3 23-09/scripts/export_journey.py
python3 23-09/tests/test_journey.py
node 23-09/tests/test_gpt_arch.cjs
node --check 23-09/web/v3/gpt_journey.js
node --check 23-09/web/v4/gpt_journey.js
```

O exportador lê os registros de `lesson/chapter4_visual_lesson.ipynb` e o código de
`lesson/gpt.py`, gravando `web/shared/gpt_journey_data.js`. V3 e v4 carregam esse
mesmo snapshot; o exportador não executa inferência.
