# Utilidades compartilhadas pelos notebooks da aula 30-09.
# Todo o modelo vem do pacote do livro em ../pkg/llms_from_scratch (sem copiar código).

import os
import sys

import requests
import torch
import torch.nn as nn

AULA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(AULA_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "pkg"))

from llms_from_scratch.ch04 import GPTModel  # noqa: E402

CHECKPOINTS = os.path.join(AULA_DIR, "checkpoints")
DATA = os.path.join(AULA_DIR, "data")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 50.000 merges do BPE + 256 bytes + <|endoftext|>
    "context_length": 1024,  # máximo de tokens que o modelo enxerga
    "emb_dim": 768,          # tamanho de cada vetor de token
    "n_heads": 12,           # cabeças de atenção (768 / 12 = 64 por cabeça)
    "n_layers": 12,          # blocos Transformer empilhados
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT2_SIZES = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

_PTH = {"124M": "gpt2-small-124M.pth", "355M": "gpt2-medium-355M.pth",
        "774M": "gpt2-large-774M.pth", "1558M": "gpt2-xl-1558M.pth"}


def the_verdict():
    with open(os.path.join(REPO_DIR, "ch02", "01_main-chapter-code", "the-verdict.txt"), encoding="utf-8") as f:
        return f.read()


def load_gpt2(size="124M"):
    """Baixa (uma vez) e carrega os pesos oficiais da OpenAI convertidos para PyTorch."""
    cfg = {**GPT_CONFIG_124M, **GPT2_SIZES[size], "drop_rate": 0.0, "qkv_bias": True}
    path = os.path.join(CHECKPOINTS, _PTH[size])
    if not os.path.exists(path):
        url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{_PTH[size]}"
        print(f"Baixando {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.rename(path + ".part", path)
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval(), cfg


class Zero(nn.Module):
    """Substitui um sub-bloco que soma no residual: x + Zero(x) = x, ou seja, remove o sub-bloco."""
    def forward(self, x):
        return torch.zeros_like(x)


def variante(cfg, sem_norm=False, sem_ffn=False, relu=False):
    """GPTModel do livro com peças removidas/trocadas, para os experimentos 🔧."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        if sem_norm:
            block.norm1 = nn.Identity()
            block.norm2 = nn.Identity()
        if sem_ffn:
            block.ff = Zero()
        elif relu:
            block.ff.layers[1] = nn.ReLU()
    if sem_norm:
        model.final_norm = nn.Identity()
    return model


def n_params(model, treinaveis=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not treinaveis)
