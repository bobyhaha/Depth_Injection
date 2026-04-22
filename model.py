import os
import math
import json
import csv
import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # Repro / device
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False

    # Dataset / tokenizer
    dataset_name: str = "tinystories"
    hf_dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    add_eos_between_docs: bool = True

    # Storage paths
    data_dir: str = "./data_tinystories"
    tokenized_dir: str = "./data_tinystories/tokenized"
    out_dir: str = "./runs_tinystories"

    # Sequence length
    text_block_size: int = 256
    max_seq_len: int = 256

    # Preprocessing
    preprocess_batch_size: int = 1024
    num_proc: int = 8

    # Training
    batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    max_steps: int = 20000
    warmup_steps: int = 500
    eval_interval: int = 500
    eval_steps: int = 100
    log_interval: int = 20
    num_workers: int = 4
    pin_memory: bool = True

    # Main transformer
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1

    # Injection family
    #   none      : baseline transformer
    #   original  : compute M(x,l) separately for each layer
    #   shared    : compute shared M once after warmup layers, up-project to all later layers
    injection_family: str = "original"

    # Where to start injecting for shared family.
    # Example: warmup_layers=2 means first two layers are plain transformer,
    # then shared side-model is computed and injected into layers 2..n_layer-1.
    warmup_layers: int = 2

    # Small M(x,l) architecture choices for ablation
    #   mlp1, mlp2,
    #   tf1, tf2,
    #   tf1_nomlp, tf2_nomlp
    m_variant: str = "mlp1"

    # Injection operator
    #   add  -> h = h + scale * m
    #   gate -> h = h + scale * sigmoid(g(h,m)) * m
    injection_type: str = "add"
    injection_scale_init: float = 0.1

    # Depth conditioning for original family
    layer_embed_dim: int = 128
    layer_embed_mlp_dim: int = 256

    # Small model width / dropout
    m_hidden_dim: int = 512
    m_dropout: float = 0.1
    m_num_heads: int = 4

    # Shared-family projector bottleneck
    shared_projector_hidden_dim: int = 1024

    # W&B
    use_wandb: bool = True
    wandb_project: str = "depth-injection"
    wandb_entity: Optional[str] = None
    wandb_group: str = "tinystories-ablation"
    wandb_job_type: str = "train"
    wandb_tags: Tuple[str, ...] = ("tinystories", "ablation", "depth-injection")
    wandb_mode: str = "online"   # online, offline, disabled

    # Run naming
    run_name: Optional[str] = None


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.max_lr * self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + cosine * (self.max_lr - self.min_lr)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def build_run_name(cfg: Config) -> str:
    if cfg.run_name is not None:
        return cfg.run_name

    return (
        f"{cfg.dataset_name}_"
        f"{cfg.injection_family}_{cfg.m_variant}_{cfg.injection_type}_"
        f"WU{cfg.warmup_layers}_"
        f"L{cfg.n_layer}_H{cfg.n_head}_D{cfg.d_model}_"
        f"FF{cfg.d_ff}_BS{cfg.batch_size}_T{cfg.text_block_size}"
    )


def maybe_init_wandb(cfg: Config, run_name: str):
    if not cfg.use_wandb or cfg.wandb_mode == "disabled":
        return None

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        group=cfg.wandb_group,
        job_type=cfg.wandb_job_type,
        tags=list(cfg.wandb_tags),
        config=asdict(cfg),
        mode=cfg.wandb_mode,
    )


def maybe_log_wandb(data: Dict[str, Any], step: int):
    if wandb.run is not None:
        wandb.log(data, step=step)


def maybe_finish_wandb():
    if wandb.run is not None:
        wandb.finish()


# ============================================================
# Tokenization / preprocessing to memmap
# ============================================================

def preprocess_tinystories_to_memmap(cfg: Config):
    ensure_dir(cfg.data_dir)
    ensure_dir(cfg.tokenized_dir)

    train_bin = os.path.join(cfg.tokenized_dir, "train.bin")
    val_bin = os.path.join(cfg.tokenized_dir, "val.bin")
    meta_path = os.path.join(cfg.tokenized_dir, "meta.json")

    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(meta_path):
        print("Found existing tokenized memmaps. Skipping preprocessing.")
        with open(meta_path, "r") as f:
            return json.load(f)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        use_fast=True,
        model_max_length=10**9,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have eos_token_id for document separation.")

    print("Loading TinyStories dataset from Hugging Face...")
    ds = load_dataset(cfg.hf_dataset_name)

    def tok_fn(batch):
        texts = batch["text"]
        out = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        if cfg.add_eos_between_docs:
            out = [ids + [eos_id] for ids in out]
        return {"ids": out, "len": [len(x) for x in out]}

    print("Tokenizing train split...")
    train_tok = ds["train"].map(
        tok_fn,
        batched=True,
        batch_size=cfg.preprocess_batch_size,
        num_proc=cfg.num_proc,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing train",
    )

    print("Tokenizing validation split...")
    val_tok = ds["validation"].map(
        tok_fn,
        batched=True,
        batch_size=cfg.preprocess_batch_size,
        num_proc=max(1, min(cfg.num_proc, 4)),
        remove_columns=ds["validation"].column_names,
        desc="Tokenizing val",
    )

    train_n_tokens = int(np.sum(train_tok["len"], dtype=np.int64))
    val_n_tokens = int(np.sum(val_tok["len"], dtype=np.int64))

    print(f"Train tokens: {train_n_tokens:,}")
    print(f"Val tokens:   {val_n_tokens:,}")

    dtype = np.uint16
    train_arr = np.memmap(train_bin, dtype=dtype, mode="w+", shape=(train_n_tokens,))
    val_arr = np.memmap(val_bin, dtype=dtype, mode="w+", shape=(val_n_tokens,))

    print("Writing train.bin...")
    idx = 0
    for ids in tqdm(train_tok["ids"], desc="train.bin"):
        arr = np.asarray(ids, dtype=dtype)
        train_arr[idx: idx + len(arr)] = arr
        idx += len(arr)
    train_arr.flush()

    print("Writing val.bin...")
    idx = 0
    for ids in tqdm(val_tok["ids"], desc="val.bin"):
        arr = np.asarray(ids, dtype=dtype)
        val_arr[idx: idx + len(arr)] = arr
        idx += len(arr)
    val_arr.flush()

    meta = {
        "dataset_name": cfg.dataset_name,
        "hf_dataset_name": cfg.hf_dataset_name,
        "tokenizer_name": cfg.tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "train_tokens": train_n_tokens,
        "val_tokens": val_n_tokens,
        "dtype": "uint16",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved metadata to {meta_path}")
    return meta


# ============================================================
# Dataset
# ============================================================

class MemmapTokenDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.bin_path = bin_path
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

    def __len__(self):
        return max(0, self.n_tokens - self.block_size - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx: idx + self.block_size], dtype=np.int64))
        y = torch.from_numpy(np.array(self.data[idx + 1: idx + self.block_size + 1], dtype=np.int64))
        return x, y


def build_datasets(cfg: Config):
    meta = preprocess_tinystories_to_memmap(cfg)
    cfg.vocab_size = meta["vocab_size"]
    cfg.max_seq_len = cfg.text_block_size

    train_bin = os.path.join(cfg.tokenized_dir, "train.bin")
    val_bin = os.path.join(cfg.tokenized_dir, "val.bin")
    return MemmapTokenDataset(train_bin, cfg.text_block_size), MemmapTokenDataset(val_bin, cfg.text_block_size), meta


# ============================================================
# Embeddings
# ============================================================

class SinusoidalLayerEmbedding(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, dim),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, layer_idx: torch.Tensor):
        raw = self.timestep_embedding(layer_idx, self.dim)
        return self.mlp(raw)


# ============================================================
# Main transformer blocks
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# Small-model variants for M
# ============================================================

class MLP1M(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MLP2M(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformerSubBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, use_ffn: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_ffn:
            x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerM(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_head: int, d_ff: int, dropout: float, num_layers: int, use_ffn: bool):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            TinyTransformerSubBlock(d_model, n_head, d_ff, dropout, use_ffn=use_ffn)
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.out_ln(h))


class OriginalSmallModel(nn.Module):
    """
    Computes M(x,l) for one layer at a time.
    Input shape:
        x0:         (B, T, d_model)
        layer_cond: (B, layer_embed_dim)
    Output:
        m:          (B, T, d_model)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_model + cfg.layer_embed_dim
        mv = cfg.m_variant

        if mv == "mlp1":
            self.model = MLP1M(in_dim, cfg.d_model, cfg.m_hidden_dim, cfg.m_dropout)
        elif mv == "mlp2":
            self.model = MLP2M(in_dim, cfg.d_model, cfg.m_hidden_dim, cfg.m_dropout)
        elif mv == "tf1":
            self.model = TinyTransformerM(in_dim, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=1, use_ffn=True)
        elif mv == "tf2":
            self.model = TinyTransformerM(in_dim, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=2, use_ffn=True)
        elif mv == "tf1_nomlp":
            self.model = TinyTransformerM(in_dim, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=1, use_ffn=False)
        elif mv == "tf2_nomlp":
            self.model = TinyTransformerM(in_dim, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=2, use_ffn=False)
        else:
            raise ValueError(f"Unknown m_variant: {mv}")

    def forward(self, x0, layer_cond):
        B, T, _ = x0.shape
        lc = layer_cond[:, None, :].expand(B, T, -1)
        return self.model(torch.cat([x0, lc], dim=-1))


class SharedSmallModel(nn.Module):
    """
    Computes one shared representation after warmup layers, then up-projects to
    all later layers at once.

    Input:
        h_shared: (B, T, d_model)
    Output:
        m_all:    (B, T, n_target_layers, d_model)
    """
    def __init__(self, cfg: Config, n_target_layers: int):
        super().__init__()
        self.n_target_layers = n_target_layers
        mv = cfg.m_variant

        if mv == "mlp1":
            self.shared = MLP1M(cfg.d_model, cfg.d_model, cfg.m_hidden_dim, cfg.m_dropout)
        elif mv == "mlp2":
            self.shared = MLP2M(cfg.d_model, cfg.d_model, cfg.m_hidden_dim, cfg.m_dropout)
        elif mv == "tf1":
            self.shared = TinyTransformerM(cfg.d_model, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=1, use_ffn=True)
        elif mv == "tf2":
            self.shared = TinyTransformerM(cfg.d_model, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=2, use_ffn=True)
        elif mv == "tf1_nomlp":
            self.shared = TinyTransformerM(cfg.d_model, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=1, use_ffn=False)
        elif mv == "tf2_nomlp":
            self.shared = TinyTransformerM(cfg.d_model, cfg.d_model, cfg.m_num_heads, cfg.m_hidden_dim, cfg.m_dropout, num_layers=2, use_ffn=False)
        else:
            raise ValueError(f"Unknown m_variant: {mv}")

        self.up_project = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.shared_projector_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.m_dropout),
            nn.Linear(cfg.shared_projector_hidden_dim, n_target_layers * cfg.d_model),
        )

    def forward(self, h_shared):
        B, T, D = h_shared.shape
        z = self.shared(h_shared)
        out = self.up_project(z)
        return out.view(B, T, self.n_target_layers, D)


# ============================================================
# Injector
# ============================================================

class Injector(nn.Module):
    def __init__(self, d_model: int, injection_type: str, scale_init: float):
        super().__init__()
        self.injection_type = injection_type
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if injection_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        elif injection_type != "add":
            raise ValueError(f"Unknown injection_type: {injection_type}")

    def forward(self, h, m):
        if self.injection_type == "add":
            return h + self.scale * m
        gate = self.gate(torch.cat([h, m], dim=-1))
        return h + self.scale * gate * m


# ============================================================
# Full model
# ============================================================

class DepthInjectionGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.layer_embed = SinusoidalLayerEmbedding(cfg.layer_embed_dim, cfg.layer_embed_mlp_dim)
        self.injectors = nn.ModuleList([
            Injector(cfg.d_model, cfg.injection_type, cfg.injection_scale_init)
            for _ in range(cfg.n_layer)
        ])

        if cfg.injection_family == "none":
            self.small_model = None
            self.shared_small_model = None
        elif cfg.injection_family == "original":
            self.small_model = OriginalSmallModel(cfg)
            self.shared_small_model = None
        elif cfg.injection_family == "shared":
            n_target = cfg.n_layer - cfg.warmup_layers
            if n_target <= 0:
                raise ValueError("warmup_layers must be < n_layer for shared family.")
            self.small_model = None
            self.shared_small_model = SharedSmallModel(cfg, n_target_layers=n_target)
        else:
            raise ValueError(f"Unknown injection_family: {cfg.injection_family}")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x0 = self.token_emb(idx)
        h = self.drop(x0 + self.pos_emb(pos)[None, :, :])

        if self.cfg.injection_family == "none":
            for block in self.blocks:
                h = block(h)

        elif self.cfg.injection_family == "original":
            for l, block in enumerate(self.blocks):
                layer_idx = torch.full((B,), l, device=idx.device, dtype=torch.long)
                layer_cond = self.layer_embed(layer_idx)
                m = self.small_model(x0, layer_cond)
                h = self.injectors[l](h, m)
                h = block(h)

        elif self.cfg.injection_family == "shared":
            # Warmup layers: plain transformer
            for l in range(self.cfg.warmup_layers):
                h = self.blocks[l](h)

            # Shared side computation once, then split into per-layer chunks
            m_all = self.shared_small_model(h)  # (B, T, n_target, d_model)
            target_idx = 0
            for l in range(self.cfg.warmup_layers, self.cfg.n_layer):
                m = m_all[:, :, target_idx, :]
                h = self.injectors[l](h, m)
                h = self.blocks[l](h)
                target_idx += 1

        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ============================================================
# Training / eval
# ============================================================

@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"loss": mean_loss, "ppl": ppl}


def make_optimizer(model: nn.Module, cfg: Config):
    decay_params = []
    no_decay_params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=cfg.betas)


def save_history(history: Dict[str, List[Any]], log_dir: str):
    json_path = os.path.join(log_dir, "history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(log_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr"])
        for s, tr, va, ppl, lr_ in zip(
            history["step"], history["train_loss"], history["val_loss"], history["val_ppl"], history["lr"]
        ):
            writer.writerow([s, tr, va, ppl, lr_])


def save_loss_plot(history: Dict[str, List[Any]], title: str, log_dir: str):
    plt.figure(figsize=(8, 5))
    train_steps = [s for s, v in zip(history["step"], history["train_loss"]) if v is not None]
    train_losses = [v for v in history["train_loss"] if v is not None]
    val_steps = [s for s, v in zip(history["step"], history["val_loss"]) if v is not None]
    val_losses = [v for v in history["val_loss"] if v is not None]

    plt.plot(train_steps, train_losses, label="train loss")
    plt.plot(val_steps, val_losses, label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig_path = os.path.join(log_dir, "loss.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def train_one_run(cfg: Config):
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds, val_ds, meta = build_datasets(cfg)
    cfg.vocab_size = meta["vocab_size"]
    cfg.max_seq_len = cfg.text_block_size

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=False,
    )

    run_name = build_run_name(cfg)
    wandb_run = maybe_init_wandb(cfg, run_name)

    model = DepthInjectionGPT(cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = make_optimizer(model, cfg)
    scheduler = CosineWarmupScheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)

    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    run_dir = os.path.join(cfg.out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if wandb_run is not None:
        wandb.summary["train_tokens"] = meta["train_tokens"]
        wandb.summary["val_tokens"] = meta["val_tokens"]
        wandb.summary["model_parameters"] = n_params
        wandb.summary["run_dir"] = run_dir

    history = {"step": [], "train_loss": [], "val_loss": [], "val_ppl": [], "lr": []}

    model.train()
    best_val = float("inf")
    step = 0
    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        lr = scheduler.step()
        step += 1

        if step % cfg.log_interval == 0:
            train_loss_value = float(loss.item())
            history["step"].append(step)
            history["train_loss"].append(train_loss_value)
            history["val_loss"].append(None)
            history["val_ppl"].append(None)
            history["lr"].append(float(lr))

            print(f"step {step:6d} | train loss {train_loss_value:.4f} | lr {lr:.6e}")
            maybe_log_wandb(
                {
                    "train/loss": train_loss_value,
                    "train/lr": float(lr),
                    "train/grad_norm": float(grad_norm),
                    "global_step": step,
                },
                step=step,
            )

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            metrics = estimate_loss(model, val_loader, device, max_batches=cfg.eval_steps)
            val_loss = float(metrics["loss"])
            val_ppl = float(metrics["ppl"])

            history["step"].append(step)
            history["train_loss"].append(None)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["lr"].append(float(lr))

            print(f"[eval] step {step:6d} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")
            maybe_log_wandb(
                {
                    "eval/loss": val_loss,
                    "eval/ppl": val_ppl,
                    "eval/best_val_loss_so_far": min(best_val, val_loss),
                    "global_step": step,
                },
                step=step,
            )

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "meta": meta,
                    "step": step,
                    "val_loss": best_val,
                    "history": history,
                }
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save(ckpt, best_path)
                print(f"Saved checkpoint to {best_path}")
                if wandb_run is not None:
                    wandb.summary["best_val_loss"] = best_val
                    wandb.summary["best_step"] = step
                    wandb.summary["best_checkpoint"] = best_path

    save_history(history, log_dir)
    loss_png_path = save_loss_plot(history, run_name, log_dir)

    if wandb_run is not None and os.path.exists(loss_png_path):
        wandb.log({"plots/loss_curve": wandb.Image(loss_png_path)})
        wandb.save(os.path.join(run_dir, "config.json"))
        wandb.save(os.path.join(log_dir, "history.csv"))
        wandb.save(os.path.join(log_dir, "history.json"))
        wandb.save(loss_png_path)

    maybe_finish_wandb()
    return model, history, run_dir, n_params


# ============================================================
# Ablation runner
# ============================================================

ABLATION_VARIANTS = [
    "mlp1",
    "mlp2",
    "tf1",
    "tf2",
    "tf1_nomlp",
    "tf2_nomlp",
]


def run_ablation_suite(base_cfg: Config):
    results = {}

    # Baseline
    print("\n" + "=" * 90)
    print("Running baseline: plain transformer")
    print("=" * 90)
    baseline_cfg = Config(**asdict(base_cfg))
    baseline_cfg.injection_family = "none"
    baseline_cfg.m_variant = "mlp1"
    baseline_cfg.run_name = None
    _, history, run_dir, n_params = train_one_run(baseline_cfg)
    results[build_run_name(baseline_cfg)] = {
        "history": history,
        "run_dir": run_dir,
        "n_params": n_params,
        "family": "none",
        "variant": "baseline",
    }

    # Original family
    for mv in ABLATION_VARIANTS:
        print("\n" + "=" * 90)
        print(f"Running original family ablation: {mv}")
        print("=" * 90)
        cfg = Config(**asdict(base_cfg))
        cfg.injection_family = "original"
        cfg.m_variant = mv
        cfg.run_name = None
        _, history, run_dir, n_params = train_one_run(cfg)
        results[build_run_name(cfg)] = {
            "history": history,
            "run_dir": run_dir,
            "n_params": n_params,
            "family": "original",
            "variant": mv,
        }

    # Shared / up-projection family
    for mv in ABLATION_VARIANTS:
        print("\n" + "=" * 90)
        print(f"Running shared family ablation: {mv}")
        print("=" * 90)
        cfg = Config(**asdict(base_cfg))
        cfg.injection_family = "shared"
        cfg.m_variant = mv
        cfg.run_name = None
        _, history, run_dir, n_params = train_one_run(cfg)
        results[build_run_name(cfg)] = {
            "history": history,
            "run_dir": run_dir,
            "n_params": n_params,
            "family": "shared",
            "variant": mv,
        }

    return results


def plot_ablation_results(results: Dict[str, Dict[str, Any]], save_path: str):
    plt.figure(figsize=(12, 7))
    for run_name, result in results.items():
        history = result["history"]
        val_steps = [s for s, v in zip(history["step"], history["val_loss"]) if v is not None]
        val_losses = [v for v in history["val_loss"] if v is not None]
        if len(val_steps) > 0:
            plt.plot(val_steps, val_losses, label=run_name)

    plt.xlabel("step")
    plt.ylabel("validation loss")
    plt.title("TinyStories ablation comparison")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_ablation_summary(results: Dict[str, Dict[str, Any]], save_path: str):
    rows = []
    for run_name, result in results.items():
        history = result["history"]
        val_pairs = [(s, v) for s, v in zip(history["step"], history["val_loss"]) if v is not None]
        ppl_values = [v for v in history["val_ppl"] if v is not None]

        if len(val_pairs) == 0:
            best_step, best_val, final_val, final_ppl = None, None, None, None
        else:
            best_step, best_val = min(val_pairs, key=lambda x: x[1])
            final_val = val_pairs[-1][1]
            final_ppl = ppl_values[-1] if len(ppl_values) > 0 else None

        rows.append({
            "run_name": run_name,
            "family": result["family"],
            "variant": result["variant"],
            "n_params": result["n_params"],
            "best_val_loss": best_val,
            "best_step": best_step,
            "final_val_loss": final_val,
            "final_val_ppl": final_ppl,
            "run_dir": result["run_dir"],
        })

    rows.sort(key=lambda x: float("inf") if x["best_val_loss"] is None else x["best_val_loss"])

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "family",
                "variant",
                "n_params",
                "best_val_loss",
                "best_step",
                "final_val_loss",
                "final_val_ppl",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_familywise_plots(results: Dict[str, Dict[str, Any]], out_dir: str):
    families = ["none", "original", "shared"]
    for family in families:
        plt.figure(figsize=(10, 6))
        plotted = False
        for run_name, result in results.items():
            if result["family"] != family:
                continue
            history = result["history"]
            val_steps = [s for s, v in zip(history["step"], history["val_loss"]) if v is not None]
            val_losses = [v for v in history["val_loss"] if v is not None]
            if len(val_steps) > 0:
                plotted = True
                plt.plot(val_steps, val_losses, label=run_name)
        if plotted:
            plt.xlabel("step")
            plt.ylabel("validation loss")
            plt.title(f"{family} family comparison")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            save_path = os.path.join(out_dir, f"ablation_{family}.png")
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = Config(
        dataset_name="tinystories",
        hf_dataset_name="roneneldan/TinyStories",
        tokenizer_name="gpt2",
        data_dir="./data_tinystories",
        tokenized_dir="./data_tinystories/tokenized",
        out_dir="./runs_tinystories",
        text_block_size=256,
        batch_size=32,
        eval_batch_size=32,
        n_layer=12,
        n_head=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        max_steps=20000,
        warmup_steps=500,
        eval_interval=500,
        eval_steps=100,
        log_interval=20,
        injection_type="add",
        injection_scale_init=0.1,
        warmup_layers=2,
        m_hidden_dim=512,
        m_num_heads=4,
        shared_projector_hidden_dim=1024,
        num_proc=8,
        num_workers=4,
        compile_model=False,
        use_wandb=True,
        wandb_project="depth-injection",
        wandb_entity=None,
        wandb_group="tinystories-ablation-full",
        wandb_job_type="train",
        wandb_tags=("tinystories", "ablation", "depth-injection", "shared-up-project"),
        wandb_mode="online",
    )

    # pip install wandb datasets transformers tqdm matplotlib
    # wandb login
    results = run_ablation_suite(cfg)

    ensure_dir(cfg.out_dir)
    ablation_plot_path = os.path.join(cfg.out_dir, "ablation_compare_all.png")
    ablation_summary_path = os.path.join(cfg.out_dir, "ablation_summary_full.csv")

    plot_ablation_results(results, save_path=ablation_plot_path)
    save_ablation_summary(results, save_path=ablation_summary_path)
    save_familywise_plots(results, out_dir=cfg.out_dir)

    if cfg.use_wandb and cfg.wandb_mode != "disabled":
        summary_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.dataset_name}-ablation-summary-full",
            group=cfg.wandb_group,
            job_type="summary",
            tags=list(cfg.wandb_tags) + ["summary"],
            config=asdict(cfg),
            mode=cfg.wandb_mode,
        )
        if os.path.exists(ablation_plot_path):
            wandb.log({"plots/ablation_compare_all": wandb.Image(ablation_plot_path)})
            wandb.save(ablation_plot_path)
        if os.path.exists(ablation_summary_path):
            wandb.save(ablation_summary_path)
        for family in ["none", "original", "shared"]:
            p = os.path.join(cfg.out_dir, f"ablation_{family}.png")
            if os.path.exists(p):
                wandb.log({f"plots/{family}_family": wandb.Image(p)})
                wandb.save(p)
        wandb.finish()
