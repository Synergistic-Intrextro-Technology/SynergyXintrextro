#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intrextro Learning â€“ Unified Adaptive & Multimodal Intelligence (Unified)
=======================================================================

This file merges Version (1) backbone + memory + federated learning with
Version (2)'s richer multimodal fusion and extended cores, under a single
stable API and device/state management layer.

Highlights
----------
- Canonical spine: SynergyModule + LearningConfig (extended)
- Memories: EpisodicMemory, SemanticMemory, WorkingMemory
- FederatedLearning: initialize/train_client/aggregate
- Fusion: MultiModalFusionCore (early/late/tensor/gated/transformer) +
  cross-modal attention, imputers, alignment scores
- Cores: MetaCore, TransferCore, OnlineCore, FewShotCore, FeedbackLoop, RLCore
- Quantum: QuantumCircuit (learnable depth) with explicit complex boundary
- Uniform I/O: module.forward(dict) and consistent get_state/set_state

Run:
    python intrextro_learning_core.py
"""
from __future__ import annotations

import math
import random
import copy
import heapq
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional metrics
try:
    from sklearn.metrics import mutual_info_score  # noqa: F401
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False

# --------------------------- Logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("intrextro")

# --------------------------- Utilities -------------------------

def _to_tensor_1d(x: Union[np.ndarray, List[float], torch.Tensor], length: int,
                  device: Optional[str] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to 1D tensor of fixed length by padding/truncating."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().view(-1).numpy()
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float32).reshape(-1)
    else:
        arr = np.array(x, dtype=np.float32).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="constant")
    elif arr.size > length:
        arr = arr[:length]
    t = torch.from_numpy(arr).to(dtype)
    if device:
        t = t.to(device)
    return t


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy/tensors to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj

# --------------------------- Configuration ----------------------
@dataclass
class LearningConfig:
    """Unified configuration for Intrextro and SynergyX modules."""
    # Model dimensions
    state_dim: int = 64
    hidden_size: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout_rate: float = 0.2

    # Embeddings / modalities
    embedding_dim: int = 128
    modalities: Optional[List[str]] = None  # defaulted in __post_init__
    supported_modalities: Optional[List[str]] = None  # for fusion core
    fusion_type: str = "transformer_fusion"  # early/late/tensor/gated/transformer_fusion

    # Optimization
    learning_rate: float = 2e-3
    batch_size: int = 32
    regularization_strength: float = 0.0

    # Memory
    memory_size: int = 1000
    episodic_memory_size: int = 500

    # Quantum and federated
    quantum_depth: int = 4
    superposition_dim: int = 64
    entanglement_factor: float = 0.5

    # RL
    epsilon: float = 0.1
    gamma: float = 0.99

    # Federated
    num_clients: int = 5

    # System
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["text", "image", "numerical", "audio", "time_series"]
        if self.supported_modalities is None:
            self.supported_modalities = list(self.modalities)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            self.device = "cpu"
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        logger.info(
            "Config: hidden_size=%d, heads=%d, lr=%.4g, mem=%d, qdepth=%d, device=%s, fusion=%s",
            self.hidden_size, self.num_heads, self.learning_rate,
            self.memory_size, self.quantum_depth, self.device, self.fusion_type
        )

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]):
        return cls(**dict_data)

# --------------------------- Base Module ------------------------
class SynergyModule(nn.Module, ABC):
    """Base class for all modules with device/state plumbing."""

    def __init__(self, config: LearningConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self._initialize()
        self.to(self.device)

    @abstractmethod
    def _initialize(self) -> None:
        pass

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def adapt(self, feedback: Dict[str, Any]) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"model_state": self.state_dict(), "config": self.config.to_dict()}

    def set_state(self, state: Dict[str, Any]) -> None:
        if "model_state" in state:
            self.load_state_dict(state["model_state"])
        if "config" in state:
            self.config = LearningConfig.from_dict(state["config"])  # type: ignore
            self.device = self.config.device
            self.to(self.device)

# --------------------- Memories -------------------------
class MemoryItem:
    def __init__(self, key: str, content: torch.Tensor, metadata: Dict[str, Any] = None,
                 importance: float = 0.0, timestamp: int = 0):
        self.key = key
        self.content = content
        self.metadata = metadata or {}
        self.importance = importance
        self.timestamp = timestamp

    def __lt__(self, other: "MemoryItem"):
        return self.importance < other.importance


class EpisodicMemory(SynergyModule):
    def _initialize(self) -> None:
        self.memory: Dict[str, MemoryItem] = {}
        self.priority_queue: List[MemoryItem] = []
        self.size: int = 0
        self.max_size: int = self.config.episodic_memory_size

    def store(self, key: str, content: torch.Tensor, metadata: Dict[str, Any] = None, importance: float = 0.0) -> None:
        timestamp = self.size
        item = MemoryItem(key, content.detach().to(self.device), metadata, importance, timestamp)
        if self.size >= self.max_size:
            self._forget_least_important()
        self.memory[key] = item
        heapq.heappush(self.priority_queue, item)
        self.size += 1

    def retrieve(self, key: str) -> Optional[torch.Tensor]:
        item = self.memory.get(key)
        return item.content if item else None

    def _forget_least_important(self) -> None:
        if self.priority_queue:
            item = heapq.heappop(self.priority_queue)
            if item.key in self.memory:
                del self.memory[item.key]
                self.size -= 1

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs.get("store_memory", False):
            content = inputs["content"]
            if not isinstance(content, torch.Tensor):
                content = _to_tensor_1d(content, self.config.hidden_size, self.device)
            self.store(
                inputs.get("memory_key", f"memory_{self.size}"),
                content,
                inputs.get("metadata", None),
                float(inputs.get("importance", 0.0)),
            )
        if "retrieve_key" in inputs:
            retrieved = self.retrieve(inputs["retrieve_key"])  # may be None
            return {"retrieved_memory": retrieved}
        return {"memory_size": self.size}


class SemanticMemory(SynergyModule):
    def _initialize(self) -> None:
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.embedding_dim = self.config.embedding_dim
        self.index_matrix = torch.zeros((0, self.embedding_dim), device=self.device)
        self.keys: List[str] = []

    def store(self, key: str, embedding: torch.Tensor, metadata: Dict[str, Any] = None) -> None:
        emb = embedding.to(self.device)
        emb = emb.unsqueeze(0) if emb.dim() == 1 else emb
        self.memory[key] = {"embedding": emb, "metadata": metadata or {}}
        self.index_matrix = torch.cat([self.index_matrix, emb], dim=0)
        self.keys.append(key)

    def retrieve_similar(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.keys:
            return []
        q = query_embedding.to(self.device)
        q = q.unsqueeze(0) if q.dim() == 1 else q
        similarities = torch.cosine_similarity(q, self.index_matrix)
        top_vals, top_idxs = torch.topk(similarities, min(top_k, len(self.keys)))
        return [(self.keys[idx], top_vals[i].item()) for i, idx in enumerate(top_idxs.detach().cpu().numpy())]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs.get("store_semantic", False):
            self.store(
                inputs.get("semantic_key", f"semantic_{len(self.keys)}"),
                inputs["embedding"],
                inputs.get("metadata", None),
            )
        if "query_embedding" in inputs:
            top_k = int(inputs.get("top_k", 5))
            similar_items = self.retrieve_similar(inputs["query_embedding"], top_k)
            return {"similar_items": similar_items}
        return {"semantic_memory_size": len(self.keys)}


class WorkingMemory(SynergyModule):
    def _initialize(self) -> None:
        self.buffer = deque(maxlen=max(1, self.config.memory_size // 10))
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=min(self.config.num_heads, 8),
            dropout=self.config.dropout_rate,
            batch_first=False,
        )

    def update(self, content: torch.Tensor) -> None:
        self.buffer.append(content.detach().to(self.device))

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs.get("update_working_memory", False):
            content = inputs["content"]
            if not isinstance(content, torch.Tensor):
                content = _to_tensor_1d(content, self.config.hidden_size, self.device)
            self.update(content)
        if not self.buffer:
            return {"working_memory_output": None}
        memory_stack = torch.stack(list(self.buffer), dim=0).contiguous()
        if "query" in inputs:
            query = inputs["query"]
            if not isinstance(query, torch.Tensor):
                query = _to_tensor_1d(query, self.config.hidden_size, self.device)
            query = query.unsqueeze(0).contiguous()
            out, attn_weights = self.attention(query, memory_stack, memory_stack)
            return {"working_memory_output": out.squeeze(0), "attention_weights": attn_weights}
        return {"working_memory_output": self.buffer[-1]}

# --------------------- Fusion (from v2, wired as SynergyModule) ------------------------
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class CrossModalAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(query_dim, hidden_dim)
        self.key = nn.Linear(key_dim, hidden_dim)
        self.value = nn.Linear(value_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, query, key, value):
        if query.dim() == 1: query = query.unsqueeze(0)
        if key.dim() == 1: key = key.unsqueeze(0)
        if value.dim() == 1: value = value.unsqueeze(0)
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        scores = (q @ k.T) / math.sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=-1)
        ctx = attn @ v
        out = self.out(ctx)
        if out.shape == query.shape:
            out = self.norm(out + query)
        else:
            out = self.norm(out)
        return out.squeeze(0)


class TensorFusionNetwork(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.out = nn.Sequential(
            nn.Linear(1 + sum(input_dims), output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, inputs: List[torch.Tensor]):
        aug = []
        for i, x in enumerate(inputs):
            if x is None:
                x = torch.zeros(self.input_dims[i], device=inputs[0].device if inputs else None)
            x = x.view(-1)
            aug.append(torch.cat([torch.ones(1, dtype=x.dtype, device=x.device), x], dim=0))
        fused = torch.cat(aug, dim=0)
        return self.out(fused)


class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(sum(input_dims), output_dim), nn.Sigmoid())
        self.out = nn.Linear(sum(input_dims), output_dim)
    def forward(self, inputs: List[torch.Tensor]):
        concat = torch.cat([x.view(-1) for x in inputs], dim=0)
        gate = self.gate(concat)
        return self.out(concat) * gate


class CrossModalTransformer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=False)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = []
        for v in modality_embeddings.values():
            v = v.view(1, -1) if v.dim() == 1 else v
            embs.append(v[0].view(1, -1))
        seq = torch.stack(embs, dim=0).contiguous()  # [S, 1, E]
        out = self.enc(seq)
        return out.mean(dim=0).squeeze(0)


class MultiModalFusionCore(SynergyModule):
    """Comprehensive multimodal fusion with encoders, imputation, attention, fusion strategies."""
    def _initialize(self) -> None:
        hs, dr = self.config.hidden_size, self.config.dropout_rate
        self.supported_modalities = list(self.config.supported_modalities or [])
        if not self.supported_modalities:
            self.supported_modalities = ["text", "image", "numerical", "audio", "time_series"]
        # Modality-specific encoders
        self.modality_encoders: nn.ModuleDict = nn.ModuleDict({
            "text": nn.Sequential(
                nn.Linear(768, hs * 2), nn.LayerNorm(hs * 2), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(hs * 2, hs), nn.LayerNorm(hs)
            ),
            "image": nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(128, hs), nn.LayerNorm(hs)
            ),
            "numerical": nn.Sequential(
                nn.Linear(64, hs * 2), nn.BatchNorm1d(hs * 2), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(hs * 2, hs), nn.BatchNorm1d(hs)
            ),
            "audio": nn.Sequential(
                nn.Conv1d(1, 16, 7, 2, 3), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, 2, 2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                nn.Linear(64 * 8, hs), nn.LayerNorm(hs)
            ),
            "time_series": nn.Sequential(
                nn.GRU(input_size=32, hidden_size=hs, num_layers=2, batch_first=True, dropout=dr),
                LambdaLayer(lambda x: x[0][:, -1, :]),
                nn.LayerNorm(hs)
            )
        })
        # Fusion networks
        self.fusion_networks = nn.ModuleDict({
            "early_fusion": nn.Sequential(
                nn.Linear(hs * len(self.supported_modalities), hs * 2),
                nn.LayerNorm(hs * 2), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(hs * 2, hs), nn.LayerNorm(hs)
            ),
            "late_fusion": nn.Sequential(
                nn.Linear(len(self.supported_modalities), len(self.supported_modalities)),
                nn.Softmax(dim=1)
            ),
            "tensor_fusion": TensorFusionNetwork([hs] * len(self.supported_modalities), hs),
            "gated_fusion": GatedMultimodalUnit([hs] * len(self.supported_modalities), hs),
            "transformer_fusion": CrossModalTransformer(hs, num_heads=min(self.config.num_heads, 8), num_layers=2, dropout=dr),
        })
        # Cross-modal attentions
        self.modality_attention: Dict[str, Dict[str, CrossModalAttention]] = {}
        for m1 in self.supported_modalities:
            self.modality_attention[m1] = {}
            for m2 in self.supported_modalities:
                if m1 == m2:
                    continue
                att = CrossModalAttention(hs, hs, hs, hs, hs, num_heads=min(self.config.num_heads, 8))
                self.add_module(f"att_{m1}_{m2}", att)
                self.modality_attention[m1][m2] = att
        # Imputers
        self.modality_imputers: nn.ModuleDict = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(hs * (len(self.supported_modalities) - 1), hs * 2),
                nn.LayerNorm(hs * 2), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(hs * 2, hs), nn.LayerNorm(hs)
            )
            for m in self.supported_modalities
        })
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.regularization_strength)

    # --- preprocessing ---
    def _preprocess_modality(self, modality: str, data: Any) -> torch.Tensor:
        dev = self.device
        if modality == "text":
            if isinstance(data, str):
                tokens = data.lower().split()
                bow = torch.zeros(768, device=dev)
                for t in tokens:
                    bow[hash(t) % 768] += 1.0
                return bow
            return _to_tensor_1d(data, 768, dev)
        if modality == "image":
            x = np.array(data, dtype=np.float32)
            if x.ndim == 3:
                x = np.transpose(x, (2, 0, 1))
            if x.ndim == 2:
                x = np.expand_dims(x, 0)
            return torch.from_numpy(x).to(dev).float().unsqueeze(0)
        if modality == "numerical":
            arr = np.array(data, dtype=np.float32).reshape(-1)
            arr = np.pad(arr, (0, max(0, 64 - arr.size)))[:64]
            return torch.from_numpy(arr).to(dev).float()
        if modality == "audio":
            arr = np.array(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return torch.from_numpy(arr).to(dev).float().unsqueeze(0)
        if modality == "time_series":
            arr = np.array(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[1] < 32:
                pad = np.zeros((arr.shape[0], 32 - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > 32:
                arr = arr[:, :32]
            return torch.from_numpy(arr).to(dev).float().unsqueeze(0)
        return torch.zeros(self.config.hidden_size, device=dev)

    def encode_modality(self, modality: str, data: Any) -> torch.Tensor:
        x = self._preprocess_modality(modality, data)
        enc = self.modality_encoders.get(modality)
        if enc is None:
            logger.warning("Unsupported modality: %s", modality)
            return torch.zeros(self.config.hidden_size, device=self.device)
        out = enc(x)
        if isinstance(out, tuple):  # GRU returns (out, h)
            out = out[0]
        if out.dim() >= 2:
            out = out.view(-1, self.config.hidden_size)[0]
        return out.contiguous()

    def apply_cross_attention(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enhanced = {}
        for tgt, tvec in embeddings.items():
            acc = tvec.clone()
            for src, svec in embeddings.items():
                if src == tgt:
                    continue
                att = self.modality_attention[tgt].get(src)
                if att is not None:
                    acc = acc + att(tvec, svec, svec)
            enhanced[tgt] = acc
        return enhanced

    def impute_missing(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        present = {m: e for m, e in embeddings.items() if e is not None}
        if len(present) == len(self.supported_modalities):
            return embeddings
        out = dict(embeddings)
        for m in self.supported_modalities:
            if out.get(m) is not None:
                continue
            if present:
                cat = torch.cat([v.view(-1) for v in present.values()], dim=0).contiguous()
                imp = self.modality_imputers[m](cat)
                out[m] = imp.view(-1)
            else:
                out[m] = torch.zeros(self.config.hidden_size, device=self.device)
        return out

    def _alignment(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = list(embeddings.keys())
        out = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                v1 = embeddings[a].view(1, -1)
                v2 = embeddings[b].view(1, -1)
                out[f"{a}_{b}"] = float(F.cosine_similarity(v1, v2).item())
        return out

    def fuse(self, embeddings: Dict[str, torch.Tensor], kind: Optional[str] = None) -> torch.Tensor:
        hs = self.config.hidden_size
        if not embeddings:
            return torch.zeros(hs, device=self.device)
        kind = kind or self.config.fusion_type
        net = self.fusion_networks.get(kind, self.fusion_networks["early_fusion"])
        if kind == "early_fusion":
            stacked = torch.cat([embeddings.get(m, torch.zeros(hs, device=self.device)) for m in self.supported_modalities], dim=0).contiguous()
            return net(stacked)
        if kind == "late_fusion":
            stacked = torch.stack([embeddings.get(m, torch.zeros(hs, device=self.device)) for m in self.supported_modalities], dim=0).contiguous()
            weights = net(torch.ones(1, len(self.supported_modalities), device=self.device))[0]
            return (stacked * weights.unsqueeze(1)).sum(dim=0)
        if kind in ("tensor_fusion", "gated_fusion"):
            return net([embeddings.get(m, torch.zeros(hs, device=self.device)) for m in self.supported_modalities])
        if kind == "transformer_fusion":
            return net(embeddings)
        return torch.mean(torch.stack(list(embeddings.values()), dim=0), dim=0)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = inputs.get("data", {})
        if not isinstance(data, dict):
            return {"error": "data must be a dict of modality -> payload", "available_modalities": []}
        embeddings: Dict[str, torch.Tensor] = {}
        available: List[str] = []
        for m in self.supported_modalities:
            if m in data and data[m] is not None:
                embeddings[m] = self.encode_modality(m, data[m])
                available.append(m)
        if not embeddings:
            return {"error": "No valid modality data provided", "available_modalities": []}
        full = self.impute_missing(embeddings)
        attended = self.apply_cross_attention(full)
        fused = self.fuse(attended, inputs.get("fusion_type"))
        align = self._alignment(attended)
        return {
            "fused_embedding": fused,
            "modality_embeddings": attended,
            "available_modalities": available,
            "imputed_modalities": [m for m in full if m not in embeddings],
            "alignment_scores": align,
            "fusion_type": inputs.get("fusion_type", self.config.fusion_type),
        }

# -------------------- Cores (from v2, now SynergyModule) -----------------------
class NeuralEnsemble(nn.Module):
    def __init__(self, config: LearningConfig):
        super().__init__()
        hs, dr = config.hidden_size, config.dropout_rate
        self.encoder = nn.Sequential(
            nn.Linear(hs, hs * 2), nn.ReLU(), nn.Dropout(dr), nn.Linear(hs * 2, hs), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hs, hs * 2), nn.ReLU(), nn.Dropout(dr), nn.Linear(hs * 2, hs)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    def process_features(self, input_data: Union[np.ndarray, List[float], torch.Tensor], hidden_size: int, device: str) -> Dict[str, Any]:
        x = _to_tensor_1d(input_data, hidden_size, device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        recon_error = F.mse_loss(decoded, x).item()
        return {"encoded_features": encoded.detach(), "reconstructed_features": decoded.detach(), "reconstruction_error": recon_error}


class MetaCore(SynergyModule):
    def _initialize(self) -> None:
        self.adaptation_history: deque = deque(maxlen=self.config.memory_size)
        self.neural_net = NeuralEnsemble(self.config).to(self.device)
    def _preprocess_data(self, arr: np.ndarray) -> np.ndarray:
        std = float(np.std(arr)) or 1.0
        return (((arr - float(np.mean(arr))) / std).astype(np.float32) * 0.85)
    def _extract_meta_patterns(self, processed: np.ndarray) -> np.ndarray:
        fft = np.fft.fft(processed, n=self.config.hidden_size)
        return (np.abs(fft) ** 2).astype(np.float32)
    def _optimize_patterns(self, meta: np.ndarray) -> np.ndarray:
        threshold = float(np.mean(meta) + np.std(meta))
        out = np.where(meta > threshold, meta, 0.0).astype(np.float32)
        m = float(np.max(np.abs(out))) if out.size else 0.0
        return out if m < 1e-10 else (out / m).astype(np.float32)
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = inputs.get("data", [])
        arr = np.array(data, dtype=np.float32).reshape(-1)
        processed = self._preprocess_data(arr)
        spectra = self._extract_meta_patterns(processed)
        optimized = self._optimize_patterns(spectra)
        nf = self.neural_net.process_features(optimized, self.config.hidden_size, self.device)
        self.adaptation_history.append({"reconstruction_error": nf["reconstruction_error"]})
        return {"features": nf["encoded_features"], "reconstruction_error": nf["reconstruction_error"]}


class DeepCore(SynergyModule):
    def _initialize(self) -> None:
        hs, dr = self.config.hidden_size, self.config.dropout_rate
        self.model = nn.Sequential(nn.Linear(hs, hs * 2), nn.ReLU(), nn.Dropout(dr), nn.Linear(hs * 2, hs)).to(self.device)
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = _to_tensor_1d(inputs.get("data", []), self.config.hidden_size, self.device)
        y = self.model(x)
        return {"features": y}


class TransferCore(SynergyModule):
    def _initialize(self) -> None:
        self.knowledge_base: Dict[str, Any] = {}
        self.transfer_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=min(self.config.num_heads, 8),
                dim_feedforward=self.config.hidden_size * 4,
                batch_first=False,
                dropout=self.config.dropout_rate,
            ),
            num_layers=self.config.num_layers,
        )
    def _embed_domain(self, domain: Any) -> torch.Tensor:
        hs = self.config.hidden_size
        if isinstance(domain, (np.ndarray, torch.Tensor, list)):
            return _to_tensor_1d(domain, hs, self.device)
        return torch.full((hs,), float(domain), dtype=torch.float32, device=self.device)
    def _compute_transfer_map(self, source: torch.Tensor) -> torch.Tensor:
        sbe = source.view(-1, 1, self.config.hidden_size).contiguous()
        feat = self.transfer_model(sbe).squeeze(1).mean(dim=0)
        return feat
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        src = self._embed_domain(inputs.get("source", 0.0))
        fmap = self._compute_transfer_map(src)
        key = f"transfer_{len(self.knowledge_base)}"
        self.knowledge_base[key] = fmap.detach().cpu().tolist()
        return {"features": fmap, "confidence": float(torch.mean(torch.abs(fmap)))}


class RLCore(SynergyModule):
    def _initialize(self) -> None:
        hs = self.config.hidden_size
        self.value_network = nn.Sequential(nn.Linear(hs, hs * 2), nn.ReLU(), nn.Linear(hs * 2, 1)).to(self.device)
        self.policy_network = nn.Sequential(nn.Linear(hs, hs * 2), nn.ReLU(), nn.Linear(hs * 2, hs), nn.Softmax(dim=-1)).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.value_network.parameters()) + list(self.policy_network.parameters()),
            lr=self.config.learning_rate, weight_decay=self.config.regularization_strength,
        )
    def _process_state(self, state: Union[Dict, np.ndarray, torch.Tensor, List[float]]) -> torch.Tensor:
        if isinstance(state, dict):
            v = state.get("state")
            arr = v if isinstance(v, (np.ndarray, torch.Tensor, list)) else list(state.values())
        else:
            arr = state
        return _to_tensor_1d(arr, self.config.hidden_size, self.device)
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        st = self._process_state(inputs.get("state", []))
        action_probs = self.policy_network(st)
        if torch.isnan(action_probs).any() or action_probs.sum() <= 0:
            action_probs = torch.full_like(action_probs, 1.0 / action_probs.numel())
        idx = int(torch.multinomial(action_probs, 1).item())
        value = self.value_network(st).item()
        out = {"action": idx, "value": value, "action_probs": action_probs.detach()}
        if "reward" in inputs:
            reward = float(inputs.get("reward", 0.0))
            target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            v = self.value_network(st)
            p = self.policy_network(st)
            v_loss = F.mse_loss(v.view_as(target), target)
            p_loss = -torch.log(p[idx].clamp_min(1e-12)) * reward
            loss = v_loss + p_loss
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            out.update({"training_loss": float(loss.item())})
        return out


class OnlineCore(SynergyModule):
    def _initialize(self) -> None:
        self.streaming_buffer: deque = deque(maxlen=self.config.memory_size)
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout_rate if self.config.num_layers > 1 else 0.0,
            batch_first=True,
        )
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = _to_tensor_1d(inputs.get("stream", []), self.config.hidden_size, self.device)
        self.streaming_buffer.append(x.detach().cpu().numpy())
        x_b = x.view(1, 1, -1).contiguous()
        feat, _ = self.lstm(x_b)
        feat = feat.view(-1)
        pred = feat * 0.5
        return {"features": feat.detach(), "predictions": pred.detach(), "buffer_fill_ratio": len(self.streaming_buffer) / self.config.memory_size}


class FewShotCore(SynergyModule):
    def _initialize(self) -> None:
        hs, dr = self.config.hidden_size, self.config.dropout_rate
        self.prototypes: Dict[int, List[float]] = {}
        self.net = nn.Sequential(nn.Linear(hs, hs * 2), nn.ReLU(), nn.Dropout(dr), nn.Linear(hs * 2, hs))
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        examples = inputs.get("examples", [])
        ex = np.array(examples, dtype=np.float32)
        if ex.ndim == 1: ex = ex[None, :]
        if ex.shape[1] < self.config.hidden_size:
            ex = np.pad(ex, ((0, 0), (0, self.config.hidden_size - ex.shape[1])), mode="constant")
        elif ex.shape[1] > self.config.hidden_size:
            ex = ex[:, :self.config.hidden_size]
        t = torch.from_numpy(ex).to(self.device).float()
        emb = self.net(t)
        proto = emb.mean(dim=0)
        pid = len(self.prototypes)
        self.prototypes[pid] = proto.detach().cpu().tolist()
        d = torch.linalg.norm(emb - proto, dim=1).clamp_min(1e-10)
        sim = 1.0 / (1.0 + d)
        return {"embeddings": emb.detach(), "prototype": proto.detach(), "similarity": {"mean": float(sim.mean().item()), "max": float(sim.max().item()), "min": float(sim.min().item())}, "num_prototypes": len(self.prototypes)}


class FeedbackLoop(SynergyModule):
    def _initialize(self) -> None:
        hs, dr = self.config.hidden_size, self.config.dropout_rate
        self.history: deque = deque(maxlen=self.config.memory_size)
        self.net = nn.Sequential(nn.Linear(hs * 2, hs), nn.ReLU(), nn.Dropout(dr), nn.Linear(hs, hs))
    @staticmethod
    def _to_numeric(state: Dict[str, Any]) -> np.ndarray:
        vals: List[float] = []
        if isinstance(state, dict):
            for v in state.values():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    vals.extend([float(x) for x in v[:10]])
        return np.array(vals, dtype=np.float32)
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        src_state = inputs.get("source_state", {})
        tgt_state = inputs.get("target_state", {})
        s = self._to_numeric(src_state)
        t = self._to_numeric(tgt_state)
        st = np.concatenate([s, t], axis=0)
        st_t = _to_tensor_1d(st, self.config.hidden_size * 2, self.device)
        feat = self.net(st_t)
        out = {"features": feat.detach(), "source_performance": float(np.mean(s) if s.size else 0.0), "target_performance": float(np.mean(t) if t.size else 0.0)}
        self.history.append({"src": src_state, "tgt": tgt_state})
        return out

# --------------------- Quantum (from v1) -------------------------
class QuantumInspiredLayer(SynergyModule):
    def _initialize(self) -> None:
        self.superposition_dim = self.config.superposition_dim
        self.input_dim = self.config.hidden_size
        self.phase_shift = nn.Linear(self.input_dim, self.superposition_dim)
        self.amplitude = nn.Linear(self.input_dim, self.superposition_dim)
        self.entanglement = nn.Parameter(
            torch.randn(self.superposition_dim, self.superposition_dim, dtype=torch.float32, device=self.device)
            * self.config.entanglement_factor
        )
        self.output_projection = nn.Linear(self.superposition_dim, self.input_dim)
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["features"].to(self.device).to(torch.float32)
        phase = self.phase_shift(x)
        amplitude = torch.sigmoid(self.amplitude(x))
        # complex boundary
        superposition = torch.complex(amplitude, torch.zeros_like(amplitude)) * torch.exp(1j * phase)
        entangled = torch.matmul(superposition, self.entanglement.to(torch.complex64))
        measured = torch.abs(entangled)
        output = self.output_projection(measured.float())
        return {"quantum_features": output, "quantum_state": {"amplitude": amplitude, "phase": phase}}


class QuantumCircuit(SynergyModule):
    def _initialize(self) -> None:
        self.depth = self.config.quantum_depth
        self.layers = nn.ModuleList([QuantumInspiredLayer(self.config) for _ in range(self.depth)])
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        features = inputs["features"].to(self.device).to(torch.float32)
        quantum_states = []
        layer_input = {"features": features}
        for layer in self.layers:
            layer_output = layer(layer_input)
            layer_input = {"features": layer_output["quantum_features"]}
            quantum_states.append(layer_output["quantum_state"])
        return {"features": layer_input["features"], "quantum_states": quantum_states}

# --------------------- Federated Learning (from v1) ----------------------
class FederatedLearning(SynergyModule):
    def _initialize(self) -> None:
        self.num_clients = self.config.num_clients
        self.client_models: Dict[int, nn.Module] = {}
        self.global_model: Optional[nn.Module] = None
        self.client_data: Dict[int, Any] = {}
    def initialize_global_model(self, model: nn.Module) -> None:
        self.global_model = copy.deepcopy(model).to(self.device)
    def distribute_to_clients(self) -> None:
        for client_id in range(self.num_clients):
            self.client_models[client_id] = copy.deepcopy(self.global_model).to(self.device)
    def train_client(self, client_id: int, data: Dict[str, Any], epochs: int = 1) -> Dict[str, Any]:
        if client_id not in self.client_models:
            raise ValueError(f"Client {client_id} not initialized")
        model = self.client_models[client_id]
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        losses = []
        for _ in range(epochs):
            outputs = model(data["inputs"])  # model is expected to accept tensors
            loss = data["loss_fn"](outputs, data["targets"])  # callable
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        return {"client_id": client_id, "final_loss": losses[-1], "avg_loss": sum(losses) / len(losses)}
    def aggregate_models(self, client_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        if not self.client_models:
            raise ValueError("No client models to aggregate")
        if client_weights is None:
            client_weights = {cid: 1.0 / self.num_clients for cid in self.client_models}
        total_weight = sum(client_weights.values())
        normalized_weights = {k: v / total_weight for k, v in client_weights.items()}
        global_dict = copy.deepcopy(self.global_model.state_dict())
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
        for cid, weight in normalized_weights.items():
            client_dict = self.client_models[cid].state_dict()
            for key in global_dict:
                global_dict[key] = global_dict[key] + client_dict[key] * weight
        self.global_model.load_state_dict(global_dict)
        return {"num_clients_aggregated": len(normalized_weights), "client_weights": normalized_weights}
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        command = inputs.get("command", "")
        if command == "initialize":
            self.initialize_global_model(inputs["model"])  # type: ignore
            self.distribute_to_clients()
            return {"status": "initialized", "num_clients": self.num_clients}
        elif command == "train_client":
            return self.train_client(int(inputs["client_id"]), inputs["data"], int(inputs.get("epochs", 1)))
        elif command == "aggregate":
            return self.aggregate_models(inputs.get("client_weights"))
        elif command == "get_global_model":
            return {"global_model": self.global_model}
        return {"error": "Unknown command"}

# --------------------- Ensemble (from v2, fixed & unified) ----------------------
class EnsembleCore(SynergyModule):
    def _initialize(self) -> None:
        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}
        self.diversity_metrics: Dict[str, float] = {}
        self.last_predictions: Dict[str, Optional[torch.Tensor]] = {}
        self.history: deque = deque(maxlen=self.config.memory_size)
    def register_model(self, model_id: str, model: nn.Module, initial_weight: float = 1.0) -> bool:
        if model_id in self.models:
            logger.warning("Ensemble: model %s already exists", model_id)
            return False
        self.models[model_id] = model.to(self.device)
        self.weights[model_id] = float(initial_weight)
        self.diversity_metrics[model_id] = 0.0
        self.last_predictions[model_id] = None
        return True
    def unregister_model(self, model_id: str) -> bool:
        if model_id not in self.models:
            return False
        del self.models[model_id]
        self.weights.pop(model_id, None)
        self.diversity_metrics.pop(model_id, None)
        self.last_predictions.pop(model_id, None)
        return True
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.models:
            return {"error": "No models in ensemble", "prediction": None}
        x = inputs.get("x")
        if not isinstance(x, torch.Tensor):
            x = _to_tensor_1d(x, self.config.hidden_size, self.device)
        pred_sum = torch.zeros_like(x)
        total_weight = 0.0
        individual = {}
        for mid, m in self.models.items():
            try:
                p = m(x)
                self.last_predictions[mid] = p.detach().clone()
                w = float(self.weights.get(mid, 1.0))
                pred_sum = pred_sum + p * w
                total_weight += w
                individual[mid] = p.detach().cpu().numpy()
            except Exception as e:
                logger.error("Ensemble forward error [%s]: %s", mid, e)
        y = pred_sum / total_weight if total_weight > 0 else pred_sum
        return {"ensemble_prediction": y.detach(), "individual_predictions": individual, "weights": dict(self.weights)}
    def update_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        if not self.models:
            return {}
        self.history.append(performance)
        total_perf = sum(max(1e-3, v) for v in performance.values())
        if total_perf <= 0:
            return dict(self.weights)
        for mid, perf in performance.items():
            if mid in self.models:
                new_w = max(1e-3, perf) / total_perf
                self.weights[mid] = 0.7 * new_w + 0.3 * self.weights.get(mid, 1.0)
        s = sum(self.weights.values())
        if s > 0:
            for k in list(self.weights.keys()):
                self.weights[k] /= s
        return dict(self.weights)
    def calculate_diversity(self) -> Dict[str, float]:
        if len(self.models) < 2:
            return {}
        scores: Dict[str, float] = {}
        ids = list(self.models.keys())
        for i, a in enumerate(ids):
            p1 = self.last_predictions.get(a)
            if p1 is None:
                continue
            acc = 0.0
            count = 0
            for j, b in enumerate(ids):
                if i == j:
                    continue
                p2 = self.last_predictions.get(b)
                if p2 is None:
                    continue
                sim = F.cosine_similarity(p1.view(1, -1), p2.view(1, -1)).item()
                acc += 1.0 - sim
                count += 1
            scores[a] = acc / count if count else 0.0
        self.diversity_metrics = scores
        return scores

# --------------------- Simple CLI smoke ----------------------
if __name__ == "__main__":
    cfg = LearningConfig(device="cpu")
    # Fusion smoke
    fusion = MultiModalFusionCore(cfg)
    out = fusion({"data": {"text": "Hello world from Intrextro", "numerical": [1,2,3,4], "time_series": np.random.randn(10, 5)}})
    logger.info("Fusion output keys: %s", list(out.keys()))
    # Memory smoke
    epi = EpisodicMemory(cfg)
    epi({"store_memory": True, "content": np.random.randn(cfg.hidden_size)})
    logger.info("Episodic size=%s", epi({}).get("memory_size"))
    # RL smoke
    rl = RLCore(cfg)
    rl_out = rl({"state": np.random.randn(cfg.hidden_size), "reward": 0.5})
    logger.info("RL action=%s value=%.4f", rl_out["action"], rl_out["value"]) }
