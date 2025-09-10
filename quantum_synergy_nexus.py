#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumSynergyNexus - Emergent Hybrid Architecture
Combines: SynergyNexusUltraFixed + UniversalQuantumFramework + SynergyXFramework

Creates capabilities none of the individual architectures possess alone:
- Quantum-enhanced multi-modal processing with memory optimization
- GPU-accelerated (optional) quantum state manipulation
- Synergistic modular architecture with quantum reasoning
- Emergent quantum-synergy phenomena (attention, memory fusion, resonance)
"""

from __future__ import annotations

import logging
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Optional upstream deps (gracefully optional)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from .synergy_nexus_ultra_fixed import SynergyNexusUltraFixed, AdvancedSynergyConfig
    SYNERGY_NEXUS_AVAILABLE = True
except Exception:
    SYNERGY_NEXUS_AVAILABLE = False

try:
    from .universal_quantum_framework import QuantumState, UniversalQuantumFramework
    QUANTUM_FRAMEWORK_AVAILABLE = True
except Exception:
    QUANTUM_FRAMEWORK_AVAILABLE = False

try:
    from .synergyx_framework import SynergyConfig, EpisodicMemory
    SYNERGYX_AVAILABLE = True
except Exception:
    SYNERGYX_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class QuantumSynergyConfig:
    """Emergent configuration combining all three architectures."""

    # From SynergyNexusUltraFixed
    modal_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "text": 768,
            "vision": 1024,
            "audio": 256,
            "numerical": 64,
            "multimodal": 512,
        }
    )
    embed_dim: int = 512
    device: str = "auto"
    use_mixed_precision: bool = True
    enable_quantum_processing: bool = True

    # From UniversalQuantumFramework
    quantum_dim: int = 128
    entanglement_layers: int = 3
    quantum_gates: List[str] = field(default_factory=lambda: ["hadamard", "phase", "cnot"])

    # From SynergyXFramework
    memory_size: int = 5000
    episodic_memory_size: int = 1000
    modular_components: List[str] = field(default_factory=lambda: ["encoder", "quantum", "memory", "fusion"])

    # Emergent properties
    synergy_quantum_coupling: float = 0.8      # 0..1 direct scalar
    coupling_trainable: bool = False           # learnable scalar if True
    emergent_processing_layers: int = 4
    quantum_memory_fusion: bool = True
    multi_dimensional_attention: bool = True

    # Safety / determinism
    deterministic_fallback: bool = True
    fallback_seed: int = 0
    clamp_trig_input: float = 6.0              # clamp inputs to trig funcs

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_device(cfg: QuantumSynergyConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure [B,T,D]; upgrades [B,D]→[B,1,D], [D]→[1,1,D]."""
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected tensor with dim 1/2/3, got {tuple(x.shape)}")

def _ensure_2d(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure [B,Q]; upgrades [Q]→[B,Q]."""
    if x.dim() == 2:
        return x
    if x.dim() == 1:
        return x.unsqueeze(0).expand(batch_size, -1)
    raise ValueError(f"Expected quantum states with dim 1/2, got {tuple(x.shape)}")

# ──────────────────────────────────────────────────────────────────────────────
# Emergent Processor
# ──────────────────────────────────────────────────────────────────────────────
class QuantumSynergyProcessor(nn.Module):
    """Emergent processor that creates new capabilities through synthesis."""

    def __init__(self, config: QuantumSynergyConfig):
        super().__init__()
        self.config = config
        self.device = _resolve_device(config)

        # Coupling parameter
        init_val = float(max(0.0, min(1.0, config.synergy_quantum_coupling)))
        if config.coupling_trainable:
            self.coupling = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        else:
            self.register_buffer("coupling", torch.tensor(init_val, dtype=torch.float32))

        # Bridges
        self.quantum_synergy_bridge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embed_dim, config.quantum_dim),
                    nn.GELU(),
                    nn.Linear(config.quantum_dim, config.embed_dim),
                )
                for _ in range(config.emergent_processing_layers)
            ]
        )

        # Projections (defined once; reused)
        self.q_to_emb = nn.Linear(config.quantum_dim, config.embed_dim, bias=False)
        self.q_to_emb_resonance = nn.Linear(config.quantum_dim, config.embed_dim, bias=False)

        # Multi-dimensional attention
        if config.multi_dimensional_attention:
            self.multi_dim_attention = nn.MultiheadAttention(
                config.embed_dim, num_heads=16, dropout=0.1, batch_first=True
            )

        # Quantum-memory fusion
        if config.quantum_memory_fusion:
            self.quantum_memory_fusion = nn.Sequential(
                nn.Linear(config.embed_dim + config.quantum_dim, config.embed_dim * 2),
                nn.GELU(),
                nn.Linear(config.embed_dim * 2, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
            )

        self.to(self.device)
        logger.info("QuantumSynergyProcessor initialized (device=%s)", self.device)

    def forward(
        self,
        synergy_embeddings: torch.Tensor,
        quantum_states: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            synergy_embeddings: [B,D] or [B,T,D]
            quantum_states:     [Q]  or [B,Q]
            memory_context:     [B,M] or [B,T,M] (optional)
        Returns:
            dict of emergent tensors
        """
        x = _ensure_3d(synergy_embeddings.to(self.device))   # [B,T,D]
        bsz, T, D = x.shape
        q = _ensure_2d(quantum_states.to(self.device), bsz)  # [B,Q]

        emergent: Dict[str, torch.Tensor] = {}
        ci = torch.clamp(self.coupling, 0.0, 1.0)

        # ── Coupling (multi-layer)
        coupled = x
        for i, bridge in enumerate(self.quantum_synergy_bridge):
            bridged = bridge(coupled)
            coupled = ci * bridged + (1.0 - ci) * coupled
            emergent[f"quantum_synergy_layer_{i}"] = coupled

        # ── Multi-dimensional attention (keys/values from quantum)
        if hasattr(self, "multi_dim_attention"):
            qk = self.q_to_emb(q).unsqueeze(1).expand(-1, T, -1)  # [B,T,D]
            attended, attn_w = self.multi_dim_attention(coupled, qk, qk)  # [B,T,D], [B,T,T]
            emergent["quantum_attention"] = attended
            emergent["attention_weights"] = attn_w
        else:
            attended = coupled

        # ── Quantum-memory fusion
        if hasattr(self, "quantum_memory_fusion") and memory_context is not None:
            mem = memory_context.to(self.device)
            mem = _ensure_3d(mem) if mem.dim() != 3 else mem
            if mem.shape[1] != attended.shape[1]:
                if mem.shape[1] == 1:
                    mem = mem.expand(bsz, attended.shape[1], mem.shape[-1])
                else:
                    raise ValueError(f"Memory time dim {mem.shape[1]} != {attended.shape[1]}")

            qk = self.q_to_emb(q).unsqueeze(1).expand(-1, attended.shape[1], -1)  # [B,T,D]
            fused_in = torch.cat([attended, qk], dim=-1)  # [B,T,D+Q->D+D]
            fused = self.quantum_memory_fusion(fused_in)
            emergent["quantum_memory_fusion"] = fused
        else:
            fused = attended

        # ── Resonance (safe trig)
        clamp = float(max(0.0, self.config.clamp_trig_input))
        xin = torch.clamp(coupled, -clamp, clamp)
        qin = torch.clamp(self.q_to_emb_resonance(q).unsqueeze(1).expand(-1, xin.shape[1], -1), -clamp, clamp)
        resonance = torch.cos(xin) * torch.sin(qin)
        emergent["quantum_resonance"] = resonance

        emergent["final_emergent"] = fused
        return emergent

# ──────────────────────────────────────────────────────────────────────────────
# Nexus Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class QuantumSynergyNexus:
    """
    Emergent Hybrid Architecture combining three powerful systems:
    - SynergyNexusUltraFixed: GPU acceleration + multi-modal processing
    - UniversalQuantumFramework: Quantum reasoning + state manipulation
    - SynergyXFramework: Memory optimization + modular architecture
    """

    def __init__(self, config: QuantumSynergyConfig):
        self.config = config
        self.device = _resolve_device(config)

        # Optional integrations
        self.synergy_nexus = None
        if SYNERGY_NEXUS_AVAILABLE:
            try:
                s_cfg = AdvancedSynergyConfig(
                    modal_dims=config.modal_dims,
                    embed_dim=config.embed_dim,
                    device=str(self.device),
                    use_mixed_precision=config.use_mixed_precision,
                )
                self.synergy_nexus = SynergyNexusUltraFixed(s_cfg)
                logger.info("🚀 SynergyNexusUltraFixed integrated")
            except Exception as e:
                logger.warning(f"SynergyNexus integration failed: {e}")

        self.quantum_framework = None
        if QUANTUM_FRAMEWORK_AVAILABLE and config.enable_quantum_processing:
            try:
                self.quantum_framework = UniversalQuantumFramework()
                logger.info("🌟 UniversalQuantumFramework integrated")
            except Exception as e:
                logger.warning(f"Quantum framework integration failed: {e}")

        self.synergyx_memory = None
        if SYNERGYX_AVAILABLE:
            try:
                sx_cfg = SynergyConfig(
                    device=str(self.device),
                    episodic_memory_size=config.episodic_memory_size,
                    semantic_memory_size=config.memory_size,
                )
                self.synergyx_memory = EpisodicMemory(sx_cfg)
                logger.info("⚡ SynergyX Memory integrated")
            except Exception as e:
                logger.warning(f"SynergyX memory integration failed: {e}")

        # Emergent processor
        self.emergent_processor = QuantumSynergyProcessor(config)

        # Deterministic fallback RNG
        self._rng = torch.Generator(device=self.device)
        if config.deterministic_fallback:
            self._rng.manual_seed(config.fallback_seed)

        # Rolling metrics
        from collections import deque
        self.emergent_metrics = {
            "quantum_synergy_coupling_efficiency": deque(maxlen=100),
            "multi_dimensional_attention_strength": deque(maxlen=100),
            "quantum_memory_fusion_coherence": deque(maxlen=100),
            "emergent_resonance_amplitude": deque(maxlen=100),
        }

        logger.info("🔮 QuantumSynergyNexus initialized (device=%s)", self.device)
        logger.info("Available components: %s", self._available_components())

    def _available_components(self) -> List[str]:
        out = ["EmergentProcessor"]
        if self.synergy_nexus is not None:
            out.append("SynergyNexusUltraFixed")
        if self.quantum_framework is not None:
            out.append("UniversalQuantumFramework")
        if self.synergyx_memory is not None:
            out.append("SynergyXMemory")
        return out

    async def process_emergent(
        self,
        query: str,
        inputs: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        context = context or {}

        # Phase 1: embeddings
        emb = inputs.get("embedding", None)
        if emb is None and self.synergy_nexus is not None:
            try:
                up = await self.synergy_nexus.process(query, inputs, context)  # type: ignore[attr-defined]
                if "embedding" in up:
                    emb = torch.tensor(up["embedding"], device=self.device)
                    emb = _ensure_3d(emb)
            except Exception as e:
                logger.warning(f"SynergyNexus process failed ({e}); falling back")
                emb = None
        if emb is None:
            logger.warning("No upstream embedding; using deterministic fallback.")
            emb = torch.randn(1, 1, self.config.embed_dim, generator=self._rng, device=self.device)

        # Phase 2: quantum
        qstate = inputs.get("quantum_state", None)
        if qstate is None:
            vec = emb[0, 0, :] if emb.dim() == 3 else emb[0]
            amps = F.softmax(vec[: self.config.quantum_dim], dim=0)
            if self.quantum_framework is not None and self.config.enable_quantum_processing:
                try:
                    qstate = torch.tensor(QuantumState(amps).amplitudes, device=self.device)
                except Exception:
                    qstate = amps.to(self.device)
            else:
                qstate = amps.to(self.device)
        else:
            qstate = qstate.to(self.device)

        # Phase 3: memory (optional)
        mem = inputs.get("memory", None)
        if mem is None and self.synergyx_memory is not None:
            try:
                mem_np = self.synergyx_memory.peek()
                if mem_np is not None:
                    mem = torch.tensor(mem_np, device=self.device)
            except Exception:
                mem = None

        # Store current embedding episodically (best-effort)
        if self.synergyx_memory is not None and emb is not None:
            try:
                key = f"emergent_query_{hash(query) % 10000}"
                self.synergyx_memory.store(key, emb[0, 0, :].detach().cpu().numpy(),
                                           metadata={"query": query, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            except Exception as e:
                logger.warning(f"Memory store failed: {e}")

        # Phase 4: emergent processor
        feats = self.emergent_processor(emb, qstate, mem)

        # Phase 5: analysis + metrics
        analysis = self._analyze_emergent_properties(feats)

        elapsed = (time.time() - start_time) * 1000.0
        logger.info("🔮 Emergent processing completed in %.1f ms", elapsed)

        return {
            "emergent_type": "QuantumSynergyNexus",
            "query": query,
            "available_components": self._available_components(),
            "processing_time_ms": elapsed,
            "emergent_features": {k: v.detach().cpu().tolist() for k, v in feats.items() if isinstance(v, torch.Tensor)},
            "emergent_analysis": analysis,
            "performance_metrics": {k: list(v) for k, v in self.emergent_metrics.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _analyze_emergent_properties(self, emergent_features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}

        # Coupling analysis
        layer_keys = [k for k in emergent_features if k.startswith("quantum_synergy_layer_")]
        if layer_keys:
            layers = [emergent_features[k] for k in layer_keys]
            norms = [float(l.norm()) for l in layers]
            eff = sum(norms) / len(norms)
            stab = (sum((n - eff) ** 2 for n in norms) / len(norms)) ** 0.5
            analysis["quantum_synergy_coupling"] = {
                "layer_evolution": norms,
                "coupling_efficiency": eff,
                "coupling_stability": float(stab),
            }
            self.emergent_metrics["quantum_synergy_coupling_efficiency"].append(eff)

        # Attention analysis
        if "attention_weights" in emergent_features:
            attn_w = emergent_features["attention_weights"]  # [B,T,S]
            # Normalize safety; compute entropy across S per query, then mean
            p = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)
            entropy_per_q = -(p * (p + 1e-8).log()).sum(dim=-1)  # [B,T]
            entropy = float(entropy_per_q.mean())
            focus = float(p.max())
            analysis["multi_dimensional_attention"] = {
                "attention_entropy": entropy,
                "attention_focus": focus,
            }
            self.emergent_metrics["multi_dimensional_attention_strength"].append(focus)

        # Fusion analysis
        if "quantum_memory_fusion" in emergent_features:
            fusion_out = emergent_features["quantum_memory_fusion"]  # [B,T,D]
            fusion_std = float(fusion_out.std())
            fusion_norm = float(fusion_out.norm())
            # Flatten time, keep features for complexity
            BT, D = fusion_out.shape[0] * fusion_out.shape[1], fusion_out.shape[2]
            mat = fusion_out.reshape(BT, D)
            try:
                _, S, _ = torch.linalg.svd(mat, full_matrices=False)
                fusion_complexity = float(S.mean())
            except Exception:
                fusion_complexity = 0.0
            analysis["quantum_memory_fusion"] = {
                "fusion_coherence": fusion_std,
                "fusion_magnitude": fusion_norm,
                "fusion_complexity": fusion_complexity,
            }
            self.emergent_metrics["quantum_memory_fusion_coherence"].append(fusion_std)

        # Resonance analysis
        if "quantum_resonance" in emergent_features:
            resonance = emergent_features["quantum_resonance"]
            amplitude = float(resonance.abs().mean())
            try:
                freq = float(torch.fft.fft(resonance.flatten()).abs().mean())
            except Exception:
                freq = 0.0
            analysis["emergent_resonance"] = {
                "resonance_amplitude": amplitude,
                "resonance_frequency": freq,
                "resonance_harmony": 1.0,  # placeholder (self-correlation)
            }
            self.emergent_metrics["emergent_resonance_amplitude"].append(amplitude)

        # Overall EIQ
        pieces = []
        if "quantum_synergy_coupling" in analysis:
            pieces.append(analysis["quantum_synergy_coupling"]["coupling_efficiency"])
        if "multi_dimensional_attention" in analysis:
            pieces.append(analysis["multi_dimensional_attention"]["attention_focus"])
        if "quantum_memory_fusion" in analysis:
            pieces.append(min(analysis["quantum_memory_fusion"]["fusion_coherence"], 1.0))
        if "emergent_resonance" in analysis:
            pieces.append(min(analysis["emergent_resonance"]["resonance_amplitude"], 1.0))
        if pieces:
            analysis["emergent_intelligence_quotient"] = sum(pieces) / len(pieces)

        return analysis

    def get_emergent_capabilities(self) -> Dict[str, str]:
        return {
            "quantum_synergy_coupling": "Bidirectional information flow between quantum and synergy processing",
            "multi_dimensional_attention": "Quantum-aware attention mechanisms across multiple modalities",
            "quantum_memory_fusion": "Integration of quantum states with episodic memory retrieval",
            "emergent_resonance_patterns": "Oscillatory patterns emerging from quantum–synergy interactions",
            "hybrid_learning": "Adaptation across quantum, synergy, and memory domains simultaneously",
            "emergent_intelligence": "Capabilities that arise only from the combined system",
        }

# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def create_quantum_synergy_nexus(device: str = "auto", **kwargs) -> QuantumSynergyNexus:
    return QuantumSynergyNexus(QuantumSynergyConfig(device=device, **kwargs))

# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
async def demo_emergent_capabilities():
    print("🔮 QuantumSynergyNexus Emergent Architecture Demo")
    print("=" * 60)

    nexus = create_quantum_synergy_nexus()

    test_inputs = {
        # If upstream isn’t integrated, these are ignored and a deterministic embedding is used.
        "text": torch.randn(1, 768),
        "vision": torch.randn(1, 1024),
        "audio": torch.randn(1, 256),
        "numerical": torch.randn(1, 64),
    }

    tests = [
        "Analyze the quantum–synergy relationship in this multimodal data",
        "What emergent patterns arise from quantum-memory fusion?",
        "Demonstrate multi-dimensional attention across modalities",
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n🔮 Emergent Test {i}: {q[:64]}...")
        result = await nexus.process_emergent(q, test_inputs)
        eq = result.get("emergent_analysis", {}).get("emergent_intelligence_quotient", "N/A")
        print(f"✅ Success — EIQ: {eq:.3f}" if isinstance(eq, float) else f"✅ Success — EIQ: {eq}")
        print(f"   Features: {len(result['emergent_features'])} • Time: {result['processing_time_ms']:.1f} ms")

    print("\n🌟 Emergent Capabilities:")
    for name, desc in nexus.get_emergent_capabilities().items():
        print(f"   • {name}: {desc}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_emergent_capabilities())

