#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Synergy-Enhanced AGI Framework
======================================

Fully upgraded Core AGI system with complete integration of:
1. Memory-Optimized Synergy Field (from previous system)
2. Quantum-Bio-Neuromorphic processing capabilities
3. Advanced metacognitive architecture
4. Multi-modal synergy optimization
5. Self-evolving intelligence substrate

This represents a complete fusion of controlled creation philosophy
with cutting-edge adaptive intelligence systems.
"""

import time
import json
import logging
import hashlib
import warnings
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedSynergyAGI")

# -----------------------------------------------------------------------------
# Config & Modes
# -----------------------------------------------------------------------------
class ProcessingMode(Enum):
    QUANTUM_ENHANCED = "quantum"
    NEUROMORPHIC = "neuromorphic"
    HYBRID_SYNERGY = "hybrid"
    BIO_INSPIRED = "bio"
    METACOGNITIVE = "meta"

@dataclass
class AdvancedAGIConfig:
    """Comprehensive configuration for the advanced AGI system."""
    # Core processing
    feature_dim: int = 512
    num_synergy_units: int = 8
    max_iterations: int = 6
    processing_mode: ProcessingMode = ProcessingMode.HYBRID_SYNERGY

    # Memory & optimization
    memory_budget_mb: float = 2048
    cache_capacity: int = 1000
    quantization_bits: int = 8
    use_gradient_checkpointing: bool = True

    # Quantum-inspired
    num_superposition_states: int = 16
    quantum_coherence_threshold: float = 0.85
    entanglement_strength: float = 0.7

    # Neuromorphic
    spike_threshold: float = 1.0
    membrane_decay: float = 0.9
    synaptic_plasticity: float = 0.1

    # Bio-inspired
    dna_sequence_length: int = 200
    protein_folding_complexity: int = 50
    evolutionary_pressure: float = 0.05

    # Metacognitive
    self_reflection_depth: int = 3
    improvement_threshold: float = 0.02
    architecture_mutation_rate: float = 0.01

    # Swarm & distributed
    num_swarm_agents: int = 32
    communication_range: float = 15.0
    swarm_coherence_target: float = 0.8

    # Health & monitoring
    performance_history_size: int = 1000
    health_check_interval: float = 30.0
    auto_optimization_enabled: bool = True

# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
class SynergyResult:
    """Enhanced synergy operation result with comprehensive metrics."""
    def __init__(self, synergy_type: str, confidence: float, components: Dict[str, Any]):
        self.synergy_type = synergy_type
        self.confidence = float(confidence)
        self.components = components
        self.timestamp = time.time()

# -----------------------------------------------------------------------------
# Memory Systems
# -----------------------------------------------------------------------------
class QuantumInspiredMemory:
    """Quantum-inspired memory with superposition and entanglement."""
    def __init__(self, feature_dim: int, capacity: int = 1000):
        self.feature_dim = feature_dim
        self.capacity = capacity
        self.cache: Dict[str, torch.Tensor] = {}
        self.entanglement_network: Dict[str, set] = {}

    def store(self, signature: str, tensor: torch.Tensor):
        if len(self.cache) >= self.capacity:
            self._quantum_eviction()
        self.cache[signature] = tensor.detach().clone()
        self._update_entanglement(signature)

    def retrieve(self, signature: str, similarity_threshold: float = 0.8) -> Optional[torch.Tensor]:
        if signature in self.cache:
            return self.cache[signature]
        for stored_sig in self.cache.keys():
            if self._quantum_similarity(signature, stored_sig) > similarity_threshold:
                return self.cache[stored_sig]
        return None

    def _quantum_eviction(self):
        if not self.entanglement_network:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            return
        ent_scores = {k: len(v) for k, v in self.entanglement_network.items()}
        least = min(ent_scores, key=ent_scores.get)
        if least in self.cache: del self.cache[least]
        if least in self.entanglement_network: del self.entanglement_network[least]

    def _update_entanglement(self, signature: str):
        self.entanglement_network.setdefault(signature, set())
        for other_sig in list(self.cache.keys()):
            if other_sig != signature and self._quantum_similarity(signature, other_sig) > 0.7:
                self.entanglement_network[signature].add(other_sig)
                self.entanglement_network.setdefault(other_sig, set()).add(signature)

    def _quantum_similarity(self, a: str, b: str) -> float:
        if len(a) != len(b): return 0.0
        return sum(c1 == c2 for c1, c2 in zip(a, b)) / len(a)

class BiologicalMemorySystem:
    """Biological memory with evolution and selection pressure."""
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.population: List[Dict[str, Any]] = []
        self.fitness_history: List[float] = []
        self.generation = 0
        self.max_population = 100

    def evolve_population(self, encoding: Dict[str, Any], fitness: float):
        individual = {"encoding": encoding, "fitness": float(fitness), "generation": self.generation, "timestamp": time.time()}
        self.population.append(individual)
        self.fitness_history.append(float(fitness))
        if len(self.population) > self.max_population:
            self._natural_selection()
        self.generation += 1

    def _natural_selection(self):
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        survivors = self.population[: self.max_population // 2]
        tail = self.population[self.max_population // 2 :]
        if tail:
            keep = min(10, len(tail))
            idx = np.random.choice(len(tail), size=keep, replace=False)
            survivors += [tail[i] for i in idx]
        self.population = survivors

    def get_best_individuals(self, n: int = 5) -> List[Dict[str, Any]]:
        if not self.population: return []
        return sorted(self.population, key=lambda x: x["fitness"], reverse=True)[:n]

class MetacognitiveMemory:
    """Metacognitive memory for self-reflection and insights."""
    def __init__(self):
        self.insights = deque(maxlen=1000)
        self.performance_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.improvement_strategies: List[Dict[str, Any]] = []

    def record_insight(self, insight: Dict[str, Any]):
        insight["insight_id"] = len(self.insights)
        self.insights.append(insight)
        self._analyze_patterns(insight)

    def _analyze_patterns(self, insight: Dict[str, Any]):
        eff = insight.get("processing_efficiency", 1.0)
        cx = insight.get("input_complexity", 0.5)
        if eff > 1.2 and cx < 0.3:
            key = "efficient_simple"
        elif eff < 0.8 and cx > 0.7:
            key = "inefficient_complex"
        else:
            key = "balanced"
        self.performance_patterns.setdefault(key, []).append(insight)

    def generate_improvement_strategies(self) -> List[Dict[str, Any]]:
        strategies: List[Dict[str, Any]] = []
        for key, ins in self.performance_patterns.items():
            if len(ins) >= 5:
                avg_eff = float(np.mean([i.get("processing_efficiency", 1.0) for i in ins]))
                avg_perf = float(np.mean([i.get("predicted_performance", 0.5) for i in ins]))
                strategies.append({
                    "pattern": key,
                    "recommendation": self._recommend(key, avg_eff, avg_perf),
                    "confidence": min(len(ins) / 20.0, 1.0),
                    "sample_size": len(ins)
                })
        self.improvement_strategies = strategies
        return strategies

    def _recommend(self, key: str, eff: float, perf: float) -> str:
        if key == "efficient_simple":
            return "Maintain high-efficiency path; slowly raise tolerated complexity with guardrails."
        if key == "inefficient_complex":
            return "Reduce complexity (dim trim/sparsity), then re-tune attention depth."
        return "Stay balanced; nudge exploration when variance < 0.05."

# -----------------------------------------------------------------------------
# Advanced Synergy Module
# -----------------------------------------------------------------------------
class AdvancedSynergyModule(nn.Module):
    """Advanced synergy module integrating quantum, biological, and metacognitive processing."""
    def __init__(self, config: AdvancedAGIConfig):
        super().__init__()
        self.config = config
        D = config.feature_dim

        # Multi-modal processors
        self.quantum_processor = self._build_quantum_processor(D)
        self.neuromorphic_processor = self._build_neuromorphic_processor(D)
        self.bio_processor = self._build_bio_processor(D)
        self.metacognitive_processor = self._build_metacognitive_processor(D)

        # Synergy integration network
        self.synergy_integration = nn.ModuleDict({
            'quantum_neural_bridge': nn.Sequential(
                nn.Linear(D * 2, D), nn.GELU(), nn.LayerNorm(D)
            ),
            'bio_quantum_fusion': nn.Sequential(
                nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(0.1), nn.Linear(D, D)
            ),
            'meta_integration': nn.MultiheadAttention(embed_dim=D, num_heads=8, dropout=0.1, batch_first=True),
            'final_synthesis': nn.Sequential(
                nn.Linear(D * 4, D * 2), nn.GELU(), nn.Linear(D * 2, D), nn.LayerNorm(D)
            )
        })

        # Memories
        self.quantum_memory = QuantumInspiredMemory(D, config.cache_capacity)
        self.bio_memory = BiologicalMemorySystem(D)
        self.metacognitive_memory = MetacognitiveMemory()

        # Performance tracking
        self.synergy_history = deque(maxlen=config.performance_history_size)
        self.performance_tracker = defaultdict(list)

        # Mode routing
        self.processing_modes = {
            ProcessingMode.QUANTUM_ENHANCED: self._quantum_enhanced_processing,
            ProcessingMode.NEUROMORPHIC: self._neuromorphic_processing,
            ProcessingMode.HYBRID_SYNERGY: self._hybrid_synergy_processing,
            ProcessingMode.BIO_INSPIRED: self._bio_inspired_processing,
            ProcessingMode.METACOGNITIVE: self._metacognitive_processing
        }

        # Fixed RP for signature stability
        torch.manual_seed(1337)
        self._rp_dim = min(64, D)
        self.register_buffer("_rp_matrix", torch.randn(D, self._rp_dim) / (self._rp_dim ** 0.5))

    # ---- builders ----
    def _build_quantum_processor(self, feature_dim: int) -> nn.Module:
        return nn.ModuleDict({
            'superposition_layer': nn.Linear(feature_dim, feature_dim * 2),
            'entanglement_matrix': nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.1),
            'measurement_basis': nn.Linear(feature_dim * 2, feature_dim),
            'coherence_controller': nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4), nn.Tanh(), nn.Linear(feature_dim // 4, feature_dim)
            )
        })

    def _build_neuromorphic_processor(self, feature_dim: int) -> nn.Module:
        return nn.ModuleDict({
            'spike_encoder': nn.Linear(feature_dim, feature_dim),
            'membrane_dynamics': nn.GRUCell(feature_dim, feature_dim),
            'synaptic_weights': nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.05),
            'spike_decoder': nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        })

    def _build_bio_processor(self, feature_dim: int) -> nn.Module:
        return nn.ModuleDict({
            'dna_encoder': nn.Embedding(4, feature_dim // 4),  # A,T,G,C
            'protein_folder': nn.LSTM(feature_dim // 4, feature_dim // 2, batch_first=True),
            'evolutionary_selector': nn.MultiheadAttention(feature_dim // 2, 4, batch_first=True),
            'fitness_evaluator': nn.Sequential(
                nn.Linear(feature_dim // 2, feature_dim // 4), nn.Sigmoid(), nn.Linear(feature_dim // 4, 1)
            )
        })

    def _build_metacognitive_processor(self, feature_dim: int) -> nn.Module:
        return nn.ModuleDict({
            'self_monitor': nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Linear(feature_dim // 2, feature_dim)),
            'performance_predictor': nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Dropout(0.2), nn.Linear(feature_dim, 1), nn.Sigmoid()),
            'strategy_optimizer': nn.Transformer(d_model=feature_dim, nhead=8, num_encoder_layers=2,
                                                 num_decoder_layers=2, dim_feedforward=feature_dim * 2, batch_first=True)
        })

    # ---- forward & modes ----
    def forward(self, x: torch.Tensor, processing_mode: Optional[ProcessingMode] = None) -> SynergyResult:
        x = x.to(torch.float32)
        mode = processing_mode or self.config.processing_mode
        processor_fn = self.processing_modes.get(mode, self._hybrid_synergy_processing)
        result = processor_fn(x)
        self._record_performance(result, mode)
        return result

    def _quantum_enhanced_processing(self, x: torch.Tensor) -> SynergyResult:
        B, D = x.shape
        superposition = self.quantum_processor['superposition_layer'](x).view(B, 2, D)
        entangled = torch.einsum('bij,jk->bik', superposition, self.quantum_processor['entanglement_matrix'])
        coherence = self.quantum_processor['coherence_controller'](x).clamp(-1.0, 1.0)
        measured_state = self.quantum_processor['measurement_basis'](entangled.flatten(1))
        output = coherence * measured_state + (1 - coherence) * x

        quantum_signature = self._compute_quantum_signature(output)
        self.quantum_memory.store(quantum_signature, output)

        return SynergyResult("quantum_enhanced", torch.mean(coherence).item(), {
            "quantum_state": output, "coherence": coherence, "entanglement": entangled, "quantum_signature": quantum_signature
        })

    def _neuromorphic_processing(self, x: torch.Tensor) -> SynergyResult:
        B, D = x.shape
        spike_input = torch.sigmoid(self.neuromorphic_processor['spike_encoder'](x))
        hidden_state = torch.zeros(B, D, device=x.device)
        spike_outputs = []
        for _ in range(5):
            hidden_state = 0.95 * hidden_state + self.neuromorphic_processor['membrane_dynamics'](spike_input, hidden_state)
            spikes = (hidden_state > self.config.spike_threshold).float()
            spike_outputs.append(spikes)
            hidden_state = hidden_state * (1 - spikes) + 0.01 * torch.randn_like(hidden_state)
            spike_input = spikes @ self.neuromorphic_processor['synaptic_weights']
        spike_train = torch.stack(spike_outputs, dim=1)
        output = self.neuromorphic_processor['spike_decoder'](torch.mean(spike_train, dim=1))
        return SynergyResult("neuromorphic", torch.mean(spike_train).item(), {
            "spike_output": output, "spike_train": spike_train, "final_membrane_state": hidden_state
        })

    def _bio_inspired_processing(self, x: torch.Tensor) -> SynergyResult:
        B, D = x.shape
        dna_idx = (torch.sigmoid(x) * 4).long().clamp(0, 3)                       # [B,D]
        dna_emb = self.bio_processor['dna_encoder'](dna_idx)                      # [B,D,Emb]
        folded, _ = self.bio_processor['protein_folder'](dna_emb)                 # [B,D,Hidden]
        attended, att = self.bio_processor['evolutionary_selector'](folded, folded, folded)  # [B,D,H], [B,4,D,D] under the hood
        fitness = self.bio_processor['fitness_evaluator'](attended)               # [B,D,1]

        best_idx = torch.argmax(fitness.squeeze(-1), dim=1, keepdim=True)         # [B,1]
        selected = torch.gather(
            attended, 1, best_idx.unsqueeze(-1).expand(-1, -1, attended.size(-1))
        ).squeeze(1)                                                              # [B,H]

        bio_encoding = self._compute_bio_encoding(selected)
        self.bio_memory.evolve_population(bio_encoding, fitness.mean().item())

        return SynergyResult("bio_inspired", fitness.mean().item(), {
            "evolved_output": selected, "fitness_scores": fitness, "attention_weights": att, "bio_encoding": bio_encoding
        })

    def _metacognitive_processing(self, x: torch.Tensor) -> SynergyResult:
        B, D = x.shape
        self_state = self.metacognitive_processor['self_monitor'](x)
        combined = torch.cat([x, self_state], dim=-1)
        predicted = self.metacognitive_processor['performance_predictor'](combined)
        src = x.unsqueeze(1)
        tgt = self_state.unsqueeze(1)
        optimized = self.metacognitive_processor['strategy_optimizer'](src, tgt).squeeze(1)
        insights = self._extract_metacognitive_insights(x, optimized, predicted)
        self.metacognitive_memory.record_insight(insights)
        return SynergyResult("metacognitive", predicted.mean().item(), {
            "metacognitive_output": optimized, "self_state": self_state, "predicted_performance": predicted, "insights": insights
        })

    def _hybrid_synergy_processing(self, x: torch.Tensor) -> SynergyResult:
        q = self._quantum_enhanced_processing(x)
        n = self._neuromorphic_processing(x)
        b = self._bio_inspired_processing(x)
        m = self._metacognitive_processing(x)

        q_out = q.components["quantum_state"]
        n_out = n.components["spike_output"]
        b_out = b.components["evolved_output"]
        m_out = m.components["metacognitive_output"]

        qn = self.synergy_integration['quantum_neural_bridge'](torch.cat([q_out, n_out], dim=-1))
        bq = self.synergy_integration['bio_quantum_fusion'](torch.cat([b_out, q_out], dim=-1))
        meta_inputs = torch.stack([qn, bq, m_out], dim=1)
        meta_int, att = self.synergy_integration['meta_integration'](meta_inputs, meta_inputs, meta_inputs)
        meta_final = meta_int.mean(dim=1)

        all_modal = torch.cat([qn, bq, meta_final, x], dim=-1)
        final_output = self.synergy_integration['final_synthesis'](all_modal)

        confidences = [q.confidence, n.confidence, b.confidence, m.confidence]
        overall_conf = float(np.mean(confidences))

        return SynergyResult("hybrid_synergy", overall_conf, {
            "final_output": final_output,
            "quantum_component": q.components,
            "neuromorphic_component": n.components,
            "bio_component": b.components,
            "metacognitive_component": m.components,
            "attention_weights": att
        })

    # ---- helpers ----
    @torch.no_grad()
    def _compute_quantum_signature(self, tensor: torch.Tensor) -> str:
        """Stable & fast signature via fixed random projection stats."""
        z = tensor @ self._rp_matrix  # [B, k]
        stats = torch.stack([z.mean(dim=0), z.std(dim=0).clamp_min(1e-8)], dim=0).flatten()
        return hashlib.md5(stats.detach().cpu().numpy().tobytes()).hexdigest()

    def _compute_bio_encoding(self, tensor: torch.Tensor) -> Dict[str, Any]:
        return {
            "dna_sequence": self._tensor_to_dna_sequence(tensor),
            "protein_signature": self._compute_protein_signature(tensor),
            "evolutionary_fitness": float(torch.norm(tensor).item())
        }

    def _tensor_to_dna_sequence(self, tensor: torch.Tensor) -> str:
        bases = ['A', 'T', 'G', 'C']
        q = (torch.sigmoid(tensor.flatten()) * 4).long().clamp(0, 3)
        return ''.join(bases[i.item()] for i in q[: self.config.dna_sequence_length])

    def _compute_protein_signature(self, tensor: torch.Tensor) -> str:
        amino = 'ACDEFGHIKLMNPQRSTVWY'
        q = (torch.sigmoid(tensor.flatten()) * 20).long().clamp(0, 19)
        return ''.join(amino[i.item()] for i in q[: self.config.protein_folding_complexity])

    def _extract_metacognitive_insights(self, inp: torch.Tensor, out: torch.Tensor, perf: torch.Tensor) -> Dict[str, Any]:
        return {
            "input_complexity": float(torch.std(inp).item()),
            "output_complexity": float(torch.std(out).item()),
            "transformation_magnitude": float(torch.norm(out - inp).item()),
            "predicted_performance": float(perf.mean().item()),
            "processing_efficiency": float(torch.norm(out).item() / (torch.norm(inp).item() + 1e-8)),
            "timestamp": time.time()
        }

    def _record_performance(self, result: SynergyResult, mode: ProcessingMode):
        self.synergy_history.append({"timestamp": result.timestamp, "mode": mode.value, "confidence": result.confidence,
                                     "synergy_type": result.synergy_type})
        self.performance_tracker[mode.value].append(result.confidence)

    def get_synergy_stats(self) -> Dict[str, Any]:
        if not self.synergy_history:
            return {"error": "No synergy operations recorded"}
        mode_stats = {}
        for mode in ProcessingMode:
            vals = self.performance_tracker.get(mode.value, [])
            if vals:
                mode_stats[mode.value] = {
                    "count": len(vals),
                    "mean_confidence": float(np.mean(vals)),
                    "std_confidence": float(np.std(vals)),
                    "max_confidence": float(np.max(vals)),
                    "min_confidence": float(np.min(vals))
                }
        return {
            "total_operations": len(self.synergy_history),
            "mode_statistics": mode_stats,
            "recent_performance": list(self.synergy_history)[-10:],
            "memory_usage": {
                "quantum_memory": len(self.quantum_memory.cache),
                "bio_memory": len(self.bio_memory.population),
                "metacognitive_insights": len(self.metacognitive_memory.insights)
            }
        }

# -----------------------------------------------------------------------------
# Swarm Intelligence
# -----------------------------------------------------------------------------
class SwarmAgent:
    """Individual agent in the swarm intelligence system."""
    def __init__(self, agent_id: str, config: AdvancedAGIConfig):
        self.agent_id = agent_id
        self.config = config
        self.position = torch.randn(config.feature_dim) * 0.1
        self.velocity = torch.zeros(config.feature_dim)
        self.best_position = self.position.clone()
        self.best_fitness = float('-inf')
        self.fitness_history = deque(maxlen=100)
        self.specialization = np.random.choice(['quantum', 'neuromorphic', 'bio', 'meta'])
        self.learning_rate = 0.01 + np.random.random() * 0.09
        self.exploration_factor = 0.1 + np.random.random() * 0.4

    def update_position(self, global_best: torch.Tensor, local_neighbors: List['SwarmAgent']):
        if local_neighbors:
            neighbor_center = torch.stack([n.position for n in local_neighbors]).mean(dim=0)
            neighbor_influence = 0.3 * (neighbor_center - self.position)
        else:
            neighbor_influence = torch.zeros_like(self.position)

        inertia = 0.7
        cognitive_factor = 2.0
        social_factor = 2.0
        cog = cognitive_factor * torch.rand_like(self.position) * (self.best_position - self.position)
        soc = social_factor * torch.rand_like(self.position) * (global_best - self.position)
        self.velocity = inertia * self.velocity + cog + soc + neighbor_influence
        self.velocity += torch.randn_like(self.position) * self.exploration_factor
        self.position = torch.clamp(self.position + self.learning_rate * self.velocity, -2.0, 2.0)

    def evaluate_fitness(self, test_input: torch.Tensor, synergy_module: AdvancedSynergyModule) -> float:
        modified_input = test_input + 0.1 * self.position.unsqueeze(0)
        mode_map = {'quantum': ProcessingMode.QUANTUM_ENHANCED,
                    'neuromorphic': ProcessingMode.NEUROMORPHIC,
                    'bio': ProcessingMode.BIO_INSPIRED,
                    'meta': ProcessingMode.METACOGNITIVE}
        mode = mode_map.get(self.specialization, ProcessingMode.HYBRID_SYNERGY)
        with torch.no_grad():
            result = synergy_module(modified_input, mode)
        base = result.components.get("final_output", modified_input)
        fitness = float(result.confidence) * (1.0 + 0.1 * torch.norm(base).item())
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.clone()
        self.fitness_history.append(fitness)
        return fitness

class SwarmIntelligenceSystem:
    """Distributed swarm intelligence for parameter optimization."""
    def __init__(self, config: AdvancedAGIConfig):
        self.config = config
        self.agents = [SwarmAgent(f"agent_{i}", config) for i in range(config.num_swarm_agents)]
        self.global_best_position = torch.randn(config.feature_dim) * 0.1
        self.global_best_fitness = float('-inf')
        self.communication_graph = self._build_communication_graph()
        self.generation = 0
        self.convergence_history: List[float] = []
        self.diversity_history: List[float] = []

    def _build_communication_graph(self) -> Dict[str, List[str]]:
        graph = defaultdict(list)
        for a in self.agents:
            for b in self.agents:
                if a.agent_id != b.agent_id:
                    distance = torch.norm(a.position - b.position).item()
                    if distance <= self.config.communication_range:
                        graph[a.agent_id].append(b.agent_id)
        return graph

    def evolve_swarm(self, test_input: torch.Tensor, synergy_module: AdvancedSynergyModule) -> Dict[str, Any]:
        fitness_scores = []
        for agent in self.agents:
            fitness = agent.evaluate_fitness(test_input, synergy_module)
            fitness_scores.append(fitness)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = agent.position.clone()

        for agent in self.agents:
            neighbor_ids = self.communication_graph.get(agent.agent_id, [])
            neighbors = [a for a in self.agents if a.agent_id in neighbor_ids]
            agent.update_position(self.global_best_position, neighbors)

        if self.generation % 10 == 0:
            self.communication_graph = self._build_communication_graph()

        swarm_diversity = self._calculate_diversity()
        convergence_rate = self._calculate_convergence()
        self.generation += 1
        self.diversity_history.append(swarm_diversity)
        self.convergence_history.append(convergence_rate)

        return {
            "generation": self.generation,
            "global_best_fitness": self.global_best_fitness,
            "mean_fitness": float(np.mean(fitness_scores)),
            "fitness_std": float(np.std(fitness_scores)),
            "swarm_diversity": swarm_diversity,
            "convergence_rate": convergence_rate,
            "best_agent_specialization": self._get_best_agent_specialization()
        }

    def _calculate_diversity(self) -> float:
        positions = torch.stack([a.position for a in self.agents])
        center = positions.mean(dim=0)
        distances = torch.norm(positions - center.unsqueeze(0), dim=1)
        return float(distances.std().item())

    def _calculate_convergence(self) -> float:
        if len(self.convergence_history) < 2:
            return 0.0
        recent = [
            max(0, self.convergence_history[-i] - self.convergence_history[-i-1])
            for i in range(1, min(6, len(self.convergence_history)))
        ]
        return float(np.mean(recent)) if recent else 0.0

    def _get_best_agent_specialization(self) -> str:
        best_agent = max(self.agents, key=lambda a: a.best_fitness)
        return best_agent.specialization

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        recs = []
        perf = defaultdict(list)
        for a in self.agents:
            perf[a.specialization].append(a.best_fitness)
        for spec, vals in perf.items():
            m = float(np.mean(vals)) if vals else 0.0
            recs.append({
                "type": "specialization_performance",
                "specialization": spec,
                "mean_fitness": m,
                "recommendation": f"{'Increase' if m > self.global_best_fitness * 0.8 else 'Decrease'} focus on {spec} processing"
            })
        if len(self.convergence_history) >= 10:
            rc = float(np.mean(self.convergence_history[-5:]))
            if rc < 0.01:
                recs.append({
                    "type": "convergence_analysis",
                    "issue": "low_convergence",
                    "recommendation": "Increase exploration factors or restart swarm with new initialization"
                })
        return recs

# -----------------------------------------------------------------------------
# Distributed Processing System
# -----------------------------------------------------------------------------
import queue

class ProcessingNode:
    """Individual processing node in distributed system."""
    def __init__(self, node_id: str, config: AdvancedAGIConfig):
        self.node_id = node_id
        self.config = config
        self.task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self.result_cache: Dict[str, Any] = {}
        self.processing_stats = {
            "tasks_completed": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.is_active = True

    def process_task(self, task: Dict[str, Any], synergy_module: AdvancedSynergyModule) -> Dict[str, Any]:
        t0 = time.time()
        task_id = task.get("task_id", f"task_{int(time.time())}")
        # Cache key from raw bytes (fast)
        arr = task["input_data"].detach().cpu().numpy()
        input_hash = hashlib.md5(arr.tobytes()).hexdigest()
        mode_hash = task.get("processing_mode", "hybrid")
        cache_key = f"{input_hash}_{mode_hash}"

        if cache_key in self.result_cache:
            self.processing_stats["cache_hits"] += 1
            res = self.result_cache[cache_key].copy()
            res["from_cache"] = True
            res["node_id"] = self.node_id
            return res
        self.processing_stats["cache_misses"] += 1

        try:
            mode = ProcessingMode(task.get("processing_mode", "hybrid"))
            with torch.no_grad():
                result = synergy_module(task["input_data"], mode)
            processed = {
                "task_id": task_id,
                "node_id": self.node_id,
                "result": result,
                "processing_time": time.time() - t0,
                "from_cache": False,
                "status": "success"
            }
            self.result_cache[cache_key] = processed.copy()
            if len(self.result_cache) > 1000:
                self.result_cache.pop(next(iter(self.result_cache)))
            self.processing_stats["tasks_completed"] += 1
            self.processing_stats["total_processing_time"] += processed["processing_time"]
            tc = self.processing_stats["tasks_completed"]
            self.processing_stats["average_response_time"] = self.processing_stats["total_processing_time"] / max(1, tc)
            return processed
        except Exception as e:
            return {"task_id": task_id, "node_id": self.node_id, "error": str(e),
                    "processing_time": time.time() - t0, "status": "error"}

    def get_load_factor(self) -> float:
        qload = self.task_queue.qsize() / (self.task_queue.maxsize or 1)
        presp = min(self.processing_stats["average_response_time"] / 1.0, 1.0)
        return float((qload + presp) / 2)

class LoadBalancer:
    """Intelligent load balancer for distributed processing."""
    def __init__(self, nodes: List[ProcessingNode]):
        self.nodes = nodes
        self.routing_history = deque(maxlen=1000)

    def select_node(self, task: Dict[str, Any]) -> ProcessingNode:
        active = [n for n in self.nodes if n.is_active]
        if not active: raise RuntimeError("No active nodes available")
        scores = {}
        for n in active:
            scores[n.node_id] = self._score(n)
        best_id = max(scores, key=scores.get)
        chosen = next(n for n in active if n.node_id == best_id)
        self.routing_history.append({
            "timestamp": time.time(),
            "selected_node": best_id,
            "node_scores": scores,
            "task_type": task.get("processing_mode", "unknown")
        })
        return chosen

    def _score(self, node: ProcessingNode) -> float:
        load = 1.0 - node.get_load_factor()
        perf = 1.0 / (1.0 + node.processing_stats["average_response_time"])
        hits = node.processing_stats["cache_hits"]
        total = hits + node.processing_stats["cache_misses"]
        hit_rate = hits / total if total > 0 else 0.0
        return 0.4 * load + 0.4 * perf + 0.2 * hit_rate

class TaskScheduler:
    """Advanced task scheduler with priority and optimization."""
    def __init__(self):
        self.scheduling_history = deque(maxlen=1000)

    def schedule_tasks(self, tasks: List[Dict[str, Any]], nodes: List[ProcessingNode]) -> Dict[str, List[Dict[str, Any]]]:
        lb = LoadBalancer(nodes)
        scheduled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for task in sorted(tasks, key=lambda t: t.get("priority", 0), reverse=True):
            node = lb.select_node(task)
            scheduled[node.node_id].append(task)
        return dict(scheduled)

class DistributedProcessingSystem:
    """Distributed processing system for parallel AGI operations."""
    def __init__(self, config: AdvancedAGIConfig, num_nodes: int = 4):
        self.config = config
        self.nodes = [ProcessingNode(f"node_{i}", config) for i in range(num_nodes)]
        self.load_balancer = LoadBalancer(self.nodes)
        self.task_scheduler = TaskScheduler()
        self.system_stats = {
            "total_tasks_processed": 0,
            "total_processing_time": 0.0,
            "system_throughput": 0.0,
            "node_utilization": {}
        }

    def submit_batch_tasks(self, tasks: List[Dict[str, Any]], synergy_module: AdvancedSynergyModule) -> List[Dict[str, Any]]:
        results = []
        scheduled = self.task_scheduler.schedule_tasks(tasks, self.nodes)
        for node_id, node_tasks in scheduled.items():
            node = next(n for n in self.nodes if n.node_id == node_id)
            for task in node_tasks:
                results.append(node.process_task(task, synergy_module))
        self._update_system_stats(results)
        return results

    def _update_system_stats(self, results: List[Dict[str, Any]]):
        succ = [r for r in results if r.get("status") == "success"]
        self.system_stats["total_tasks_processed"] += len(results)
        total_time = sum(r.get("processing_time", 0.0) for r in succ)
        self.system_stats["total_processing_time"] += total_time
        self.system_stats["system_throughput"] = (len(succ) / total_time) if total_time > 0 else 0.0
        for n in self.nodes:
            hits = n.processing_stats["cache_hits"]
            total = hits + n.processing_stats["cache_misses"]
            self.system_stats["node_utilization"][n.node_id] = {
                "load_factor": n.get_load_factor(),
                "tasks_completed": n.processing_stats["tasks_completed"],
                "cache_hit_rate": hits / total if total > 0 else 0.0
            }

    def get_system_health(self) -> Dict[str, Any]:
        active = [n for n in self.nodes if n.is_active]
        avg_load = float(np.mean([n.get_load_factor() for n in active])) if active else 0.0
        return {
            "active_nodes": len(active),
            "total_nodes": len(self.nodes),
            "average_system_load": avg_load,
            "total_tasks_processed": self.system_stats["total_tasks_processed"],
            "system_throughput": self.system_stats["system_throughput"],
            "node_health": {
                n.node_id: {"active": n.is_active, "load": n.get_load_factor(),
                            "avg_response_time": n.processing_stats["average_response_time"]}
                for n in self.nodes
            }
        }

# -----------------------------------------------------------------------------
# Monitoring / Optimization
# -----------------------------------------------------------------------------
class PerformanceMonitor:
    def __init__(self):
        self.operation_history = deque(maxlen=1000)
        self.performance_trends = defaultdict(list)

    def record_operation(self, op: Dict[str, Any]):
        op["timestamp"] = time.time()
        self.operation_history.append(op)
        for k in ["processing_time", "confidence", "output_quality"]:
            if k in op: self.performance_trends[k].append(op[k])

    def get_recent_performance(self, n: int = 10) -> List[Dict[str, Any]]:
        return list(self.operation_history)[-n:]

    def analyze_trends(self) -> Dict[str, Any]:
        out = {}
        for metric, values in self.performance_trends.items():
            if len(values) >= 5:
                recent_avg = float(np.mean(values[-10:]))
                overall_avg = float(np.mean(values))
                trend = "improving" if recent_avg > overall_avg else "declining"
                out[metric] = {"recent_average": recent_avg, "overall_average": overall_avg,
                               "trend_direction": trend,
                               "volatility": float(np.std(values[-10:])) if len(values) >= 10 else 0.0}
        return out

class AutoOptimizer:
    def __init__(self, config: AdvancedAGIConfig):
        self.config = config
        self.optimization_history = deque(maxlen=100)

    def analyze_and_recommend(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(performance_data) < 5:
            return []
        recs = []
        times = [op.get("processing_time", 0.0) for op in performance_data]
        t_avg = float(np.mean(times))
        if t_avg > 2.0:
            recs.append({"type": "performance_optimization", "issue": "high_processing_time",
                         "action": "reduce_complexity",
                         "current_value": t_avg, "recommended_change": "Reduce max_iterations or enable gradient checkpointing"})
        conf = [op.get("confidence", 0.0) for op in performance_data]
        c_avg = float(np.mean(conf))
        if c_avg < 0.7:
            recs.append({"type": "quality_optimization", "issue": "low_confidence",
                         "action": "increase_complexity",
                         "current_value": c_avg, "recommended_change": "Increase synergy units or enable multi-modal processing"})
        qual = [op.get("output_quality", 0.0) for op in performance_data]
        if qual and float(np.mean(qual)) < 0.6:
            recs.append({"type": "quality_optimization", "issue": "low_output_quality",
                         "action": "enhance_processing",
                         "current_value": float(np.mean(qual)),
                         "recommended_change": "Enable hybrid synergy mode or increase feature dimensions"})
        return recs

class HealthMonitor:
    def __init__(self):
        self.health_metrics = {"last_check": time.time(), "system_errors": [], "warning_conditions": [], "performance_alerts": []}

    def get_health_status(self) -> Dict[str, Any]:
        now = time.time()
        delta = now - self.health_metrics["last_check"]
        errors = len(self.health_metrics["system_errors"])
        warns = len(self.health_metrics["warning_conditions"])
        if errors > 5: health = "critical"
        elif warns > 10: health = "degraded"
        elif delta > 300: health = "stale"
        else: health = "healthy"
        self.health_metrics["last_check"] = now
        return {
            "overall_health": health,
            "last_check": now,
            "error_count": errors,
            "warning_count": warns,
            "uptime_minutes": delta / 60,
            "recent_errors": self.health_metrics["system_errors"][-5:],
            "health_score": self._score()
        }

    def _score(self) -> float:
        base = 100.0
        return max(0.0, base - len(self.health_metrics["system_errors"]) * 10 - len(self.health_metrics["warning_conditions"]) * 2)

# -----------------------------------------------------------------------------
# High-Level Controller
# -----------------------------------------------------------------------------
class AdvancedAGIController:
    """Main controller orchestrating all AGI components with advanced capabilities."""
    def __init__(self, config: AdvancedAGIConfig):
        self.config = config
        logger.info(f"Initializing Advanced AGI Controller with {config.processing_mode.value} mode")
        self.synergy_module = AdvancedSynergyModule(config)
        self.swarm_system = SwarmIntelligenceSystem(config)
        self.distributed_system = DistributedProcessingSystem(config)
        self.performance_monitor = PerformanceMonitor()
        self.auto_optimizer = AutoOptimizer(config)
        self.health_monitor = HealthMonitor()
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.operation_count = 0
        self.total_processing_time = 0.0
        self._initialize_systems()

    def _initialize_systems(self):
        logger.info("Initializing AGI subsystems...")
        test_input = torch.randn(4, self.config.feature_dim)
        with torch.no_grad():
            warm = self.synergy_module(test_input)
        logger.info(f"Warmup completed - Confidence: {warm.confidence:.3f}")
        logger.info("All subsystems initialized successfully")

    def process_advanced(self, input_data: torch.Tensor, processing_mode: Optional[ProcessingMode] = None,
                         use_swarm_optimization: bool = True, use_distributed_processing: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        self.operation_count += 1
        mode = processing_mode or self.config.processing_mode
        results: Dict[str, Any] = {
            "session_id": self.session_id,
            "operation_id": self.operation_count,
            "processing_mode": mode.value,
            "input_shape": list(input_data.shape),
            "timestamp": time.time()
        }
        try:
            if use_swarm_optimization:
                swarm = self.swarm_system.evolve_swarm(input_data, self.synergy_module)
                results["swarm_optimization"] = swarm
                mod_strength = min(swarm["global_best_fitness"] / 10.0, 0.2) if swarm["global_best_fitness"] > 0 else 0.0
                optimized_input = input_data + mod_strength * self.swarm_system.global_best_position.unsqueeze(0)
            else:
                optimized_input = input_data

            if use_distributed_processing:
                tasks = [{"task_id": f"task_{i}", "input_data": optimized_input[i:i+1],
                          "processing_mode": mode.value} for i in range(optimized_input.shape[0])]
                dist_res = self.distributed_system.submit_batch_tasks(tasks, self.synergy_module)
                succ = [r for r in dist_res if r.get("status") == "success"]
                if succ:
                    avg_conf = float(np.mean([r["result"].confidence for r in succ]))
                else:
                    avg_conf = 0.0
                synergy_result = SynergyResult("distributed_combined", avg_conf, {
                    "distributed_results": [r["result"].components for r in succ],
                    "num_successful": len(succ),
                    "processing_nodes": [r["node_id"] for r in succ]
                })
                results["distributed_processing"] = {
                    "num_tasks": len(tasks),
                    "successful_tasks": len(succ),
                    "total_processing_time": sum(r.get("processing_time", 0.0) for r in dist_res)
                }
            else:
                synergy_result = self.synergy_module(optimized_input, mode)

            proc_time = time.time() - t0
            self.total_processing_time += proc_time
            self.performance_monitor.record_operation({
                "operation_id": self.operation_count,
                "processing_time": proc_time,
                "confidence": synergy_result.confidence,
                "mode": mode.value,
                "input_complexity": float(torch.std(input_data).item()),
                "output_quality": self._assess_output_quality(synergy_result)
            })

            if self.config.auto_optimization_enabled and self.operation_count % 10 == 0:
                results["optimization_recommendations"] = self.auto_optimizer.analyze_and_recommend(
                    self.performance_monitor.get_recent_performance(10)
                )

            results.update({
                "synergy_result": synergy_result,
                "processing_time": proc_time,
                "total_session_time": self.total_processing_time,
                "system_health": self.health_monitor.get_health_status(),
                "performance_metrics": self._get_comprehensive_metrics()
            })
            return results

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            results.update({"error": str(e), "processing_time": time.time() - t0, "status": "failed"})
            return results

    def _assess_output_quality(self, result: SynergyResult) -> float:
        base = float(result.confidence)
        emergent_bonus = 0.0
        if "emergent_properties" in result.components:
            emergent_bonus = float(result.components["emergent_properties"].get("synergy_strength", 0.0)) * 0.1
        integration_bonus = 0.1 if result.synergy_type == "hybrid_synergy" else 0.0
        return float(min(base + emergent_bonus + integration_bonus, 1.0))

    def _get_comprehensive_metrics(self) -> Dict[str, Any]:
        return {
            "session_metrics": {
                "operations_completed": self.operation_count,
                "total_processing_time": self.total_processing_time,
                "average_operation_time": self.total_processing_time / max(1, self.operation_count),
                "session_throughput": self.operation_count / max(self.total_processing_time, 1e-9)
            },
            "synergy_metrics": self.synergy_module.get_synergy_stats(),
            "swarm_metrics": {
                "generation": self.swarm_system.generation,
                "global_best_fitness": self.swarm_system.global_best_fitness,
                "swarm_diversity": self.swarm_system._calculate_diversity(),
                "active_agents": len([a for a in self.swarm_system.agents if a.best_fitness > 0])
            },
            "distributed_metrics": self.distributed_system.get_system_health()
        }

    def optimize_system(self) -> Dict[str, Any]:
        logger.info("Starting system optimization...")
        res = {"timestamp": time.time(), "optimizations_applied": []}
        swarm_recs = self.swarm_system.get_optimization_recommendations()
        auto_recs = self.auto_optimizer.analyze_and_recommend(self.performance_monitor.get_recent_performance(50))
        for r in swarm_recs + auto_recs:
            if self._apply_optimization(r):
                res["optimizations_applied"].append(r)
        res["post_optimization_health"] = self.health_monitor.get_health_status()
        logger.info(f"System optimization completed - {len(res['optimizations_applied'])} changes applied")
        return res

    def _apply_optimization(self, rec: Dict[str, Any]) -> bool:
        try:
            rtype = rec.get("type", "unknown")
            if rtype == "specialization_performance":
                target = rec["specialization"]
                if rec["recommendation"].startswith("Increase"):
                    for a in self.swarm_system.agents[:5]:
                        if np.random.random() < 0.3:
                            a.specialization = target
                    return True
            elif rtype == "performance_optimization":
                action = rec.get("action", "")
                if "increase_complexity" in action:
                    self.config.max_iterations = min(self.config.max_iterations + 1, 10)
                    return True
                if "reduce_complexity" in action:
                    self.config.max_iterations = max(self.config.max_iterations - 1, 3)
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to apply optimization {rec}: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "session_info": {
                "session_id": self.session_id,
                "operations_completed": self.operation_count,
                "total_runtime": self.total_processing_time,
                "current_mode": self.config.processing_mode.value
            },
            "component_status": {
                "synergy_module": "active",
                "swarm_system": f"generation_{self.swarm_system.generation}",
                "distributed_system": f"{len([n for n in self.distributed_system.nodes if n.is_active])}_active_nodes",
                "performance_monitor": f"{len(self.performance_monitor.operation_history)}_operations_tracked",
                "health_monitor": self.health_monitor.get_health_status()["overall_health"]
            },
            "performance_summary": self._get_comprehensive_metrics(),
            "memory_usage": self._get_memory_usage(),
            "optimization_status": {
                "auto_optimization_enabled": self.config.auto_optimization_enabled
            }
        }

    def _get_memory_usage(self) -> Dict[str, Any]:
        return {
            "estimated_total_mb": self.config.memory_budget_mb,
            "synergy_module": {
                "quantum_memory": len(self.synergy_module.quantum_memory.cache),
                "bio_memory": len(self.synergy_module.bio_memory.population),
                "metacognitive_memory": len(self.synergy_module.metacognitive_memory.insights)
            },
            "swarm_system": {
                "agents": len(self.swarm_system.agents),
                "communication_graph_size": sum(len(v) for v in self.swarm_system.communication_graph.values())
            },
            "distributed_system": {
                "total_cache_entries": sum(len(n.result_cache) for n in self.distributed_system.nodes)
            }
        }

# -----------------------------------------------------------------------------
# Demo / CLI
# -----------------------------------------------------------------------------
def create_demo_config() -> AdvancedAGIConfig:
    return AdvancedAGIConfig(
        feature_dim=256,
        num_synergy_units=6,
        max_iterations=4,
        processing_mode=ProcessingMode.HYBRID_SYNERGY,
        memory_budget_mb=1024,
        cache_capacity=500,
        quantization_bits=8,
        num_swarm_agents=16,
        auto_optimization_enabled=True
    )

def run_comprehensive_demo():
    print("🤖 Advanced Synergy-Enhanced AGI Framework Demo")
    print("=" * 60)
    config = create_demo_config()
    agi = AdvancedAGIController(config)
    print(f"\n✅ System initialized! Session ID: {agi.session_id}")

    batch_size = 8
    test_inputs = [
        torch.randn(batch_size, config.feature_dim) * 0.5,
        torch.randn(batch_size, config.feature_dim) * 1.5,
        torch.sin(torch.linspace(0, 10, batch_size * config.feature_dim)).reshape(batch_size, config.feature_dim),
    ]
    modes = [
        ProcessingMode.QUANTUM_ENHANCED,
        ProcessingMode.NEUROMORPHIC,
        ProcessingMode.BIO_INSPIRED,
        ProcessingMode.METACOGNITIVE,
        ProcessingMode.HYBRID_SYNERGY
    ]

    all_results = []
    for i, tin in enumerate(test_inputs):
        print(f"\n--- Test Input #{i+1} ---")
        for mode in modes:
            print(f"  Testing {mode.value} processing...")
            out = agi.process_advanced(
                tin,
                processing_mode=mode,
                use_swarm_optimization=(i == 2),
                use_distributed_processing=(mode == ProcessingMode.HYBRID_SYNERGY)
            )
            conf = out["synergy_result"].confidence
            t = out["processing_time"]
            print(f"    ✓ Confidence: {conf:.3f}, Time: {t:.3f}s")
            all_results.append(out)

    print(f"\n🔧 Performing system optimization...")
    opt = agi.optimize_system()
    print(f"  ✓ Applied {len(opt['optimizations_applied'])} optimizations")

    print(f"\n📊 Final System Status:")
    status = agi.get_system_status()
    print(f"  • Operations completed: {status['session_info']['operations_completed']}")
    print(f"  • Total processing time: {status['session_info']['total_runtime']:.2f}s")
    print(f"  • System health: {status['component_status']['health_monitor']}")
    print(f"  • Quantum memory entries: {status['performance_summary']['synergy_metrics']['memory_usage']['quantum_memory']}")
    sm = status['performance_summary']['session_metrics']
    print(f"  • Avg operation time: {sm['average_operation_time']:.3f}s | Throughput: {sm['session_throughput']:.2f} ops/sec")

    print(f"\n🎉 Demo completed successfully!")
    print(f"Advanced AGI Framework executed {len(all_results)} operations across {len(modes)} modalities.")
    return agi, all_results

if __name__ == "__main__":
    import argparse, sys
    # Determinism knobs for repeatable demos
    torch.manual_seed(42); np.random.seed(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Advanced Synergy-Enhanced AGI Framework Demo")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=[m.value for m in ProcessingMode],
                        help="Processing mode")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for demo step")
    parser.add_argument("--iters", type=int, default=1, help="Number of demo iterations")
    args = parser.parse_args()

    # If args requested one-off runs, do that; else run full demo
    if args.iters > 1 or args.mode != "hybrid":
        cfg = AdvancedAGIConfig(feature_dim=256, processing_mode=ProcessingMode(args.mode))
        core = AdvancedAGIController(cfg)
        all_stats = []
        for _ in range(max(1, args.iters)):
            x = torch.randn(args.batch, cfg.feature_dim)
            out = core.process_advanced(x, processing_mode=ProcessingMode(args.mode),
                                        use_swarm_optimization=True,
                                        use_distributed_processing=(args.mode == "hybrid"))
            all_stats.append({
                "confidence": out["synergy_result"].confidence,
                "latency_sec": out["processing_time"]
            })
        print(json.dumps({
            "iterations": len(all_stats),
            "summary": all_stats[-1],
            "session": core.get_system_status()
        }, indent=2))
    else:
        controller, results = run_comprehensive_demo()
        print(f"\n📈 Summary Statistics:")
        confs = [r["synergy_result"].confidence for r in results]
        times = [r["processing_time"] for r in results]
        print(f"  • Average confidence: {np.mean(confs):.3f} ± {np.std(confs):.3f}")
        print(f"  • Average processing time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  • Fastest: {np.min(times):.3f}s | Slowest: {np.max(times):.3f}s")
        print("\n✅ Run complete.")

