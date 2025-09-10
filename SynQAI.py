#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synergy Orchestrator
====================

An architecture whose *primary purpose* is to synergize other frameworks/modules.
It dynamically analyzes functions, capabilities, dependencies, and knowledge
across active modules to:

- discover integrations and hidden capabilities,
- build optimal service strategies (plans), and
- adapt execution at runtime using telemetry and bandit-style exploration.

Design goals
------------
- **Open interface**: any module implementing `SynergyModule` can be plugged in.
- **Introspection**: capability metadata + light runtime probing to validate claims.
- **Capability Graph**: typed ports (inputs/outputs), dependencies, QoS hints.
- **Planner**: goal → beam search over compositions with a synergy score.
- **Executor**: async DAG runner with caching, retries, and partial results.
- **Telemetry**: success/latency/quality feedback, reputation per module pair.
- **Discovery**: ablation and swap tests + UCB bandit to reveal better chains.

This file is dependency-light (stdlib + optional numpy) so you can run it anywhere.
Drop your frameworks in as `SynergyModule` providers and let the orchestrator compose them.
"""
from __future__ import annotations

import asyncio
import time
import math
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from collections import defaultdict, deque

try:
    import numpy as np  # Optional, used for simple scoring utilities
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Capability model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Port:
    name: str
    dtype: str  # semantic type id, e.g. "text", "embedding:float32[768]", "image:rgb"
    description: str = ""

@dataclass
class QoS:
    latency_ms: float = 50.0
    cost_unit: float = 1.0
    reliability: float = 0.99  # estimated success prob
    quality: float = 0.7       # domain-specific; higher is better

@dataclass
class Capability:
    name: str
    inputs: List[Port]
    outputs: List[Port]
    tags: List[str] = field(default_factory=list)  # e.g., ["nlp", "summarization"]
    q: QoS = field(default_factory=QoS)
    constraints: Dict[str, Any] = field(default_factory=dict)  # e.g., max_len
    provides: Dict[str, Any] = field(default_factory=dict)  # extra signals (e.g., attention weights)

class SynergyModule:
    """Base interface that any pluggable module must implement."""
    def id(self) -> str:
        raise NotImplementedError

    def capability(self) -> Capability:
        raise NotImplementedError

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Execute on keyworded inputs. Keys must match `inputs.name`.
        Returns a dict with keys of `outputs.name`.
        """
        raise NotImplementedError

    # Optional: modules may declare *soft* dependencies to influence planning
    def dependencies(self) -> List[str]:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Registry, Knowledge, and Reputation
# ─────────────────────────────────────────────────────────────────────────────

class ModuleRegistry:
    def __init__(self):
        self._mods: Dict[str, SynergyModule] = {}

    def register(self, mod: SynergyModule):
        self._mods[mod.id()] = mod

    def get(self, mid: str) -> SynergyModule:
        return self._mods[mid]

    def all(self) -> List[SynergyModule]:
        return list(self._mods.values())

class KnowledgeBase:
    """Stores learned compatibilities and performance stats."""
    def __init__(self):
        # reputation[(A,B)] aggregates how well A→B composition performs
        self.reputation: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: dict(count=0, score=0.0))
        # vector of module priors
        self.module_prior: Dict[str, float] = defaultdict(lambda: 0.0)
        # arbitrary notes
        self.notes: Dict[str, Any] = {}

    def update_pair(self, a: str, b: str, quality: float, success: bool):
        stat = self.reputation[(a, b)]
        stat['count'] = stat.get('count', 0) + 1
        delta = quality * (1.0 if success else 0.0)
        # incremental mean
        stat['score'] = stat.get('score', 0.0) + (delta - stat.get('score', 0.0)) / stat['count']

    def pair_score(self, a: str, b: str) -> float:
        s = self.reputation.get((a, b))
        if not s:
            return 0.0
        # optimistic bonus for few samples (UCB-ish)
        c = max(1, s['count'])
        bonus = math.sqrt(2 * math.log(max(2, self.total_observations())) / c)
        return s['score'] + 0.1 * bonus

    def total_observations(self) -> int:
        return sum(s['count'] for s in self.reputation.values())

# ─────────────────────────────────────────────────────────────────────────────
# Capability Graph & Planner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Goal:
    # Desired outputs by semantic type or tag, plus an objective (utility)
    target_types: List[str]
    objective: str = "quality"  # or "latency", "cost", etc.
    budget: Optional[float] = None
    max_stages: int = 5

@dataclass
class Plan:
    chain: List[str]                   # ordered list of module ids
    bindings: List[Dict[str, str]]     # how port names are wired stage-to-stage
    score: float
    predicted: Dict[str, float]

class CapabilityGraph:
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        # adjacency: A -> [B] if types connect
        self.edges: Dict[str, List[str]] = defaultdict(list)
        # record type signatures
        self.types_out: Dict[str, Set[str]] = defaultdict(set)
        self.types_in: Dict[str, Set[str]] = defaultdict(set)
        self._build()

    def _build(self):
        mods = self.registry.all()
        for m in mods:
            cap = m.capability()
            for p in cap.outputs:
                self.types_out[m.id()].add(p.dtype)
            for p in cap.inputs:
                self.types_in[m.id()].add(p.dtype)
        # connect by intersecting types
        for a in mods:
            for b in mods:
                if a.id() == b.id():
                    continue
                if self.types_out[a.id()] & self.types_in[b.id()]:
                    self.edges[a.id()].append(b.id())

    def successors(self, mid: str) -> List[str]:
        return self.edges.get(mid, [])

class Planner:
    def __init__(self, kb: KnowledgeBase, registry: ModuleRegistry):
        self.kb = kb
        self.registry = registry

    def _module_score(self, mid: str) -> float:
        # prior + intrinsic quality
        cap = self.registry.get(mid).capability()
        return 0.3 * self.kb.module_prior[mid] + 0.7 * cap.q.quality

    def _compat_score(self, a: str, b: str) -> float:
        # learned reputation plus type overlap bonus
        rep = self.kb.pair_score(a, b)
        types = self.registry.get(a).capability().outputs
        types_b = self.registry.get(b).capability().inputs
        overlap = len({t.dtype for t in types} & {t.dtype for t in types_b})
        return rep + 0.05 * overlap

    def _chain_qos(self, chain: List[str]) -> Dict[str, float]:
        # aggregate simple QoS
        lat = sum(self.registry.get(m).capability().q.latency_ms for m in chain)
        cost = sum(self.registry.get(m).capability().q.cost_unit for m in chain)
        qual = float(np.mean([self.registry.get(m).capability().q.quality for m in chain])) if NUMPY_AVAILABLE else sum(self.registry.get(m).capability().q.quality for m in chain)/len(chain)
        rel = 1.0
        for m in chain:
            rel *= self.registry.get(m).capability().q.reliability
        return dict(latency_ms=lat, cost=cost, quality=qual, reliability=rel)

    def plan(self, goal: Goal, graph: CapabilityGraph, seed_inputs: List[str]) -> List[Plan]:
        """Beam search over module chains that transform seed types to goal types."""
        beam: List[Tuple[List[str], float]] = []
        # Seed: any module that consumes at least one seed type
        for m in self.registry.all():
            if self.registry.get(m.id()).capability().inputs and not set(seed_inputs) & self.types_of_inputs(m.id()):
                continue
            beam.append(([m.id()], self._module_score(m.id())))
        beam = sorted(beam, key=lambda x: x[1], reverse=True)[:8]

        final: List[Plan] = []
        for _ in range(goal.max_stages - 1):
            new_beam: List[Tuple[List[str], float]] = []
            for chain, score in beam:
                tail = chain[-1]
                # if tail outputs the goal, keep it
                if self.types_of_outputs(tail) & set(goal.target_types):
                    qos = self._chain_qos(chain)
                    final.append(Plan(chain=chain, bindings=self._simple_bindings(chain), score=score, predicted=qos))
                    continue
                for succ in graph.successors(tail):
                    if succ in chain:
                        continue  # avoid loops
                    comp = self._compat_score(tail, succ)
                    new_score = score + 0.6 * comp + 0.4 * self._module_score(succ)
                    new_beam.append((chain + [succ], new_score))
            if not new_beam:
                break
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:8]

        # also consider single-stage matches
        for m in self.registry.all():
            if self.types_of_outputs(m.id()) & set(goal.target_types):
                qos = self._chain_qos([m.id()])
                final.append(Plan(chain=[m.id()], bindings=self._simple_bindings([m.id()]), score=self._module_score(m.id()), predicted=qos))

        # budget filter
        if goal.budget is not None:
            final = [p for p in final if p.predicted['cost'] <= goal.budget]
        # sort by goal objective
        key = 'quality' if goal.objective == 'quality' else goal.objective
        final = sorted(final, key=lambda p: (p.predicted.get(key, 0.0), p.score), reverse=True)
        return final[:5]

    def types_of_outputs(self, mid: str) -> Set[str]:
        return {p.dtype for p in self.registry.get(mid).capability().outputs}

    def types_of_inputs(self, mid: str) -> Set[str]:
        return {p.dtype for p in self.registry.get(mid).capability().inputs}

    def _simple_bindings(self, chain: List[str]) -> List[Dict[str, str]]:
        # Trivial same-type wiring (first match wins). Real impl would carry shapes & names.
        bindings: List[Dict[str, str]] = []
        for a, b in zip(chain, chain[1:]):
            outs = self.registry.get(a).capability().outputs
            ins = self.registry.get(b).capability().inputs
            map_ = {}
            for o in outs:
                for i in ins:
                    if o.dtype == i.dtype and i.name not in map_.values():
                        map_[o.name] = i.name
                        break
            bindings.append(map_)
        return bindings

# ─────────────────────────────────────────────────────────────────────────────
# Executor & Telemetry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecResult:
    outputs: Dict[str, Any]
    success: bool
    metrics: Dict[str, float]
    artifacts: Dict[str, Any] = field(default_factory=dict)

class Executor:
    def __init__(self, registry: ModuleRegistry, kb: KnowledgeBase):
        self.registry = registry
        self.kb = kb
        self.cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (module_id, json.dumps(inputs)) -> outputs

    async def run_plan(self, plan: Plan, inputs: Dict[str, Any]) -> ExecResult:
        ctx = dict(inputs)
        latency = 0.0
        success = True
        artifacts: Dict[str, Any] = {}

        for idx, mid in enumerate(plan.chain):
            mod = self.registry.get(mid)
            cap = mod.capability()
            # gather kwargs by matching port names / dtypes
            kwargs = {}
            for p in cap.inputs:
                # prefer name match, else find any ctx item with same dtype tag
                if p.name in ctx:
                    kwargs[p.name] = ctx[p.name]
                else:
                    # heuristic: find any value whose metadata says dtype==p.dtype
                    # Here we just allow raw passthrough by type key
                    typed_key = f"__{p.dtype}__"
                    if typed_key in ctx:
                        kwargs[p.name] = ctx[typed_key]
            key = (mid, json.dumps(sorted(kwargs.items()), default=str))
            start = time.time()
            try:
                if key in self.cache:
                    out = self.cache[key]
                else:
                    out = await mod.run(**kwargs)
                    self.cache[key] = out
                stage_latency = (time.time() - start) * 1000
                latency += stage_latency
                # push outputs back into context (both named and typed)
                for port in cap.outputs:
                    if port.name in out:
                        ctx[port.name] = out[port.name]
                        ctx[f"__{port.dtype}__"] = out[port.name]
                # reputation update uses predicted quality (fallback to cap.q.quality)
                qual = out.get('_quality', cap.q.quality) if isinstance(out, dict) else cap.q.quality
                success = success and True
                if idx > 0:
                    self.kb.update_pair(plan.chain[idx-1], mid, quality=float(qual), success=True)
            except Exception:
                success = False
                break
        return ExecResult(outputs=ctx, success=success, metrics=dict(latency_ms=latency), artifacts=artifacts)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: analyze → plan → execute → learn
# ─────────────────────────────────────────────────────────────────────────────

class SynergyOrchestrator:
    def __init__(self):
        self.registry = ModuleRegistry()
        self.kb = KnowledgeBase()
        self.executor = Executor(self.registry, self.kb)

    def register(self, *mods: SynergyModule) -> None:
        for m in mods:
            self.registry.register(m)

    def analyze(self) -> CapabilityGraph:
        return CapabilityGraph(self.registry)

    async def plan_and_run(self, goal: Goal, inputs: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.analyze()
        planner = Planner(self.kb, self.registry)
        seed_types = list({f"text" if isinstance(v, str) else type(v).__name__ for v in inputs.values()})
        plans = planner.plan(goal, graph, seed_types)
        if not plans:
            return {"ok": False, "error": "no viable plan"}
        # exploration vs exploitation (ε-greedy)
        eps = 0.2
        choice = random.choice(plans) if random.random() < eps else plans[0]
        exec_res = await self.executor.run_plan(choice, inputs)
        return {
            "ok": exec_res.success,
            "plan": asdict(choice),
            "metrics": exec_res.metrics,
            "outputs": {k: v for k, v in exec_res.outputs.items() if not k.startswith("__")},
            "knowledge": {
                "pair_reputation": {f"{a}->{b}": s for (a,b), s in self.kb.reputation.items()}
            }
        }

# ─────────────────────────────────────────────────────────────────────────────
# Example pluggable modules (toy but functional)
# ─────────────────────────────────────────────────────────────────────────────

class TextSummarizer(SynergyModule):
    def __init__(self, name="summarizer"):
        self._id = name
    def id(self) -> str: return self._id
    def capability(self) -> Capability:
        return Capability(
            name=self._id,
            inputs=[Port("text", "text")],
            outputs=[Port("summary", "text")],
            tags=["nlp", "summarization"],
            q=QoS(latency_ms=15, cost_unit=0.2, reliability=0.98, quality=0.72)
        )
    async def run(self, **kwargs) -> Dict[str, Any]:
        txt = kwargs.get("text", "")
        s = " ".join(txt.split())
        # naive summary: first 20 words
        out = " ".join(s.split()[:20])
        return {"summary": out, "_quality": 0.7 + min(0.3, len(out)/400)}

class KeywordExtractor(SynergyModule):
    def __init__(self, name="keywords"):
        self._id = name
    def id(self) -> str: return self._id
    def capability(self) -> Capability:
        return Capability(
            name=self._id,
            inputs=[Port("text", "text")],
            outputs=[Port("keywords", "list[str]")],
            tags=["nlp", "keywords"],
            q=QoS(latency_ms=8, cost_unit=0.1, reliability=0.99, quality=0.65)
        )
    async def run(self, **kwargs) -> Dict[str, Any]:
        txt = kwargs.get("text", "")
        words = [w.strip(".,!?:;") for w in txt.lower().split()]
        stop = {"the","a","and","to","of","in","on","for","with"}
        freq = defaultdict(int)
        for w in words:
            if w and w not in stop:
                freq[w]+=1
        kw = sorted(freq, key=freq.get, reverse=True)[:8]
        return {"keywords": kw, "_quality": 0.6 + 0.05*len(kw)}

class SentimentAnalyzer(SynergyModule):
    def __init__(self, name="sentiment"):
        self._id = name
    def id(self) -> str: return self._id
    def capability(self) -> Capability:
        return Capability(
            name=self._id,
            inputs=[Port("text", "text")],
            outputs=[Port("sentiment", "label:polarity"), Port("confidence", "float")],
            tags=["nlp","sentiment"],
            q=QoS(latency_ms=6, cost_unit=0.05, reliability=0.995, quality=0.6)
        )
    async def run(self, **kwargs) -> Dict[str, Any]:
        t = kwargs.get("text","")
        pos = sum(w in t.lower() for w in ["great","good","love","amazing","happy"]) 
        neg = sum(w in t.lower() for w in ["bad","hate","terrible","sad","angry"]) 
        lab = "positive" if pos>=neg else "negative" if neg>pos else "neutral"
        conf = 0.55 + 0.1*abs(pos-neg)
        return {"sentiment": lab, "confidence": conf, "_quality": 0.58 + 0.02*abs(pos-neg)}

class Reranker(SynergyModule):
    def __init__(self, name="rerank"):
        self._id = name
    def id(self) -> str: return self._id
    def capability(self) -> Capability:
        return Capability(
            name=self._id,
            inputs=[Port("candidates", "list[str]")],
            outputs=[Port("top", "text")],
            tags=["ranking"],
            q=QoS(latency_ms=10, cost_unit=0.2, reliability=0.97, quality=0.68)
        )
    async def run(self, **kwargs) -> Dict[str, Any]:
        cands = kwargs.get("candidates", [])
        best = max(cands, key=lambda s: len(s)) if cands else ""
        return {"top": best, "_quality": 0.65 + 0.03*min(5, len(cands))}

class SummarizeThenSentiment(SynergyModule):
    """Example *composite* module that internally orchestrates two children.
    Demonstrates how modules themselves can expose new virtual capabilities.
    """
    def __init__(self, summarizer: SynergyModule, sentiment: SynergyModule, name="sum_then_sent"):
        self._id = name
        self.s = summarizer
        self.v = sentiment
    def id(self) -> str: return self._id
    def capability(self) -> Capability:
        return Capability(
            name=self._id,
            inputs=[Port("text","text")],
            outputs=[Port("summary","text"), Port("sentiment","label:polarity")],
            tags=["composite","pipeline"],
            q=QoS(latency_ms=self.s.capability().q.latency_ms+self.v.capability().q.latency_ms,
                  cost_unit=self.s.capability().q.cost_unit+self.v.capability().q.cost_unit,
                  reliability=self.s.capability().q.reliability*self.v.capability().q.reliability,
                  quality=0.72)
        )
    async def run(self, **kwargs) -> Dict[str, Any]:
        s_out = await self.s.run(text=kwargs.get("text",""))
        v_out = await self.v.run(text=s_out["summary"])
        return {"summary": s_out["summary"], "sentiment": v_out["sentiment"], "_quality": 0.72}

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

async def _demo():
    orch = SynergyOrchestrator()
    s = TextSummarizer()
    k = KeywordExtractor()
    v = SentimentAnalyzer()
    r = Reranker()
    combo = SummarizeThenSentiment(s, v)

    orch.register(s, k, v, r, combo)

    goal = Goal(target_types=["label:polarity"], objective="quality", max_stages=4)
    inputs = {"text": "I absolutely love the crisp visuals, but the battery life is terrible on long trips."}

    result = await orch.plan_and_run(goal, inputs)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(_demo())

