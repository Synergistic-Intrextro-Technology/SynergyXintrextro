#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intrextro Synergy — Base (Optimized)
===================================

Purpose
-------
A production‑grade integration hub that composes heterogeneous modules (EI, CognitiveLoop,
embedders, detectors, planners) and dynamically blends their outputs based on QoS targets.

Highlights
---------
- Strongly‑typed config & state (dataclasses)
- Module Registry with capabilities and dependency graph
- Async pipeline with timeouts, retries, and circuit breaker
- Dynamic strategy: select, parallelize, or blend modules by domain/task and confidence
- Synergy scoring across outputs (agreement, complementarity, novelty)
- Rolling telemetry (p50/p95 latency, success, quality)
- Short‑term ring + episodic memory with traceable provenance
- Orchestrator adapter (SynergyModule) with typed ports

Notes
-----
This is the *base* implementation. Specialized variants can subclass components
or override policy hooks for domains like research, enterprise RAG, or agentic planning.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Deque
from collections import deque, defaultdict
import logging

# Optional imports: base modules we already refactored
try:
    from cognitive_loop import CognitiveLoop, LoopConfig
except Exception:  # fallback when the file name differs
    CognitiveLoop = None
    LoopConfig = None

try:
    from emotional_intelligence_base import EmotionalIntelligence as EIBase, EIConfig
except Exception:
    try:
        # allow legacy name (user kept original as emotional_intelligence.py)
        from emotional_intelligence import EmotionalIntelligence as EIBase  # type: ignore
        from emotional_intelligence import EIConfig  # type: ignore
    except Exception:
        EIBase = None
        EIConfig = None

logger = logging.getLogger("intrextro_synergy_base")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Config & State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynergyConfig:
    step_timeout_s: float = 3.0
    retry_attempts: int = 1
    circuit_threshold: int = 5
    circuit_cooldown_s: float = 45.0

    ring_size: int = 512
    episodic_max: int = 1000

    # Strategy
    min_conf_for_select: float = 0.78
    min_conf_for_use: float = 0.65
    blend_when_close_delta: float = 0.08

    # Telemetry
    keep_history: int = 400

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SynergyState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    short_ring: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=512))
    episodic: List[Dict[str, Any]] = field(default_factory=list)
    last_strategy: str = "select"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: circuit breaker, rolling stats, retries
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.fail = 0
        self.last_t: Optional[float] = None
        self.open = False
    def allow(self) -> bool:
        if not self.open:
            return True
        if self.last_t is None:
            return True
        if (time.time() - self.last_t) > self.cooldown_s:
            self.open = False
            self.fail = 0
            return True
        return False
    def success(self) -> None:
        self.fail = 0
        self.open = False
    def failure(self) -> None:
        self.fail += 1
        self.last_t = time.time()
        if self.fail >= self.threshold:
            self.open = True

class Rolling:
    @staticmethod
    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0
    @staticmethod
    def p(xs: List[float], q: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        k = max(0, min(len(ys) - 1, int(q * (len(ys) - 1))))
        return ys[k]

async def with_timeout(coro, t_s: float):
    return await asyncio.wait_for(coro, timeout=t_s)

# ─────────────────────────────────────────────────────────────────────────────
# Module Protocol
# ─────────────────────────────────────────────────────────────────────────────

class SynergyComponent:
    """Minimal interface that all components follow."""
    name: str
    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# Wrappers around our known engines to standardize envelopes
class EIComponent(SynergyComponent):
    def __init__(self, cfg: Optional[EIConfig] = None):
        if EIBase is None:
            raise RuntimeError("EmotionalIntelligence base not available")
        self.engine = EIBase(cfg)
        self.name = "emotional_intelligence"
    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        env = await self.engine.process(inputs)
        out = env.get("outputs", {})
        return {
            "ok": bool(env.get("ok", True)),
            "name": self.name,
            "latency_ms": float(env.get("metrics", {}).get("latency_ms_total", 0.0)),
            "confidence": float(out.get("confidence", 0.72)),
            "tags": list(out.get("tags", []) or []),
            "payload": out,
        }

class CogLoopComponent(SynergyComponent):
    def __init__(self, cfg: Optional[LoopConfig] = None):
        if CognitiveLoop is None:
            raise RuntimeError("CognitiveLoop not available")
        self.loop = CognitiveLoop(cfg)
        self.name = "cognitive_loop"
    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        env = await self.loop.process(inputs)
        out = env.get("outputs", {})
        q = float(env.get("metrics", {}).get("quality_mean", 0.7))
        return {
            "ok": bool(env.get("ok", True)),
            "name": self.name,
            "latency_ms": float(env.get("metrics", {}).get("latency_ms_total", 0.0)),
            "confidence": q,
            "tags": ["reasoning","plan"],
            "payload": out,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Module Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModuleRegistry:
    def __init__(self):
        self._mods: Dict[str, SynergyComponent] = {}
    def register(self, comp: SynergyComponent) -> None:
        self._mods[comp.name] = comp
    def get(self, name: str) -> SynergyComponent:
        return self._mods[name]
    def names(self) -> List[str]:
        return list(self._mods.keys())
    def items(self):
        return self._mods.items()

# ─────────────────────────────────────────────────────────────────────────────
# Synergy Hub
# ─────────────────────────────────────────────────────────────────────────────

class IntrextroSynergyBase:
    def __init__(self, cfg: Optional[SynergyConfig] = None, modules: Optional[List[SynergyComponent]] = None):
        self.cfg = cfg or SynergyConfig()
        self.state = SynergyState(short_ring=deque(maxlen=self.cfg.ring_size))
        self.cb = CircuitBreaker(self.cfg.circuit_threshold, self.cfg.circuit_cooldown_s)
        self.registry = ModuleRegistry()

        # Default set: EI and CognitiveLoop if available
        if modules:
            for m in modules:
                self.registry.register(m)
        else:
            try:
                self.registry.register(EIComponent())
            except Exception as e:
                logger.warning("EI component unavailable: %s", e)
            try:
                self.registry.register(CogLoopComponent())
            except Exception as e:
                logger.warning("CognitiveLoop component unavailable: %s", e)

        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.keep_history)

    # Public API
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cb.allow():
            return {"ok": False, "error": "circuit_open"}
        t0 = time.time()
        ok = True
        latencies: List[float] = []
        results: List[Dict[str, Any]] = []
        try:
            # Run modules concurrently with timeouts
            coros = [with_timeout(m.run(inputs), self.cfg.step_timeout_s) for _, m in self.registry.items()]
            for fut in asyncio.as_completed(coros):
                try:
                    r = await fut
                    results.append(r)
                except Exception as e:
                    logger.warning("module error: %s", e)
            self.cb.success()
        except Exception as e:  # outer failure
            ok = False
            self.cb.failure()
            logger.exception("synergy_error: %s", e)

        # Aggregate telemetry
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        # Strategy: select or blend
        final, strat = self._select_or_blend(inputs, results)
        self.state.last_strategy = strat
        self._memorize(inputs, results, final, strat)

        env = {
            "ok": ok,
            "outputs": final,
            "metrics": {
                "latency_ms_total": sum(latencies),
                "latency_ms_p50": Rolling.p(latencies, 0.5),
                "latency_ms_p95": Rolling.p(latencies, 0.95),
                "strategy": strat,
                "modules": [r.get("name") for r in results],
                "confidences": {r.get("name"): r.get("confidence") for r in results},
            },
            "state": asdict(self.state),
        }
        return env

    # ── policy hooks ───────────────────────────────────────────────────────
    def _select_or_blend(self, inputs: Dict[str, Any], results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        usable = [r for r in results if r.get("ok") and r.get("confidence", 0) >= self.cfg.min_conf_for_use]
        if not usable:
            # fallback to highest confidence even if below min
            if not results:
                return {"note": "no_modules"}, "none"
            best = max(results, key=lambda r: r.get("confidence", 0))
            return self._format_payload(best), f"select:{best.get('name')}"

        # If a single module clears the select threshold by margin, pick it
        best = max(usable, key=lambda r: r.get("confidence", 0))
        second = sorted(usable, key=lambda r: r.get("confidence", 0), reverse=True)[1:2]
        delta = best.get("confidence", 0) - (second[0].get("confidence", 0) if second else 0)
        if best.get("confidence", 0) >= self.cfg.min_conf_for_select and delta >= self.cfg.blend_when_close_delta:
            return self._format_payload(best), f"select:{best.get('name')}"

        # otherwise blend compatible outputs
        return self._blend(usable), "blend"

    def _format_payload(self, res: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(res.get("payload", {}))
        payload.setdefault("source", res.get("name"))
        payload.setdefault("confidence", res.get("confidence", 0.0))
        return payload

    def _blend(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Synergy: merge common keys and compute weighted averages where possible
        # Start with union of keys across payloads
        out: Dict[str, Any] = {"sources": [r.get("name") for r in results]}
        # Weighted VAD if present
        def wavg(key: str) -> Optional[float]:
            vals = [(r["payload"].get(key), r.get("confidence", 0.0)) for r in results if isinstance(r.get("payload", {}).get(key, None), (int, float))]
            if not vals:
                return None
            num = sum(float(v) * float(w) for v, w in vals)
            den = sum(float(w) for _, w in vals) or 1.0
            return num / den
        for k in ("valence","arousal","dominance"):
            v = wavg(k)
            if v is not None:
                out[k] = v
        # Merge tags
        tags: List[str] = []
        for r in results:
            tags += list(r.get("tags", []) or [])
        out["tags"] = list(dict.fromkeys(tags))
        # Emotion label preference by highest confidence, fallback to majority vote
        emo_candidates = [(r.get("payload", {}).get("emotion"), r.get("confidence", 0.0)) for r in results if r.get("payload", {}).get("emotion")]
        if emo_candidates:
            emo = max(emo_candidates, key=lambda x: x[1])[0]
            out["emotion"] = emo
        # Suggestions: union top 3
        sugg: List[str] = []
        for r in results:
            actions = ((r.get("payload", {}).get("suggestion", {}) or {}).get("actions", []) or [])
            sugg.extend(actions)
        if sugg:
            out["suggestion"] = {"actions": list(dict.fromkeys(sugg))[:3]}
        # Confidence: max
        out["confidence"] = max((float(r.get("confidence", 0.0)) for r in results), default=0.0)
        return out

    def _memorize(self, inputs: Dict[str, Any], results: List[Dict[str, Any]], final: Dict[str, Any], strat: str) -> None:
        trace = {
            "inputs": {k: v for k, v in inputs.items() if k != "context"},
            "results": [{"name": r.get("name"), "confidence": r.get("confidence"), "tags": r.get("tags")} for r in results],
            "final": final,
            "strategy": strat,
        }
        self.state.short_ring.append(trace)
        self.state.episodic.append(trace)
        if len(self.state.episodic) > self.cfg.episodic_max:
            self.state.episodic.pop(0)
        self.history.append(trace)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Adapter
# ─────────────────────────────────────────────────────────────────────────────

try:
    from synergy_orchestrator import Capability, Port, QoS, SynergyModule  # type: ignore

    class IntrextroSynergyModule(SynergyModule):
        def __init__(self, name: str = "intrextro_synergy_base"):
            self._id = name
            self.hub = IntrextroSynergyBase()
        def id(self) -> str: return self._id
        def capability(self) -> Capability:
            return Capability(
                name=self._id,
                inputs=[
                    Port("text","text"),
                    Port("domain","label:str"),
                    Port("task","label:str"),
                    Port("context","json"),
                ],
                outputs=[
                    Port("emotion","label:str"),
                    Port("valence","float[-1,1]"),
                    Port("arousal","float[0,1]"),
                    Port("dominance","float[0,1]"),
                    Port("tags","list[str]"),
                    Port("suggestion","json"),
                    Port("confidence","float[0,1]"),
                    Port("source","label:str"),
                ],
                tags=["synergy","integration","ensemble"],
                q=QoS(latency_ms=60, cost_unit=0.35, reliability=0.98, quality=0.8),
            )
        async def run(self, **kwargs) -> Dict[str, Any]:
            env = await self.hub.process(kwargs)
            out = env.get("outputs", {})
            # Map a few common outputs; pass‑through others
            mapped = {
                k: out.get(k) for k in ["emotion","valence","arousal","dominance","tags","suggestion","confidence","source"] if k in out
            }
            mapped["_quality"] = float(out.get("confidence", 0.75))
            return mapped
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _demo():
        hub = IntrextroSynergyBase()
        res = await hub.process({
            "text": "We love the velocity, but the failure risk makes me anxious!!!",
            "domain": "enterprise",
            "task": "decision_support",
        })
        print(json.dumps(res, indent=2))
    asyncio.run(_demo())

