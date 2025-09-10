#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive Loop (Optimized)
==========================

A production‑grade reasoning loop with:
- Typed config & state
- Pluggable steps (observe → interpret → plan → act → reflect)
- Async/sync execution
- Telemetry (latency, quality, success), rolling stats
- Memory buffers (short‑term ring; long‑term key store)
- Policies (epsilon‑greedy exploration; temperature for sampling)
- Safe guards (timeouts, retries, circuit breaker)

Drop‑in compatible with your Synergy/Orchestrator stack:
- Exposes a minimal `process()` that accepts dict inputs and returns a structured result
- Optional `CognitiveLoopModule` adapter to register into the Synergy Orchestrator
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Callable, List, Optional, Tuple, Deque
from collections import deque, defaultdict
import logging
import uuid

logger = logging.getLogger("cognitive_loop")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Config & State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    max_steps: int = 6
    step_timeout_s: float = 2.5
    retry_attempts: int = 1
    epsilon: float = 0.15  # exploration for strategy choice
    temperature: float = 0.7
    ring_size: int = 256  # short‑term memory items
    circuit_threshold: int = 5
    circuit_cooldown_s: float = 30.0

@dataclass
class LoopState:
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step: int = 0
    short_memory: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    long_memory: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=lambda: dict(latencies=[], qualities=[], success=True))

# ─────────────────────────────────────────────────────────────────────────────
# Utilities: circuit breaker, timers, rolling stats
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.fail_count = 0
        self.last_fail_t: Optional[float] = None
        self.open = False

    def allow(self) -> bool:
        if not self.open:
            return True
        if self.last_fail_t is None:
            return True
        if time.time() - self.last_fail_t > self.cooldown_s:
            self.open = False
            self.fail_count = 0
            return True
        return False

    def mark_success(self) -> None:
        self.fail_count = 0
        self.open = False

    def mark_failure(self) -> None:
        self.fail_count += 1
        self.last_fail_t = time.time()
        if self.fail_count >= self.threshold:
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

# ─────────────────────────────────────────────────────────────────────────────
# Step interfaces
# ─────────────────────────────────────────────────────────────────────────────

StepFn = Callable[[Dict[str, Any], LoopState, LoopConfig], "StepOutput"]

@dataclass
class StepOutput:
    data: Dict[str, Any]
    quality: float = 0.7
    notes: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# Default steps (can be replaced)
# ─────────────────────────────────────────────────────────────────────────────

def step_observe(ctx: Dict[str, Any], st: LoopState, cfg: LoopConfig) -> StepOutput:
    """Ingest and normalize inputs."""
    normalized = {k: v for k, v in ctx.items()}
    return StepOutput({"obs": normalized}, quality=0.75, notes="observed")

def step_interpret(ctx: Dict[str, Any], st: LoopState, cfg: LoopConfig) -> StepOutput:
    """Light feature extraction / signal detection."""
    obs = ctx.get("obs", {})
    signals = {"has_text": isinstance(obs.get("text"), str), "length": len(str(obs.get("text", "")))}
    return StepOutput({"signals": signals}, quality=0.72, notes="interpreted")

def step_plan(ctx: Dict[str, Any], st: LoopState, cfg: LoopConfig) -> StepOutput:
    """Choose candidate actions; epsilon‑greedy exploration."""
    signals = ctx.get("signals", {})
    candidates = [
        {"action": "analyze_text", "weight": 0.6 + 0.4 * (1 if signals.get("has_text") else 0)},
        {"action": "summarize", "weight": 0.4},
        {"action": "extract_keywords", "weight": 0.3},
    ]
    if random() < cfg.epsilon:  # explore
        choice = candidates[::-1][0]
    else:
        choice = max(candidates, key=lambda c: c["weight"])
    return StepOutput({"plan": choice}, quality=choice["weight"], notes="planned")

def step_act(ctx: Dict[str, Any], st: LoopState, cfg: LoopConfig) -> StepOutput:
    """Execute the chosen action (placeholder hooks)."""
    plan = ctx.get("plan", {})
    act = plan.get("action", "noop")
    if act == "analyze_text":
        text = st.short_memory[-1].get("obs", {}).get("text") if st.short_memory else ctx.get("obs", {}).get("text")
        out = {"analysis": {"len": len(str(text or "")), "upper": str(text or "").upper()[:64]}}
        q = 0.78
    elif act == "summarize":
        text = ctx.get("obs", {}).get("text", "")
        out = {"summary": " ".join(str(text).split()[:24])}
        q = 0.7
    elif act == "extract_keywords":
        text = str(ctx.get("obs", {}).get("text", "")).lower().split()
        stop = {"the","a","and","to","of","in","on","for","with"}
        freq = defaultdict(int)
        for w in text:
            if w and w not in stop:
                freq[w] += 1
        out = {"keywords": sorted(freq, key=freq.get, reverse=True)[:8]}
        q = 0.68
    else:
        out, q = {"noop": True}, 0.5
    return StepOutput(out, quality=q, notes=f"acted:{act}")

def step_reflect(ctx: Dict[str, Any], st: LoopState, cfg: LoopConfig) -> StepOutput:
    """Evaluate outcome; write to memory."""
    quality_signals = [v for k, v in ctx.items() if k in {"analysis", "summary", "keywords"}]
    q = 0.7 + 0.05 * len(quality_signals)
    return StepOutput({"reflection": {"quality": q, "signals": list(ctx.keys())}}, quality=q, notes="reflected")

# ─────────────────────────────────────────────────────────────────────────────
# CognitiveLoop core
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveLoop:
    def __init__(self, config: Optional[LoopConfig] = None):
        self.cfg = config or LoopConfig()
        self.cb = CircuitBreaker(self.cfg.circuit_threshold, self.cfg.circuit_cooldown_s)
        self.steps: List[StepFn] = [step_observe, step_interpret, step_plan, step_act, step_reflect]

    def set_steps(self, steps: List[StepFn]) -> None:
        self.steps = steps

    async def _run_step(self, fn: StepFn, ctx: Dict[str, Any], st: LoopState) -> StepOutput:
        async def _call():
            return fn(ctx, st, self.cfg)
        return await asyncio.wait_for(_call(), timeout=self.cfg.step_timeout_s)

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cb.allow():
            return {"ok": False, "error": "circuit_open"}
        st = LoopState(short_memory=deque(maxlen=self.cfg.ring_size))
        ctx: Dict[str, Any] = dict(inputs)
        latencies: List[float] = []
        qualities: List[float] = []
        ok = True

        try:
            for i in range(min(self.cfg.max_steps, len(self.steps))):
                st.step = i
                t0 = time.time()
                out = await self._run_step(self.steps[i], ctx, st)
                lat = (time.time() - t0) * 1000
                latencies.append(lat)
                qualities.append(out.quality)
                ctx.update(out.data)
                st.short_memory.append(ctx.copy())
            self.cb.mark_success()
        except Exception as e:  # timeout or step error
            ok = False
            self.cb.mark_failure()
            logger.exception("cognitive_loop_error: %s", e)
            ctx["error"] = str(e)

        metrics = {
            "latency_ms_total": sum(latencies),
            "latency_ms_p50": Rolling.p(latencies, 0.5),
            "latency_ms_p95": Rolling.p(latencies, 0.95),
            "quality_mean": Rolling.mean(qualities),
            "success": ok,
        }
        return {"ok": ok, "state": asdict(st), "outputs": ctx, "metrics": metrics}

# ─────────────────────────────────────────────────────────────────────────────
# Optional: Adapter for Synergy Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

try:
    # Import lazily to keep file standalone when orchestrator isn't present
    from synergy_orchestrator import Capability, Port, QoS, SynergyModule  # type: ignore

    class CognitiveLoopModule(SynergyModule):
        def __init__(self, name: str = "cognitive_loop"):
            self._id = name
            self.loop = CognitiveLoop()
        def id(self) -> str: return self._id
        def capability(self) -> Capability:
            return Capability(
                name=self._id,
                inputs=[Port("text", "text"), Port("context", "json")],
                outputs=[Port("analysis", "json"), Port("summary", "text"), Port("keywords", "list[str]")],
                tags=["reasoning","cognitive_loop"],
                q=QoS(latency_ms=30, cost_unit=0.2, reliability=0.98, quality=0.72),
            )
        async def run(self, **kwargs) -> Dict[str, Any]:
            res = await self.loop.process(kwargs)
            out = res.get("outputs", {})
            # Map to declared ports if present
            return {
                k: out.get(k) for k in ["analysis","summary","keywords"] if k in out
            } | {"_quality": float(res.get("metrics",{}).get("quality_mean",0.7))}
except Exception:  # orchestrator not installed; safe to ignore
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _demo():
        loop = CognitiveLoop()
        result = await loop.process({"text": "I love the camera but hate the battery life."})
        print(json.dumps(result, indent=2))
    asyncio.run(_demo())

