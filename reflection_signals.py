#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affect Hub — Composite Orchestrator Module
=========================================

Implements the user's integration strategy:

1) **Mandatory core**: always run the Base Emotional Intelligence engine.
2) **Extended layer**: optionally run an Enhanced Affect engine (detector/processor/regulation).
3) **Memory merge**: unify base short‑term ring with enhanced contextual memory.
4) **Dynamic selection/blending**: choose or blend outputs by domain/task sophistication.
5) **Fallbacks**: if enhanced fails or is low‑confidence, use core only.

This file depends on `emotional_intelligence.py — optimized` (Base EI).
It optionally consumes a user‑provided Enhanced engine via a simple protocol.

Exports both a plain async API and a Synergy Orchestrator adapter.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List, Deque, Tuple, Protocol
from collections import deque
import logging

# Base EI
from emotional_intelligence import EmotionalIntelligence, EIConfig, EIState

logger = logging.getLogger("affect_hub")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Engine Protocol (plug‑in)
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedAffect(Protocol):
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return envelope { ok, outputs:{emotion,valence,arousal,dominance,tags,suggestion,confidence}, metrics, state }"""
        ...

# ─────────────────────────────────────────────────────────────────────────────
# Config & State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HubConfig:
    step_timeout_s: float = 2.0
    low_conf_threshold: float = 0.68      # below → do not trust enhanced
    blend_threshold: float = 0.78         # between thresholds → blend; above → prefer enhanced
    ring_size: int = 512

@dataclass
class HubState:
    hub_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_ring: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=512))
    last_choice: str = "core_only"  # core_only | blended | enhanced_only

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(out: Dict[str, Any], key: str, default: Any = None):
    return (out or {}).get(key, default)

# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

class AffectHub:
    def __init__(self, core_cfg: Optional[EIConfig] = None, hub_cfg: Optional[HubConfig] = None, enhanced: Optional[EnhancedAffect] = None):
        self.core = EmotionalIntelligence(core_cfg)
        self.hcfg = hub_cfg or HubConfig()
        self.state = HubState(memory_ring=deque(maxlen=self.hcfg.ring_size))
        self.enhanced = enhanced  # may be None

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run core EI, optionally enhanced EI, then select/blend according to strategy."""
        t0 = time.time()
        # 1) Mandatory core
        core = await asyncio.wait_for(self.core.process(inputs), timeout=self.hcfg.step_timeout_s)
        core_out = core.get("outputs", {})
        core_conf = float(_safe_get(core_out, "confidence", 0.72))

        # 2) Optional enhanced
        enh_env: Optional[Dict[str, Any]] = None
        enh_out: Dict[str, Any] = {}
        enh_conf: float = 0.0
        if self.enhanced:
            try:
                enh_env = await asyncio.wait_for(self.enhanced.process(inputs), timeout=self.hcfg.step_timeout_s)
                enh_out = _safe_get(enh_env, "outputs", {})
                enh_conf = float(_safe_get(enh_out, "confidence", 0.0))
            except Exception as e:  # swallow and fallback
                logger.warning("enhanced affect failed: %s", e)
                enh_env = {"ok": False, "error": str(e)}

        # 3) Decide selection/blend
        choice = self._decide_choice(inputs, core_out, core_conf, enh_out, enh_conf)
        merged = self._merge_memory(core.get("state", {}), _safe_get(enh_env, "state", {}))

        # 4) Produce final outputs
        final = self._compose_outputs(choice, core_out, enh_out)

        # 5) Record
        self.state.last_choice = choice
        self.state.memory_ring.append({
            "inputs": {k: v for k, v in inputs.items() if k != "context"},
            "core": {k: core_out.get(k) for k in ["emotion","valence","arousal","dominance","tags","suggestion","confidence"]},
            "enh": {k: enh_out.get(k) for k in ["emotion","valence","arousal","dominance","tags","suggestion","confidence"]},
            "choice": choice,
            "final": final,
        })

        latency_ms = (time.time() - t0) * 1000
        return {
            "ok": True,
            "outputs": final,
            "metrics": {
                "latency_ms_total": latency_ms,
                "choice": choice,
                "core_conf": core_conf,
                "enh_conf": enh_conf,
            },
            "state": {**asdict(self.state), **merged},
        }

    # ── policy helpers ─────────────────────────────────────────────────────
    def _decide_choice(self, inputs: Dict[str, Any], core_out: Dict[str, Any], core_conf: float,
                       enh_out: Dict[str, Any], enh_conf: float) -> str:
        domain = str(inputs.get("domain",""))
        task = str(inputs.get("task",""))
        sophisticated = (domain in {"clinical","enterprise","safety"} or len(str(inputs.get("text",""))) > 240)
        if not self.enhanced or enh_conf < self.hcfg.low_conf_threshold:
            return "core_only"
        if enh_conf >= self.hcfg.blend_threshold and sophisticated:
            return "enhanced_only"
        # Else blend in the middle or for general domains
        return "blended"

    def _merge_memory(self, core_state: Dict[str, Any], enh_state: Dict[str, Any]) -> Dict[str, Any]:
        # Merge last VAD and carry persona/context if present
        merged = {}
        for k in ("last_valence","last_arousal","last_dominance","persona"):
            if k in core_state:
                merged[k] = core_state[k]
        if enh_state:
            merged.update({f"enh_{k}": v for k, v in enh_state.items() if k in ("last_valence","last_arousal","last_dominance")})
        return merged

    def _compose_outputs(self, choice: str, core_out: Dict[str, Any], enh_out: Dict[str, Any]) -> Dict[str, Any]:
        if choice == "core_only" or not enh_out:
            return core_out
        if choice == "enhanced_only":
            return enh_out
        # blended: weighted average on VAD; merge tags; merge suggestions (dedupe)
        wc, we = 0.4, 0.6
        def _w(k, default=0.0):
            return wc*float(_safe_get(core_out,k,default)) + we*float(_safe_get(enh_out,k,default))
        tags = list({*(_safe_get(core_out,"tags",[]) or []), *(_safe_get(enh_out,"tags",[]) or [])})
        sugg_c = (_safe_get(core_out, "suggestion", {}) or {}).get("actions", [])
        sugg_e = (_safe_get(enh_out, "suggestion", {}) or {}).get("actions", [])
        suggestion = {"goal": (_safe_get(enh_out,"suggestion",{}) or {}).get("goal", (_safe_get(core_out,"suggestion",{}) or {}).get("goal","soothe")),
                      "actions": (list(dict.fromkeys(sugg_e + sugg_c)))[:3]}
        # choose emotion by highest magnitude valence or enhanced label if confidence close
        emo = (_safe_get(enh_out,"emotion") or _safe_get(core_out,"emotion") or "neutral")
        confidence = max(float(_safe_get(core_out,"confidence",0.0)), float(_safe_get(enh_out,"confidence",0.0)))
        return {
            "emotion": emo,
            "valence": _w("valence"),
            "arousal": _w("arousal"),
            "dominance": _w("dominance"),
            "tags": tags,
            "suggestion": suggestion,
            "confidence": confidence,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Adapter (composite)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from synergy_orchestrator import Capability, Port, QoS, SynergyModule  # type: ignore

    class AffectHubModule(SynergyModule):
        """Always executes Base EI; optionally augments with Enhanced EI.
        The orchestrator treats this as a single capability that internally enforces the mandatory core.
        """
        def __init__(self, enhanced: Optional[EnhancedAffect] = None, name: str = "affect_hub"):
            self._id = name
            self.hub = AffectHub(enhanced=enhanced)
        def id(self) -> str: return self._id
        def capability(self) -> Capability:
            return Capability(
                name=self._id,
                inputs=[
                    Port("text","text"),
                    Port("goal","label:str"),
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
                ],
                tags=["affect","composite","regulation"],
                q=QoS(latency_ms=35, cost_unit=0.2, reliability=0.985, quality=0.8),
            )
        async def run(self, **kwargs) -> Dict[str, Any]:
            res = await self.hub.process(kwargs)
            out = res.get("outputs", {})
            return out | {"_quality": float(out.get("confidence", 0.75))}
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def _demo():
        hub = AffectHub()
        res = await hub.process({
            "text": "I'm excited about the launch but anxious about the risks!!!",
            "goal": "soothe",
            "domain": "enterprise",
            "task": "decision_support",
        })
        import json
        print(json.dumps(res, indent=2))
    asyncio.run(_demo())

