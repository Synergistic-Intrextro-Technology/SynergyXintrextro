# kernel_router.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class KernelRequest:
    task: str
    modality: str
    sla_ms: int | None = None
    hints: Dict[str, Any] | None = None

class KernelRouter:
    """SLA-aware router with health/fallback and per-kernel stats (minimal)."""
    def __init__(self, kernels: Dict[str, Dict[str, Any]], default_fallback: str | None = None):
        self.kernels = kernels
        self.default_fallback = default_fallback

    def list_kernels(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for name, info in self.kernels.items():
            st = info.setdefault("stats", {"calls":0,"ok":0,"fail":0,"avg_latency_ms":None,"healthy":None,"last_error":None})
            out[name] = {"tags": info.get("tags", []), "stats": st}
        return out

    def _pick_candidates(self, req: KernelRequest) -> List[str]:
        tags = []
        if req.hints and isinstance(req.hints, dict) and "mode" in req.hints:
            tags.append(req.hints["mode"])
        candidates = []
        for name, info in self.kernels.items():
            ktags = set(info.get("tags", []))
            if not tags or (set(tags) & ktags):
                candidates.append(name)
        return candidates or list(self.kernels.keys())

    def _is_healthy(self, name: str) -> bool:
        k = self.kernels[name]["obj"]
        try:
            ok = bool(k.healthy())
        except Exception:
            ok = False
        self.kernels[name]["stats"]["healthy"] = ok
        return ok

    def _pick_best(self, req: KernelRequest, candidates: List[str]) -> str | None:
        healthy = [c for c in candidates if self._is_healthy(c)]
        pool = healthy or candidates
        if not pool:
            return None
        if req.sla_ms:
            pool = sorted(pool, key=lambda n: self.kernels[n].get("timeout_ms", 30000))
        return pool[hash((req.task, req.modality, tuple(sorted((req.hints or {}).items())))) % len(pool)]

    def route(self, req: KernelRequest, payload: Any = None) -> Dict[str, Any]:
        tried = []
        start = time.time()
        candidates = self._pick_candidates(req)
        chosen = self._pick_best(req, candidates)
        fallbacks = []
        if chosen:
            fb = self.kernels.get(chosen, {}).get("fallback")
            if fb: fallbacks.append(fb)
        if self.default_fallback:
            fallbacks.append(self.default_fallback)
        chain = [c for c in [chosen] if c] + [f for f in fallbacks if f]
        if not chain:
            return {"status":"error","error":"no kernels available","tried":[]}

        last_err = None
        for name in chain:
            tried.append(name)
            info = self.kernels[name]
            obj = info["obj"]
            t0 = time.time()
            try:
                out = obj.run(req, payload)
                lat = (time.time() - t0) * 1000.0
                st = info.setdefault("stats", {"calls":0,"ok":0,"fail":0,"avg_latency_ms":None,"healthy":None,"last_error":None})
                st["calls"] += 1
                st["ok"] += 1 if out.get("status") == "ok" else 0
                prev = st.get("avg_latency_ms")
                st["avg_latency_ms"] = lat if prev is None else (0.9*prev + 0.1*lat)
                st["healthy"] = True
                out.setdefault("kernel", name)
                out["tried"] = tried
                return out
            except Exception as e:
                lat = (time.time() - t0) * 1000.0
                st = info.setdefault("stats", {"calls":0,"ok":0,"fail":0,"avg_latency_ms":None,"healthy":None,"last_error":None})
                st["calls"] += 1; st["fail"] += 1; st["healthy"] = False; st["last_error"] = str(e)
                last_err = f"{name}: {e}"
        return {"status":"error","error": last_err or "unknown error","tried": tried, "latency_ms": (time.time()-start)*1000.0}
