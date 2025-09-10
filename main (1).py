# app/main.py
from __future__ import annotations
import os, time, statistics
from typing import Any, Dict, Optional
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kernel_config import build_router_from_config
from kernel_router import KernelRequest

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
CFG_YAML = os.getenv("KERNELS_YAML", "kernels.yaml")
CFG_JSON = os.getenv("KERNELS_JSON", "kernels.json")
METRIC_WINDOW = int(os.getenv("METRIC_WINDOW", "500"))

app = FastAPI(title="Cognitive OS Router", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = build_router_from_config(CFG_YAML, CFG_JSON)

class RouteIn(BaseModel):
    task: str
    modality: str
    sla_ms: Optional[int] = None
    hints: Dict[str, Any] = Field(default_factory=dict)
    payload: Any = None

class RouteOut(BaseModel):
    status: str
    kernel: Optional[str] = None
    device: Optional[str] = None
    latency_ms: Optional[float] = None
    result: Dict[str, Any] | None = None
    tried: list[str] | None = None
    reason: Optional[str] = None
    error: Optional[str] = None

class MetricStore:
    def __init__(self, capacity: int = 500):
        self.latencies = deque(maxlen=capacity)
        self.successes = deque(maxlen=capacity)
        self.total = 0
        self.errors = 0
        self.started = time.time()

    def record(self, ok: bool, latency_ms: float):
        self.total += 1
        self.latencies.append(latency_ms)
        self.successes.append(1 if ok else 0)
        if not ok:
            self.errors += 1

    def snapshot(self):
        lats = list(self.latencies)
        succ = list(self.successes)
        def pctl(vals, p):
            if not vals: return None
            vals2 = sorted(vals)
            idx = max(0, min(len(vals2)-1, int(round(p * (len(vals2)-1)))))
            return vals2[idx]
        return {
            "window": len(lats),
            "p50_ms": pctl(lats, 0.50),
            "p95_ms": pctl(lats, 0.95),
            "avg_ms": (sum(lats)/len(lats)) if lats else None,
            "success_rate": (sum(succ)/len(succ)) if succ else None,
            "uptime_s": time.time() - self.started,
            "total": self.total,
            "errors": self.errors
        }

metrics = MetricStore(capacity=METRIC_WINDOW)

@app.post("/route", response_model=RouteOut)
def route_endpoint(inb: RouteIn):
    try:
        req = KernelRequest(task=inb.task, modality=inb.modality, sla_ms=inb.sla_ms, hints=inb.hints)
        t0 = time.time()
        out = router.route(req, payload=inb.payload)
        lat_ms = (time.time() - t0) * 1000.0
        ok = out.get("status") == "ok"
        metrics.record(ok=ok, latency_ms=lat_ms)
        return out
    except Exception as e:
        metrics.record(ok=False, latency_ms=0.0)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {
        "app_version": APP_VERSION,
        "kernels": router.list_kernels(),
        "config": {"yaml": CFG_YAML, "json": CFG_JSON},
        "metric_window": METRIC_WINDOW
    }

@app.get("/metrics")
def metrics_endpoint():
    snap = metrics.snapshot()
    kernels = router.list_kernels()
    per_kernel = {}
    for name, info in kernels.items():
        st = info.get("stats", {})
        per_kernel[name] = {
            "avg_ms": st.get("avg_latency_ms"),
            "ok": st.get("ok"), "fail": st.get("fail"), "calls": st.get("calls"),
            "healthy": st.get("healthy"), "last_error": st.get("last_error")
        }
    return {"global": snap, "kernels": per_kernel}

@app.post("/reload")
def reload_kernels():
    global router
    router = build_router_from_config(CFG_YAML, CFG_JSON)
    return {"status": "reloaded", "kernels": list(router.list_kernels().keys())}
