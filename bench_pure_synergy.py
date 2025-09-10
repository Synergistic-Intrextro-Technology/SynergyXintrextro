import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kernel_config import build_router_from_config
from kernel_router import KernelRequest

router = build_router_from_config("/mnt/data/patch_pure_synergy/kernels.yaml", "/mnt/data/patch_pure_synergy/kernels.json")

def pctl(a, p):
    s = sorted(a); i = max(0, min(len(s)-1, int(round(p*(len(s)-1)))))
    return s[i] if s else None

if __name__ == "__main__":
    lats, ok = [], 0
    for _ in range(10):
        req = KernelRequest(task="qa", modality="text", sla_ms=3000, hints={})
        t0 = time.time()
        out = router.route(req, payload="bench")
        lats.append((time.time()-t0)*1000.0)
        ok += 1 if out.get("status") == "ok" else 0
    metrics = {"n": len(lats), "ok": ok, "p50_ms": pctl(lats, 0.50), "p95_ms": pctl(lats, 0.95), "avg_ms": sum(lats)/len(lats)}
    (pathlib.Path("/mnt/data/patch_pure_synergy") / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))