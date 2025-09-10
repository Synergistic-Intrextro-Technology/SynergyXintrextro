# demo_kernel_route.py
import json, time
from kernel_config import build_router_from_config
from kernel_router import KernelRequest

router = build_router_from_config("kernels.yaml", "kernels.json")

def run_demo():
    req = KernelRequest(task="qa", modality="text", sla_ms=2000, hints={"mode":"hybrid"})
    payload = "Explain the synergy loop in one sentence."
    t0 = time.time()
    out = router.route(req, payload)
    lat = (time.time()-t0)*1000.0
    print("[demo] latency_ms=%.2f" % lat)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    run_demo()
