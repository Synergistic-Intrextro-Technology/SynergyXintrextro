# Pure Synergy — Routing Patch

Contents:
- `kernels.yaml` / `kernels.json`: merged config referencing generated adapters
- `kernel_router.py` / `kernel_config.py`: minimal router + loader
- `adapters_generated/*.py`: stable import wrappers for discovered kernels
- `demo_kernel_route.py`: quick route test
- `bench_pure_synergy.py`: micro-benchmark

Run:
```bash
python demo_kernel_route.py
python bench_pure_synergy.py
```

Notes:
- Wrappers proxy to your original classes so you can keep developing in-place.
- Edit `kernels.yaml` to adjust names/tags/timeouts or to add static routing rules.
- For production, copy this folder into your repo and point your service at `kernels.yaml`.