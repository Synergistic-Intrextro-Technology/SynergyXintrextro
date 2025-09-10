# kernel_config.py
from __future__ import annotations
import importlib, json, yaml
from typing import Any, Dict
from kernel_router import KernelRouter

def _import_from(path: str):
    mod_path, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)

def build_router_from_config(yaml_path: str="kernels.yaml", json_path: str="kernels.json") -> KernelRouter:
    cfg = {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
            cfg = {**cfg, **j}
    except FileNotFoundError:
        pass

    kernels_cfg = cfg.get("kernels", [])
    default_fallback = cfg.get("default_fallback")
    kernels = {}
    for item in kernels_cfg:
        name = item["name"]
        adapter = _import_from(item["adapter"])
        args = item.get("args", {})
        obj = adapter(**args) if isinstance(args, dict) else adapter()
        kernels[name] = {"obj": obj, "tags": item.get("tags", []),
                         "timeout_ms": item.get("timeout_ms", 30000),
                         "fallback": item.get("fallback")}
    return KernelRouter(kernels=kernels, default_fallback=default_fallback)
