
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperLearningSystem Pro â€” Hardened Edition
=========================================
Drop-in replacement for your current SLS Pro with:
- Safer SQLite (WAL, busy_timeout, pragmas), JSON export helpers
- Stronger AdaptiveCache (TTL+LRU, get_or_set, optional Redis L2)
- Smoother ResourceManager (EMA throttle, PSUTIL fallback)
- CircuitBreaker with half-open sampling + jitter
- StreamProcessor batch draining + stats
- DistributedExecutor with cooperative cancellation
- Ensemble with robust numeric extraction + error budget
- RLPolicy table decay & size cap
- Health report + plugin autoload: ./sls_plugins/*.py -> register(system)

Stdlib + numpy. Optional: psutil, redis. Type hints included where helpful.
"""

from __future__ import annotations

import argparse, asyncio, functools, hashlib, importlib.util, json, logging, math, os, random, sqlite3, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np

# ------------------------ Logging ------------------------

def _build_logger() -> logging.Logger:
    level = os.getenv("SLS_LOG_LEVEL", "INFO").upper()
    log = logging.getLogger("SLSPro")
    if getattr(log, "_configured", False):
        return log
    log.setLevel(getattr(logging, level, logging.INFO))

    fmt_json = os.getenv("SLS_LOG_JSON", "0") == "1"
    if fmt_json:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                obj = {
                    "ts": round(time.time(), 3),
                    "level": record.levelname,
                    "name": record.name,
                    "msg": record.getMessage(),
                    "pid": os.getpid(),
                    "thread": record.threadName,
                }
                return json.dumps(obj, ensure_ascii=False)
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)

    path = os.getenv("SLS_LOG_FILE")
    if path:
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    log._configured = True  # type: ignore[attr-defined]
    return log

logger = _build_logger()

# Optional deps
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available â€” limited resource telemetry")

try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    logger.warning("redis not available â€” L2 cache disabled")

# ------------------------ Resource & Perf ------------------------

class ResourceManager:
    """Monitors process resources and computes a smoothed throttle factor."""
    def __init__(self, max_memory_mb: int = 2048, max_cpu_percent: float = 85.0, ema: float = 0.25) -> None:
        self.max_memory_mb = float(max_memory_mb)
        self.max_cpu_percent = float(max_cpu_percent)
        self.ema = float(ema)
        self._mem_ema = 0.0
        self._cpu_ema = 0.0
        self.metrics: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=240))
        self._lock = threading.Lock()

    def _cpu_now(self) -> float:
        if not PSUTIL_AVAILABLE: return 0.0
        try:
            return float(psutil.Process().cpu_percent(interval=None))
        except Exception:
            return 0.0

    def _mem_now(self) -> float:
        if not PSUTIL_AVAILABLE: return 0.0
        try:
            return float(psutil.Process().memory_info().rss / (1024 * 1024))
        except Exception:
            return 0.0

    def check(self) -> Dict[str, Any]:
        mem_mb = self._mem_now()
        cpu = self._cpu_now()
        with self._lock:
            # EMA smoothing
            self._mem_ema = (1 - self.ema) * self._mem_ema + self.ema * mem_mb
            self._cpu_ema = (1 - self.ema) * self._cpu_ema + self.ema * cpu
            m = self._mem_ema or mem_mb
            c = self._cpu_ema or cpu
            m_ratio = m / max(1.0, self.max_memory_mb)
            c_ratio = c / max(1.0, self.max_cpu_percent)
            r = max(m_ratio, c_ratio)
            throttle = float(1.0 / (1.0 + max(0.0, r)))
            available = bool((mem_mb < self.max_memory_mb or not PSUTIL_AVAILABLE) and (cpu < self.max_cpu_percent or not PSUTIL_AVAILABLE))
            self.metrics["memory_mb"].append(mem_mb)
            self.metrics["cpu_percent"].append(cpu)
        return {"memory_mb": mem_mb, "cpu_percent": cpu, "available": available, "throttle": throttle}

    def trends(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out: Dict[str, Dict[str, float]] = {}
            for k, v in self.metrics.items():
                if not v: continue
                arr = np.array(list(v), dtype=float)
                slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 2 else 0.0
                out[k] = {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()), "std": float(arr.std()), "trend": slope}
            return out

# ------------------------ Circuit Breaker ------------------------

class CircuitBreaker:
    def __init__(self, failures: int = 5, recovery_timeout: float = 60.0, jitter: float = 0.2) -> None:
        self.failures = int(failures)
        self.timeout = float(recovery_timeout)
        self.jitter = float(jitter)
        self.count = 0
        self.last_failure = 0.0
        self.state = "closed"  # closed | open | half_open
        self._lock = threading.Lock()

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with self._lock:
                if self.state == "open":
                    backoff = self.timeout * (1.0 + self.jitter * (random.random() - 0.5))
                    if time.time() - self.last_failure > backoff:
                        self.state = "half_open"
                    else:
                        raise RuntimeError(f"circuit open: {fn.__name__}")
            try:
                out = fn(*args, **kwargs)
                with self._lock:
                    if self.state in ("open", "half_open"):
                        self.state = "closed"
                    self.count = 0
                return out
            except Exception:
                with self._lock:
                    self.count += 1
                    self.last_failure = time.time()
                    if self.count >= self.failures:
                        self.state = "open"
                raise
        return wrapped

    def reset(self) -> None:
        with self._lock:
            self.count = 0; self.last_failure = 0.0; self.state = "closed"

# ------------------------ Cache & Persistence ------------------------

class AdaptiveCache:
    def __init__(self, max_size: int = 2000, default_ttl: float = 3600.0) -> None:
        self.max_size = int(max_size)
        self.default_ttl = float(default_ttl)
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._order: Deque[str] = deque()
        self._hits = 0; self._miss = 0
        self._lock = threading.Lock()
        self._r = None
        if REDIS_AVAILABLE:
            try:
                self._r = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    decode_responses=True,
                )
                self._r.ping()
                logger.info("Redis connected for cache")
            except Exception:
                self._r = None
                logger.warning("Redis connection failed â€” continuing without L2 cache")

    def _evict_if_needed(self) -> None:
        while len(self._mem) > self.max_size and self._order:
            k = self._order.popleft()
            self._mem.pop(k, None)

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            e = self._mem.get(key)
            if e and e["exp"] > now:
                self._hits += 1
                try: self._order.remove(key)
                except ValueError: pass
                self._order.append(key)
                return e["val"]
            self._miss += 1
        if self._r is not None:
            try:
                raw = self._r.get(f"sls:{key}")
                if raw: return json.loads(raw)
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = float(ttl or self.default_ttl)
        exp = time.time() + ttl
        with self._lock:
            self._mem[key] = {"val": value, "exp": exp}
            try: self._order.remove(key)
            except ValueError: pass
            self._order.append(key)
            self._evict_if_needed()
        if self._r is not None:
            try: self._r.setex(f"sls:{key}", int(ttl), json.dumps(value, default=str))
            except Exception: pass

    def get_or_set(self, key: str, fn: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        v = self.get(key)
        if v is not None:
            return v
        v = fn()
        self.set(key, v, ttl=ttl)
        return v

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._mem.pop(key, None)
            try: self._order.remove(key)
            except ValueError: pass
        if self._r is not None:
            try: self._r.delete(f"sls:{key}")
            except Exception: pass

    def stats(self) -> Dict[str, float]:
        with self._lock:
            total = self._hits + self._miss or 1
            return {"size": float(len(self._mem)), "hits": float(self._hits), "miss": float(self._miss), "hit_rate": float(self._hits) / float(total)}

class PersistentState:
    def __init__(self, path: Union[str, Path] = "superlearning.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=15.0, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL,
                    created REAL NOT NULL,
                    updated REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    op TEXT,
                    ts REAL,
                    duration REAL,
                    mem_delta REAL,
                    meta TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics ON metrics(op, ts)")

    def save(self, key: str, value: Any) -> None:
        t = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO kv(k,v,created,updated)
                    VALUES(?,?,?,?)
                    ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated=excluded.updated
                """, (key, json.dumps(value, default=str), t, t))

    def load(self, key: str, default: Any = None) -> Any:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("SELECT v FROM kv WHERE k=?", (key,))
                row = cur.fetchone()
                if not row:
                    return default
                try:
                    return json.loads(row[0])
                except Exception:
                    return default

    def log_metric(self, op: str, duration: float, mem_delta: float = 0.0, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("INSERT INTO metrics (op, ts, duration, mem_delta, meta) VALUES (?, ?, ?, ?, ?)",
                             (op, time.time(), float(duration), float(mem_delta), json.dumps(meta or {}, default=str)))

    def history(self, op: str, since_s: float = 86_400.0) -> List[Dict[str, Any]]:
        cutoff = time.time() - since_s
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("SELECT ts, duration, mem_delta, meta FROM metrics WHERE op=? AND ts>? ORDER BY ts DESC", (op, cutoff))
                return [{"ts": r[0], "duration": r[1], "mem_delta": r[2], "meta": (json.loads(r[3]) if r[3] else {})} for r in cur.fetchall()]

# ------------------------ Quantum-ish State ------------------------

@dataclass
class QuantumLearningState:
    amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))
    phase: float = 0.0
    entanglement_map: Dict[str, float] = field(default_factory=dict)
    coherence_time: float = 1.0
    measurement_history: List[float] = field(default_factory=list)
    confidence: float = 1.0

    def collapse(self) -> float:
        p = abs(self.amplitude) ** 2
        self.measurement_history.append(float(p))
        self.measurement_history[:] = self.measurement_history[-64:]
        return float(p)

    def entangle_with(self, other: "QuantumLearningState", strength: float = 0.8) -> None:
        a = f"e_{id(other)}"; b = f"e_{id(self)}"
        self.entanglement_map[a] = float(strength)
        other.entanglement_map[b] = float(strength)
        if len(self.entanglement_map) > 64:
            top = sorted(self.entanglement_map.items(), key=lambda kv: kv[1], reverse=True)[:32]
            self.entanglement_map = dict(top)

    def serialize(self) -> Dict[str, Any]:
        return {
            "amp_r": self.amplitude.real, "amp_i": self.amplitude.imag, "phase": self.phase,
            "entanglement_map": self.entanglement_map, "coherence_time": self.coherence_time,
            "measurement_history": self.measurement_history[-32:], "confidence": self.confidence,
        }

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "QuantumLearningState":
        return cls(
            amplitude=complex(d.get("amp_r", 1.0), d.get("amp_i", 0.0)),
            phase=float(d.get("phase", 0.0)),
            entanglement_map=dict(d.get("entanglement_map", {})),
            coherence_time=float(d.get("coherence_time", 1.0)),
            measurement_history=list(d.get("measurement_history", [])),
            confidence=float(d.get("confidence", 1.0)),
        )

class EnhancedQuantumProcessor:
    def __init__(self, hilbert_dim: int = 16, decoherence_rate: float = 0.01) -> None:
        self.hilbert = int(max(2, min(64, hilbert_dim)))
        self.deco = float(decoherence_rate)
        self._U = None
        self.rm = ResourceManager()
        self.prof = PerformanceProfiler()
        self._pool: List[QuantumLearningState] = []
        self._lock = threading.Lock()

    def _ensure_U(self) -> None:
        if self._U is None:
            with self.prof.profile("quantum/init_U"):
                self._U = np.eye(self.hilbert, dtype=complex)

    def _get_state(self) -> QuantumLearningState:
        with self._lock:
            return self._pool.pop() if self._pool else QuantumLearningState()

    def _recycle(self, s: QuantumLearningState) -> None:
        s.amplitude = complex(1.0, 0.0); s.phase = 0.0; s.entanglement_map.clear()
        s.measurement_history.clear(); s.confidence = 1.0
        with self._lock:
            if len(self._pool) < 64: self._pool.append(s)

    @CircuitBreaker(failures=3, recovery_timeout=30.0)
    def create_superposition(self, contexts: List[Dict[str, Any]]) -> QuantumLearningState:
        with self.prof.profile("quantum/superposition"):
            res = self.rm.check()
            n = max(1, min(32 if res["available"] else 8, len(contexts)))
            amp = complex(1.0 / math.sqrt(n), 0.0)
            st = self._get_state()
            st.amplitude = amp
            st.phase = float(np.random.uniform(0.0, 2 * math.pi))
            st.entanglement_map.clear()
            for i, ctx in enumerate(contexts[:n]):
                st.entanglement_map[f"ctx_{i}_{hash(str(sorted(ctx.items()))) % 1_000_003}"] = 1.0 / n
            return st

    def evolve(self, state: QuantumLearningState, H: Optional[np.ndarray] = None) -> QuantumLearningState:
        with self.prof.profile("quantum/evolve"):
            self._ensure_U()
            dt = 0.1
            try:
                u00 = np.exp(-1j * (H[0, 0] if H is not None else 0.1) * dt)
            except Exception:
                u00 = 1.0 + 0.0j
            ns = self._get_state()
            ns.amplitude = state.amplitude * u00 * np.exp(-self.deco * dt)
            ns.phase = state.phase + dt
            ns.entanglement_map = dict(state.entanglement_map)
            ns.coherence_time = state.coherence_time * float(np.exp(-self.deco * dt))
            ns.measurement_history = list(state.measurement_history)
            return ns

# ------------------------ Performance Profiler ------------------------

class PerformanceProfiler:
    """Sampling profiler collecting durations (and mem delta if psutil)."""
    def __init__(self, sampling_rate: float = 0.15) -> None:
        self.sampling_rate = sampling_rate
        self.data: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._lock = threading.Lock()

    def profile(self, op: str):
        class Ctx:
            def __init__(self, outer, op):
                self.outer = outer; self.op = op; self.t0 = 0.0; self.m0 = 0.0; self.sample = False
            def __enter__(self):
                self.sample = (np.random.random() < self.outer.sampling_rate)
                if self.sample:
                    if PSUTIL_AVAILABLE:
                        self.m0 = float(psutil.Process().memory_info().rss)
                    self.t0 = time.time()
                return self
            def __exit__(self, exc_type, exc, tb):
                if self.sample:
                    t1 = time.time()
                    rec = {"duration": t1 - self.t0}
                    if PSUTIL_AVAILABLE:
                        rec["mem_delta"] = float(psutil.Process().memory_info().rss - self.m0)
                    with self.outer._lock:
                        self.outer.data[self.op].append(rec)
                        self.outer.data[self.op] = self.outer.data[self.op][-1000:]
        return Ctx(self, op)

    def stats(self, op: str) -> Dict[str, float]:
        with self._lock:
            arr = [r["duration"] for r in self.data.get(op, [])]
            if not arr: return {}
            a = np.array(arr, dtype=float)
            return {
                "count": float(len(a)),
                "mean": float(a.mean()),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
                "std": float(a.std()),
            }

# ------------------------ Ensemble & RL ------------------------

class Ensemble:
    def __init__(self) -> None:
        self.models: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.error_budget = 0.1  # tolerate 10% model failures without crashing

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]], weight: float = 1.0) -> None:
        self.models[name] = fn; self.weights[name] = float(weight)

    def _extract_scalars(self, obj: Any, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.update(self._extract_scalars(v, f"{prefix}{k}."))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                out.update(self._extract_scalars(v, f"{prefix}{i}."))
        elif isinstance(obj, (int, float)):
            out[prefix[:-1] or "value"] = float(obj)
        return out

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        res: Dict[str, Dict[str, Any]] = {}
        failures = 0
        for n, m in self.models.items():
            try:
                res[n] = m(data)
            except Exception as e:
                failures += 1
                res[n] = {"error": str(e)}
        fused = self._weighted(res)
        conf = self._confidence(res)
        agree = self._agreement(res)
        if self.models and failures / max(1, len(self.models)) > self.error_budget:
            logger.warning("ensemble failure rate high: %s/%s", failures, len(self.models))
        return {"ensemble_result": fused, "individual": res, "confidence": conf, "agreement": agree}

    def _weighted(self, res: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        out: Dict[str, float] = defaultdict(float); tot = 0.0
        for n, r in res.items():
            if "error" in r: continue
            w = float(self.weights.get(n, 1.0)); tot += w
            scalars = self._extract_scalars(r)
            for k, v in scalars.items(): out[k] += w * v
        if tot > 0:
            for k in list(out.keys()): out[k] /= tot
        return dict(out)

    def _confidence(self, res: Dict[str, Dict[str, Any]]) -> float:
        vals = [float(v) for r in res.values() for v in self._extract_scalars(r).values()]
        return float(np.clip(np.mean(vals) if vals else 0.5, 0.0, 1.0))

    def _agreement(self, res: Dict[str, Dict[str, Any]]) -> float:
        keys = sorted({k for r in res.values() for k in self._extract_scalars(r).keys()})
        if not keys: return 1.0
        mat = []
        for r in res.values():
            scal = self._extract_scalars(r)
            mat.append([float(scal.get(k, 0.0)) for k in keys])
        if len(mat) < 2: return 1.0
        arr = np.array(mat, dtype=float)
        var = float(np.mean(np.var(arr, axis=0)))
        return float(np.clip(1.0 / (1.0 + var), 0.0, 1.0))

class RLPolicy:
    def __init__(self, max_states: int = 5000, decay: float = 0.999) -> None:
        self.Q: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.eps = 0.1; self.alpha = 0.02; self.gamma = 0.95
        self.decay = float(decay); self.max_states = int(max_states)

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        s = self._encode(state)
        a_primary = self._eps_greedy(s)
        alt = {"explore": self._random(), "greedy": self._greedy(s), "meta": "deep_analyze" if "complex" in s else "optimize"}
        choice = self._vote({"primary": a_primary, **alt})
        conf = float(np.clip(self.Q[s].get(choice, 0.0) / 10.0 + 0.5, 0.0, 1.0))
        self._cap_and_decay()
        return {"action": choice, "confidence": conf, "alternatives": alt}

    def update(self, fb: Dict[str, Any]) -> None:
        s = self._encode(fb.get("state", {})); ns = self._encode(fb.get("next_state", {}))
        a = str(fb.get("action", "")); r = float(fb.get("reward", 0.0))
        cur = self.Q[s][a]; nxt = max(self.Q[ns].values()) if self.Q[ns] else 0.0
        self.Q[s][a] = cur + self.alpha * (r + self.gamma * nxt - cur)
        perf = float(fb.get("performance", 0.5)); self.eps = float(np.clip(0.5 - 0.4 * perf, 0.01, 0.5))
        self._cap_and_decay()

    # helpers
    def _encode(self, s: Any) -> str:
        j = json.dumps(s, sort_keys=True, default=str); return j[:1024]

    def _random(self) -> str:
        choices = ["deep_analyze", "format", "optimize", "refactor", "benchmark", "validate"]
        return str(np.random.choice(choices))

    def _greedy(self, s: str) -> str:
        return max(self.Q[s], key=self.Q[s].get) if self.Q[s] else self._random()

    def _eps_greedy(self, s: str) -> str:
        return self._random() if np.random.random() < self.eps else self._greedy(s)

    def _vote(self, actions: Dict[str, str]) -> str:
        votes: Dict[str, float] = defaultdict(float)
        for strat, act in actions.items():
            w = {"primary": 1.0, "explore": 0.6, "greedy": 0.9, "meta": 0.8}.get(strat, 0.5)
            votes[act] += w
        return max(votes, key=votes.get)

    def _cap_and_decay(self) -> None:
        # decay
        for s in list(self.Q.keys()):
            for a in list(self.Q[s].keys()):
                self.Q[s][a] *= self.decay
        # cap states
        if len(self.Q) > self.max_states:
            # drop oldest-ish deterministically
            for s in list(self.Q.keys())[: len(self.Q) - self.max_states]:
                self.Q.pop(s, None)

# ------------------------ Streaming (async) ------------------------

class StreamProcessor:
    def __init__(self, max_queue: int = 1000) -> None:
        self.q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue)
        self.buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=256))
        self.stats: Dict[str, List[float]] = defaultdict(list)

    async def push(self, item: Dict[str, Any]) -> None:
        try:
            self.q.put_nowait(item)
        except asyncio.QueueFull:
            sid = item.get("stream_id", "default")
            if self.buffers[sid]: self.buffers[sid].popleft()
            await self.q.put(item)

    async def process_once(self) -> Optional[Dict[str, Any]]:
        if self.q.empty(): return None
        it = await self.q.get()
        t0 = time.time()
        out = await self._enrich(await self._aggregate(await self._filter(await self._normalize(it))))
        self.stats["latency"].append(time.time() - t0)
        sid = it.get("stream_id", "default"); self.buffers[sid].append(out)
        return out

    async def drain(self, max_n: int = 100) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(max_n):
            if self.q.empty(): break
            r = await self.process_once()
            if r: out.append(r)
        return out

    # transforms
    async def _normalize(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        if "timestamp" in out: out["timestamp"] = float(out["timestamp"])
        for k, v in list(out.items()):
            if k != "timestamp" and isinstance(v, (int, float)): out[k] = float(v)
        return out

    async def _filter(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if not str(k).startswith("_")}

    async def _aggregate(self, d: Dict[str, Any]) -> Dict[str, Any]:
        sid = d.get("stream_id", "default"); buf = self.buffers[sid]
        if len(buf) < 2: return d
        out = dict(d); lat = self.stats.get("latency", [0.0])
        out["streaming_stats"] = {"window": len(buf), "avg_latency": float(np.mean(lat)) if lat else 0.0}
        return out

    async def _enrich(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d); out["processing_context"] = {"ts": float(time.time()), "qsize": self.q.qsize()}
        return out

# ------------------------ Distributed execution ------------------------

class DistributedExecutor:
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False) -> None:
        self.use_processes = use_processes
        self.max_workers = max_workers or max(2, os.cpu_count() or 2)
        self._tp = ThreadPoolExecutor(max_workers=self.max_workers)
        self._pp = ProcessPoolExecutor(max_workers=max(1, (self.max_workers // 2))) if use_processes else None

    def map_unordered(self, fn: Callable[[Any], Any], items: Sequence[Any], timeout_s: float = 60.0, cancel_event: Optional[threading.Event] = None) -> List[Any]:
        futures = [self._tp.submit(fn, x) for x in items]
        out: List[Any] = []
        start = time.time()
        for f in as_completed(futures, timeout=timeout_s):
            if cancel_event and cancel_event.is_set(): break
            try:
                out.append(f.result(timeout=max(0.1, timeout_s - (time.time() - start))))
            except Exception as e:
                out.append({"error": str(e)})
        return out

    def shutdown(self) -> None:
        self._tp.shutdown(wait=False, cancel_futures=True)
        if self._pp:
            self._pp.shutdown(wait=False, cancel_futures=True)

# ------------------------ SLS Pro faÃ§ade ------------------------

class SuperLearningSystemPro:
    def __init__(self, workdir: Optional[Union[str, Path]] = None) -> None:
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.store = PersistentState(self.workdir / os.getenv("SLS_DB", "superlearning.db"))
        self.cache = AdaptiveCache()
        self.meta = MetaOptimizerPro(self.store, self.cache)
        self.extract = FeatureExtractor()
        self.ensemble = Ensemble()
        self.rl = RLPolicy()
        self.stream = StreamProcessor()
        self.pool = DistributedExecutor(max_workers=int(os.getenv("SLS_WORKERS", "8")))
        self.rm = ResourceManager(max_memory_mb=int(os.getenv("SLS_MAX_MEM_MB", "3072")))
        self.prof = PerformanceProfiler()
        self._autoload_plugins()

        # Register trivial ensemble members (replace with real models)
        self.ensemble.register("mean_scorer", lambda d: {"score": float(len(str(d)) % 100) / 100.0}, weight=1.0)
        self.ensemble.register("len_scorer", lambda d: {"score": float(len(str(d)) % 50) / 50.0}, weight=0.7)

    def _autoload_plugins(self) -> None:
        plug_dir = self.workdir / "sls_plugins"
        if not plug_dir.exists(): return
        for fp in plug_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(fp.stem, fp)
                if not spec or not spec.loader: continue
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "register"):
                    mod.register(self)
                    logger.info("plugin loaded: %s", fp.name)
            except Exception as e:
                logger.warning("plugin failed: %s (%s)", fp.name, e)

    # Core ops
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        with self.prof.profile("sls/process_task"):
            res = self.rm.check()
            feats = self.extract.extract(task)
            meta = self.meta.optimize(task)
            ens = self.ensemble.run({**task, **feats.get("semantic", {})})
            decision = self.rl.act({"features": feats, "meta": meta, "resources": res})
            out = {"features": feats, "meta": meta, "ensemble": ens, "decision": decision, "resources": res}
            st = self.prof.stats("sls/process_task")
            self.store.log_metric("process_task", duration=st.get("mean", 0.0), mem_delta=0.0, meta={"sig": meta.get("task_signature", "")})
            return out

    def analyze_project(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        files = sorted([str(fp) for fp in p.rglob("*.py")]) if p.exists() else []
        return {"root": str(p), "py_files": len(files), "hash": hashlib.md5("".join(files).encode()).hexdigest()[:12]}

    def batch_process(self, tasks: Sequence[Dict[str, Any]], timeout_s: float = 120.0) -> List[Dict[str, Any]]:
        res = self.rm.check(); throttle = res.get("throttle", 1.0)
        chunk = max(1, int(10 * throttle))
        cancel = threading.Event()
        def _proc(t: Dict[str, Any]) -> Dict[str, Any]:
            return self.process_task(t)
        batched: List[Dict[str, Any]] = []
        for i in range(0, len(tasks), chunk):
            batched.extend(self.pool.map_unordered(_proc, list(tasks[i : i + chunk]), timeout_s=timeout_s, cancel_event=cancel))
        return batched

    async def stream_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        await self.stream.push(data)
        out = await self.stream.process_once()
        return out or {"status": "no-op"}

    # Health & Metrics
    def health(self) -> Dict[str, Any]:
        prof = self.prof.stats("sls/process_task")
        return {
            "ok": True,
            "resources": self.rm.check(),
            "cache": self.cache.stats(),
            "profiler": prof,
            "trends": self.rm.trends(),
            "kv_size": len(self.store.history("process_task", since_s=7*24*3600)),
        }

    def export_metrics_json(self, path: Union[str, Path] = "superlearning_metrics.json") -> str:
        data = {"health": self.health(), "meta": {"ts": time.time()}}
        out_path = str(self.workdir / path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path

    # Self-healing
    def reset_components(self) -> None:
        self.cache = AdaptiveCache()
        self.rl = RLPolicy()
        self.stream = StreamProcessor()
        logger.info("Components reset (cache, RL, stream)")

# ------------------------ Feature extraction (minimal) ------------------------

class FeatureExtractor:
    def __init__(self) -> None:
        self.layers: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.adapt_threshold = 0.8

    def extract(self, data: Any) -> Dict[str, Any]:
        feats = {"struct": self._struct(data), "semantic": self._semantic(data), "context": {"ts": time.time()}}
        out: Dict[str, Any] = {}
        for k, d in feats.items():
            out[k] = self._transform(d)
        out["interactions"] = self._interactions(out)
        if self._should_evolve(out): self._evolve()
        return out

    def _struct(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict): return {"depth": self._depth(x), "breadth": len(x), "complex": self._complex(x)}
        if isinstance(x, (list, tuple)): return {"len": len(x), "homog": float(len({type(v).__name__ for v in x}) == 1)}
        return {"type": type(x).__name__, "size": len(str(x))}

    def _semantic(self, x: Any) -> Dict[str, Any]:
        s = str(x).lower()
        return {
            "domain": ("code" if any(t in s for t in ("def ", "class ", "import ")) else "text"),
            "intent": ("opt" if "optimiz" in s else ("an" if "analy" in s else "gen")),
            "sent": float(np.clip((s.count("good") - s.count("bad")) / 10.0, -1.0, 1.0)),
        }

    def _transform(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        for i, layer in enumerate(self.layers):
            for k, v in list(out.items()):
                if isinstance(v, (int, float)):
                    out[k] = float(v) * (1 + 0.03 * i)
        return out

    def _interactions(self, feats: Dict[str, Any]) -> Dict[str, float]:
        keys = [k for k in feats if isinstance(feats[k], dict)]
        out: Dict[str, float] = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = sum(v for v in feats[keys[i]].values() if isinstance(v, (int, float)))
                b = sum(v for v in feats[keys[j]].values() if isinstance(v, (int, float)))
                out[f"{keys[i]}Ã—{keys[j]}"] = float(a * b)
        return out

    def _should_evolve(self, feats: Dict[str, Any]) -> bool:
        perf = [p for vs in self.metrics.values() for p in vs][-20:]
        ok = (np.mean(perf) if perf else 1.0) >= self.adapt_threshold
        return not ok and len(self.layers) < 6

    def _evolve(self) -> None:
        self.layers.append(object()); self.history.append({"t": time.time(), "layers": len(self.layers)})

    def _depth(self, d: Any, dep: int = 0) -> int:
        if isinstance(d, dict): return max([self._depth(v, dep + 1) for v in d.values()] + [dep])
        if isinstance(d, (list, tuple)): return max([self._depth(v, dep + 1) for v in d] + [dep])
        return dep

    def _complex(self, d: Any) -> float:
        if isinstance(d, dict): return float(len(d)) * math.log(len(d) + 1) + sum(self._complex(v) for v in d.values())
        if isinstance(d, (list, tuple)): return float(len(d)) * math.log(len(d) + 1) + sum(self._complex(v) for v in d)
        return 1.0

# ------------------------ Meta-optimizer ------------------------

class MetaOptimizerPro:
    def __init__(self, store: PersistentState, cache: AdaptiveCache) -> None:
        self.store = store; self.cache = cache; self.prof = PerformanceProfiler(); self.qp = EnhancedQuantumProcessor()
        self.learning_rates: Dict[str, float] = defaultdict(lambda: 0.001, self.store.load("lr_map", {}))
        self.meta: Dict[str, float] = defaultdict(float, self.store.load("meta_params", {
            "adaptation_speed": 0.1, "memory_decay": 0.95, "exploration_rate": 0.1, "lr_decay": 0.999, "momentum": 0.9
        }))
        self.strategy_perf: Dict[str, List[float]] = defaultdict(list)
        self._last_save = 0.0

    def optimize(self, task: Dict[str, Any]) -> Dict[str, Any]:
        with self.prof.profile("meta/optimize"):
            key = self._cache_key(task)
            hit = self.cache.get(key)
            if hit: return hit
            sig = self._sig(task); ctx = self._contexts(task)
            q = self.qp.create_superposition(ctx)
            strat = self._choose_strategy(task, q)
            lr = self._lr(sig, strat)
            plan = self._plan(task, strat)
            out = {
                "task_signature": sig, "learning_rate": lr, "strategy": strat,
                "quantum_state": q.serialize(), "learning_plan": plan, "confidence": q.collapse(),
                "meta": dict(self.meta), "ts": time.time()
            }
            self.cache.set(key, out, ttl=1800.0); self._maybe_save(); return out

    def feedback(self, fb: Dict[str, Any]) -> None:
        with self.prof.profile("meta/feedback"):
            s = fb.get("strategy")
            if s is not None and "performance" in fb:
                self.strategy_perf[s].append(float(fb["performance"])); self.strategy_perf[s] = self.strategy_perf[s][-200:]
            adj = fb.get("learning_rate_adjustment", {})
            for sig, mult in adj.items():
                self.learning_rates[sig] = float(self.learning_rates.get(sig, 0.001) * float(mult) * float(self.meta.get("lr_decay", 0.999)))
            for k, v in fb.get("meta_parameter_updates", {}).items():
                base = float(self.meta.get(k, 1.0)); self.meta[k] = float(max(1e-6, min(10.0, base * float(v))))
            self._maybe_save(force=False)

    # internals
    def _cache_key(self, task: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(task, sort_keys=True, default=str).encode()).hexdigest()[:32]

    def _sig(self, task: Dict[str, Any]) -> str:
        return f"{task.get('type','unknown')}_{hashlib.md5(json.dumps(task, sort_keys=True, default=str).encode()).hexdigest()[:12]}"

    def _contexts(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"type": "struct", "len": len(str(task)), "depth": self._depth(task)},
            {"type": "semantic", "domain": self._domain(task), "intent": self._intent(task)},
            {"type": "temporal", "ts": time.time()},
        ]

    def _choose_strategy(self, task: Dict[str, Any], q: QuantumLearningState) -> str:
        cpx = len(str(task)); conf = q.collapse()
        pool = ["hierarchical_decomposition", "transfer_learning_enhanced", "exploration_focused", "exploitation_optimized", "ensemble_boosted", "adaptive_hybrid"]
        def base(s: str) -> float:
            return {
                "hierarchical_decomposition": 1.0 + (cpx / 10000) + (1 - conf),
                "transfer_learning_enhanced": 0.9 + conf * 0.5 + min(cpx / 5000, 0.3),
                "exploration_focused": 0.6 + (1 - conf) * 0.8,
                "exploitation_optimized": 0.7 + conf * 0.6,
                "ensemble_boosted": 0.9 + min(cpx / 8000, 0.3),
                "adaptive_hybrid": 0.85 + abs(conf - 0.5) * 0.3,
            }.get(s, 0.5)
        bonus = {s: (0.5 + 0.5 * float(np.mean(self.strategy_perf[s][-20:])) if self.strategy_perf[s] else 1.0) for s in pool}
        scored = {s: base(s) * bonus[s] for s in pool}
        return max(scored, key=scored.get)

    def _lr(self, sig: str, strat: str) -> float:
        base = float(self.learning_rates.get(sig, 0.001))
        mult = {
            "hierarchical_decomposition": 0.5, "transfer_learning_enhanced": 1.2, "exploration_focused": 2.0,
            "exploitation_optimized": 0.8, "ensemble_boosted": 1.1, "adaptive_hybrid": 1.0
        }.get(strat, 1.0)
        return float(max(1e-6, base * mult))

    def _plan(self, task: Dict[str, Any], strat: str) -> Dict[str, Any]:
        phases = {
            "hierarchical_decomposition": ["decompose", "learn", "integrate", "validate"],
            "transfer_learning_enhanced": ["identify_source", "adapt", "fine_tune", "evaluate"],
            "exploration_focused": ["explore", "sample", "evaluate", "exploit"],
            "exploitation_optimized": ["exploit", "refine", "optimize", "validate"],
            "ensemble_boosted": ["route_models", "fuse", "score", "audit"],
            "adaptive_hybrid": ["branch", "run", "compare", "select"],
        }.get(strat, ["analyze", "learn", "apply", "evaluate"])
        return {
            "phases": phases,
            "resources": self._resources(task),
            "checkpoints": [
                {"phase": "init", "metrics": ["setup_time"]},
                {"phase": "learn", "metrics": ["acc", "loss", "conv_rate"]},
                {"phase": "validate", "metrics": ["perf", "robustness"]},
                {"phase": "complete", "metrics": ["score", "efficiency", "quality"]},
            ],
            "fallbacks": ["exploitation_optimized", "transfer_learning_enhanced"],
        }

    def _resources(self, task: Dict[str, Any]) -> Dict[str, int]:
        L = len(str(task))
        return {"cpu_threads": int(max(1, min(8, L // 1000))), "memory_mb": int(max(64, min(4096, L // 100))), "time_budget_s": int(max(10, min(900, L // 50)))}

    def _maybe_save(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self._last_save) > 300:
            self.store.save("lr_map", dict(self.learning_rates))
            self.store.save("meta_params", dict(self.meta))
            self._last_save = now

    # helpers
    def _depth(self, x: Any, d: int = 0) -> int:
        if isinstance(x, dict): return max([self._depth(v, d + 1) for v in x.values()] + [d])
        if isinstance(x, (list, tuple)): return max([self._depth(v, d + 1) for v in x] + [d])
        return d

    def _domain(self, task: Dict[str, Any]) -> str:
        s = str(task).lower()
        if any(k in s for k in ("code", "def ", "class ", "import ")): return "software"
        if any(k in s for k in ("bug", "exception", "traceback")): return "debugging"
        if any(k in s for k in ("optimiz", "perf", "speed")): return "optimization"
        return "general"

    def _intent(self, task: Dict[str, Any]) -> str:
        s = str(task).lower()
        if any(k in s for k in ("fix", "repair")): return "repair"
        if any(k in s for k in ("optimiz", "improv")): return "optimize"
        if any(k in s for k in ("analy", "understand")): return "analysis"
        return "process"

# ------------------------ CLI ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SLS Pro â€” Hardened")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("process", help="Process a single task (JSON)")
    p1.add_argument("task", type=str, help="JSON task or inline string")

    p2 = sub.add_parser("analyze", help="Analyze a project path")
    p2.add_argument("path", type=str)

    p3 = sub.add_parser("batch", help="Batch process tasks from JSON lines file")
    p3.add_argument("file", type=str, help="Path to .jsonl with one task per line")

    p4 = sub.add_parser("health", help="Print health JSON")

    p5 = sub.add_parser("export-metrics", help="Write metrics JSON to file")
    p5.add_argument("--out", type=str, default="superlearning_metrics.json")

    args = parser.parse_args()
    sys = SuperLearningSystemPro()

    if args.cmd == "process":
        try:
            task = json.loads(args.task)
        except Exception:
            task = {"type": "demo", "payload": args.task}
        print(json.dumps(sys.process_task(task), indent=2))
    elif args.cmd == "analyze":
        print(json.dumps(sys.analyze_project(args.path), indent=2))
    elif args.cmd == "batch":
        tasks: List[Dict[str, Any]] = []
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: tasks.append(json.loads(line))
                except Exception: tasks.append({"type": "demo", "payload": line})
        out = sys.batch_process(tasks)
        print(json.dumps(out, indent=2))
    elif args.cmd == "health":
        print(json.dumps(sys.health(), indent=2))
    elif args.cmd == "export-metrics":
        p = sys.export_metrics_json(args.out)
        print(json.dumps({"written": p}, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":  # pragma: no cover
    main()
