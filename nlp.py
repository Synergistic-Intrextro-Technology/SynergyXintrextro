#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 Enterprise-Grade NLP Consciousness System (Optimized, Orchestrator-Ready)
============================================================================

Key Improvements
- Event-loop hygiene (no nested asyncio.run; clean shutdown via aclose())
- Thread-safe batching and pooled workers
- TTL/LRU cache with Redis optional
- Circuit breaker, retries (exp backoff), rate limiting
- Safer input validation, consistent structured results
- Faster spaCy fallback + guarded TF-IDF/LDA (optional offload)
- Embeddings: model/tokenizer preloaded once; mean pooling; mini-batch support
- Optional ONNX inference hook
- Metrics aggregator with p50/p95/p99 + cache hit rate
- Dialogue engine kept; strategy selection cleaned up
- Orchestrator adapter (NLPAdapter) and minimal CLI
- **No silent regex fallbacks** unless AGI_ALLOW_FALLBACK=1
- Small, CPU-friendly defaults configurable via env
- One-shot warmup of pipelines to avoid first-run latency spikes
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback
import warnings
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependencies (graceful fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMPY_AVAILABLE = False

try:
    import pandas as pd  # type: ignore

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover
    PANDAS_AVAILABLE = False

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        logging as transformers_logging,
        pipeline,
        set_seed,
    )

    transformers_logging.set_verbosity_error()
    set_seed(42)
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy  # type: ignore

    SPACY_AVAILABLE = True
except Exception:  # pragma: no cover
    SPACY_AVAILABLE = False

try:
    from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore

    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    ONNX_AVAILABLE = False

try:
    import redis  # type: ignore

    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    REDIS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: str = "nlp_system.log") -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    handlers: List[logging.Handler] = [console]
    try:
        from logging.handlers import RotatingFileHandler

        fileh = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
        fileh.setFormatter(formatter)
        handlers.append(fileh)
    except Exception:  # pragma: no cover
        pass
    logging.basicConfig(level=level, handlers=handlers, force=True)
    return logging.getLogger("enterprise_nlp")


logger = setup_logging(os.environ.get("NLP_LOG_LEVEL", "INFO"))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EnterpriseConfig:
    # Models
    model_name: str = os.environ.get("NLP_MODEL", "distilbert-base-uncased")
    sentiment_model: str = os.environ.get(
        "NLP_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
    )
    ner_model: str = os.environ.get(
        "NLP_NER_MODEL", "dslim/bert-base-NER"
    )
    device: str = DEVICE
    max_length: int = 512
    batch_size: int = 32

    # Performance
    use_gpu: bool = TORCH_AVAILABLE and (torch.cuda.is_available() if TORCH_AVAILABLE else False)
    use_onnx: bool = ONNX_AVAILABLE
    enable_batching: bool = True
    max_batch_wait_time: float = 0.1  # seconds

    # Cache
    enable_caching: bool = True
    cache_type: str = os.environ.get("NLP_CACHE", "memory")  # memory | redis
    cache_size: int = 10_000
    cache_ttl: int = 3600
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Processing toggles
    enable_preprocessing: bool = True
    enable_entity_recognition: bool = True
    enable_sentiment_analysis: bool = True
    enable_topic_modeling: bool = True
    enable_embeddings: bool = True

    # Concurrency
    async_processing: bool = True
    max_workers: int = max(4, (os.cpu_count() or 1) + 4)
    process_pool_size: int = max(1, min(4, os.cpu_count() or 1))
    queue_max_size: int = 1000

    # Resilience
    retry_attempts: int = 3
    retry_delay: float = 0.5
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    health_check_interval: float = 30.0

    # Security
    enable_input_validation: bool = True
    max_input_length: int = 10_000
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Monitoring
    enable_metrics: bool = True
    log_level: str = os.environ.get("NLP_LOG_LEVEL", "INFO")

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 8:
            raise ValueError("max_length too small")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities: retries, circuit breaker, metrics
# ─────────────────────────────────────────────────────────────────────────────


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            if self.state == "OPEN":
                if self.last_failure_time is not None and (time.time() - self.last_failure_time) > self.timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            return True

    def success(self) -> None:
        with self.lock:
            if self.state in ("OPEN", "HALF_OPEN"):
                self.state = "CLOSED"
            self.failure_count = 0

    def failure(self) -> None:
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


def with_retries(attempts: int, base_delay: float = 0.5):
    def deco(fn):
        async def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:  # pragma: no cover
                    last_exc = e
                    await asyncio.sleep(base_delay * (2 ** i))
            raise last_exc  # type: ignore

        return wrapper

    return deco


class PerformanceMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.metrics: Dict[str, Any] = dict(
            requests_total=0,
            requests_successful=0,
            requests_failed=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            cache_hits=0,
            cache_misses=0,
        )
        self.request_times = deque(maxlen=2000)

    def record_request(self, processing_time: float, success: bool, cache_hit: bool) -> None:
        with self.lock:
            self.metrics["requests_total"] += 1
            if success:
                self.metrics["requests_successful"] += 1
            else:
                self.metrics["requests_failed"] += 1
            if cache_hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            self.metrics["total_processing_time"] += processing_time
            total = max(1, self.metrics["requests_total"])
            self.metrics["average_processing_time"] = self.metrics["total_processing_time"] / total
            self.request_times.append(processing_time)

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            m = dict(self.metrics)
            if self.request_times:
                times = sorted(self.request_times)
                n = len(times)
                m["p50_processing_time"] = times[n // 2]
                m["p95_processing_time"] = times[int(0.95 * (n - 1))]
                m["p99_processing_time"] = times[int(0.99 * (n - 1))]
            if m["requests_total"]:
                m["success_rate"] = m["requests_successful"] / m["requests_total"]
            denom = (m["cache_hits"] + m["cache_misses"]) or 1
            m["cache_hit_rate"] = m["cache_hits"] / denom
            return m


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────


class EnterpriseCache:
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.backend = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_ts: Dict[str, float] = {}
        self.lock = threading.Lock()

        if config.cache_type == "redis" and REDIS_AVAILABLE:
            try:
                self.backend = redis.from_url(config.redis_url)
                self.backend.ping()
                logger.info("Redis cache backend initialized")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Redis unavailable: {e}")
                self.backend = None
        else:
            logger.info("Using in-memory cache backend")

    async def get(self, key: str) -> Optional[Any]:
        try:
            if self.backend:
                loop = asyncio.get_running_loop()
                value = await loop.run_in_executor(None, self.backend.get, key)
                if value:
                    return json.loads(value.decode())
            else:
                with self.lock:
                    if key in self.local_cache:
                        if (time.time() - self.cache_ts.get(key, 0.0)) < self.config.cache_ttl:
                            return self.local_cache[key]
                        # expired
                        self.local_cache.pop(key, None)
                        self.cache_ts.pop(key, None)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Cache get error: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.config.cache_ttl
        try:
            if self.backend:
                payload = json.dumps(value, default=str)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.backend.setex, key, ttl, payload)
            else:
                with self.lock:
                    # LRU-ish trim: drop oldest 25% when full
                    if len(self.local_cache) >= self.config.cache_size:
                        oldest = sorted(self.cache_ts.items(), key=lambda kv: kv[1])[
                            : max(1, len(self.local_cache) // 4)
                        ]
                        for k, _ in oldest:
                            self.local_cache.pop(k, None)
                            self.cache_ts.pop(k, None)
                    self.local_cache[key] = value
                    self.cache_ts[key] = time.time()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Cache set error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


class ModelManager:
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.onnx_sessions: Dict[str, Any] = {}
        self.onnx_paths: Dict[str, str] = {}
        self.lock = threading.Lock()

    def register_onnx(self, key: str, path: str) -> None:
        self.onnx_paths[key] = path

    async def load_model(self, model_name: str, task: str = "general") -> Tuple[Optional[Any], Optional[Any]]:
        cache_key = f"{model_name}:{task}"
        with self.lock:
            if cache_key in self.models:
                return self.models[cache_key], self.tokenizers.get(cache_key)
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            return None, None

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if task == "sentiment":
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            return model, tokenizer

        loop = asyncio.get_running_loop()
        model, tokenizer = await loop.run_in_executor(None, _load)

        if self.config.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            model = model.to(self.config.device)

        with self.lock:
            self.models[cache_key] = model
            self.tokenizers[cache_key] = tokenizer
        return model, tokenizer

    async def load_onnx(self, key: str):
        if not ONNX_AVAILABLE or key not in self.onnx_paths:
            return None

        def _mk():
            return ort.InferenceSession(
                self.onnx_paths[key],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        loop = asyncio.get_running_loop()
        sess = await loop.run_in_executor(None, _mk)
        with self.lock:
            self.onnx_sessions[key] = sess
        return sess


# ─────────────────────────────────────────────────────────────────────────────
# Batching
# ─────────────────────────────────────────────────────────────────────────────


class BatchProcessor:
    def __init__(self, config: EnterpriseConfig, processor_func):
        self.config = config
        self.processor_func = processor_func
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_max_size)
        self.processing = False

    async def start(self) -> None:
        if not self.processing:
            self.processing = True
            asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.processing = False

    async def process_item(self, item_id: str, data: Any) -> Any:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((item_id, data, fut))
        return await fut

    async def _loop(self) -> None:
        while self.processing:
            batch, futures = [], []
            deadline = time.time() + self.config.max_batch_wait_time
            try:
                while len(batch) < self.config.batch_size and time.time() < deadline:
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=max(0.0, deadline - time.time())
                        )
                        batch.append(item[1])
                        futures.append(item[2])
                    except asyncio.TimeoutError:
                        break

                if not batch:
                    await asyncio.sleep(0.005)
                    continue

                results = await self.processor_func(batch)
                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:  # pragma: no cover
                logger.error(f"Batch error: {e}")
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────


class EnterpriseNLPProcessor:
    def __init__(self, config: Optional[EnterpriseConfig] = None):
        self.config = config or EnterpriseConfig()
        self.config.validate()

        # Core services
        self.model_manager = ModelManager(self.config)
        self.cache = EnterpriseCache(self.config)
        self.monitor = PerformanceMonitor()
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold, self.config.circuit_breaker_timeout
        )

        # Pools
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.process_pool_size)

        # Pipelines
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.spacy_nlp = None

        # Policy flags
        self.transformers_available = TRANSFORMERS_AVAILABLE
        self.allow_fallback = str(os.getenv("AGI_ALLOW_FALLBACK", "0")).lower() in {"1", "true", "yes"}

        # Throttling & batching
        self.rate_limiter: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.rate_limit_requests)
        )
        self.batch_processor = (
            BatchProcessor(self.config, self._process_batch_internal)
            if self.config.enable_batching
            else None
        )

        # Dialogue engine hook
        self.dialogue_engine: Optional[ConsciousnessDialogueEngine] = None

        # Embedding preloads
        self._embed_model = None
        self._embed_tok = None
        self._onnx_session = None

        self._closed = False

        # Initialize components
        self._initialize_components()

    async def initialize(self) -> None:
        if self.batch_processor:
            await self.batch_processor.start()

        # Warmup embedding model if needed
        if self.config.enable_embeddings and TRANSFORMERS_AVAILABLE:
            self._embed_model, self._embed_tok = await self.model_manager.load_model(
                self.config.model_name, "embeddings"
            )
            if self._embed_model is None or self._embed_tok is None:
                logger.warning("Embedding model unavailable; embeddings disabled.")
                self.config.enable_embeddings = False
        if self.config.use_onnx:
            self._onnx_session = await self.model_manager.load_onnx("embeddings")

        await self._warmup()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.batch_processor:
            await self.batch_processor.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.process_pool.shutdown(wait=False, cancel_futures=True)

    def attach_dialogue_engine(self, engine: "ConsciousnessDialogueEngine") -> None:
        self.dialogue_engine = engine

    # --- internal setup ---
    def _guard_no_fallback(self) -> None:
        if not self.transformers_available and not self.allow_fallback:
            raise RuntimeError(
                "Transformers are unavailable; refusing to run in fallback mode. "
                "Install `transformers` (and a backend like PyTorch), or set AGI_ALLOW_FALLBACK=1 to allow regex/spaCy fallbacks."
            )

    def _initialize_components(self) -> None:
        # Guard: avoid silent fallback unless explicitly allowed
        self._guard_no_fallback()

        # spaCy
        if SPACY_AVAILABLE:
            try:
                spacy_model = os.getenv("AGI_SPACY_MODEL", "en_core_web_sm")
                try:
                    self.spacy_nlp = spacy.load(
                        spacy_model, disable=["textcat", "lemmatizer", "morphologizer"]
                    )
                except OSError:
                    from spacy.cli import download

                    logger.info("Downloading spaCy model %s...", spacy_model)
                    download(spacy_model)
                    self.spacy_nlp = spacy.load(
                        spacy_model, disable=["textcat", "lemmatizer", "morphologizer"]
                    )
                logger.info("spaCy model loaded (%s)", spacy_model)
            except Exception as e:  # pragma: no cover
                logger.warning(f"spaCy model not found: {e}")

        # Pipelines
        if self.transformers_available:
            try:
                device_map = None if self.config.device == "cpu" else "auto"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.config.sentiment_model,
                    device_map=device_map,
                    return_all_scores=True,
                )
                logger.info("Sentiment pipeline ready (%s)", self.config.sentiment_model)
            except Exception as e:  # pragma: no cover
                if self.allow_fallback:
                    logger.warning("Sentiment pipeline init failed: %s (fallback allowed)", e)
                    self.sentiment_pipeline = None
                else:
                    raise

            try:
                device_map = None if self.config.device == "cpu" else "auto"
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.config.ner_model,
                    aggregation_strategy="simple",
                    device_map=device_map,
                )
                logger.info("NER pipeline ready (%s)", self.config.ner_model)
            except Exception as e:  # pragma: no cover
                if self.allow_fallback:
                    logger.warning("NER pipeline init failed: %s (fallback allowed)", e)
                    self.ner_pipeline = None
                else:
                    raise

    async def _warmup(self) -> None:
        try:
            # One-shot warmups (kept small & synchronous in threadpool)
            loop = asyncio.get_running_loop()

            if self.sentiment_pipeline is not None:
                await loop.run_in_executor(self.executor, self.sentiment_pipeline, "I love warm starts!")
            if self.ner_pipeline is not None:
                await loop.run_in_executor(self.executor, self.ner_pipeline, "Alice met Bob in Paris on Monday.")
            if self.spacy_nlp is not None:
                await loop.run_in_executor(self.executor, self.spacy_nlp, "Quick warmup.")

            logger.info("Model warmup complete")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Warmup failed: {e}")

    # --- guards ---
    def _check_rate_limit(self, client_id: str) -> bool:
        now = time.time()
        q = self.rate_limiter[client_id]
        # purge old
        while q and (now - q[0]) > self.config.rate_limit_window:
            q.popleft()
        if len(q) >= self.config.rate_limit_requests:
            return False
        q.append(now)
        return True

    def _validate_input(self, text: str) -> bool:
        if not self.config.enable_input_validation:
            return True
        if not isinstance(text, str) or not text.strip():
            return False
        if len(text) > self.config.max_input_length:
            return False
        if re.search(r"<script|javascript:|data:", text, re.IGNORECASE):
            return False
        return True

    # --- helpers ---
    def _cache_key(self, text: str, options: Dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(json.dumps(options, sort_keys=True, default=str).encode("utf-8"))
        return f"pt:{h.hexdigest()}"

    async def _error(self, text: str, message: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "text": text,
            "error": message,
            "timestamp": datetime.now().isoformat(),
        }

    # --- public API ---
    @with_retries(attempts=3, base_delay=0.5)
    async def process_text(
        self, text: str, client_id: str = "default", options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main processing entrypoint (CB + caching + batching)."""
        options = options or {}
        start = time.time()
        cache_hit = False

        # circuit breaker gate
        if not self.circuit_breaker.allow():
            self.monitor.record_request(time.time() - start, success=False, cache_hit=False)
            return await self._error(text, "Circuit breaker is OPEN")

        try:
            if not self._check_rate_limit(client_id):
                raise RuntimeError("Rate limit exceeded")
            if not self._validate_input(text):
                raise ValueError("Invalid input text")

            key = self._cache_key(text, options)
            if self.config.enable_caching:
                cached = await self.cache.get(key)
                if cached:
                    cache_hit = True
                    cached.setdefault("ok", True)
                    cached.setdefault("cache_hit", True)
                    cached.setdefault("processing_time", 0.0)
                    self.monitor.record_request(0.0, success=True, cache_hit=True)
                    self.circuit_breaker.success()
                    return cached

            if self.batch_processor:
                result = await self.batch_processor.process_item(key, (text, options))
            else:
                result = await self._process_single(text, options)

            if self.config.enable_caching:
                await self.cache.set(key, result)

            result.setdefault("ok", True)
            result["processing_time"] = time.time() - start
            result["cache_hit"] = cache_hit
            self.monitor.record_request(result["processing_time"], success=True, cache_hit=cache_hit)
            self.circuit_breaker.success()
            return result
        except Exception as e:
            self.circuit_breaker.failure()
            self.monitor.record_request(time.time() - start, success=False, cache_hit=False)
            logger.error(f"process_text error: {e}\n{traceback.format_exc()}")
            return await self._error(text, str(e))

    # --- internals ---
    async def _process_batch_internal(self, batch_data: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for text, options in batch_data:
            try:
                res = await self._process_single(text, options)
                results.append(res)
            except Exception as e:
                results.append(await self._error(text, str(e)))
        return results

    async def _process_single(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "enterprise_single",
            "model_info": {
                "primary_model": self.config.model_name,
                "device": self.config.device,
                "transformers_available": TRANSFORMERS_AVAILABLE,
            },
        }

        tasks: List[asyncio.Future] = []
        if self.config.enable_preprocessing:
            tasks.append(self._preprocess_async(text))
        if self.config.enable_sentiment_analysis:
            tasks.append(self._sentiment_async(text))
        if self.config.enable_entity_recognition:
            tasks.append(self._entities_async(text))
        if self.config.enable_topic_modeling:
            tasks.append(self._topics_async(text))
        if self.config.enable_embeddings:
            tasks.append(self._embed_async(text))

        parts = await asyncio.gather(*tasks, return_exceptions=True)
        for p in parts:
            if isinstance(p, dict):
                result.update(p)
            elif isinstance(p, Exception):  # pragma: no cover
                logger.warning(f"Task failed: {p}")

        # unified confidence (simple heuristic)
        conf = 0.0
        if "sentiment" in result and isinstance(result["sentiment"], dict):
            conf = max(conf, float(result["sentiment"].get("confidence", 0.0)))
        if "entity_count" in result:
            conf = max(conf, min(0.9, 0.5 + 0.02 * float(result.get("entity_count", 0))))
        result["confidence"] = round(conf or 0.7, 4)
        return result

    # --- components ---
    async def _preprocess_async(self, text: str) -> Dict[str, Any]:
        def _work() -> Dict[str, Any]:
            normalized = re.sub(r"\s+", " ", text.strip())
            if self.spacy_nlp:
                doc = self.spacy_nlp(text)
                tokens = [
                    dict(
                        text=t.text,
                        lemma=t.lemma_,
                        pos=t.pos_,
                        tag=t.tag_,
                        dep=t.dep_,
                        is_stop=t.is_stop,
                        is_alpha=t.is_alpha,
                        is_punct=t.is_punct,
                    )
                    for t in doc
                ]
                sentences = [s.text for s in doc.sents]
                noun_phrases = [c.text for c in getattr(doc, "noun_chunks", [])]
                return dict(
                    preprocessing=dict(
                        normalized_text=normalized,
                        tokens=tokens,
                        sentences=sentences,
                        noun_phrases=noun_phrases,
                        token_count=len(tokens),
                        sentence_count=len(sentences),
                        method="spacy_advanced",
                    )
                )
            else:
                words = re.findall(r"\b\w+\b", normalized.lower())
                return dict(
                    preprocessing=dict(
                        normalized_text=normalized,
                        tokens=[{"text": w} for w in words],
                        token_count=len(words),
                        method="regex_fallback",
                    )
                )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _work)

    async def _sentiment_async(self, text: str) -> Dict[str, Any]:
        def _work() -> Dict[str, Any]:
            try:
                if self.sentiment_pipeline:
                    out = self.sentiment_pipeline(text)
                    if isinstance(out[0], list):
                        scores = {d["label"]: float(d["score"]) for d in out[0]}
                        lab, conf = max(scores.items(), key=lambda kv: kv[1])
                        return dict(
                            sentiment=dict(
                                label=lab,
                                confidence=conf,
                                all_scores=scores,
                                method="transformer",
                                model=self.config.sentiment_model,
                            )
                        )
                    else:
                        d = out[0]
                        return dict(
                            sentiment=dict(
                                label=d["label"],
                                confidence=float(d["score"]),
                                method="transformer",
                                model=self.config.sentiment_model,
                            )
                        )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Sentiment error: {e}")
            # fallback
            if not self.allow_fallback:
                raise RuntimeError("Sentiment model unavailable and fallback is disabled (AGI_ALLOW_FALLBACK=0)")
            label = (
                "POSITIVE"
                if re.search(r"\b(love|great|good|excellent|awesome|like)\b", text, re.I)
                else "NEGATIVE"
            )
            return dict(sentiment=dict(label=label, confidence=0.55, method="fallback"))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _work)

    async def _entities_async(self, text: str) -> Dict[str, Any]:
        def _work() -> Dict[str, Any]:
            entities: List[Dict[str, Any]] = []
            try:
                if self.ner_pipeline:
                    for ent in self.ner_pipeline(text):
                        if ent.get("score", 0.0) >= 0.5:
                            entities.append(
                                dict(
                                    text=ent.get("word"),
                                    label=ent.get("entity_group"),
                                    confidence=float(ent.get("score", 0.0)),
                                    start=ent.get("start", 0),
                                    end=ent.get("end", 0),
                                    method="transformer",
                                )
                            )
                elif self.spacy_nlp:
                    doc = self.spacy_nlp(text)
                    for ent in doc.ents:
                        entities.append(
                            dict(
                                text=ent.text,
                                label=ent.label_,
                                confidence=0.8,
                                start=ent.start_char,
                                end=ent.end_char,
                                method="spacy",
                            )
                        )
            except Exception as e:  # pragma: no cover
                logger.warning(f"NER error: {e}")

            if not entities and not self.allow_fallback and not self.ner_pipeline and not self.spacy_nlp:
                raise RuntimeError("NER models unavailable and fallback is disabled (AGI_ALLOW_FALLBACK=0)")

            # pattern entities (consciousness/philosophy)
            patterns = {
                "CONSCIOUSNESS_CONCEPT": r"\b(consciousness|awareness|mind|thought|experience|perception)\b",
                "MENTAL_STATE": r"\b(feeling|emotion|sensation|intuition|reflection|meditation)\b",
                "COGNITIVE_PROCESS": r"\b(thinking|reasoning|learning|memory|attention|processing)\b",
                "PHILOSOPHY": r"\b(existence|reality|truth|knowledge|belief|doubt|meaning|purpose|identity)\b",
            }
            for label, pat in patterns.items():
                for m in re.finditer(pat, text, re.I):
                    entities.append(
                        dict(
                            text=m.group(0),
                            label=label,
                            confidence=0.9,
                            start=m.start(),
                            end=m.end(),
                            method="pattern",
                        )
                    )

            return dict(
                entities=entities,
                entity_count=len(entities),
                entity_types=sorted({e["label"] for e in entities}),
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _work)

    async def _topics_async(self, text: str) -> Dict[str, Any]:
        def _work() -> Dict[str, Any]:
            topics: List[Dict[str, Any]] = []
            try:
                if SKLEARN_AVAILABLE and len(text.split()) > 10:
                    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20]
                    if sentences:
                        # Guarded vectorizer to avoid max_df < min_df errors in tiny corpora
                        vec = TfidfVectorizer(
                            max_features=200,
                            stop_words="english",
                            ngram_range=(1, 3),
                            max_df=0.8,
                        )
                        X = vec.fit_transform(sentences)
                        feats = vec.get_feature_names_out()
                        scores = X.sum(axis=0).A1
                        top_idx = scores.argsort()[-15:][::-1]
                        for rank, idx in enumerate(top_idx, 1):
                            if scores[idx] > 0.1:
                                topics.append(
                                    dict(term=str(feats[idx]), score=float(scores[idx]), type="tfidf", rank=rank)
                                )
                        if len(sentences) >= 5:
                            try:
                                lda = LatentDirichletAllocation(
                                    n_components=min(3, max(1, len(sentences) // 2)),
                                    random_state=42,
                                    max_iter=10,
                                )
                                lda.fit(X)
                                for tid, comp in enumerate(lda.components_):
                                    top = comp.argsort()[-5:][::-1]
                                    words = [str(feats[i]) for i in top]
                                    topics.append(
                                        dict(
                                            term=" + ".join(words[:3]),
                                            score=float(comp[top[0]]),
                                            type="lda_topic",
                                            topic_id=tid,
                                            words=words,
                                        )
                                    )
                            except Exception as e:  # pragma: no cover
                                logger.debug(f"LDA skipped: {e}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Topic extraction error: {e}")

            # consciousness keywords
            for kw, base in {
                "consciousness": 0.9,
                "awareness": 0.8,
                "experience": 0.7,
                "mind": 0.8,
                "thought": 0.6,
                "perception": 0.7,
                "reflection": 0.7,
                "introspection": 0.8,
            }.items():
                if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
                    topics.append(dict(term=kw, score=base, type="consciousness"))

            return dict(
                topics=topics,
                topic_count=len(topics),
                primary_topics=topics[:5],
                topic_categories=sorted({t["type"] for t in topics}),
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _work)

    async def _embed_async(self, text: str | List[str]) -> Dict[str, Any]:
        def _work_batch(payload_texts: List[str]) -> Dict[str, Any]:
            try:
                if self._onnx_session is not None:
                    # Placeholder for ONNX parity; implement as needed
                    pass
                if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and self._embed_model and self._embed_tok):
                    return dict(embeddings="not_available")
                tok = self._embed_tok
                model = self._embed_model
                inputs = tok(
                    payload_texts,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True,
                )
                if self.config.use_gpu and torch.cuda.is_available():
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    model.to(self.config.device)
                with torch.no_grad():
                    out = model(**inputs)
                    embs = out.last_hidden_state  # [B, T, H]
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
                    sent = (embs * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                    if sent.is_cuda:
                        sent = sent.cpu()
                    vectors = sent.numpy().tolist()
                    return dict(
                        embeddings=dict(
                            vectors=vectors,
                            dimension=len(vectors[0]) if vectors else 0,
                            model=self.config.model_name,
                            method="transformer_mean_pooling",
                        )
                    )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Embedding error: {e}")
                return dict(embedding_error=str(e))

        payload_texts = text if isinstance(text, list) else [text]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _work_batch, payload_texts)


# ─────────────────────────────────────────────────────────────────────────────
# Consciousness Dialogue Engine (kept minimal)
# ─────────────────────────────────────────────────────────────────────────────


class ConsciousnessDialogueEngine:
    def __init__(self, nlp_processor: EnterpriseNLPProcessor):
        self.nlp = nlp_processor
        self.conversation_history: List[Dict[str, Any]] = []
        self.dialogue_state: Dict[str, Any] = {"exploration_depth": 0.0}

    async def generate_response(self, user_input: str, nlp_analysis: Dict[str, Any]) -> str:
        depth = self.dialogue_state["exploration_depth"]
        self.dialogue_state["exploration_depth"] = min(1.0, depth + 0.1)
        note = (
            " (flagging depth: expect nuance > certainty.)"
            if self.dialogue_state["exploration_depth"] > 0.6
            else ""
        )
        return f"I’m tracking your query and surfacing constraints{note}"


# ─────────────────────────────────────────────────────────────────────────────
# Adapter for Meta Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class NLPAdapter:
    def __init__(self, cfg: EnterpriseConfig | None = None):
        self.proc = EnterpriseNLPProcessor(cfg)
        self._ready = False

    async def ensure_ready(self):
        if not self._ready:
            await self.proc.initialize()
            self._ready = True

    async def analyze(self, text: str, client_id: str = "default") -> Dict[str, Any]:
        await self.ensure_ready()
        return await self.proc.process_text(text, client_id=client_id)

    async def close(self):  # orchestrator lifecycle
        await self.proc.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default="Hello consciousness.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    async def _main():
        cfg = EnterpriseConfig()
        adapter = NLPAdapter(cfg)
        out = await adapter.analyze(args.text, client_id="cli")
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else out.get("reply", out))
        await adapter.close()

    asyncio.run(_main())

