"""
Intrextro Meta Learning Optimizer
Advanced hyperparameter optimization and self-evaluation system
"""
import logging
import random
import time
from dataclasses import dataclass
from typing import Any,Callable,Dict,List,Optional,Tuple

import numpy as np
from skopt import gp_minimize
from synergyx_main import NeuralEmotionModule

from intrextro_synergyx_bridge import IntrextroSynergyXBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result container for optimization operations"""

    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Tuple[Dict, float]]
    execution_time: float
    domain: str


class EmotionAwareIntrextro:
    def __init__(self):
        self.emotion_module = NeuralEmotionModule(
            config={
                "embedding_dim": 128,
                "hidden_dim": 64,
                "emotion_categories": [
                    "optimism",
                    "pessimism",
                    "uncertainty",
                    "confidence",
                    "fear",
                    "excitement",
                ],
            }
        )

    def analyze_market_sentiment(self, financial_data):
        # Your existing confidence + emotion analysis
        emotion_context = self.emotion_module.forward({"embedding": financial_data})
        return enhanced_confidence_with_emotion


class HyperparameterOptimizer:
    """Advanced hyperparameter search for Intrextro modules with domain awareness."""

    def __init__(
        self,
        module=None,
        param_space: Dict = None,
        eval_fn: Callable = None,
        n_calls: int = 20,
    ):
        self.module = module
        self.param_space = param_space or {}
        self.eval_fn = eval_fn
        self.n_calls = n_calls
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        self.domain_specific_configs = {}

    def set_domain_config(self, domain: str, config: Dict):
        """Set domain-specific optimization configuration"""
        self.domain_specific_configs[domain] = config
        logger.info(f"Set domain config for {domain}: {config}")

    def optimize(self, domain: str = "general") -> OptimizationResult:
        """Run optimization with domain awareness"""
        start_time = time.time()

        # Use domain-specific config if available
        if domain in self.domain_specific_configs:
            domain_config = self.domain_specific_configs[domain]
            param_space = domain_config.get("param_space", self.param_space)
            n_calls = domain_config.get("n_calls", self.n_calls)
        else:
            param_space = self.param_space
            n_calls = self.n_calls

        if not param_space:
            logger.warning("No parameter space defined, using default optimization")
            return self._default_optimization(domain)

        def objective(params):
            try:
                # Set params on module if module exists
                if self.module:
                    param_dict = dict(zip(param_space.keys(), params))
                    for key, value in param_dict.items():
                        setattr(self.module, key, value)

                    # Evaluate with domain context
                    if self.eval_fn:
                        score = self.eval_fn(self.module, domain)
                    else:
                        score = self._default_eval(self.module, domain)
                else:
                    # Simulate evaluation if no module
                    score = random.uniform(0.3, 0.9)

                # Record history
                param_dict = dict(zip(param_space.keys(), params))
                self.optimization_history.append((param_dict, score))

                return -score  # Minimize negative score (maximize score)

            except Exception as e:
                logger.error(f"Error in optimization objective: {e}")
                return -0.1  # Poor score for failed evaluations

        try:
            res = gp_minimize(objective, list(param_space.values()), n_calls=n_calls)
            self.best_params = dict(zip(param_space.keys(), res.x))
            self.best_score = -res.fun

            # Apply best params to module
            if self.module:
                for key, value in self.best_params.items():
                    setattr(self.module, key, value)

            execution_time = time.time() - start_time

            logger.info(
                f"Optimization complete for domain {domain}. Best score: {self.best_score:.4f}"
            )

            return OptimizationResult(
                best_params=self.best_params,
                best_score=self.best_score,
                optimization_history=self.optimization_history,
                execution_time=execution_time,
                domain=domain,
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._default_optimization(domain)

    def _default_optimization(self, domain: str) -> OptimizationResult:
        """Fallback optimization when no proper config is available"""
        logger.info(f"Running default optimization for domain: {domain}")

        # Simulate some optimization work
        time.sleep(0.1)

        default_params = {"learning_rate": 0.001, "batch_size": 32, "dropout_rate": 0.1}

        return OptimizationResult(
            best_params=default_params,
            best_score=0.75,
            optimization_history=[(default_params, 0.75)],
            execution_time=0.1,
            domain=domain,
        )

    def _default_eval(self, module, domain: str) -> float:
        """Default evaluation function"""
        # Simulate evaluation based on domain
        domain_scores = {
            "finance": random.uniform(0.6, 0.9),
            "healthcare": random.uniform(0.5, 0.8),
            "general": random.uniform(0.4, 0.7),
        }
        return domain_scores.get(domain, random.uniform(0.3, 0.6))


class SelfEvaluator:
    """Advanced performance evaluation and drift detection system."""

    def __init__(
        self, module=None, eval_interval: int = 10, drift_threshold: float = 0.6
    ):
        self.module = module
        self.eval_interval = eval_interval
        self.drift_threshold = drift_threshold
        self.performance_history = []
        self.domain_performance = {}  # domain -> [(timestamp, score)]
        self.last_eval_time = time.time()
        self.adaptation_triggers = []

    def evaluate(self, context: Dict, domain: str = "general") -> Dict[str, Any]:
        """Evaluate module performance with context and domain awareness"""
        try:
            current_time = time.time()

            if self.module and hasattr(self.module, "process"):
                result = self.module.process(context)
                score = result.get("confidence", 0.5)
            else:
                # Simulate evaluation
                score = self._simulate_evaluation(context, domain)

            # Record performance
            self.performance_history.append((current_time, score, domain))

            # Track domain-specific performance
            if domain not in self.domain_performance:
                self.domain_performance[domain] = []
            self.domain_performance[domain].append((current_time, score))

            # Check for performance drift
            drift_detected = self._detect_drift(domain)

            evaluation_result = {
                "score": score,
                "domain": domain,
                "timestamp": current_time,
                "drift_detected": drift_detected,
                "performance_trend": self._get_performance_trend(domain),
            }

            if drift_detected:
                self._trigger_adaptation(domain, score)

            self.last_eval_time = current_time
            return evaluation_result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "score": 0.0,
                "domain": domain,
                "timestamp": time.time(),
                "drift_detected": False,
                "error": str(e),
            }

    def _simulate_evaluation(self, context: Dict, domain: str) -> float:
        """Simulate evaluation when no module is available"""
        # Base score varies by domain
        domain_base_scores = {
            "finance": 0.7,
            "healthcare": 0.65,
            "education": 0.6,
            "general": 0.5,
        }

        base_score = domain_base_scores.get(domain, 0.5)

        # Add some realistic variation
        variation = random.uniform(-0.2, 0.2)

        # Context complexity affects score
        context_complexity = len(str(context)) / 1000.0  # Simple complexity measure
        complexity_penalty = min(context_complexity * 0.1, 0.3)

        final_score = max(0.0, min(1.0, base_score + variation - complexity_penalty))
        return final_score

    def _detect_drift(self, domain: str) -> bool:
        """Detect performance drift for specific domain"""
        if domain not in self.domain_performance:
            return False

        domain_history = self.domain_performance[domain]

        if len(domain_history) < self.eval_interval:
            return False

        # Get recent performance
        recent_scores = [score for _, score in domain_history[-self.eval_interval :]]
        recent_avg = np.mean(recent_scores)

        # Compare with historical average
        if len(domain_history) > self.eval_interval * 2:
            historical_scores = [
                score for _, score in domain_history[: -self.eval_interval]
            ]
            historical_avg = np.mean(historical_scores)

            # Drift if recent performance significantly worse
            if recent_avg < historical_avg * 0.8:  # 20% degradation
                return True

        # Also check absolute threshold
        if recent_avg < self.drift_threshold:
            return True

        return False

    def _get_performance_trend(self, domain: str) -> str:
        """Get performance trend for domain"""
        if domain not in self.domain_performance:
            return "unknown"

        domain_history = self.domain_performance[domain]

        if len(domain_history) < 5:
            return "insufficient_data"

        recent_scores = [score for _, score in domain_history[-5:]]

        # Simple trend analysis
        if len(recent_scores) >= 3:
            early_avg = np.mean(recent_scores[:2])
            late_avg = np.mean(recent_scores[-2:])

            if late_avg > early_avg * 1.05:
                return "improving"
            elif late_avg < early_avg * 0.95:
                return "declining"
            else:
                return "stable"

        return "stable"

    def _trigger_adaptation(self, domain: str, current_score: float):
        """Trigger adaptation when drift is detected"""
        adaptation_event = {
            "timestamp": time.time(),
            "domain": domain,
            "trigger_score": current_score,
            "reason": "performance_drift",
        }

        self.adaptation_triggers.append(adaptation_event)

        logger.warning(
            f"Performance drift detected for domain {domain}. "
            f"Current score: {current_score:.3f}. Triggering adaptation."
        )

        # Here you could trigger re-optimization or other adaptation strategies
        # For now, we just log and record the event

    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get performance summary for a specific domain"""
        if domain not in self.domain_performance:
            return {"domain": domain, "status": "no_data"}

        domain_history = self.domain_performance[domain]
        scores = [score for _, score in domain_history]

        return {
            "domain": domain,
            "total_evaluations": len(scores),
            "average_score": np.mean(scores),
            "latest_score": scores[-1] if scores else 0.0,
            "trend": self._get_performance_trend(domain),
            "drift_events": len(
                [t for t in self.adaptation_triggers if t["domain"] == domain]
            ),
        }


class MetaFeedbackLoop:
    """Advanced meta-learning system that learns optimal strategies across domains."""

    def __init__(self, memory_limit: int = 1000):
        self.strategy_performance = {}  # domain -> {strategy: [scores]}
        self.strategy_contexts = {}  # domain -> {strategy: [contexts]}
        self.strategy_timestamps = {}  # domain -> {strategy: [timestamps]}
        self.memory_limit = memory_limit
        self.adaptation_rules = {}  # domain -> {condition: strategy}
        self.cross_domain_patterns = {}  # patterns that work across domains

    def record(self, domain: str, strategy: str, score: float, context: Dict = None):
        """Record strategy performance with context"""
        try:
            current_time = time.time()

            # Initialize domain tracking if needed
            if domain not in self.strategy_performance:
                self.strategy_performance[domain] = {}
                self.strategy_contexts[domain] = {}
                self.strategy_timestamps[domain] = {}

            # Record performance
            if strategy not in self.strategy_performance[domain]:
                self.strategy_performance[domain][strategy] = []
                self.strategy_contexts[domain][strategy] = []
                self.strategy_timestamps[domain][strategy] = []

            self.strategy_performance[domain][strategy].append(score)
            self.strategy_contexts[domain][strategy].append(context or {})
            self.strategy_timestamps[domain][strategy].append(current_time)

            # Maintain memory limit
            self._enforce_memory_limit(domain, strategy)

            # Update cross-domain patterns
            self._update_cross_domain_patterns(strategy, score, context)

            logger.debug(
                f"Recorded strategy '{strategy}' for domain '{domain}' with score {score:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to record strategy performance: {e}")

    def best_strategy(self, domain: str, context: Dict = None) -> Optional[str]:
        """Get best strategy for domain with optional context matching"""
        try:
            if domain not in self.strategy_performance:
                return self._get_cross_domain_recommendation(context)

            strategies = self.strategy_performance[domain]
            if not strategies:
                return self._get_cross_domain_recommendation(context)

            # If context provided, try context-aware selection
            if context:
                context_strategy = self._get_context_aware_strategy(domain, context)
                if context_strategy:
                    return context_strategy

            # Fallback to average performance
            avg_scores = {}
            for strategy, scores in strategies.items():
                if scores:  # Only consider strategies with recorded performance
                    # Weight recent performance more heavily
                    weights = np.exp(np.linspace(-1, 0, len(scores)))
                    weighted_avg = np.average(scores, weights=weights)
                    avg_scores[strategy] = weighted_avg

            if avg_scores:
                best_strategy = max(avg_scores, key=avg_scores.get)
                logger.info(
                    f"Best strategy for domain '{domain}': {best_strategy} "
                    f"(score: {avg_scores[best_strategy]:.3f})"
                )
                return best_strategy

            return None

        except Exception as e:
            logger.error(f"Failed to get best strategy: {e}")
            return None

    def _get_context_aware_strategy(self, domain: str, context: Dict) -> Optional[str]:
        """Find best strategy based on context similarity"""
        try:
            if domain not in self.strategy_contexts:
                return None

            best_strategy = None
            best_score = -1

            for strategy, contexts in self.strategy_contexts[domain].items():
                if not contexts:
                    continue

                # Find most similar context
                similarities = []
                for stored_context in contexts:
                    similarity = self._calculate_context_similarity(
                        context, stored_context
                    )
                    similarities.append(similarity)

                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity > 0.7:  # High similarity threshold
                        # Get corresponding performance scores
                        strategy_scores = self.strategy_performance[domain][strategy]
                        if strategy_scores:
                            avg_score = np.mean(strategy_scores)
                            combined_score = avg_score * max_similarity

                            if combined_score > best_score:
                                best_score = combined_score
                                best_strategy = strategy

            return best_strategy

        except Exception as e:
            logger.error(f"Context-aware strategy selection failed: {e}")
            return None

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        try:
            if not context1 or not context2:
                return 0.0

            # Simple similarity based on common keys and values
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0

            matches = 0
            total = len(common_keys)

            for key in common_keys:
                if context1[key] == context2[key]:
                    matches += 1
                elif isinstance(context1[key], (int, float)) and isinstance(
                    context2[key], (int, float)
                ):
                    # Numerical similarity
                    diff = abs(context1[key] - context2[key])
                    max_val = max(abs(context1[key]), abs(context2[key]), 1)
                    similarity = 1 - (diff / max_val)
                    matches += max(0, similarity)

            return matches / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"Context similarity calculation failed: {e}")
            return 0.0

    def _get_cross_domain_recommendation(self, context: Dict = None) -> Optional[str]:
        """Get strategy recommendation based on cross-domain patterns"""
        try:
            if not self.cross_domain_patterns:
                return "default_optimization"  # Fallback strategy

            # Find patterns that work well across multiple domains
            pattern_scores = {}
            for pattern, data in self.cross_domain_patterns.items():
                if data["domain_count"] >= 2:  # Works in at least 2 domains
                    pattern_scores[pattern] = data["average_score"]

            if pattern_scores:
                best_pattern = max(pattern_scores, key=pattern_scores.get)
                logger.info(f"Using cross-domain pattern: {best_pattern}")
                return best_pattern

            return "default_optimization"

        except Exception as e:
            logger.error(f"Cross-domain recommendation failed: {e}")
            return "default_optimization"

    def _update_cross_domain_patterns(self, strategy: str, score: float, context: Dict):
        """Update patterns that work across domains"""
        try:
            if strategy not in self.cross_domain_patterns:
                self.cross_domain_patterns[strategy] = {
                    "scores": [],
                    "domains": set(),
                    "average_score": 0.0,
                    "domain_count": 0,
                }

            pattern = self.cross_domain_patterns[strategy]
            pattern["scores"].append(score)

            # Update average (keep last 100 scores)
            if len(pattern["scores"]) > 100:
                pattern["scores"] = pattern["scores"][-100:]

            pattern["average_score"] = np.mean(pattern["scores"])
            pattern["domain_count"] = len(pattern["domains"])

        except Exception as e:
            logger.error(f"Failed to update cross-domain patterns: {e}")

    def _enforce_memory_limit(self, domain: str, strategy: str):
        """Enforce memory limits to prevent unbounded growth"""
        try:
            if len(self.strategy_performance[domain][strategy]) > self.memory_limit:
                # Keep most recent entries
                self.strategy_performance[domain][strategy] = self.strategy_performance[
                    domain
                ][strategy][-self.memory_limit :]
                self.strategy_contexts[domain][strategy] = self.strategy_contexts[
                    domain
                ][strategy][-self.memory_limit :]
                self.strategy_timestamps[domain][strategy] = self.strategy_timestamps[
                    domain
                ][strategy][-self.memory_limit :]

        except Exception as e:
            logger.error(f"Memory limit enforcement failed: {e}")

    def get_strategy_summary(self, domain: str = None) -> Dict[str, Any]:
        """Get comprehensive strategy performance summary"""
        try:
            if domain:
                # Domain-specific summary
                if domain not in self.strategy_performance:
                    return {"domain": domain, "strategies": {}}

                strategies = {}
                for strategy, scores in self.strategy_performance[domain].items():
                    if scores:
                        strategies[strategy] = {
                            "average_score": np.mean(scores),
                            "latest_score": scores[-1],
                            "total_uses": len(scores),
                            "trend": (
                                "improving"
                                if len(scores) > 1 and scores[-1] > scores[0]
                                else "stable"
                            ),
                        }

                return {"domain": domain, "strategies": strategies}
            else:
                # Global summary
                summary = {
                    "total_domains": len(self.strategy_performance),
                    "cross_domain_patterns": len(self.cross_domain_patterns),
                    "domains": {},
                }

                for domain in self.strategy_performance:
                    summary["domains"][domain] = self.get_strategy_summary(domain)

                return summary

        except Exception as e:
            logger.error(f"Strategy summary generation failed: {e}")
            return {"error": str(e)}


class EnhancedIntrextroMetaOptimizer:
    def __init__(self):
        # Your existing initialization
        self.synergyx_bridge = IntrextroSynergyXBridge()
        self.emotion_history = []
        self.enhanced_confidence_history = []

    def optimize_with_emotion(self, scraped_data, confidence_scores):
        """Enhanced optimization with emotion awareness"""

        # Process through SynergyX
        synergyx_result = self.synergyx_bridge.process_intrextro_data(
            scraped_data, confidence_scores
        )

        # Store results
        self.emotion_history.append(synergyx_result["emotion_analysis"])
        self.enhanced_confidence_history.append(synergyx_result["enhanced_confidence"])

        # Adapt SynergyX based on performance
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )
        self.synergyx_bridge.adapt_from_performance(avg_confidence, confidence_scores)

        return synergyx_result

    def get_emotion_insights(self):
        """Get emotion analysis insights"""
        if not self.emotion_history:
            return None

        latest = self.emotion_history[-1]
        return {
            "current_emotion": latest["dominant_emotion"],
            "emotion_trend": self._analyze_emotion_trend(),
            "confidence_improvement": self._calculate_confidence_boost(),
        }

    def _analyze_emotion_trend(self):
        """Analyze emotion trends over time"""
        if len(self.emotion_history) < 3:
            return "insufficient_data"

        recent_emotions = [e["dominant_emotion"] for e in self.emotion_history[-3:]]
        if len(set(recent_emotions)) == 1:
            return f"stable_{recent_emotions[0]}"
        else:
            return "volatile"

    def _calculate_confidence_boost(self):
        """Calculate average confidence improvement from emotion analysis"""
        if len(self.enhanced_confidence_history) < 2:
            return 0.0

        recent_avg = sum(self.enhanced_confidence_history[-3:]) / min(
            3, len(self.enhanced_confidence_history)
        )
        baseline = 0.75  # Your typical confidence baseline
        return recent_avg - baseline


class MetaLearningOptimizer:
    """Main meta-learning optimization system that coordinates all components."""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()

        # Initialize components
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.self_evaluator = SelfEvaluator(
            eval_interval=self.config.get("eval_interval", 10),
            drift_threshold=self.config.get("drift_threshold", 0.6),
        )
        self.feedback_loop = MetaFeedbackLoop(
            memory_limit=self.config.get("memory_limit", 1000)
        )

        # State tracking
        self.active_domains = set()
        self.optimization_history = {}
        self.current_strategies = {}  # domain -> current strategy
        self.performance_metrics = {}

        logger.info("MetaLearningOptimizer initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "eval_interval": 10,
            "drift_threshold": 0.6,
            "memory_limit": 1000,
            "optimization_calls": 20,
            "adaptation_threshold": 0.7,
            "cross_domain_learning": True,
        }

    def optimize_for_domain(self, domain: str, data: Dict = None) -> Dict[str, Any]:
        """Main optimization entry point for a specific domain"""
        try:
            logger.info(f"Starting optimization for domain: {domain}")
            start_time = time.time()

            # Add domain to active domains
            self.active_domains.add(domain)

            # Get recommended strategy from feedback loop
            recommended_strategy = self.feedback_loop.best_strategy(domain, data)
            if not recommended_strategy:
                recommended_strategy = "bayesian_optimization"

            logger.info(
                f"Using strategy '{recommended_strategy}' for domain '{domain}'"
            )

            # Configure optimization based on domain and strategy
            self._configure_for_domain(domain, recommended_strategy, data)

            # Run optimization
            optimization_result = self.hyperparameter_optimizer.optimize(domain)

            # Evaluate the optimization result
            evaluation_result = self.self_evaluator.evaluate(
                context={"optimization_result": optimization_result.__dict__},
                domain=domain,
            )

            # Record strategy performance
            self.feedback_loop.record(
                domain=domain,
                strategy=recommended_strategy,
                score=evaluation_result["score"],
                context=data,
            )

            # Update tracking
            self.current_strategies[domain] = recommended_strategy
            self._update_performance_metrics(
                domain, optimization_result, evaluation_result
            )

            # Prepare result
            result = {
                "domain": domain,
                "strategy_used": recommended_strategy,
                "optimization_result": optimization_result.__dict__,
                "evaluation_result": evaluation_result,
                "execution_time": time.time() - start_time,
                "status": "success",
            }

            # Store in history
            if domain not in self.optimization_history:
                self.optimization_history[domain] = []
            self.optimization_history[domain].append(result)

            logger.info(
                f"Optimization complete for domain '{domain}'. Score: {evaluation_result['score']:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Optimization failed for domain '{domain}': {e}")
            return {
                "domain": domain,
                "status": "error",
                "error": str(e),
                "execution_time": (
                    time.time() - start_time if "start_time" in locals() else 0
                ),
            }

    def _configure_for_domain(self, domain: str, strategy: str, data: Dict = None):
        """Configure optimization components for specific domain and strategy"""
        try:
            # Domain-specific parameter spaces
            domain_param_spaces = {
                "finance": {
                    "learning_rate": (0.0001, 0.01),
                    "batch_size": (16, 128),
                    "dropout_rate": (0.1, 0.5),
                    "regularization": (0.001, 0.1),
                },
                "healthcare": {
                    "learning_rate": (0.0005, 0.005),
                    "batch_size": (8, 64),
                    "dropout_rate": (0.2, 0.6),
                    "attention_heads": (4, 16),
                },
                "education": {
                    "learning_rate": (0.001, 0.01),
                    "batch_size": (32, 256),
                    "dropout_rate": (0.1, 0.4),
                    "sequence_length": (128, 1024),
                },
                "general": {
                    "learning_rate": (0.0001, 0.01),
                    "batch_size": (16, 128),
                    "dropout_rate": (0.1, 0.5),
                },
            }

            # Strategy-specific configurations
            strategy_configs = {
                "bayesian_optimization": {"n_calls": 20},
                "random_search": {"n_calls": 30},
                "grid_search": {"n_calls": 15},
                "evolutionary": {"n_calls": 25},
            }

            # Set parameter space
            param_space = domain_param_spaces.get(
                domain, domain_param_spaces["general"]
            )
            self.hyperparameter_optimizer.param_space = param_space

            # Set strategy-specific config
            strategy_config = strategy_configs.get(strategy, {"n_calls": 20})
            self.hyperparameter_optimizer.n_calls = strategy_config["n_calls"]

            # Set domain-specific config
            domain_config = {
                "param_space": param_space,
                "n_calls": strategy_config["n_calls"],
                "strategy": strategy,
            }
            self.hyperparameter_optimizer.set_domain_config(domain, domain_config)

            logger.debug(
                f"Configured optimization for domain '{domain}' with strategy '{strategy}'"
            )

        except Exception as e:
            logger.error(f"Configuration failed for domain '{domain}': {e}")

    def _update_performance_metrics(
        self, domain: str, opt_result: OptimizationResult, eval_result: Dict
    ):
        """Update performance metrics tracking"""
        try:
            if domain not in self.performance_metrics:
                self.performance_metrics[domain] = {
                    "total_optimizations": 0,
                    "average_score": 0.0,
                    "best_score": 0.0,
                    "total_time": 0.0,
                    "scores_history": [],
                }

            metrics = self.performance_metrics[domain]
            metrics["total_optimizations"] += 1
            metrics["total_time"] += opt_result.execution_time

            current_score = eval_result["score"]
            metrics["scores_history"].append(current_score)
            metrics["average_score"] = np.mean(metrics["scores_history"])
            metrics["best_score"] = max(metrics["best_score"], current_score)

            # Keep only recent scores for average calculation
            if len(metrics["scores_history"]) > 100:
                metrics["scores_history"] = metrics["scores_history"][-100:]
                metrics["average_score"] = np.mean(metrics["scores_history"])

        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")

    def initialize_for_domain(self, domain: str) -> Dict[str, Any]:
        """Initialize optimization system for a new domain"""
        try:
            logger.info(f"Initializing meta-learning for domain: {domain}")

            # Add to active domains
            self.active_domains.add(domain)

            # Initialize domain-specific configurations
            self._configure_for_domain(domain, "bayesian_optimization")

            # Initialize performance tracking
            if domain not in self.performance_metrics:
                self.performance_metrics[domain] = {
                    "total_optimizations": 0,
                    "average_score": 0.0,
                    "best_score": 0.0,
                    "total_time": 0.0,
                    "scores_history": [],
                }

            return {
                "domain": domain,
                "status": "initialized",
                "active_domains": list(self.active_domains),
                "default_strategy": "bayesian_optimization",
            }

        except Exception as e:
            logger.error(f"Domain initialization failed: {e}")
            return {"domain": domain, "status": "error", "error": str(e)}

    def get_domain_status(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive status for a domain"""
        try:
            if domain not in self.active_domains:
                return {"domain": domain, "status": "not_initialized"}

            # Get performance metrics
            metrics = self.performance_metrics.get(domain, {})

            # Get strategy summary
            strategy_summary = self.feedback_loop.get_strategy_summary(domain)

            # Get evaluation summary
            eval_summary = self.self_evaluator.get_domain_summary(domain)

            return {
                "domain": domain,
                "status": "active",
                "current_strategy": self.current_strategies.get(domain, "unknown"),
                "performance_metrics": metrics,
                "strategy_summary": strategy_summary,
                "evaluation_summary": eval_summary,
                "optimization_history_count": len(
                    self.optimization_history.get(domain, [])
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get domain status: {e}")
            return {"domain": domain, "status": "error", "error": str(e)}


def orchestrated_entry_point(domain: str, data: Dict = None) -> Dict[str, Any]:
    """
    Main entry point called by the Intrextro orchestrator.
    This function handles all meta-learning optimization requests.
    """
    try:
        logger.info(
            f"Meta-learning orchestrator entry point called for domain: {domain}"
        )

        # Initialize global optimizer if not exists
        if not hasattr(orchestrated_entry_point, "optimizer"):
            orchestrated_entry_point.optimizer = MetaLearningOptimizer()
            logger.info("Created new MetaLearningOptimizer instance")

        optimizer = orchestrated_entry_point.optimizer

        # Determine operation type from data
        operation = "optimize"  # default operation
        if data:
            operation = data.get("operation", "optimize")

        # Route to appropriate operation
        if operation == "optimize":
            result = optimizer.optimize_for_domain(domain, data)
        elif operation == "initialize":
            result = optimizer.initialize_for_domain(domain)
        elif operation == "status":
            result = optimizer.get_domain_status(domain)
        elif operation == "summary":
            result = _get_comprehensive_summary(optimizer, domain)
        else:
            logger.warning(f"Unknown operation: {operation}")
            result = optimizer.optimize_for_domain(domain, data)  # fallback

        # Add orchestrator metadata
        result.update(
            {
                "module": "meta_learning",
                "orchestrator_timestamp": time.time(),
                "operation": operation,
            }
        )

        return result

    except Exception as e:
        logger.error(f"Orchestrated entry point failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "module": "meta_learning",
            "domain": domain,
            "operation": data.get("operation", "unknown") if data else "unknown",
            "orchestrator_timestamp": time.time(),
        }


def run_orchestrated(context: Dict) -> Dict[str, Any]:
    """
    Orchestrated entry point called by IntrextroOrchestrator.
    This is the function the orchestrator is actually looking for.
    """
    try:
        logger.info("Meta-learning module called by orchestrator")

        # Extract domain and data from context
        domain = context.get("domain", "general")
        data = context.get("data", {})

        # Add orchestrator context info
        context_id = context.get("context_id", "unknown")

        logger.info(
            f"Processing meta-learning for domain: {domain}, context: {context_id}"
        )

        # Call our existing orchestrated entry point logic
        result = orchestrated_entry_point(domain, data)

        # Add context information
        result.update(
            {
                "context_id": context_id,
                "orchestrator_integration": True,
                "processed_by": "run_orchestrated",
            }
        )

        logger.info(f"Meta-learning completed for domain {domain}")
        return result

    except Exception as e:
        logger.error(f"run_orchestrated failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "module": "meta_learning",
            "context_id": context.get("context_id", "unknown"),
            "domain": context.get("domain", "unknown"),
        }


def _get_comprehensive_summary(
    optimizer: MetaLearningOptimizer, domain: str = None
) -> Dict[str, Any]:
    """Get comprehensive summary of meta-learning system"""
    try:
        if domain:
            # Domain-specific summary
            return {
                "summary_type": "domain_specific",
                "domain": domain,
                "domain_status": optimizer.get_domain_status(domain),
                "strategy_performance": optimizer.feedback_loop.get_strategy_summary(
                    domain
                ),
                "evaluation_metrics": optimizer.self_evaluator.get_domain_summary(
                    domain
                ),
            }
        else:
            # Global summary
            return {
                "summary_type": "global",
                "active_domains": list(optimizer.active_domains),
                "total_domains": len(optimizer.active_domains),
                "global_strategy_performance": optimizer.feedback_loop.get_strategy_summary(),
                "performance_overview": {
                    domain: optimizer.performance_metrics.get(domain, {})
                    for domain in optimizer.active_domains
                },
            }

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return {"error": str(e), "summary_type": "error"}


# Utility functions for external use
def create_optimizer(config: Dict = None) -> MetaLearningOptimizer:
    """Factory function to create a new optimizer instance"""
    return MetaLearningOptimizer(config)


def validate_domain_config(domain: str, config: Dict) -> bool:
    """Validate domain configuration"""
    try:
        required_fields = ["param_space"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field '{field}' in domain config")
                return False

        # Validate parameter space format
        param_space = config["param_space"]
        if not isinstance(param_space, dict):
            logger.error("Parameter space must be a dictionary")
            return False

        for param, bounds in param_space.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                logger.error(
                    f"Parameter '{param}' bounds must be a tuple/list of length 2"
                )
                return False

            if bounds[0] >= bounds[1]:
                logger.error(
                    f"Parameter '{param}' lower bound must be less than upper bound"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Domain config validation failed: {e}")
        return False


def get_supported_strategies() -> List[str]:
    """Get list of supported optimization strategies"""
    return [
        "bayesian_optimization",
        "random_search",
        "grid_search",
        "evolutionary",
        "default_optimization",
    ]


def get_supported_domains() -> List[str]:
    """Get list of pre-configured domains"""
    return ["finance", "healthcare", "education", "general"]


# Example usage and testing functions
def run_example_optimization():
    """Example usage of the meta-learning optimizer"""
    try:
        logger.info("Running example optimization...")

        # Create optimizer
        optimizer = MetaLearningOptimizer()

        # Test different domains
        test_domains = ["finance", "healthcare", "general"]

        for domain in test_domains:
            logger.info(f"Testing domain: {domain}")

            # Initialize domain
            init_result = optimizer.initialize_for_domain(domain)
            logger.info(f"Initialization result: {init_result}")

            # Run optimization
            opt_result = optimizer.optimize_for_domain(domain, {"test_data": "example"})
            logger.info(f"Optimization result: {opt_result['status']}")

            # Get status
            status = optimizer.get_domain_status(domain)
            logger.info(f"Domain status: {status['status']}")

        # Get global summary
        summary = _get_comprehensive_summary(optimizer)
        logger.info(f"Global summary: {summary['summary_type']}")

        return True

    except Exception as e:
        logger.error(f"Example optimization failed: {e}")
        return False


# Module metadata
__version__ = "1.0.0"
__author__ = "Intrextro AI System"
__description__ = "Advanced meta-learning optimization system with domain awareness"

# Export main classes and functions
__all__ = [
    "MetaLearningOptimizer",
    "HyperparameterOptimizer",
    "SelfEvaluator",
    "MetaFeedbackLoop",
    "OptimizationResult",
    "orchestrated_entry_point",
    "create_optimizer",
    "validate_domain_config",
    "get_supported_strategies",
    "get_supported_domains",
    "run_example_optimization",
]

# Main execution for testing
if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting meta-learning optimizer module test...")

    # Test orchestrated entry point
    test_result = orchestrated_entry_point("finance", {"operation": "initialize"})
    logger.info(f"Test result: {test_result}")

    # Run example optimization
    example_success = run_example_optimization()
    logger.info(f"Example optimization success: {example_success}")

    logger.info("Meta-learning optimizer module test complete.")

# Configuration constants
DEFAULT_CONFIG = {
    "eval_interval": 10,
    "drift_threshold": 0.6,
    "memory_limit": 1000,
    "optimization_calls": 20,
    "adaptation_threshold": 0.7,
    "cross_domain_learning": True,
    "logging_level": "INFO",
}

# Domain-specific optimization presets
DOMAIN_PRESETS = {
    "finance": {
        "param_space": {
            "learning_rate": (0.0001, 0.01),
            "batch_size": (16, 128),
            "dropout_rate": (0.1, 0.5),
            "regularization": (0.001, 0.1),
        },
        "optimization_calls": 25,
        "drift_threshold": 0.7,
    },
    "healthcare": {
        "param_space": {
            "learning_rate": (0.0005, 0.005),
            "batch_size": (8, 64),
            "dropout_rate": (0.2, 0.6),
            "attention_heads": (4, 16),
        },
        "optimization_calls": 30,
        "drift_threshold": 0.75,
    },
    "education": {
        "param_space": {
            "learning_rate": (0.001, 0.01),
            "batch_size": (32, 256),
            "dropout_rate": (0.1, 0.4),
            "sequence_length": (128, 1024),
        },
        "optimization_calls": 20,
        "drift_threshold": 0.65,
    },
}

# Strategy performance weights
STRATEGY_WEIGHTS = {
    "bayesian_optimization": 1.0,
    "random_search": 0.8,
    "grid_search": 0.7,
    "evolutionary": 0.9,
    "default_optimization": 0.5,
}

logger.info("Intrextro Meta Learning Optimizer module loaded successfully")
logger.info(f"Module version: {__version__}")
logger.info(f"Supported domains: {get_supported_domains()}")
logger.info(f"Supported strategies: {get_supported_strategies()}")
