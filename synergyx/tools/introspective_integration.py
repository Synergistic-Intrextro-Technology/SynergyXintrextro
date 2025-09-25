"""Introspective integration analysis inspired by reflective pathways."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .base import AnalysisTool


@dataclass
class CapabilityProfile:
    """Normalized attributes for an existing capability module."""

    name: str
    focus: str
    reliability: float
    load: float
    dependencies: Sequence[str]


@dataclass
class IntegrationCandidate:
    """Description of a potential integration expansion."""

    integration_id: str
    integrates_with: Sequence[str]
    expected_gain: float
    complexity: float
    risk_notes: Sequence[str]


FOCUS_WEIGHTS = {
    "control": 1.0,
    "stability": 0.9,
    "exploration": 0.75,
    "support": 0.85,
    "coordination": 0.8,
}


class IntrospectiveIntegrationTool(AnalysisTool):
    """Reflect on current capabilities and chart the next integration arc."""

    @property
    def name(self) -> str:
        return "introspective_integration"

    @property
    def description(self) -> str:
        return (
            "Balance existing capabilities, readiness signals, and integration candidates "
            "to surface the next deliberate expansion path."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "capabilities": {
                    "type": "array",
                    "description": "Known capabilities with reliability and load markers.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "focus": {"type": "string"},
                            "reliability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "load": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["name"],
                    },
                    "default": [],
                },
                "integration_candidates": {
                    "type": "array",
                    "description": "Potential integrations to evaluate.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "integration_id": {"type": "string"},
                            "integrates_with": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "expected_gain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "complexity": {"type": "number", "minimum": 0.0},
                            "risk_notes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["integration_id"],
                    },
                    "default": [],
                },
                "introspection_signals": {
                    "type": "object",
                    "description": "Signals such as friction, momentum, or vision weightings.",
                    "additionalProperties": {"type": "number"},
                    "default": {},
                },
                "expansion_objectives": {
                    "type": "array",
                    "description": "Strategic objectives like lunar-scouting or canopy-mapping.",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
        }

    async def execute(
        self,
        *,
        capabilities: Optional[Sequence[Mapping[str, Any]]] = None,
        integration_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        introspection_signals: Optional[Mapping[str, float]] = None,
        expansion_objectives: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        profiles = [self._parse_capability(item) for item in capabilities or []]
        candidates = [self._parse_candidate(item) for item in integration_candidates or []]
        signals = {str(k).lower(): float(v) for k, v in (introspection_signals or {}).items() if self._is_number(v)}
        objectives = [str(obj).strip().lower() for obj in expansion_objectives or [] if str(obj).strip()]

        capability_matrix = self._capability_matrix(profiles)
        canvas = self._integration_canvas(profiles, objectives)
        prioritised = self._score_candidates(candidates, profiles, signals, objectives)
        reflective = self._reflective_feedback(capability_matrix, signals)

        return {
            "capability_matrix": capability_matrix,
            "integration_canvas": canvas,
            "prioritized_integrations": prioritised,
            "introspective_feedback": reflective,
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_capability(self, payload: Mapping[str, Any]) -> CapabilityProfile:
        name = str(payload.get("name", "")).strip() or "unnamed"
        focus = str(payload.get("focus", "support")).strip().lower() or "support"
        reliability = self._bounded_float(payload.get("reliability"), fallback=0.5)
        load = self._bounded_float(payload.get("load"), fallback=0.4)
        dependencies = [
            str(dep).strip()
            for dep in payload.get("dependencies", []) or []
            if str(dep).strip()
        ]
        return CapabilityProfile(name=name, focus=focus, reliability=reliability, load=load, dependencies=dependencies)

    def _parse_candidate(self, payload: Mapping[str, Any]) -> IntegrationCandidate:
        integration_id = str(payload.get("integration_id", "")).strip() or "candidate"
        integrates_with = [
            str(name).strip()
            for name in payload.get("integrates_with", []) or []
            if str(name).strip()
        ]
        expected_gain = self._bounded_float(payload.get("expected_gain"), fallback=0.35)
        complexity = max(self._safe_float(payload.get("complexity")) or 1.0, 0.1)
        risk_notes = [
            str(note).strip()
            for note in payload.get("risk_notes", []) or []
            if str(note).strip()
        ]
        return IntegrationCandidate(
            integration_id=integration_id,
            integrates_with=integrates_with,
            expected_gain=expected_gain,
            complexity=complexity,
            risk_notes=risk_notes,
        )

    # ------------------------------------------------------------------
    # Capability analysis
    # ------------------------------------------------------------------

    def _capability_matrix(self, profiles: Sequence[CapabilityProfile]) -> Dict[str, Any]:
        if not profiles:
            return {
                "focus_distribution": {},
                "average_reliability": None,
                "load_pressure": None,
                "dependency_clusters": [],
            }

        focus_counts: MutableMapping[str, int] = {}
        reliability_values: List[float] = []
        load_values: List[float] = []
        dependency_map: MutableMapping[str, List[str]] = {}

        for profile in profiles:
            focus_counts[profile.focus] = focus_counts.get(profile.focus, 0) + 1
            reliability_values.append(profile.reliability)
            load_values.append(profile.load)
            for dependency in profile.dependencies:
                dependency_map.setdefault(dependency, []).append(profile.name)

        dependency_clusters = [
            {"dependency": dep, "supported_by": sorted(set(sources))}
            for dep, sources in dependency_map.items()
            if len(sources) > 1
        ]

        return {
            "focus_distribution": focus_counts,
            "average_reliability": round(mean(reliability_values), 3),
            "load_pressure": round(mean(load_values), 3),
            "dependency_clusters": dependency_clusters,
        }

    def _integration_canvas(
        self,
        profiles: Sequence[CapabilityProfile],
        objectives: Sequence[str],
    ) -> Dict[str, Any]:
        """Generate a symbolic canvas linking anchors (stone, canopy, lunar, solar)."""

        anchors = {
            "bedrock": [profile.name for profile in profiles if profile.focus in {"control", "stability"}],
            "canopy": [profile.name for profile in profiles if profile.focus in {"exploration", "support"}],
            "luminous": [profile.name for profile in profiles if profile.reliability >= 0.7],
            "frontier": [profile.name for profile in profiles if profile.load < 0.4],
        }
        motive_themes = []
        if any("lunar" in obj or "night" in obj for obj in objectives):
            motive_themes.append("lunar-reflection")
        if any("solar" in obj or "radiant" in obj or "day" in obj for obj in objectives):
            motive_themes.append("solar-burst")
        if any("forest" in obj or "canopy" in obj for obj in objectives):
            motive_themes.append("canopy-resonance")
        if any("path" in obj or "trail" in obj for obj in objectives):
            motive_themes.append("trail-continuum")
        if not motive_themes:
            motive_themes.append("equilibrium-axis")

        return {
            "anchors": anchors,
            "motive_themes": motive_themes,
            "objectives": objectives,
        }

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: Sequence[IntegrationCandidate],
        profiles: Sequence[CapabilityProfile],
        signals: Mapping[str, float],
        objectives: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        profile_lookup = {profile.name: profile for profile in profiles}
        control_bias = signals.get("control", 0.0)
        discovery_bias = signals.get("discovery", signals.get("exploration", 0.0))
        friction = signals.get("friction", 0.0)
        momentum = signals.get("momentum", 0.0)
        clarity = signals.get("clarity", 0.0)

        themed_objectives = {
            "lunar": any("lunar" in obj or "moon" in obj for obj in objectives),
            "canopy": any("forest" in obj or "canopy" in obj for obj in objectives),
            "solar": any("solar" in obj or "sun" in obj for obj in objectives),
        }

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            anchors = [profile_lookup.get(name) for name in candidate.integrates_with]
            anchors = [profile for profile in anchors if profile is not None]
            anchor_weight = mean(
                FOCUS_WEIGHTS.get(profile.focus, 0.7) * profile.reliability
                for profile in anchors
            ) if anchors else 0.5

            load_penalty = mean(profile.load for profile in anchors) if anchors else 0.3
            load_penalty = min(load_penalty * 0.6, 0.4)

            complexity_penalty = min(candidate.complexity / 10.0, 0.7)
            bias_bonus = (control_bias * 0.2) + (discovery_bias * 0.15) + (momentum * 0.1) + (clarity * 0.1)
            friction_offset = max(0.0, 1.0 - friction * 0.5)

            thematic_bonus = 0.0
            if themed_objectives["lunar"] and any("night" in rn or "shadow" in rn for rn in candidate.risk_notes):
                thematic_bonus += 0.05
            if themed_objectives["canopy"] and any(
                name for name in candidate.integrates_with if "canopy" in name.lower() or "forest" in name.lower()
            ):
                thematic_bonus += 0.05
            if themed_objectives["solar"] and candidate.expected_gain >= 0.6:
                thematic_bonus += 0.05

            score = (
                candidate.expected_gain
                * (0.8 + anchor_weight)
                * friction_offset
            ) - (complexity_penalty + load_penalty) + bias_bonus + thematic_bonus
            score = max(score, 0.0)

            lane = self._lane_alignment(candidate, anchors, signals)
            rationale = self._build_rationale(candidate, anchors, score, lane)
            scored.append((score, {
                "integration_id": candidate.integration_id,
                "score": round(score, 3),
                "lane_alignment": lane,
                "rationale": rationale,
                "phased_path": self._phased_path(candidate, anchors),
            }))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored]

    def _lane_alignment(
        self,
        candidate: IntegrationCandidate,
        anchors: Sequence[CapabilityProfile],
        signals: Mapping[str, float],
    ) -> Dict[str, float]:
        stability_lane = mean(profile.reliability for profile in anchors) if anchors else 0.5
        exploration_lane = signals.get("discovery", signals.get("exploration", 0.0)) + candidate.expected_gain
        synthesis_lane = (signals.get("clarity", 0.0) * 0.5) + (signals.get("momentum", 0.0) * 0.5)

        return {
            "stability_lane": round(min(stability_lane, 1.0), 3),
            "exploration_lane": round(min(exploration_lane, 1.5), 3),
            "synthesis_lane": round(min(synthesis_lane + 0.3, 1.2), 3),
        }

    def _build_rationale(
        self,
        candidate: IntegrationCandidate,
        anchors: Sequence[CapabilityProfile],
        score: float,
        lane: Mapping[str, float],
    ) -> str:
        phrases = []
        if anchors:
            primary = max(anchors, key=lambda prof: prof.reliability)
            phrases.append(
                f"Anchored by {primary.name} ({primary.focus}) with reliability {primary.reliability:.2f}."
            )
        else:
            phrases.append("Integration would establish a new anchor node.")

        if score >= 0.8:
            phrases.append("Projected to expand control surface meaningfully.")
        elif score >= 0.45:
            phrases.append("Delivers balanced growth with manageable risk.")
        else:
            phrases.append("Treat as exploratory spike; monitor for signal drift.")

        if lane["exploration_lane"] > 1.0:
            phrases.append("Exploration lane is energized; schedule a discovery sprint.")
        if lane["stability_lane"] < 0.6:
            phrases.append("Stability lane light—stage safety harness tests.")

        if candidate.risk_notes:
            phrases.append("Risk notes: " + "; ".join(candidate.risk_notes))

        return " ".join(phrases)

    def _phased_path(
        self,
        candidate: IntegrationCandidate,
        anchors: Sequence[CapabilityProfile],
    ) -> List[str]:
        phases = [
            "Diagnostic echo — validate dependencies and telemetry hooks.",
            "Pilot braid — weave integration alongside primary anchor for two cycles.",
            "Load shift — gradually elevate responsibility while monitoring control signals.",
        ]

        if not anchors:
            phases.insert(0, "Foundation rune — prototype baseline scaffolding before pairing.")
        if candidate.complexity > 6:
            phases.append("Red moon watch — keep fallback routine live during ramp.")
        if candidate.expected_gain >= 0.7:
            phases.append("Aurora unlock — broadcast learning across collaborating modules.")
        return phases

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _reflective_feedback(
        self,
        matrix: Mapping[str, Any],
        signals: Mapping[str, float],
    ) -> Dict[str, Any]:
        focus_distribution = matrix.get("focus_distribution", {})
        if not focus_distribution:
            return {
                "summary": "No capabilities registered; establish a core stone before branching.",
                "control_core": None,
                "expansion_vector": None,
                "recommended_checkpoint": "map-first-node",
            }

        dominant_focus = max(focus_distribution.items(), key=lambda item: item[1])[0]
        control_core = "stable" if dominant_focus in {"control", "stability"} else "adaptive"
        expansion_vector = "trail" if signals.get("momentum", 0) > 0.5 else "stillness"

        summary_parts = [
            f"Control core is {control_core} with dominant focus on {dominant_focus}.",
            f"Reliability mean at {matrix.get('average_reliability')} and load pressure at {matrix.get('load_pressure')}.",
        ]

        if signals.get("friction", 0) > 0.4:
            summary_parts.append("Friction elevated — introduce deliberate pauses for recalibration.")
        if signals.get("clarity", 0) >= 0.6:
            summary_parts.append("Clarity signal strong; time to chart a longer arc.")

        return {
            "summary": " ".join(summary_parts),
            "control_core": control_core,
            "expansion_vector": expansion_vector,
            "recommended_checkpoint": "lunar-gate" if expansion_vector == "trail" else "stone-circle",
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _bounded_float(value: Any, *, fallback: float) -> float:
        numeric = IntrospectiveIntegrationTool._safe_float(value)
        if numeric is None:
            return fallback
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

