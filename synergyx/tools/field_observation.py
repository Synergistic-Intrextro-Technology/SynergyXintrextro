"""Field observation analysis inspired by nocturnal control missions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .base import AnalysisTool


VisibilityLevel = {
    "clear": 0.05,
    "light_fog": 0.15,
    "haze": 0.25,
    "low_light": 0.35,
    "shadowed": 0.45,
    "dense_fog": 0.55,
    "storm": 0.65,
    "blackout": 0.8,
}

SPECIES_THREAT = {
    "raven": 0.45,
    "wolf": 0.6,
    "spider": 0.35,
    "bat": 0.25,
    "moth": 0.2,
    "owl": 0.3,
}


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return mean(values) if values else None


def _bounded(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _visibility_penalty(level: str) -> float:
    level = (level or "").lower().replace(" ", "_")
    if level in VisibilityLevel:
        return VisibilityLevel[level]
    if "moon" in level:
        return 0.35
    if "shadow" in level:
        return 0.45
    if "rain" in level or "storm" in level:
        return 0.6
    return 0.1 if level else 0.2


def _species_pressure(entries: Mapping[str, float]) -> float:
    total = 0.0
    weight = 0.0
    for species, count in entries.items():
        base = SPECIES_THREAT.get(species, 0.3)
        if any(keyword in species for keyword in ("venom", "predator", "wolf")):
            base = max(base, 0.6)
        total += base * count
        weight += count
    return _bounded(total / weight if weight else 0.0)


def _normalize_signal(signal: Optional[float]) -> Optional[float]:
    if signal is None:
        return None
    return _bounded(signal / 100.0)


def _link_weight(edge: Mapping[str, Any]) -> float:
    weight = edge.get("weight")
    try:
        return float(weight)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class WindowSnapshot:
    """Normalized metrics for a single observation window."""

    window_id: str
    crew: str
    throughput_per_hour: Optional[float]
    incidents: int
    signal_strength: Optional[float]
    visibility: str
    species_totals: Mapping[str, float]
    ambient_noise: Optional[float]
    notes: Optional[str]


class NocturnalFieldInsightTool(AnalysisTool):
    """Fuse crew cadence, network loads, and wildlife pressure at night."""

    @property
    def name(self) -> str:
        return "nocturnal_field_insight"

    @property
    def description(self) -> str:
        return (
            "Evaluate night operations observations to derive risk, limiting factors, "
            "and recommended control adjustments."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "observation_windows": {
                    "type": "array",
                    "description": "Time-sliced field notes from the shift.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "window_id": {"type": "string"},
                            "crew": {"type": "string"},
                            "tasks_completed": {"type": "number"},
                            "duration_minutes": {"type": "number", "minimum": 0},
                            "incidents": {"type": "integer", "minimum": 0},
                            "signal_strength": {"type": "number"},
                            "visibility": {"type": "string"},
                            "wildlife": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "species": {"type": "string"},
                                        "count": {"type": "number", "minimum": 0},
                                    },
                                    "required": ["species"],
                                },
                                "default": [],
                            },
                            "ambient": {
                                "type": "object",
                                "properties": {
                                    "noise_db": {"type": "number"},
                                    "temperature_c": {"type": "number"},
                                },
                            },
                            "notes": {"type": "string"},
                        },
                    },
                    "default": [],
                },
                "relay_links": {
                    "type": "array",
                    "description": "Communication or supply relays with weightings.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["source", "target", "weight"],
                    },
                    "default": [],
                },
                "environment_markers": {
                    "type": "array",
                    "description": "Environment cues such as red-moon, ground-mist, etc.",
                    "items": {"type": "string"},
                    "default": [],
                },
                "support_assets": {
                    "type": "object",
                    "description": "Portable power, thermal, or rescue asset status.",
                    "properties": {
                        "battery_hours": {"type": "number"},
                        "spare_power_ratio": {"type": "number"},
                        "thermal_margin": {"type": "number"},
                    },
                },
            },
        }

    async def execute(
        self,
        *,
        observation_windows: Optional[Sequence[Mapping[str, Any]]] = None,
        relay_links: Optional[Sequence[Mapping[str, Any]]] = None,
        environment_markers: Optional[Sequence[str]] = None,
        support_assets: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        windows = [self._parse_window(item) for item in observation_windows or []]
        crews = self._aggregate_crews(windows)
        wildlife_pressure = _species_pressure(self._merge_species(windows))
        network_diagnostics = self._network_health(relay_links or [])
        environment_summary = self._environment_profile(environment_markers or [], support_assets or {})

        risk_score, limiting_factors = self._compose_risk(
            crews=crews,
            wildlife_pressure=wildlife_pressure,
            network=network_diagnostics,
            environment=environment_summary,
        )

        return {
            "field_state": {
                "crews": crews,
                "network": network_diagnostics,
                "environment": environment_summary,
                "wildlife_pressure": wildlife_pressure,
                "window_count": len(windows),
            },
            "control_focus": {
                "risk_score": risk_score,
                "limiting_factors": limiting_factors,
                "recommendations": self._recommendations(
                    risk=risk_score,
                    crews=crews,
                    environment=environment_summary,
                    network=network_diagnostics,
                    wildlife_pressure=wildlife_pressure,
                ),
            },
            "notes": self._collect_notes(windows),
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_window(self, payload: Mapping[str, Any]) -> WindowSnapshot:
        duration_minutes = max(float(payload.get("duration_minutes") or 0.0), 0.0)
        tasks = float(payload.get("tasks_completed") or 0.0)
        throughput = None
        if duration_minutes > 0:
            throughput = (tasks / duration_minutes) * 60.0

        wildlife_entries = {}
        for entry in payload.get("wildlife", []) or []:
            species = str(entry.get("species", "unknown")).strip().lower()
            if not species:
                continue
            count = entry.get("count", 1)
            try:
                wildlife_entries[species] = wildlife_entries.get(species, 0.0) + max(float(count), 0.0)
            except (TypeError, ValueError):
                wildlife_entries[species] = wildlife_entries.get(species, 0.0) + 1.0

        ambient = payload.get("ambient") or {}
        ambient_noise = ambient.get("noise_db")
        try:
            ambient_noise = float(ambient_noise) if ambient_noise is not None else None
        except (TypeError, ValueError):
            ambient_noise = None

        signal = payload.get("signal_strength")
        try:
            signal = float(signal) if signal is not None else None
        except (TypeError, ValueError):
            signal = None

        return WindowSnapshot(
            window_id=str(payload.get("window_id", "")) or "window",
            crew=str(payload.get("crew", "unknown")).strip() or "unknown",
            throughput_per_hour=throughput,
            incidents=max(int(payload.get("incidents") or 0), 0),
            signal_strength=_normalize_signal(signal),
            visibility=str(payload.get("visibility", "unknown")).strip().lower(),
            species_totals=wildlife_entries,
            ambient_noise=ambient_noise,
            notes=str(payload.get("notes")) if payload.get("notes") else None,
        )

    def _merge_species(self, windows: Sequence[WindowSnapshot]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for window in windows:
            for species, count in window.species_totals.items():
                totals[species] = totals.get(species, 0.0) + count
        return totals

    def _aggregate_crews(self, windows: Sequence[WindowSnapshot]) -> Dict[str, Dict[str, Any]]:
        grouped: MutableMapping[str, Dict[str, Any]] = {}
        for window in windows:
            crew = window.crew
            stats = grouped.setdefault(
                crew,
                {
                    "windows": 0,
                    "throughput_per_hour": [],
                    "incidents": 0,
                    "signal_strength": [],
                    "visibility_penalties": [],
                    "ambient_noise": [],
                },
            )
            stats["windows"] += 1
            if window.throughput_per_hour is not None:
                stats["throughput_per_hour"].append(window.throughput_per_hour)
            stats["incidents"] += window.incidents
            if window.signal_strength is not None:
                stats["signal_strength"].append(window.signal_strength)
            stats["visibility_penalties"].append(_visibility_penalty(window.visibility))
            if window.ambient_noise is not None:
                stats["ambient_noise"].append(window.ambient_noise)

        summary: Dict[str, Dict[str, Any]] = {}
        for crew, stats in grouped.items():
            total_incidents = stats["incidents"]
            window_count = max(stats["windows"], 1)
            incident_rate = total_incidents / window_count
            summary[crew] = {
                "windows": stats["windows"],
                "incident_rate": round(incident_rate, 2),
                "total_incidents": total_incidents,
                "avg_throughput_per_hour": _safe_mean(stats["throughput_per_hour"]),
                "avg_signal_strength": _safe_mean(stats["signal_strength"]),
                "avg_visibility_penalty": _safe_mean(stats["visibility_penalties"]),
                "avg_ambient_noise": _safe_mean(stats["ambient_noise"]),
            }
        return summary

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _network_health(self, links: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not links:
            return {
                "nodes": 0,
                "links": 0,
                "avg_weight": None,
                "bottleneck": None,
                "stability": 0.5,
            }

        node_weights: MutableMapping[str, List[float]] = {}
        link_weights: List[float] = []
        for edge in links:
            source = str(edge.get("source", "")).strip() or "unknown"
            target = str(edge.get("target", "")).strip() or "unknown"
            weight = _link_weight(edge)
            node_weights.setdefault(source, []).append(weight)
            node_weights.setdefault(target, []).append(weight)
            link_weights.append(weight)

        avg_weight = _safe_mean(link_weights)
        weakest = min(link_weights) if link_weights else None
        stability = _bounded((avg_weight or 0.0) / (abs(weakest) + 1.0)) if weakest is not None else 0.6

        if weakest is not None and weakest < 0:
            stability *= 0.5

        return {
            "nodes": len(node_weights),
            "links": len(link_weights),
            "avg_weight": avg_weight,
            "bottleneck": weakest,
            "stability": round(_bounded(stability), 3),
        }

    def _environment_profile(
        self,
        markers: Sequence[str],
        assets: Mapping[str, Any],
    ) -> Dict[str, Any]:
        lowered = [marker.strip().lower() for marker in markers if isinstance(marker, str)]
        penalties = []
        if any("red moon" in marker for marker in lowered):
            penalties.append(0.3)
        if any(marker for marker in lowered if "rain" in marker or "wet" in marker):
            penalties.append(0.25)
        if any("drone" in marker for marker in lowered):
            penalties.append(0.2)
        if any("predator" in marker for marker in lowered):
            penalties.append(0.35)

        support = {
            "battery_hours": self._safe_float(assets.get("battery_hours")),
            "spare_power_ratio": self._safe_float(assets.get("spare_power_ratio")),
            "thermal_margin": self._safe_float(assets.get("thermal_margin")),
        }

        resilience = 0.5
        available_metrics = [value for value in support.values() if value is not None]
        if available_metrics:
            resilience = _bounded(mean(available_metrics) / 10.0)

        return {
            "markers": lowered,
            "support": support,
            "environment_penalty": round(_bounded(mean(penalties) if penalties else 0.0), 3),
            "support_resilience": round(resilience, 3),
        }

    def _compose_risk(
        self,
        *,
        crews: Mapping[str, Mapping[str, Any]],
        wildlife_pressure: float,
        network: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> Tuple[float, List[str]]:
        factors: List[Tuple[str, float]] = []

        avg_visibility = _safe_mean(
            value.get("avg_visibility_penalty")
            for value in crews.values()
            if value.get("avg_visibility_penalty") is not None
        )
        if avg_visibility is not None:
            factors.append(("visibility", avg_visibility))

        incident_pressure = _safe_mean(value.get("incident_rate") for value in crews.values())
        if incident_pressure is not None:
            factors.append(("incident_rate", min(incident_pressure / 5.0, 0.6)))

        if network:
            stability = network.get("stability")
            if stability is not None:
                factors.append(("network", 0.6 - (stability * 0.4)))

        if wildlife_pressure:
            factors.append(("wildlife", wildlife_pressure * 0.6))

        env_penalty = environment.get("environment_penalty")
        if env_penalty:
            factors.append(("environment", env_penalty))

        support_resilience = environment.get("support_resilience")
        if support_resilience is not None:
            factors.append(("support_gap", max(0.2 - support_resilience * 0.2, 0.0)))

        if not factors:
            return 0.2, []

        total = _bounded(mean(score for _, score in factors))
        limiting = [name for name, score in factors if score >= total * 0.9 and score > 0.1]
        return round(total, 3), limiting

    def _recommendations(
        self,
        *,
        risk: float,
        crews: Mapping[str, Mapping[str, Any]],
        environment: Mapping[str, Any],
        network: Mapping[str, Any],
        wildlife_pressure: float,
    ) -> List[str]:
        actions: List[str] = []

        high_incident_crews = [
            crew
            for crew, stats in crews.items()
            if stats.get("incident_rate", 0) >= 1.0
        ]
        if high_incident_crews:
            actions.append(
                "Rotate crews with elevated incident rates (" + ", ".join(high_incident_crews) + ") to reduce fatigue."
            )

        if network.get("stability") is not None and network["stability"] < 0.4:
            actions.append(
                "Re-route through alternate relays; current weakest link weight %.2f is degrading comms." % network.get("bottleneck", 0.0)
            )

        if wildlife_pressure >= 0.4:
            actions.append(
                "Deploy acoustic deterrents or adjust patrol paths to avoid concentrated wildlife pressure zones."
            )

        if environment.get("environment_penalty", 0) >= 0.3:
            actions.append("Escalate weather countermeasures and tighten safety perimeter for low visibility.")

        support = environment.get("support", {})
        if support.get("battery_hours") is not None and support["battery_hours"] < 4:
            actions.append("Stage recharge kits; battery reserves below 4h could compromise response depth.")

        if not actions:
            if risk <= 0.25:
                actions.append("Maintain current cadence; conditions stable with moderate buffers.")
            else:
                actions.append("Increase monitoring cadence; assess crew telemetry every 30 minutes for drift.")

        return actions

    def _collect_notes(self, windows: Sequence[WindowSnapshot]) -> List[str]:
        notes = []
        for window in windows:
            if window.notes:
                notes.append(f"[{window.crew}] {window.notes}")
        return notes

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
