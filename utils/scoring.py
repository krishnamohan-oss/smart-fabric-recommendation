"""
Multi-criteria scoring algorithm and dynamic weight calculation for fabric recommendation.
Dynamically adjusts weights based on user priorities and normalizes them to 100%.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


BASELINE_WEIGHTS = {
    "sustainability": 0.30,
    "climate": 0.20,
    "comfort": 0.15,
    "durability": 0.15,
    "cost": 0.10,
    "performance": 0.10,
}

PRIORITY_MULTIPLIERS = {
    "Low": 0.60,
    "Medium": 1.00,
    "High": 1.60,
}

BUDGET_COST_MULTIPLIERS = {
    "Low": 2.00,       # Budget is tight; affordability is critical
    "Medium": 1.00,    # Balanced cost sensitivity
    "High": 0.50,      # Premium budget; cost is less constrained
}

CLIMATE_COLUMN_MAP = {
    "Hot": "suitability_hot",
    "Hot & Humid": "suitability_hot_humid",
    "Moderate": "suitability_moderate",
    "Cold": "suitability_cold",
    "Rainy": "suitability_rainy",
}


def calculate_dynamic_weights(
    sustainability_priority: str = "Medium",
    comfort_priority: str = "Medium",
    durability_priority: str = "Medium",
    budget: str = "Medium",
    has_performance_focus: bool = True,
) -> Dict[str, float]:
    """
    Calculate dynamically adjusted and normalized criterion weights.
    Guarantees that the sum of all weights strictly equals 1.0 (100%).
    """
    sust_mult = PRIORITY_MULTIPLIERS.get(sustainability_priority, 1.0)
    comf_mult = PRIORITY_MULTIPLIERS.get(comfort_priority, 1.0)
    dur_mult = PRIORITY_MULTIPLIERS.get(durability_priority, 1.0)
    cost_mult = BUDGET_COST_MULTIPLIERS.get(budget, 1.0)
    perf_mult = 1.30 if has_performance_focus else 1.00

    raw_weights = {
        "sustainability": BASELINE_WEIGHTS["sustainability"] * sust_mult,
        "climate": BASELINE_WEIGHTS["climate"],
        "comfort": BASELINE_WEIGHTS["comfort"] * comf_mult,
        "durability": BASELINE_WEIGHTS["durability"] * dur_mult,
        "cost": BASELINE_WEIGHTS["cost"] * cost_mult,
        "performance": BASELINE_WEIGHTS["performance"] * perf_mult,
    }

    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return {k: round(v, 4) for k, v in BASELINE_WEIGHTS.items()}

    normalized_weights = {
        k: round(v / total_weight, 4) for k, v in raw_weights.items()
    }

    # Ensure slight rounding difference sums cleanly to 1.0
    diff = 1.0 - sum(normalized_weights.values())
    normalized_weights["sustainability"] = round(
        normalized_weights["sustainability"] + diff, 4
    )

    return normalized_weights


def extract_performance_score(row: pd.Series, performance_requirement: str) -> float:
    """Extract or synthesize performance score based on specific requirement."""
    req = performance_requirement.strip().lower()

    if "breathab" in req:
        return float(row.get("breathability", 70))
    elif "moisture" in req or "quick dry" in req or "wicking" in req:
        return float(row.get("moisture_management", 70))
    elif "thermal" in req or "warm" in req or "insulat" in req:
        return float(row.get("thermal_insulation", 70))
    elif "water resist" in req or "rain" in req or "repell" in req:
        return float(row.get("water_resistance", 50))
    elif "stretch" in req or "flexib" in req:
        return float(row.get("stretchability", 50))
    else:
        # Balanced everyday performance metric
        b = row.get("breathability", 70)
        m = row.get("moisture_management", 70)
        d = row.get("durability", 70)
        return float(0.4 * b + 0.3 * m + 0.3 * d)


def calculate_fabric_scores(
    fabric_row: pd.Series,
    weights: Dict[str, float],
    climate: str,
    performance_requirement: str,
) -> Dict[str, float]:
    """
    Calculate individual criterion scores (0-100) and weighted overall score for a fabric.
    """
    climate_col = CLIMATE_COLUMN_MAP.get(climate, "suitability_moderate")
    climate_score = float(fabric_row.get(climate_col, 70))

    sustainability_score = float(fabric_row.get("sustainability_score", 70))

    # Comfort synthesized from tactile softness and breathability
    raw_comfort = float(fabric_row.get("comfort", 70))
    raw_breathability = float(fabric_row.get("breathability", 70))
    comfort_score = round(0.70 * raw_comfort + 0.30 * raw_breathability, 2)

    durability_score = float(fabric_row.get("durability", 70))

    # Affordability score represents cost friendliness (100 = low cost, 0 = luxury expensive)
    cost_score = float(fabric_row.get("affordability_score", 70))

    performance_score = round(extract_performance_score(fabric_row, performance_requirement), 2)

    overall_general_score = (
        weights["sustainability"] * sustainability_score
        + weights["climate"] * climate_score
        + weights["comfort"] * comfort_score
        + weights["durability"] * durability_score
        + weights["cost"] * cost_score
        + weights["performance"] * performance_score
    )

    return {
        "sustainability_score": round(sustainability_score, 1),
        "climate_score": round(climate_score, 1),
        "comfort_score": round(comfort_score, 1),
        "durability_score": round(durability_score, 1),
        "cost_score": round(cost_score, 1),
        "performance_score": round(performance_score, 1),
        "general_recommendation_score": round(min(100.0, max(0.0, overall_general_score)), 2),
    }
