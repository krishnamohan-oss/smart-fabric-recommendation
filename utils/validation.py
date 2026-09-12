"""
Input validation and bounds checking for recommendation requests.
Ensures application resilience against missing, malformed, or out-of-range user input.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd

VALID_CLIMATES = ["Hot", "Hot & Humid", "Moderate", "Cold", "Rainy"]
VALID_BUDGETS = ["Low", "Medium", "High"]
VALID_PRIORITIES = ["Low", "Medium", "High"]
DEFAULT_GARMENT = "T-shirt"
DEFAULT_CLIMATE = "Moderate"
DEFAULT_BUDGET = "Medium"


def validate_recommendation_inputs(
    garment_type: Any,
    climate: Any,
    budget: Any,
    sustainability_priority: Any,
    comfort_priority: Any,
    durability_priority: Any,
    available_garments: List[str],
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validate and sanitize user recommendation inputs.
    Returns:
        (is_valid, sanitized_inputs, warning_messages)
    """
    warnings: List[str] = []
    sanitized: Dict[str, Any] = {}

    # Validate Garment Type
    if not garment_type or str(garment_type).strip() == "":
        warnings.append(f"Garment type was missing; defaulted to '{DEFAULT_GARMENT}'.")
        sanitized["garment_type"] = DEFAULT_GARMENT
    elif str(garment_type).strip() not in available_garments:
        warnings.append(f"Garment '{garment_type}' not in recognized list; using closest default.")
        sanitized["garment_type"] = available_garments[0] if available_garments else DEFAULT_GARMENT
    else:
        sanitized["garment_type"] = str(garment_type).strip()

    # Validate Climate
    if not climate or str(climate).strip() not in VALID_CLIMATES:
        warnings.append(f"Climate '{climate}' is invalid; defaulted to '{DEFAULT_CLIMATE}'.")
        sanitized["climate"] = DEFAULT_CLIMATE
    else:
        sanitized["climate"] = str(climate).strip()

    # Validate Budget
    if not budget or str(budget).strip().capitalize() not in VALID_BUDGETS:
        warnings.append(f"Budget '{budget}' is invalid; defaulted to '{DEFAULT_BUDGET}'.")
        sanitized["budget"] = DEFAULT_BUDGET
    else:
        sanitized["budget"] = str(budget).strip().capitalize()

    # Validate Priorities
    for key, val in [
        ("sustainability_priority", sustainability_priority),
        ("comfort_priority", comfort_priority),
        ("durability_priority", durability_priority),
    ]:
        if not val or str(val).strip().capitalize() not in VALID_PRIORITIES:
            sanitized[key] = "Medium"
        else:
            sanitized[key] = str(val).strip().capitalize()

    return (len(warnings) == 0, sanitized, warnings)


def check_dataframe_integrity(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """Check whether a dataset is present, non-empty, and contains required schema."""
    if df is None or df.empty:
        return (False, "Dataset is empty or could not be loaded.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return (False, f"Dataset is missing required columns: {', '.join(missing_cols)}")

    return (True, "Integrity check passed.")
