"""
Explainable AI (XAI) engine for fabric recommendation.
Provides transparent rationale:
1. 'Why this fabric?' - specific strengths and priority alignment
2. 'Trade-offs' - what is gained vs sacrificed
3. 'Why #2 and #3 were not selected'
Supports optional Google Gemini LLM synthesis if API key is provided, with complete offline fallback.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd


def generate_fabric_explanation(
    top_fabric: pd.Series,
    user_inputs: Dict[str, Any],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generate structured, metric-grounded explanations for the #1 recommended fabric.
    """
    reasons: List[str] = []
    garment = user_inputs.get("garment_type", "Garment")
    climate = user_inputs.get("climate", "Moderate")
    budget = user_inputs.get("budget", "Medium")

    name = top_fabric.get("fabric_name", "Recommended Fabric")
    sust_score = top_fabric.get("sustainability_score", 0)
    comf_score = top_fabric.get("comfort", 0)
    breath_score = top_fabric.get("breathability", 0)
    dur_score = top_fabric.get("durability", 0)
    afford_score = top_fabric.get("affordability_score", 0)
    climate_col = f"suitability_{climate.lower().replace(' & ', '_').replace(' ', '_')}"
    clim_score = top_fabric.get(climate_col, 75)

    # 1. Sustainability explanation
    if sust_score >= 85:
        reasons.append(f"Outstanding sustainability ({sust_score}/100): Excellent low-impact profile with high eco-responsibility.")
    elif sust_score >= 70:
        reasons.append(f"Solid sustainability balance ({sust_score}/100): Fulfills environmental expectations while meeting performance needs.")

    # 2. Climate suitability
    if clim_score >= 85:
        reasons.append(f"Excellent {climate} climate suitability ({clim_score}/100): Keeps the wearer comfortable and regulated.")
    elif clim_score >= 70:
        reasons.append(f"Good climate compatibility ({clim_score}/100) for {climate} conditions.")

    # 3. Comfort & Breathability
    if comf_score >= 85 and breath_score >= 85:
        reasons.append(f"Superior tactile comfort ({comf_score}/100) and breathability ({breath_score}/100) tailored for {garment}.")
    elif comf_score >= 80:
        reasons.append(f"High comfort rating ({comf_score}/100) delivering smooth drape and pleasant skin feel.")

    # 4. Durability
    if dur_score >= 85:
        reasons.append(f"High structural durability ({dur_score}/100): Highly resistant to everyday wear, friction, and washing.")
    elif dur_score >= 70:
        reasons.append(f"Reliable fabric strength ({dur_score}/100) suited for standard lifecycle demands.")

    # 5. Budget & Cost
    if afford_score >= 80:
        reasons.append(f"High cost-efficiency ({afford_score}/100): Highly accessible for a {budget.lower()} budget.")
    elif afford_score >= 50:
        reasons.append(f"Balanced price-to-value ratio ({afford_score}/100) aligned with the {budget.lower()} budget tier.")
    else:
        reasons.append(f"Premium grade investment ({afford_score}/100): High quality offsetting the higher cost tier.")

    # 6. Specific fabric advantages from dataset
    advantages = str(top_fabric.get("advantages", "")).strip()
    if advantages:
        specific_adv = [adv.strip() for adv in advantages.split(";") if adv.strip()]
        for adv in specific_adv[:2]:
            reasons.append(f"Key material property: {adv}.")

    # Limitations to watch for
    limitations = str(top_fabric.get("limitations", "")).strip()
    key_limitation = limitations if limitations else "Requires standard gentle garment care."

    return {
        "fabric_name": name,
        "score_bullet_points": reasons,
        "key_advantages": advantages,
        "key_limitations": key_limitation,
        "care_instructions": top_fabric.get("care_instructions", "Follow standard care labels."),
    }


def generate_tradeoff_analysis(
    top_fabric: pd.Series,
    alternatives: List[pd.Series],
    user_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate comprehensive trade-off comparisons between the top recommendation
    and runner-up alternatives (#2 and #3).
    """
    tradeoffs: List[Dict[str, str]] = []
    top_name = top_fabric.get("fabric_name", "Top Fabric")

    for rank, alt in enumerate(alternatives, start=2):
        alt_name = alt.get("fabric_name", f"Alternative #{rank}")

        # Find key metric differences
        deltas = {
            "Sustainability": alt.get("sustainability_score", 0) - top_fabric.get("sustainability_score", 0),
            "Breathability": alt.get("breathability", 0) - top_fabric.get("breathability", 0),
            "Comfort": alt.get("comfort", 0) - top_fabric.get("comfort", 0),
            "Durability": alt.get("durability", 0) - top_fabric.get("durability", 0),
            "Affordability": alt.get("affordability_score", 0) - top_fabric.get("affordability_score", 0),
        }

        # Identify where alternative is better and where it falls short
        better_metrics = [k for k, v in deltas.items() if v >= 6]
        worse_metrics = [k for k, v in deltas.items() if v <= -6]

        why_not_first = ""
        tradeoff_statement = ""

        if better_metrics and worse_metrics:
            tradeoff_statement = (
                f"{alt_name} offers higher {', '.join(better_metrics)} "
                f"({'+' if deltas[better_metrics[0]] > 0 else ''}{deltas[better_metrics[0]]:.0f} pts), "
                f"but sacrifices {', '.join(worse_metrics)} compared to {top_name}."
            )
            why_not_first = (
                f"Ranked #{rank} because the loss in {worse_metrics[0].lower()} outweighed its "
                f"gains given the selected priorities."
            )
        elif worse_metrics:
            tradeoff_statement = (
                f"{alt_name} has lower {', '.join(worse_metrics[:2])} than {top_name}."
            )
            why_not_first = (
                f"Ranked #{rank} due to slightly lower composite suitability across user criteria."
            )
        else:
            tradeoff_statement = (
                f"{alt_name} is a very close peer to {top_name} with slightly differing texture and origin."
            )
            why_not_first = f"Ranked #{rank} with a very close competitive score."

        limitation = alt.get("limitations", "")

        tradeoffs.append({
            "rank": rank,
            "fabric_name": alt_name,
            "overall_score": alt.get("final_score", alt.get("general_recommendation_score", 0)),
            "tradeoff_statement": tradeoff_statement,
            "why_not_selected": why_not_first,
            "limitation": limitation,
        })

    return {
        "top_fabric": top_name,
        "runner_up_comparisons": tradeoffs,
    }


def generate_ai_llm_explanation(
    top_fabric_name: str,
    top_score: float,
    garment: str,
    climate: str,
    reasons: List[str],
    tradeoffs: List[Dict[str, str]],
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Optional LLM generation using google-genai SDK if an API key is available.
    Returns None if no API key or upon exception, gracefully falling back to rule-based output.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=key)

        prompt = f"""
You are an expert textile scientist and circular fashion consultant for the Smart Fabric Recommendation System.
Provide a concise, engaging, 2-3 paragraph explanation of why {top_fabric_name} was chosen as the #1 fabric recommendation for a {garment} in a {climate} climate.

Key data:
- Recommendation score: {top_score:.1f}/100
- Top attributes: {'; '.join(reasons[:3])}
- Alternative trade-offs: {tradeoffs[0]['tradeoff_statement'] if tradeoffs else 'N/A'}

Format with clear, professional tone highlighting both the environmental merits and functional advantages. Avoid generic fluff.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception:
        # Seamless fallback
        return None

    return None
