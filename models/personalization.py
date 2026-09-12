"""
Personalization Engine for Smart Fabric Recommendation.
Learns from historical user interactions, ratings, and choices to adapt recommendations.
Implements the 70% General Score + 30% User Preference Score blending formula.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from database.feedback import (
    get_user_history,
    get_user_fabric_affinity,
    init_db
)


class PersonalizationEngine:
    """
    Computes personalized fabric preference scores from SQLite user feedback history.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        init_db(self.db_path)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Extract detailed user profile including interaction count, preferred fabrics,
        category affinities, and active personalization learning weight.
        """
        if not user_id or user_id.strip() == "":
            return {
                "user_id": "Guest",
                "interaction_count": 0,
                "has_sufficient_history": False,
                "learning_weight": 0.0,
                "fabric_affinities": {},
                "top_categories": [],
                "avg_rating": 0.0,
            }

        df = get_user_history(user_id, self.db_path)
        interaction_count = len(df)
        has_sufficient_history = interaction_count >= 2

        # Learning weight ramps up from 0 to 30%
        if interaction_count == 0:
            learning_weight = 0.0
        elif interaction_count == 1:
            learning_weight = 0.15
        else:
            learning_weight = 0.30

        fabric_affinities = get_user_fabric_affinity(user_id, self.db_path)

        avg_rating = round(float(df["rating"].mean()), 2) if not df.empty else 0.0

        # Extract top categories if available
        top_categories: List[str] = []
        if not df.empty and "decision_factors" in df.columns:
            # Check factors mentioned in history
            factors = df["decision_factors"].dropna().tolist()
            top_categories = list(set([f.strip() for f in ",".join(factors).split(",") if f.strip()]))

        return {
            "user_id": user_id,
            "interaction_count": interaction_count,
            "has_sufficient_history": has_sufficient_history,
            "learning_weight": learning_weight,
            "fabric_affinities": fabric_affinities,
            "top_categories": top_categories,
            "avg_rating": avg_rating,
        }

    def compute_fabric_preference_score(
        self,
        fabric_name: str,
        fabric_category: str,
        user_id: str,
        general_score: float,
    ) -> Tuple[float, float, str]:
        """
        Compute the user's preference score (0-100) for a given fabric.
        Returns:
            (personalized_final_score, user_preference_score, explanation_note)
        """
        profile = self.get_user_profile(user_id)
        alpha = profile["learning_weight"]  # 0.0 to 0.30

        # Fallback to general recommendation engine if no or insufficient history
        if alpha == 0.0:
            return (
                general_score,
                general_score,
                "Standard recommendation (insufficient interaction history for personalization)",
            )

        fabric_affinities = profile["fabric_affinities"]

        if fabric_name in fabric_affinities:
            # Direct fabric affinity from past ratings
            raw_pref = fabric_affinities[fabric_name]
            note = f"Personalized boost from past user rating & selection ({raw_pref:.0f}/100 affinity)"
        else:
            # Check if user has demonstrated strong preference for other fabrics in this category
            cat_affinities = []
            for past_fabric, score in fabric_affinities.items():
                # Neutral baseline if unvisited
                cat_affinities.append(score)

            if cat_affinities:
                # Modest category transfer affinity
                avg_cat_pref = sum(cat_affinities) / len(cat_affinities)
                # Dampen towards 50 to avoid over-generalization
                raw_pref = 50.0 * 0.5 + avg_cat_pref * 0.5
                note = f"Category tendency inference ({raw_pref:.0f}/100 affinity)"
            else:
                raw_pref = general_score
                note = "Neutral baseline preference"

        # Apply the required formula:
        # Final = (1 - alpha) * General + alpha * User Preference
        final_score = (1.0 - alpha) * general_score + alpha * raw_pref
        final_score = round(max(0.0, min(100.0, final_score)), 2)
        pref_score = round(raw_pref, 1)

        return (final_score, pref_score, note)
