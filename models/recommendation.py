"""
Core Recommendation Engine for Smart Fabric Recommendation System.
Coordinates garment compatibility filtering, multi-criteria decision analysis (MCDA),
dynamic priority weighting, and personalization score blending.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from utils.scoring import calculate_dynamic_weights, calculate_fabric_scores
from utils.explanations import generate_fabric_explanation, generate_tradeoff_analysis
from models.personalization import PersonalizationEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FABRICS_CSV = os.path.join(DATA_DIR, "fabrics.csv")
COMPATIBILITY_CSV = os.path.join(DATA_DIR, "garment_compatibility.csv")


class FabricRecommendationEngine:
    """
    Intelligent multi-criteria recommendation engine for sustainable textile selection.
    """

    def __init__(
        self,
        fabrics_path: Optional[str] = None,
        compatibility_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.fabrics_path = fabrics_path or FABRICS_CSV
        self.compatibility_path = compatibility_path or COMPATIBILITY_CSV
        self.db_path = db_path
        self.personalization_engine = PersonalizationEngine(db_path=self.db_path)
        self.fabrics_df = self._load_fabrics()
        self.compatibility_df = self._load_compatibility()

    def _load_fabrics(self) -> pd.DataFrame:
        """Load fabrics dataset from CSV."""
        if not os.path.exists(self.fabrics_path):
            raise FileNotFoundError(f"Fabrics dataset file not found at: {self.fabrics_path}")
        return pd.read_csv(self.fabrics_path)

    def _load_compatibility(self) -> pd.DataFrame:
        """Load garment-to-fabric compatibility dataset."""
        if not os.path.exists(self.compatibility_path):
            raise FileNotFoundError(
                f"Compatibility rules file not found at: {self.compatibility_path}"
            )
        return pd.read_csv(self.compatibility_path)

    def get_available_garments(self) -> List[str]:
        """Return unique list of supported garment types."""
        return sorted(self.compatibility_df["garment_type"].unique().tolist())

    def get_all_fabrics(self) -> pd.DataFrame:
        """Return full fabrics DataFrame."""
        return self.fabrics_df.copy()

    def filter_by_garment(self, garment_type: str) -> pd.DataFrame:
        """
        Filter fabrics to only those compatible with the specified garment.
        """
        compat_matches = self.compatibility_df[
            (self.compatibility_df["garment_type"] == garment_type)
            & (self.compatibility_df["is_compatible"] == 1)
        ]

        if compat_matches.empty:
            # Fallback: if unknown garment, return all fabrics with a neutral compatibility flag
            return self.fabrics_df.copy()

        compatible_fabric_ids = set(compat_matches["fabric_id"].tolist())
        filtered = self.fabrics_df[self.fabrics_df["fabric_id"].isin(compatible_fabric_ids)].copy()

        # Merge compatibility metadata
        merged = pd.merge(
            filtered,
            compat_matches[["fabric_id", "compatibility_score", "suitability_notes"]],
            on="fabric_id",
            how="left",
        )
        return merged

    def recommend(
        self,
        garment_type: str = "T-shirt",
        climate: str = "Moderate",
        budget: str = "Medium",
        sustainability_priority: str = "Medium",
        comfort_priority: str = "Medium",
        durability_priority: str = "Medium",
        performance_requirement: str = "Everyday balance",
        preferred_category: str = "Any",
        user_id: str = "Guest",
    ) -> Dict[str, Any]:
        """
        Generate ranked recommendations, detailed scoring breakdown,
        explanations, and trade-off comparisons.
        """
        # 1. Compatibility filtering
        candidate_df = self.filter_by_garment(garment_type)

        if candidate_df.empty:
            candidate_df = self.fabrics_df.copy()

        # Optional category filtering
        if preferred_category and preferred_category != "Any":
            cat_filtered = candidate_df[
                (candidate_df["category"].str.contains(preferred_category, case=False, na=False))
                | (candidate_df["origin_type"].str.contains(preferred_category, case=False, na=False))
            ]
            if not cat_filtered.empty:
                candidate_df = cat_filtered

        # 2. Dynamic weights
        weights = calculate_dynamic_weights(
            sustainability_priority=sustainability_priority,
            comfort_priority=comfort_priority,
            durability_priority=durability_priority,
            budget=budget,
            has_performance_focus=(performance_requirement != "Everyday balance"),
        )

        # 3. Compute scores
        scored_records = []
        user_profile = self.personalization_engine.get_user_profile(user_id)

        for _, row in candidate_df.iterrows():
            scores = calculate_fabric_scores(
                fabric_row=row,
                weights=weights,
                climate=climate,
                performance_requirement=performance_requirement,
            )

            gen_score = scores["general_recommendation_score"]
            fabric_name = str(row["fabric_name"])
            category = str(row["category"])

            final_score, pref_score, pref_note = (
                self.personalization_engine.compute_fabric_preference_score(
                    fabric_name=fabric_name,
                    fabric_category=category,
                    user_id=user_id,
                    general_score=gen_score,
                )
            )

            combined_record = row.to_dict()
            combined_record.update(scores)
            combined_record["user_preference_score"] = pref_score
            combined_record["final_score"] = final_score
            combined_record["personalization_note"] = pref_note
            scored_records.append(combined_record)

        results_df = pd.DataFrame(scored_records)

        # Rank by final_score descending
        results_df = results_df.sort_values(by=["final_score", "sustainability_score"], ascending=[False, False]).reset_index(drop=True)
        results_df["rank"] = range(1, len(results_df) + 1)

        # Extract top recommendation and alternatives
        top_fabric = results_df.iloc[0]
        alternatives = [results_df.iloc[i] for i in range(1, min(4, len(results_df)))]

        # 4. Generate Explainable AI outputs
        user_inputs = {
            "garment_type": garment_type,
            "climate": climate,
            "budget": budget,
            "sustainability_priority": sustainability_priority,
            "comfort_priority": comfort_priority,
            "durability_priority": durability_priority,
            "performance_requirement": performance_requirement,
        }

        explanation = generate_fabric_explanation(top_fabric, user_inputs, weights)
        tradeoffs = generate_tradeoff_analysis(top_fabric, alternatives, user_inputs)

        return {
            "top_recommendation": top_fabric,
            "alternatives": alternatives,
            "ranked_df": results_df,
            "weights": weights,
            "user_inputs": user_inputs,
            "explanation": explanation,
            "tradeoffs": tradeoffs,
            "user_profile": user_profile,
        }
