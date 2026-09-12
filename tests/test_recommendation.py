"""
Comprehensive test suite for Smart Fabric Recommendation System.
Tests fabric filtering, dynamic weighting, multi-criteria scoring,
personalization blending (70/30), SQLite feedback, and fallback handling.
"""

import os
import tempfile
import unittest
import pandas as pd

from utils.scoring import calculate_dynamic_weights, calculate_fabric_scores, BASELINE_WEIGHTS
from utils.validation import validate_recommendation_inputs
from utils.explanations import generate_fabric_explanation, generate_tradeoff_analysis
from database.feedback import init_db, save_feedback, get_user_history, get_user_fabric_affinity
from models.personalization import PersonalizationEngine
from models.recommendation import FabricRecommendationEngine


class TestFabricRecommendation(unittest.TestCase):
    """Unit tests for fabric recommendation modules."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_db = os.path.join(cls.temp_dir, "test_feedback.db")
        init_db(cls.temp_db)
        cls.engine = FabricRecommendationEngine(db_path=cls.temp_db)

    def test_garment_filtering(self):
        """Verify fabrics are properly filtered based on garment compatibility."""
        tshirt_fabrics = self.engine.filter_by_garment("T-shirt")
        self.assertFalse(tshirt_fabrics.empty)
        names = tshirt_fabrics["fabric_name"].tolist()

        # T-shirts should include breathable cottons / linens / tencel
        self.assertIn("Organic Cotton", names)
        self.assertIn("Linen", names)

        # T-shirts should NOT include heavy virgin wool or heavy jute
        self.assertNotIn("Virgin Wool", names)
        self.assertNotIn("Jute", names)

        # Winter jacket should include Wool
        jacket_fabrics = self.engine.filter_by_garment("Jacket")
        jacket_names = jacket_fabrics["fabric_name"].tolist()
        self.assertIn("Recycled Wool", jacket_names)

    def test_dynamic_weight_normalization(self):
        """Verify that weights always sum to exactly 1.0 under diverse priority configurations."""
        scenarios = [
            {"sustainability_priority": "High", "comfort_priority": "Low", "durability_priority": "Medium", "budget": "Low"},
            {"sustainability_priority": "Low", "comfort_priority": "High", "durability_priority": "High", "budget": "High"},
            {"sustainability_priority": "Medium", "comfort_priority": "Medium", "durability_priority": "Medium", "budget": "Medium"},
        ]

        for s in scenarios:
            weights = calculate_dynamic_weights(**s)
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=3, msg=f"Weights do not sum to 1.0: {weights}")

    def test_dynamic_priority_shifts(self):
        """Verify that high sustainability boosts sustainability weight and low budget boosts cost weight."""
        base_weights = calculate_dynamic_weights(
            sustainability_priority="Medium", budget="Medium"
        )
        high_sust_weights = calculate_dynamic_weights(
            sustainability_priority="High", budget="Medium"
        )
        low_budget_weights = calculate_dynamic_weights(
            sustainability_priority="Medium", budget="Low"
        )

        self.assertGreater(
            high_sust_weights["sustainability"], base_weights["sustainability"],
            "High priority failed to boost sustainability weight"
        )
        self.assertGreater(
            low_budget_weights["cost"], base_weights["cost"],
            "Low budget failed to boost affordability/cost weight"
        )

    def test_scoring_and_ranking_order(self):
        """Verify overall recommendation pipeline generates valid scores and ranked ordering."""
        rec = self.engine.recommend(
            garment_type="T-shirt",
            climate="Hot & Humid",
            budget="Medium",
            sustainability_priority="High",
            comfort_priority="High",
            durability_priority="Medium",
            user_id="test_user_ranking",
        )

        ranked_df = rec["ranked_df"]
        self.assertGreater(len(ranked_df), 0)

        # Check ranking monotonic decrease
        scores = ranked_df["final_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check score bounds
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)

        # Check top recommendation
        top_fab = rec["top_recommendation"]
        self.assertIsNotNone(top_fab)
        self.assertIn(top_fab["fabric_name"], ["Organic Cotton", "Linen", "Tencel / Lyocell", "Banana Fiber"])

    def test_personalization_learning_and_blending(self):
        """Verify that saving positive feedback in SQLite updates user preference and shifts score."""
        user_id = "test_user_learn_01"

        # Baseline recommendation before feedback
        initial_rec = self.engine.recommend(
            garment_type="Shirt",
            climate="Hot",
            budget="Medium",
            user_id=user_id,
        )
        initial_profile = self.engine.personalization_engine.get_user_profile(user_id)
        self.assertFalse(initial_profile["has_sufficient_history"])
        self.assertEqual(initial_profile["learning_weight"], 0.0)

        # User chooses and highly rates Linen multiple times
        save_feedback(
            user_id=user_id,
            garment="Shirt",
            climate="Hot",
            budget="Medium",
            recommended_fabric="Organic Cotton",
            selected_fabric="Linen",
            rating=5,
            would_choose=True,
            decision_factors="Breathability, Natural feel",
            db_path=self.temp_db,
        )
        save_feedback(
            user_id=user_id,
            garment="Trousers",
            climate="Hot",
            budget="Medium",
            recommended_fabric="Linen",
            selected_fabric="Linen",
            rating=5,
            would_choose=True,
            decision_factors="High cooling comfort",
            db_path=self.temp_db,
        )

        # Check updated profile
        updated_profile = self.engine.personalization_engine.get_user_profile(user_id)
        self.assertTrue(updated_profile["has_sufficient_history"])
        self.assertEqual(updated_profile["learning_weight"], 0.30)
        self.assertIn("Linen", updated_profile["fabric_affinities"])
        self.assertGreaterEqual(updated_profile["fabric_affinities"]["Linen"], 80.0)

        # Re-run recommendation with learning enabled
        personalized_rec = self.engine.recommend(
            garment_type="Shirt",
            climate="Hot",
            budget="Medium",
            user_id=user_id,
        )
        ranked_df = personalized_rec["ranked_df"]
        linen_row = ranked_df[ranked_df["fabric_name"] == "Linen"].iloc[0]

        # Verify 70/30 score calculation
        gen_score = linen_row["general_recommendation_score"]
        pref_score = linen_row["user_preference_score"]
        expected_final = round(0.70 * gen_score + 0.30 * pref_score, 2)
        self.assertAlmostEqual(linen_row["final_score"], expected_final, places=1)

    def test_input_validation_and_fallbacks(self):
        """Verify invalid inputs default safely without throwing exceptions."""
        is_valid, sanitized, warnings = validate_recommendation_inputs(
            garment_type="NonExistentSpacesuit",
            climate="MartianArctic",
            budget="InfiniteGold",
            sustainability_priority="UltraMega",
            comfort_priority=None,
            durability_priority="",
            available_garments=self.engine.get_available_garments(),
        )

        self.assertFalse(is_valid)
        self.assertGreater(len(warnings), 0)
        self.assertEqual(sanitized["climate"], "Moderate")
        self.assertEqual(sanitized["budget"], "Medium")
        self.assertEqual(sanitized["sustainability_priority"], "Medium")

    def test_explainable_ai_and_tradeoffs(self):
        """Verify generated explanations contain 'Why this fabric?' points and trade-offs."""
        rec = self.engine.recommend(
            garment_type="T-shirt",
            climate="Hot",
            budget="Medium",
            user_id="test_xai",
        )

        explanation = rec["explanation"]
        self.assertIn("fabric_name", explanation)
        self.assertGreater(len(explanation["score_bullet_points"]), 0)

        tradeoffs = rec["tradeoffs"]
        self.assertIn("runner_up_comparisons", tradeoffs)
        self.assertGreater(len(tradeoffs["runner_up_comparisons"]), 0)

        first_alt = tradeoffs["runner_up_comparisons"][0]
        self.assertIn("tradeoff_statement", first_alt)
        self.assertIn("why_not_selected", first_alt)
    def test_empty_or_missing_dataset_handling(self):
        """Verify engine raises or handles missing and empty datasets gracefully."""
        from utils.validation import check_dataframe_integrity

        empty_df = pd.DataFrame()
        valid, msg = check_dataframe_integrity(empty_df, ["fabric_id", "fabric_name"])
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

        incomplete_df = pd.DataFrame({"fabric_id": [1, 2]})
        valid_inc, msg_inc = check_dataframe_integrity(incomplete_df, ["fabric_id", "sustainability_score"])
        self.assertFalse(valid_inc)
        self.assertIn("missing required columns", msg_inc.lower())

        with self.assertRaises(FileNotFoundError):
            FabricRecommendationEngine(fabrics_path="non_existent_fabrics_path.csv")


if __name__ == "__main__":
    unittest.main()
