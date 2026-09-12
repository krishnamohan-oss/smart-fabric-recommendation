"""
Smart Fabric Recommendation System - Streamlit Web Application
Track 1 — FUTURE FABRIC (Hackathon Ready)
A multi-criteria, explainable, personalized decision-support platform for sustainable textiles.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from models.recommendation import FabricRecommendationEngine
from database.feedback import (
    init_db,
    save_feedback,
    get_user_history,
    get_all_feedback,
    get_feedback_summary_stats,
    reset_user_history,
)
from utils.explanations import generate_ai_llm_explanation

# Streamlit Page Config
st.set_page_config(
    page_title="Smart Fabric Recommender | Future Fabric",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional Dark & Emerald Theme CSS
st.markdown(
    """
    <style>
    /* Main Dark Theme Variables */
    :root {
        --bg-dark: #0f172a;
        --card-dark: #1e293b;
        --border-dark: #334155;
        --text-light: #f8fafc;
        --text-muted: #94a3b8;
        --emerald: #10b981;
        --emerald-dark: #065f46;
        --emerald-light: #34d399;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #34d399;
        margin-bottom: 1.5rem;
    }
    
    .trophy-badge {
        background: linear-gradient(135deg, #059669, #065f46);
        color: #ffffff;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    .confidence-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .top-match-card {
        background-color: #1e293b;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }
    
    .alt-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #f8fafc;
    }

    .stat-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pill-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .pill-blue { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .pill-orange { background-color: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.4); }
    
    .score-circle {
        font-size: 2.3rem;
        font-weight: 800;
        color: #34d399;
    }

    .progress-row {
        margin-bottom: 10px;
    }
    .progress-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.86rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 3px;
    }
    .progress-bar-bg {
        background: #334155;
        border-radius: 9999px;
        height: 9px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    .pipeline-step {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #f8fafc;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Cache recommendation engine instance across sessions."""
    return FabricRecommendationEngine()


engine = get_engine()

# ==============================================================================
# SIDEBAR SETUP (High-Contrast Dark Aesthetic & Personalization Summary)
# ==============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&q=80",
        caption="Future Fabric AI | Track 1",
        use_container_width=True,
    )
    st.title("🌿 Smart Fabric AI")
    st.caption("Intelligent Sustainable Textile Decision System")

    # Download project bundle button
    zip_path = os.path.join(os.path.dirname(__file__), "smart-fabric-recommendation-github-upload.zip")
    if not os.path.exists(zip_path):
        zip_path = os.path.join(os.path.dirname(__file__), "smart-fabric-recommendation.zip")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as fp:
            st.download_button(
                label="📦 Download Project (ZIP)",
                data=fp.read(),
                file_name="smart-fabric-recommendation.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("👤 User Profile Session")

    user_id_options = ["User_001", "User_002", "Eco_Designer_Elena", "Fashion_Brand_Lead", "Guest"]
    selected_user = st.selectbox(
        "Active User ID",
        user_id_options,
        index=0,
        help="Recommendations dynamically adapt as this user rates fabrics.",
    )
    custom_user = st.text_input("Or enter custom user ID", value="").strip()
    active_user = custom_user if custom_user else selected_user

    # High-contrast, dark professional personalization summary card
    user_prof = engine.personalization_engine.get_user_profile(active_user)
    learned = user_prof.get("learned_priorities", {
        "sustainability": "Medium",
        "comfort": "Medium",
        "durability": "Medium",
        "cost": "Medium",
    })

    st.markdown(
        f"""
        <div style='background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-top: 8px; color: #f8fafc;'>
            <div style='font-size: 0.9rem; font-weight: 800; color: #34d399; letter-spacing: 0.05em; margin-bottom: 6px;'>
                👤 PERSONALIZATION
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-bottom: 3px;'>
                <b>User:</b> <span style='color: #ffffff; font-weight: 600;'>{active_user}</span>
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-bottom: 3px;'>
                <b>History:</b> {user_prof['interaction_count']} selections logged
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-bottom: 3px;'>
                <b>Personalization Weight:</b> <span style='color: #34d399; font-weight: 700;'>{int(user_prof['learning_weight'] * 100)}%</span>
            </div>
            <div style='font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;'>
                <b>Avg User Rating:</b> {'⭐ ' + str(user_prof['avg_rating']) if user_prof['avg_rating'] > 0 else 'No ratings yet'}
            </div>
            <hr style='border-color: #334155; margin: 8px 0;'>
            <div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; margin-bottom: 6px;'>
                Learned Preferences
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.78rem; color: #e2e8f0;'>
                <div>🌱 Sust: <b style='color: #34d399;'>{learned['sustainability']}</b></div>
                <div>☁️ Comf: <b style='color: #38bdf8;'>{learned['comfort']}</b></div>
                <div>💪 Durab: <b style='color: #fbbf24;'>{learned['durability']}</b></div>
                <div>💰 Cost: <b style='color: #f472b6;'>{learned['cost']}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Reset This User Profile", help="Clear past ratings for this user"):
        deleted = reset_user_history(active_user)
        st.success(f"Reset {deleted} historical records for {active_user}!")
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Optional AI Assistant")
    gemini_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Leave blank to use full offline rule-based explanation engine.",
    )

    st.markdown("---")
    stats = get_feedback_summary_stats()
    st.markdown(
        f"""
        <small style='color: #94a3b8;'>
        <b>Global DB Stats:</b><br>
        • Total Reviews: {stats['total_reviews']}<br>
        • Overall Acceptance: {stats['acceptance_rate']}%<br>
        • Top Chosen Fabric: {stats['top_selected_fabric']}
        </small>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN TABS NAVIGATION
# ==============================================================================
tabs = st.tabs([
    "🏠 Overview",
    "🎯 Recommendation",
    "📊 Leaderboard",
    "⚖️ Compare Fabrics",
    "🌱 Sustainability Hub",
    "👤 Personalization Profile",
    "📜 Feedback History",
    "ℹ️ Methodology",
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.markdown("<h1 class='main-title'>Smart Fabric Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Track 1 — FUTURE FABRIC | Balancing Multi-Dimensional Sustainability, Garment Functionality & User Preferences</p>",
        unsafe_allow_html=True,
    )

    # Download banner
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as fp:
            st.download_button(
                label="⬇️ Click Here to Download Complete Project (.ZIP Archive)",
                data=fp.read(),
                file_name="smart-fabric-recommendation.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    col_hero1, col_hero2 = st.columns([3, 2])
    with col_hero1:
        st.markdown(
            """
            ### The Core Challenge in Sustainable Fashion
            Most sustainability tools simply recommend the fabric with the lowest carbon footprint or highest organic content. **In real garment production, this leads to failures:**
            - **Unsuitable drape**: Choosing stiff hemp for delicate sarees or structured silk for rugged workwear.
            - **Misaligned climate resilience**: Specifying virgin wool in sweltering humid monsoons, or lightweight linen in freezing winters.
            - **Hidden trade-offs**: Conventional cotton consumes massive amounts of irrigation water; synthetics like recycled polyester shed persistent microplastics; bamboo viscose can involve heavy chemical processing.

            ### Our Solution: Multi-Criteria Intelligent Decision Support
            Our recommendation platform calculates a holistic, dynamically weighted compatibility score balancing:
            1. **Garment Compatibility**: Pre-filtering fabrics engineered for the garment's functional demands.
            2. **Climate Suitability**: Temperature regulation, breathability, and moisture wicking.
            3. **User Priorities**: Sliders dynamically re-normalize weights to your specific sustainability, comfort, and durability goals.
            4. **Adaptive Personalization**: SQLite-backed feedback learning with a **70% General + 30% User Preference** model.
            5. **Explainable AI (XAI)**: Comprehensive breakdown of *Why this fabric?* and explicit *Trade-offs* with runner-up alternatives.
            """
        )
    with col_hero2:
        st.markdown(
            """
            <div style='background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
                <h4 style='color: #34d399; margin-top: 0;'>🚀 Quick Start Workflow</h4>
                <ol style='padding-left: 20px; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;'>
                    <li>Open the <b>🎯 Recommendation</b> tab.</li>
                    <li>Select your garment type (e.g., <i>T-shirt, Jacket, Saree</i>).</li>
                    <li>Specify your climate and priority sliders.</li>
                    <li>Explore the <b>🏆 BEST MATCH</b>, detailed rationale, and trade-offs.</li>
                    <li>Provide feedback below the result to train your personalized user profile!</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Key System Metrics")
        
        # 4 responsive cards ensuring no text truncation (specifically fixing "14+ Criteria")
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(
                """
                <div class='stat-card'>
                    <div style='font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;'>Curated Fabrics</div>
                    <div style='font-size: 1.35rem; color: #34d399; font-weight: 800;'>22</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_cols[1]:
            st.markdown(
                """
                <div class='stat-card'>
                    <div style='font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;'>Garment Types</div>
                    <div style='font-size: 1.35rem; color: #38bdf8; font-weight: 800;'>13</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_cols[2]:
            st.markdown(
                """
                <div class='stat-card'>
                    <div style='font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;'>Evaluation Criteria</div>
                    <div style='font-size: 1.15rem; color: #10b981; font-weight: 800; white-space: nowrap;'>14+ Criteria</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_cols[3]:
            st.markdown(
                """
                <div class='stat-card'>
                    <div style='font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;'>Recommendation Factors</div>
                    <div style='font-size: 1.15rem; color: #fbbf24; font-weight: 800; white-space: nowrap;'>Multi-Criteria</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 2: RECOMMENDATION (CORE SHOWCASE)
# ==============================================================================
with tabs[1]:
    st.markdown("### 🎯 Find Your Ideal Sustainable Fabric")
    st.caption("Specify garment context, climate conditions, and design priorities to calculate the optimal fabric.")

    # A. USER INPUTS
    with st.form("recommendation_form"):
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            garments_list = engine.get_available_garments()
            sel_garment = st.selectbox("1. Garment Type", garments_list, index=garments_list.index("T-shirt") if "T-shirt" in garments_list else 0)
            sel_climate = st.selectbox("2. Climate Condition", ["Hot", "Hot & Humid", "Moderate", "Cold", "Rainy"], index=2)
            sel_budget = st.selectbox("3. Cost Priority (Budget)", ["Low", "Medium", "High"], index=1)

        with col_in2:
            sel_sustainability = st.select_slider("4. Sustainability Priority", options=["Low", "Medium", "High"], value="High")
            sel_comfort = st.select_slider("5. Comfort & Breathability Priority", options=["Low", "Medium", "High"], value="Medium")
            sel_durability = st.select_slider("6. Durability & Longevity Priority", options=["Low", "Medium", "High"], value="Medium")

        with col_in3:
            sel_performance = st.selectbox(
                "7. Performance Requirement",
                [
                    "Everyday balance",
                    "High breathability",
                    "Moisture wicking & quick dry",
                    "Thermal warmth & insulation",
                    "Water resistance & weather protection",
                    "Stretch & flexibility",
                ],
                index=0,
            )
            sel_fiber_pref = st.selectbox(
                "8. Preferred Fabric Origin",
                ["Any", "Natural Plant", "Regenerated Cellulose", "Natural Animal", "Synthetic", "Next-Gen Bio-Material"],
                index=0,
            )

        submit_rec = st.form_submit_button("✨ Generate AI Fabric Recommendation", use_container_width=True)

    # Calculate recommendation
    rec_result = engine.recommend(
        garment_type=sel_garment,
        climate=sel_climate,
        budget=sel_budget,
        sustainability_priority=sel_sustainability,
        comfort_priority=sel_comfort,
        durability_priority=sel_durability,
        performance_requirement=sel_performance,
        preferred_category=sel_fiber_pref,
        user_id=active_user,
    )

    top = rec_result["top_recommendation"]
    alts = rec_result["alternatives"]
    weights = rec_result["weights"]
    explanation = rec_result["explanation"]
    tradeoffs = rec_result["tradeoffs"]

    # Calculate transparent Data Confidence indicator from availability of fabric fields
    required_eval_fields = [
        "sustainability_score", "water_efficiency", "carbon_score", "recyclability",
        "biodegradability", "breathability", "comfort", "durability", "stretchability",
        "moisture_management", "thermal_insulation", "water_resistance", "affordability_score"
    ]
    present_fields = sum(1 for field in required_eval_fields if field in top and pd.notna(top[field]))
    completeness_pct = int((present_fields / len(required_eval_fields)) * 100)
    
    if completeness_pct >= 95:
        confidence_label = "🟢 High Confidence"
        confidence_desc = f"{completeness_pct}% Data Complete"
    elif completeness_pct >= 75:
        confidence_label = "🟡 Moderate Confidence"
        confidence_desc = f"{completeness_pct}% Data Complete"
    else:
        confidence_label = "🔴 Limited Data"
        confidence_desc = f"{completeness_pct}% Data Complete"

    st.markdown("---")

    # B. BEST MATCH CARD
    garment_compat_score = int(top.get("compatibility_score", 95))
    climate_suit_score = int(top["climate_score"])
    sust_score = int(top["sustainability_score"])
    comf_score = int(top["comfort_score"])
    dur_score = int(top["durability_score"])
    cost_score = int(top["cost_score"])

    st.markdown(
        f"""
        <div class='top-match-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;'>
                <div>
                    <span class='trophy-badge'>🏆 BEST MATCH</span>
                    <span class='confidence-badge' title='Confidence reflects the completeness of available fabric information used for this recommendation.'>
                        {confidence_label} ({confidence_desc})
                    </span>
                    <h2 style='color: #ffffff; margin: 8px 0 2px 0;'>{top['fabric_name']}</h2>
                    <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                        <b>Category:</b> {top['category']} | <b>Origin:</b> {top['origin_type']} | <b>Target Garment:</b> {sel_garment}
                    </p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600;'>Overall Compatibility</div>
                    <div class='score-circle'>{top['final_score']:.1f}<span style='font-size: 1.1rem; color: #94a3b8;'> / 100</span></div>
                </div>
            </div>
            <div style='margin-top: 14px;'>
                <span class='metric-pill pill-green'>🌱 Sustainability: {sust_score}/100</span>
                <span class='metric-pill pill-blue'>🌤️ Climate Fit: {climate_suit_score}/100</span>
                <span class='metric-pill pill-blue'>☁️ Comfort: {comf_score}/100</span>
                <span class='metric-pill pill-green'>🛡️ Durability: {dur_score}/100</span>
                <span class='metric-pill pill-orange'>💰 Cost Fit: {cost_score}/100</span>
                <span class='metric-pill pill-green'>🎯 Garment Compatibility: {garment_compat_score}/100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # C. SCORE BREAKDOWN (Clean Horizontal Progress Bars with Real Calculated Values)
    st.markdown("#### 📊 Score Breakdown by Major Criteria")
    st.caption("Actual calculated contributions based on garment compatibility and multi-criteria utility weighting:")

    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        st.markdown(
            f"""
            <div class='progress-row'>
                <div class='progress-header'><span>Garment Compatibility</span><span>{garment_compat_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {garment_compat_score}%;'></div></div>
            </div>
            <div class='progress-row'>
                <div class='progress-header'><span>Climate Suitability ({sel_climate})</span><span>{climate_suit_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {climate_suit_score}%;'></div></div>
            </div>
            <div class='progress-row'>
                <div class='progress-header'><span>Sustainability Score</span><span>{sust_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {sust_score}%;'></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sb_col2:
        st.markdown(
            f"""
            <div class='progress-row'>
                <div class='progress-header'><span>Comfort & Breathability</span><span>{comf_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {comf_score}%;'></div></div>
            </div>
            <div class='progress-row'>
                <div class='progress-header'><span>Durability & Strength</span><span>{dur_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {dur_score}%;'></div></div>
            </div>
            <div class='progress-row'>
                <div class='progress-header'><span>Cost / Affordability</span><span>{cost_score}/100</span></div>
                <div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {cost_score}%;'></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # D. "WHY THIS FABRIC?" XAI SECTION & E. TRADE-OFFS WITH RUNNER-UP
    col_why, col_tradeoffs = st.columns([1, 1])

    with col_why:
        st.markdown("#### 💡 Why this fabric?")
        for pt in explanation["score_bullet_points"]:
            st.markdown(f"- ✅ {pt}")

        st.markdown(f"**Key Material Advantages:** {explanation['key_advantages']}")
        st.markdown(f"**Recommended Care:** `{explanation['care_instructions']}`")

        if top.get("personalization_note") and "Standard" not in str(top["personalization_note"]):
            st.info(f"✨ **Personalized Insight:** {top['personalization_note']}")

    with col_tradeoffs:
        st.markdown("#### ⚖️ Trade-offs (Best Match vs. Runner-Up)")
        if alts:
            runner_up = alts[0]
            st.markdown(
                f"""
                <div style='background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 12px; color: #f8fafc;'>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;'>
                        <div style='border-right: 1px solid #334155; padding-right: 8px;'>
                            <b style='color: #34d399;'>🏆 Best Match: {top['fabric_name']}</b><br>
                            • Overall Score: <b>{top['final_score']:.1f}/100</b><br>
                            • Sustainability: <b>{sust_score}/100</b><br>
                            • Climate Suitability: <b>{climate_suit_score}/100</b><br>
                            • Comfort: <b>{comf_score}/100</b><br>
                            • Durability: <b>{dur_score}/100</b><br>
                            <span style='color: #94a3b8; font-size: 0.78rem;'>Advantage: {str(top['advantages'])[:60]}...</span>
                        </div>
                        <div style='padding-left: 4px;'>
                            <b style='color: #cbd5e1;'>🥈 Runner-up: {runner_up['fabric_name']}</b><br>
                            • Overall Score: <b>{runner_up['final_score']:.1f}/100</b><br>
                            • Sustainability: <b>{int(runner_up['sustainability_score'])}/100</b><br>
                            • Climate Suitability: <b>{int(runner_up['climate_score'])}/100</b><br>
                            • Comfort: <b>{int(runner_up['comfort_score'])}/100</b><br>
                            • Durability: <b>{int(runner_up['durability_score'])}/100</b><br>
                            <span style='color: #94a3b8; font-size: 0.78rem;'>Limitation: {str(runner_up['limitations'])[:60]}...</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Concrete explanation of why Best Match ranked higher
            if tradeoffs["runner_up_comparisons"]:
                first_t = tradeoffs["runner_up_comparisons"][0]
                st.markdown(
                    f"**Comparison Summary:** {top['fabric_name']} ranked higher overall ({top['final_score']:.1f} vs {runner_up['final_score']:.1f}) for **{sel_garment}** in **{sel_climate}** climate. {first_t['tradeoff_statement']} *{first_t['why_not_selected']}*"
                )
        else:
            st.write("No direct runner-up candidate meets this strict compatibility criteria.")

        # Optional Gemini AI narrative
        if gemini_key:
            ai_narrative = generate_ai_llm_explanation(
                top_fabric_name=top["fabric_name"],
                top_score=top["final_score"],
                garment=sel_garment,
                climate=sel_climate,
                reasons=explanation["score_bullet_points"],
                tradeoffs=tradeoffs["runner_up_comparisons"],
                api_key=gemini_key,
            )
            if ai_narrative:
                with st.expander("🤖 Gemini AI Detailed Textile Narrative", expanded=True):
                    st.write(ai_narrative)

    st.markdown("---")

    # F. PERSONALIZATION EXPLANATION SECTION
    active_weight_pct = int(user_prof['learning_weight'] * 100)
    general_weight_pct = 100 - active_weight_pct

    st.markdown(
        f"""
        <div style='background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 20px;'>
            <h5 style='color: #34d399; margin: 0 0 6px 0;'>🧠 How Personalization Works</h5>
            <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>
                {general_weight_pct}% General Recommendation + {active_weight_pct}% User Preference (Target Model: 70% / 30%)
            </div>
            <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0;'>
                The recommendation combines general fabric suitability with your learned preferences from previous feedback. 
                {"<b>Adaptive learning is currently active!</b>" if active_weight_pct > 0 else "<i>Currently using 100% general recommendation. Submit feedback below to activate your 30% preference learning weight.</i>"}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Alternatives Display
    st.markdown("#### 🥈 Top Alternative Fabrics")
    if alts:
        alt_cols = st.columns(len(alts))
        for idx, (col_alt, alt_row) in enumerate(zip(alt_cols, alts), start=2):
            with col_alt:
                st.markdown(
                    f"""
                    <div class='alt-card'>
                        <span style='background: #334155; color: #f8fafc; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;'>#{idx} ALTERNATIVE</span>
                        <h4 style='color: #34d399; margin: 6px 0;'>{alt_row['fabric_name']}</h4>
                        <div style='font-size: 1.5rem; font-weight: 800; color: #f8fafc;'>{alt_row['final_score']:.1f}<span style='font-size: 0.9rem; color: #94a3b8;'> / 100</span></div>
                        <p style='font-size: 0.82rem; color: #cbd5e1; margin: 4px 0;'><b>Category:</b> {alt_row['category']}</p>
                        <p style='font-size: 0.8rem; color: #34d399;'>🌱 Sust: <b>{alt_row['sustainability_score']}</b> | ☁️ Comf: <b>{alt_row['comfort_score']}</b></p>
                        <p style='font-size: 0.8rem; color: #38bdf8;'>🛡️ Dura: <b>{alt_row['durability_score']}</b> | 💰 Cost: <b>{alt_row['cost_score']}</b></p>
                        <p style='font-size: 0.78rem; color: #94a3b8; margin-top: 6px;'>{alt_row['advantages'][:75]}...</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ==============================================================================
    # USER FEEDBACK SECTION (IMPROVED VISIBILITY & BUTTONS)
    # ==============================================================================
    st.markdown("### ⭐ Was this recommendation useful?")
    st.caption("Your feedback updates your personal profile and trains the 70/30 adaptive learning model.")

    with st.form("feedback_form"):
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])

        with fb_col1:
            fb_useful = st.radio("Was this recommendation useful?", ["👍 Yes", "👎 No"], index=0, horizontal=True)

        with fb_col2:
            fb_rating = st.select_slider("Rate this recommendation", options=[1, 2, 3, 4, 5], value=5)

        with fb_col3:
            all_candidate_names = [top["fabric_name"]] + [a["fabric_name"] for a in alts]
            fb_selected_fabric = st.selectbox("Which fabric would you actually choose?", all_candidate_names, index=0)

        fb_factors = st.multiselect(
            "What influenced your decision?",
            ["High comfort", "Exceptional breathability", "Eco-friendly sustainability", "Durability & strength", "Budget affordability", "Soft drape", "Ease of care"],
            default=["High comfort", "Eco-friendly sustainability"],
        )
        fb_notes = st.text_input("Optional notes / feedback", placeholder="e.g., Exactly what I was looking for...")

        submit_feedback = st.form_submit_button("💾 Submit Feedback & Update Personalization", use_container_width=True)

        if submit_feedback:
            row_id = save_feedback(
                user_id=active_user,
                garment=sel_garment,
                climate=sel_climate,
                budget=sel_budget,
                recommended_fabric=top["fabric_name"],
                selected_fabric=fb_selected_fabric,
                rating=fb_rating,
                would_choose=("Yes" in fb_useful),
                decision_factors=", ".join(fb_factors),
                user_notes=fb_notes,
            )
            st.success(
                f"✅ Thank you! Your feedback helps personalize future recommendations. Profile for '{active_user}' updated!"
            )
            st.rerun()

# ==============================================================================
# TAB 3: LEADERBOARD & BREAKDOWN
# ==============================================================================
with tabs[2]:
    st.markdown("### 📊 Ranked Fabric Leaderboard")
    st.caption(f"Full ranking of all compatible fabrics for **{sel_garment}** in **{sel_climate}** climate.")

    ranked_display_df = rec_result["ranked_df"][[
        "rank", "fabric_name", "category", "final_score",
        "sustainability_score", "climate_score", "comfort_score",
        "durability_score", "cost_score", "performance_score", "origin_type"
    ]].rename(columns={
        "rank": "Rank",
        "fabric_name": "Fabric",
        "category": "Category",
        "final_score": "Final Score",
        "sustainability_score": "Sustainability",
        "climate_score": "Climate",
        "comfort_score": "Comfort",
        "durability_score": "Durability",
        "cost_score": "Affordability",
        "performance_score": "Performance",
        "origin_type": "Origin",
    })

    st.dataframe(ranked_display_df, use_container_width=True, hide_index=True)

    # Plotly Stacked / Multi-Bar Score Breakdown
    st.markdown("#### 📈 Multi-Criteria Score Breakdown by Fabric")
    plot_df = rec_result["ranked_df"].head(8).copy()

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(name="Sustainability", x=plot_df["fabric_name"], y=plot_df["sustainability_score"], marker_color="#10b981"))
    fig_bars.add_trace(go.Bar(name="Climate Fit", x=plot_df["fabric_name"], y=plot_df["climate_score"], marker_color="#34d399"))
    fig_bars.add_trace(go.Bar(name="Comfort", x=plot_df["fabric_name"], y=plot_df["comfort_score"], marker_color="#38bdf8"))
    fig_bars.add_trace(go.Bar(name="Durability", x=plot_df["fabric_name"], y=plot_df["durability_score"], marker_color="#fbbf24"))
    fig_bars.add_trace(go.Bar(name="Affordability", x=plot_df["fabric_name"], y=plot_df["cost_score"], marker_color="#f472b6"))
    fig_bars.add_trace(go.Bar(name="Performance", x=plot_df["fabric_name"], y=plot_df["performance_score"], marker_color="#a78bfa"))

    fig_bars.update_layout(
        template="plotly_dark",
        barmode="group",
        xaxis_title="Fabric",
        yaxis_title="Score (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=50),
        height=420,
    )
    st.plotly_chart(fig_bars, use_container_width=True)

# ==============================================================================
# TAB 4: FABRIC COMPARISON (CLEAN TABLE & RADAR)
# ==============================================================================
with tabs[3]:
    st.markdown("### ⚖️ Side-by-Side Fabric Comparison")
    st.caption("Compare trade-offs across multiple textiles with clean matrix tables and interactive radar charts.")

    all_fabrics = engine.get_all_fabrics()
    fabric_names = all_fabrics["fabric_name"].tolist()

    default_selection = [top["fabric_name"]]
    if alts:
        default_selection.append(alts[0]["fabric_name"])
    if len(alts) > 1:
        default_selection.append(alts[1]["fabric_name"])

    selected_compare = st.multiselect(
        "Select 2 to 4 Fabrics to Compare",
        fabric_names,
        default=default_selection[:3],
    )

    if len(selected_compare) < 2:
        st.warning("Please select at least 2 fabrics to generate a comparison.")
    else:
        compare_df = all_fabrics[all_fabrics["fabric_name"].isin(selected_compare)]

        # Clean Structured Comparison Table matching requirement 8:
        # Rows: Sustainability, Comfort, Durability, Breathability, Climate Fit, Cost, Overall Score
        # Columns: Fabric A, Fabric B, ...
        comp_metrics_rows = [
            "Sustainability",
            "Comfort",
            "Durability",
            "Breathability",
            "Climate Fit (Moderate)",
            "Cost (Affordability)",
            "Composite Eco Score",
        ]
        
        comp_table_data = {"Criterion": comp_metrics_rows}
        for _, row in compare_df.iterrows():
            f_name = row["fabric_name"]
            comp_table_data[f_name] = [
                f"{row['sustainability_score']}/100",
                f"{row['comfort']}/100",
                f"{row['durability']}/100",
                f"{row['breathability']}/100",
                f"{row.get('suitability_moderate', 75)}/100",
                f"{row['affordability_score']}/100",
                f"{row['sustainability_score']}/100",
            ]
        
        clean_matrix_df = pd.DataFrame(comp_table_data)

        col_tbl, col_rad = st.columns([1, 1])

        with col_tbl:
            st.markdown("#### Direct Parameter Comparison Matrix")
            st.dataframe(clean_matrix_df, use_container_width=True, hide_index=True)

            st.markdown("#### Advantages & Limitations")
            for _, r in compare_df.iterrows():
                st.markdown(f"**{r['fabric_name']}:**")
                st.markdown(f"- *Advantages:* {r['advantages']}")
                st.markdown(f"- *Limitations:* {r['limitations']}")

        with col_rad:
            st.markdown("#### Visual Radar Analysis")
            radar_categories = [
                "Sustainability",
                "Water Efficiency",
                "Low Carbon",
                "Breathability",
                "Comfort",
                "Durability",
                "Affordability",
                "Moisture Mgmt",
            ]

            fig_radar = go.Figure()
            colors = ["#10b981", "#38bdf8", "#fbbf24", "#f472b6"]

            for idx, (_, row) in enumerate(compare_df.iterrows()):
                values = [
                    row["sustainability_score"],
                    row["water_efficiency"],
                    row["carbon_score"],
                    row["breathability"],
                    row["comfort"],
                    row["durability"],
                    row["affordability_score"],
                    row["moisture_management"],
                ]
                values.append(values[0])

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=radar_categories + [radar_categories[0]],
                        fill="toself",
                        name=row["fabric_name"],
                        line=dict(color=colors[idx % len(colors)]),
                        opacity=0.55,
                    )
                )

            fig_radar.update_layout(
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=30, r=30, t=30, b=30),
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ==============================================================================
# TAB 5: SUSTAINABILITY HUB
# ==============================================================================
with tabs[4]:
    st.markdown("### 🌱 Multi-Dimensional Sustainability Hub")
    st.markdown(
        """
        > [!IMPORTANT]
        > **Sustainability is Multi-Dimensional:** Evaluating textiles solely on a single factor (like biodegradability) leads to trade-off blindspots. Conventional cotton decomposes naturally, but requires massive freshwater irrigation and pesticides. Recycled polyester preserves bottles from landfills, but sheds persistent synthetic microplastics. Our system tracks the multi-dimensional lifecycle footprint.
        """
    )

    fab_df = engine.get_all_fabrics()

    # Fabric Dimensions Deep Dive Table
    st.markdown("#### 🔬 Detailed Fabric Sustainability Dimensions")
    st.caption("Data derived from lifecycle assessments (LCAs). Non-tracked criteria are accurately flagged as unavailable.")

    hub_rows = []
    for _, r in fab_df.iterrows():
        hub_rows.append({
            "Fabric": r["fabric_name"],
            "Origin Category": r["category"],
            "Sustainability Score": f"{r['sustainability_score']} / 100",
            "🌍 Carbon Impact": f"{r['carbon_score']} / 100",
            "💧 Water Impact": f"{r['water_efficiency']} / 100",
            "♻️ Recyclability": f"{r['recyclability']} / 100",
            "🌱 Biodegradability": f"{r['biodegradability']} / 100",
            "⚡ Energy Impact": "Data unavailable",
            "🧪 Chemical Processing": "Data unavailable",
        })

    hub_df = pd.DataFrame(hub_rows)
    st.dataframe(hub_df, use_container_width=True, hide_index=True)

    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        st.markdown("#### 📈 Sustainability vs. Affordability Trade-off Landscape")
        fig_scatter = px.scatter(
            fab_df,
            x="affordability_score",
            y="sustainability_score",
            size="durability",
            color="origin_type",
            hover_name="fabric_name",
            text="fabric_name",
            labels={
                "affordability_score": "Cost Friendliness / Affordability (0 = Luxury, 100 = Budget)",
                "sustainability_score": "Composite Sustainability Score (0-100)",
                "origin_type": "Origin Type",
                "durability": "Durability",
            },
            color_discrete_map={"Natural": "#10b981", "Regenerated": "#38bdf8", "Synthetic": "#fbbf24"},
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            template="plotly_dark",
            height=440,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_s2:
        st.markdown("#### 📊 Rank Fabrics by Dimension")
        metric_choice = st.selectbox(
            "Select Dimension to Rank",
            ["water_efficiency", "carbon_score", "recyclability", "biodegradability", "sustainability_score"],
            format_func=lambda x: {
                "water_efficiency": "💧 Water Efficiency & Conservation",
                "carbon_score": "📉 Low Carbon Footprint",
                "recyclability": "🔄 Closed-Loop Recyclability",
                "biodegradability": "🍂 Natural Biodegradability",
                "sustainability_score": "🌱 Composite Sustainability",
            }[x],
        )

        ranked_dim = fab_df.sort_values(by=metric_choice, ascending=True)
        fig_dim = px.bar(
            ranked_dim,
            x=metric_choice,
            y="fabric_name",
            orientation="h",
            color=metric_choice,
            color_continuous_scale="Greens",
            labels={metric_choice: "Score (0-100)", "fabric_name": "Fabric"},
            height=440,
        )
        fig_dim.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dim, use_container_width=True)

# ==============================================================================
# TAB 6: PERSONALIZATION & LEARNING PROFILE
# ==============================================================================
with tabs[5]:
    st.markdown(f"### 👤 Personalization Profile for: `{active_user}`")
    st.markdown(
        """
        Our personalization engine applies an adaptive Bayesian-inspired learning formula:
        $$\\text{Final Score} = (1 - \\alpha) \\times \\text{General Recommendation Score} + \\alpha \\times \\text{User Preference Score}$$
        - **Initial State ($\alpha = 0.0$):** 100% general recommendation engine.
        - **Ramp-Up ($\alpha = 0.15$ after 1 interaction):** 85% General + 15% User Preference.
        - **Full Adaptation ($\alpha = 0.30$ after 2+ interactions):** 70% General + 30% User Preference score.
        """
    )

    prof = engine.personalization_engine.get_user_profile(active_user)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Past Interactions", prof["interaction_count"])
    m2.metric("Personalization Weight (α)", f"{int(prof['learning_weight'] * 100)}%")
    m3.metric("Avg Given Rating", f"{prof['avg_rating']} ⭐" if prof['avg_rating'] > 0 else "N/A")
    m4.metric("Learning Status", "Active (30%)" if prof["has_sufficient_history"] else ("Ramping (15%)" if prof["interaction_count"] == 1 else "Collecting Data (0%)"))

    st.markdown("---")

    col_aff1, col_aff2 = st.columns([1, 1])

    with col_aff1:
        st.markdown("#### Learned Fabric Affinities")
        affinities = prof["fabric_affinities"]
        if not affinities:
            st.info(f"No ratings yet for user '{active_user}'. Submit feedback in the Recommendation tab to build this profile!")
        else:
            aff_df = pd.DataFrame([{"Fabric": k, "Affinity Score": v} for k, v in affinities.items()]).sort_values(
                by="Affinity Score", ascending=False
            )
            fig_aff = px.bar(
                aff_df,
                x="Affinity Score",
                y="Fabric",
                orientation="h",
                color="Affinity Score",
                color_continuous_scale="Viridis",
                range_x=[0, 100],
            )
            fig_aff.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_aff, use_container_width=True)

    with col_aff2:
        st.markdown("#### User History Records")
        user_history = get_user_history(active_user)
        if user_history.empty:
            st.write("No historical feedback stored.")
        else:
            st.dataframe(
                user_history[[
                    "created_at", "garment", "climate", "selected_fabric", "rating", "would_choose"
                ]].rename(columns={
                    "created_at": "Timestamp",
                    "garment": "Garment",
                    "climate": "Climate",
                    "selected_fabric": "Chosen Fabric",
                    "rating": "Rating",
                    "would_choose": "Accepted",
                }),
                use_container_width=True,
                hide_index=True,
            )

# ==============================================================================
# TAB 7: FEEDBACK HISTORY (SQLITE DATA)
# ==============================================================================
with tabs[6]:
    st.markdown("### 📜 System-Wide Feedback Database")
    st.caption("All user submissions stored in SQLite (`database/fabrics_feedback.db`).")

    all_fb = get_all_feedback()
    if all_fb.empty:
        st.info("No feedback has been submitted across any users yet.")
    else:
        st.dataframe(all_fb, use_container_width=True, hide_index=True)

        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            st.markdown("#### Ratings Distribution")
            rating_counts = all_fb["rating"].value_counts().reset_index()
            rating_counts.columns = ["Rating", "Count"]
            fig_ratings = px.bar(rating_counts, x="Rating", y="Count", color="Rating", color_continuous_scale="Greens")
            fig_ratings.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_ratings, use_container_width=True)

        with col_fb2:
            st.markdown("#### Most Chosen Fabrics")
            chosen_counts = all_fb["selected_fabric"].value_counts().reset_index()
            chosen_counts.columns = ["Fabric", "Selections"]
            fig_chosen = px.pie(chosen_counts, values="Selections", names="Fabric", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_chosen.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_chosen, use_container_width=True)

# ==============================================================================
# TAB 8: METHODOLOGY & RECOMMENDATION PIPELINE
# ==============================================================================
with tabs[7]:
    st.markdown("### ℹ️ Methodology & Recommendation Pipeline")
    st.markdown(
        """
        #### Project Overview
        - **Project Title:** Smart Fabric Recommendation System
        - **Track:** Track 1 — FUTURE FABRIC
        - **Objective:** An intelligent, explainable, personalized decision-support engine assisting designers, brands, and conscious consumers to choose the most sustainable and functionally appropriate fabric for any garment.
        """
    )

    st.markdown("#### 🔄 Visual Recommendation Pipeline")
    st.caption("Step-by-step decision architecture evaluated for every recommendation request:")

    steps = [
        ("1. User Inputs", "Collects garment type, climate conditions, budget, sustainability, comfort, and durability priorities."),
        ("2. Garment Compatibility Filter", "Pre-filters fabrics against garment_compatibility.csv to eliminate unviable candidates."),
        ("3. Climate Suitability Evaluation", "Maps ambient conditions (Hot, Humid, Moderate, Cold, Rainy) to empirical thermal regulation."),
        ("4. Sustainability Evaluation", "Calculates composite footprint across water conservation, carbon footprint, recyclability, and biodegradability."),
        ("5. User Preference Weighting", "Dynamically re-normalizes criteria weights based on user priority sliders so weights strictly sum to 1.0."),
        ("6. Multi-Criteria Ranking (MCDA)", "Computes multi-attribute general utility scores and ranks fabrics descending."),
        ("7. Best Match & Runner-Up Identification", "Extracts the #1 Best Match, runner-up alternative, and secondary candidates."),
        ("8. Explainable AI (XAI) Synthesis", "Generates human-readable 'Why this fabric?' rationale and head-to-head trade-off analysis."),
        ("9. User Feedback Logging", "Captures user rating (1-5 stars) and acceptance decision in local SQLite database."),
        ("10. Personalized Profile Update", "Updates user fabric affinity and ramps personalization blending weight (up to 70% General + 30% User Preference)."),
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class='pipeline-step'>
                <b style='color: #34d399;'>{title}</b>
                <p style='color: #cbd5e1; margin: 3px 0 0 0; font-size: 0.88rem;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        #### Mathematical Scoring Model
        Recommendations are computed via a dynamically weighted utility function:

        $$\\text{Score}_{\\text{general}} = \\sum_{i} w_i \\cdot S_i$$

        Where criteria $S_i$ include:
        1. **Garment Compatibility:** Empirical structural suitability for target clothing item.
        2. **Climate Suitability ($w_{\\text{climate}} = 20\\%$ baseline):** Temperature and humidity regulation index.
        3. **Sustainability Score ($w_{\\text{sust}} = 30\\%$ baseline):** Composite of water efficiency, low carbon footprint, recyclability, and biodegradability.
        4. **Comfort & Breathability ($w_{\\text{comf}} = 15\\%$ baseline):** Softness, skin feel, and airflow.
        5. **Durability ($w_{\\text{dur}} = 15\\%$ baseline):** Tensile strength and laundering resistance.
        6. **Cost / Affordability ($w_{\\text{cost}} = 10\\%$ baseline):** Normalized price tier.
        7. **Performance Requirement ($w_{\\text{perf}} = 10\\%$ baseline):** Moisture wicking, stretch, or thermal insulation.

        #### Personalization Blending Formula
        When feedback is logged in SQLite:
        $$\\text{Score}_{\\text{final}} = (1 - \\alpha) \\times \\text{Score}_{\\text{general}} + \\alpha \\times \\text{Score}_{\\text{user preference}}$$
        where $\\alpha = 0.30$ ($70\\% / 30\\%$) once sufficient interaction history is established.
        """
    )
