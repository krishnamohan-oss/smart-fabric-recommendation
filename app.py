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

# Custom Eco-Modern CSS
st.markdown(
    """
    <style>
    /* Main Theme Variables */
    :root {
        --eco-dark: #1b4332;
        --eco-primary: #2d6a4f;
        --eco-light: #52b788;
        --eco-bg: #f4f9f4;
        --eco-card: #ffffff;
        --eco-accent: #74c69e;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    
    .trophy-badge {
        background: linear-gradient(135deg, #2d6a4f, #1b4332);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .top-match-card {
        background-color: #ffffff;
        border: 2px solid #52b788;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 14px rgba(45, 106, 79, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .alt-card {
        background-color: #ffffff;
        border: 1px solid #d8f3dc;
        border-radius: 10px;
        padding: 1.1rem;
        height: 100%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    
    .metric-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pill-green { background-color: #d8f3dc; color: #1b4332; }
    .pill-blue { background-color: #e0f2fe; color: #0369a1; }
    .pill-orange { background-color: #fef3c7; color: #92400e; }
    
    .score-circle {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2d6a4f;
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

# Sidebar Setup
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&q=80",
        caption="Future Fabric AI | Track 1",
        use_container_width=True,
    )
    st.title("🌿 Smart Fabric AI")
    st.caption("Intelligent Sustainable Textile Decision System")

    zip_path = os.path.join(os.path.dirname(__file__), "smart-fabric-recommendation-github-upload.zip")
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

    # Display active user profile summary
    user_prof = engine.personalization_engine.get_user_profile(active_user)
    st.markdown(
        f"""
        <div style='background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px; margin-top: 8px;'>
            <small><b>History:</b> {user_prof['interaction_count']} selections</small><br>
            <small><b>Personalization Weight:</b> {int(user_prof['learning_weight'] * 100)}% active</small><br>
            <small><b>Avg User Rating:</b> {'⭐ ' + str(user_prof['avg_rating']) if user_prof['avg_rating'] > 0 else 'No ratings yet'}</small>
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
        <small style='color: #6b7280;'>
        <b>Global DB Stats:</b><br>
        • Total Reviews: {stats['total_reviews']}<br>
        • Overall Acceptance: {stats['acceptance_rate']}%<br>
        • Top Chosen Fabric: {stats['top_selected_fabric']}
        </small>
        """,
        unsafe_allow_html=True,
    )

# App Navigation Tabs
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

    zip_path = os.path.join(os.path.dirname(__file__), "smart-fabric-recommendation-github-upload.zip")
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
            <div style='background: #ffffff; border: 1px solid #d8f3dc; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                <h4 style='color: #2d6a4f; margin-top: 0;'>🚀 Quick Start Workflow</h4>
                <ol style='padding-left: 20px; color: #374151; font-size: 0.95rem; line-height: 1.6;'>
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

        st.markdown("#### Key System Metrics")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Curated Fabrics", len(engine.get_all_fabrics()))
        with kpi2:
            st.metric("Garment Types", len(engine.get_available_garments()))
        with kpi3:
            st.metric("Evaluation Criteria", "14+ Metrics")

# ==============================================================================
# TAB 2: RECOMMENDATION (CORE ENGINE)
# ==============================================================================
with tabs[1]:
    st.markdown("### 🎯 Find Your Ideal Sustainable Fabric")
    st.caption("Select your garment context, climate conditions, and design priorities below.")

    with st.form("recommendation_form"):
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            garments_list = engine.get_available_garments()
            sel_garment = st.selectbox("1. Garment Type", garments_list, index=garments_list.index("T-shirt") if "T-shirt" in garments_list else 0)
            sel_climate = st.selectbox("2. Climate Condition", ["Hot", "Hot & Humid", "Moderate", "Cold", "Rainy"], index=2)
            sel_budget = st.selectbox("3. Budget Tier", ["Low", "Medium", "High"], index=1)

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

    st.markdown("---")

    # Dynamic Weights Visual Bar
    st.markdown("##### ⚖️ Dynamic Criteria Weights for This Request")
    w_cols = st.columns(6)
    w_cols[0].metric("Sustainability", f"{int(weights['sustainability'] * 100)}%")
    w_cols[1].metric("Climate", f"{int(weights['climate'] * 100)}%")
    w_cols[2].metric("Comfort", f"{int(weights['comfort'] * 100)}%")
    w_cols[3].metric("Durability", f"{int(weights['durability'] * 100)}%")
    w_cols[4].metric("Cost/Affordability", f"{int(weights['cost'] * 100)}%")
    w_cols[5].metric("Performance", f"{int(weights['performance'] * 100)}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # 🏆 BEST MATCH CARD
    st.markdown(
        f"""
        <div class='top-match-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                <div>
                    <span class='trophy-badge'>🏆 BEST MATCH</span>
                    <h2 style='color: #1b4332; margin: 0;'>{top['fabric_name']}</h2>
                    <p style='color: #4b5563; margin-top: 4px; font-size: 0.95rem;'>
                        <b>Category:</b> {top['category']} | <b>Origin:</b> {top['origin_type']} | <b>Garment:</b> {sel_garment}
                    </p>
                </div>
                <div style='text-align: right;'>
                    <div class='score-circle'>{top['final_score']:.1f}<span style='font-size: 1.1rem; color: #6b7280;'>/100</span></div>
                    <small style='color: #4b5563; font-weight: 600;'>Overall Compatibility Score</small>
                </div>
            </div>
            <div style='margin-top: 12px;'>
                <span class='metric-pill pill-green'>🌱 Sustainability: {top['sustainability_score']}/100</span>
                <span class='metric-pill pill-blue'>🌤️ Climate Fit: {top['climate_score']}/100</span>
                <span class='metric-pill pill-blue'>☁️ Comfort: {top['comfort_score']}/100</span>
                <span class='metric-pill pill-green'>🛡️ Durability: {top['durability_score']}/100</span>
                <span class='metric-pill pill-orange'>💰 Affordability: {top['cost_score']}/100</span>
                <span class='metric-pill pill-blue'>⚡ Performance: {top['performance_score']}/100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Explainable AI Section
    col_why, col_tradeoffs = st.columns([1, 1])

    with col_why:
        st.markdown("#### 💡 Why This Fabric?")
        for pt in explanation["score_bullet_points"]:
            st.markdown(f"- ✅ {pt}")

        st.markdown(f"**Key Material Advantages:** {explanation['key_advantages']}")
        st.markdown(f"**Recommended Care:** `{explanation['care_instructions']}`")

        # Check for personalization note
        if top.get("personalization_note") and "Standard" not in str(top["personalization_note"]):
            st.info(f"✨ **Personalized Insight:** {top['personalization_note']}")

    with col_tradeoffs:
        st.markdown("#### ⚖️ Trade-offs & Considerations")
        st.markdown(f"- ⚠️ **Watch out for:** {explanation['key_limitations']}")

        if tradeoffs["runner_up_comparisons"]:
            st.markdown("**Comparison vs. Runner-Up Alternatives:**")
            for alt_t in tradeoffs["runner_up_comparisons"]:
                st.markdown(
                    f"- **#{alt_t['rank']} {alt_t['fabric_name']}** ({alt_t['overall_score']:.1f}/100): {alt_t['tradeoff_statement']} *{alt_t['why_not_selected']}*"
                )

        # Optional Gemini AI generated narrative
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

    # Top 3 Alternatives Display
    st.markdown("#### 🥈 Top Alternative Fabrics")
    if alts:
        alt_cols = st.columns(len(alts))
        for idx, (col_alt, alt_row) in enumerate(zip(alt_cols, alts), start=2):
            with col_alt:
                st.markdown(
                    f"""
                    <div class='alt-card'>
                        <span style='background: #e2e8f0; color: #334155; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;'>#{idx} ALTERNATIVE</span>
                        <h4 style='color: #2d6a4f; margin: 6px 0;'>{alt_row['fabric_name']}</h4>
                        <div style='font-size: 1.5rem; font-weight: 800; color: #1b4332;'>{alt_row['final_score']:.1f}<span style='font-size: 0.9rem; color: #64748b;'>/100</span></div>
                        <p style='font-size: 0.82rem; color: #475569; margin: 4px 0;'><b>Category:</b> {alt_row['category']}</p>
                        <p style='font-size: 0.8rem; color: #166534;'>🌱 Sust: <b>{alt_row['sustainability_score']}</b> | ☁️ Comf: <b>{alt_row['comfort_score']}</b></p>
                        <p style='font-size: 0.8rem; color: #1e3a8a;'>🛡️ Dura: <b>{alt_row['durability_score']}</b> | 💰 Cost: <b>{alt_row['cost_score']}</b></p>
                        <p style='font-size: 0.78rem; color: #64748b; margin-top: 6px;'>{alt_row['advantages'][:80]}...</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ==============================================================================
    # USER FEEDBACK FORM (LEARNING LOOP)
    # ==============================================================================
    st.markdown("### 💬 Was this recommendation helpful? (Train Personalization)")
    st.caption("Your feedback updates the SQLite persistence layer and improves recommendations for your user profile.")

    with st.form("feedback_form"):
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])

        with fb_col1:
            fb_rating = st.select_slider("Rating (1-5 stars)", options=[1, 2, 3, 4, 5], value=5)

        with fb_col2:
            fb_would_choose = st.radio("Would you choose this fabric?", ["Yes", "No"], index=0, horizontal=True)

        with fb_col3:
            # Options to pick which fabric they actually preferred
            all_candidate_names = [top["fabric_name"]] + [a["fabric_name"] for a in alts]
            fb_selected_fabric = st.selectbox("Which fabric would you actually pick?", all_candidate_names, index=0)

        fb_factors = st.multiselect(
            "What influenced your decision?",
            ["High comfort", "Exceptional breathability", "Eco-friendly sustainability", "Durability & strength", "Budget affordability", "Soft drape", "Ease of care"],
            default=["High comfort", "Eco-friendly sustainability"],
        )
        fb_notes = st.text_input("Optional notes / feedback", placeholder="e.g., Perfect for summer daily wear...")

        submit_feedback = st.form_submit_button("💾 Save Feedback & Update Personalization Model", use_container_width=True)

        if submit_feedback:
            row_id = save_feedback(
                user_id=active_user,
                garment=sel_garment,
                climate=sel_climate,
                budget=sel_budget,
                recommended_fabric=top["fabric_name"],
                selected_fabric=fb_selected_fabric,
                rating=fb_rating,
                would_choose=(fb_would_choose == "Yes"),
                decision_factors=", ".join(fb_factors),
                user_notes=fb_notes,
            )
            st.success(
                f"✅ Feedback logged! Profile for '{active_user}' updated. Future recommendations will adapt to your preference for {fb_selected_fabric}."
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
    fig_bars.add_trace(go.Bar(name="Sustainability", x=plot_df["fabric_name"], y=plot_df["sustainability_score"], marker_color="#2d6a4f"))
    fig_bars.add_trace(go.Bar(name="Climate Fit", x=plot_df["fabric_name"], y=plot_df["climate_score"], marker_color="#52b788"))
    fig_bars.add_trace(go.Bar(name="Comfort", x=plot_df["fabric_name"], y=plot_df["comfort_score"], marker_color="#74c69e"))
    fig_bars.add_trace(go.Bar(name="Durability", x=plot_df["fabric_name"], y=plot_df["durability_score"], marker_color="#95d5b2"))
    fig_bars.add_trace(go.Bar(name="Affordability", x=plot_df["fabric_name"], y=plot_df["cost_score"], marker_color="#b7e4c7"))
    fig_bars.add_trace(go.Bar(name="Performance", x=plot_df["fabric_name"], y=plot_df["performance_score"], marker_color="#0077b6"))

    fig_bars.update_layout(
        barmode="group",
        xaxis_title="Fabric",
        yaxis_title="Score (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=50),
        height=420,
    )
    st.plotly_chart(fig_bars, use_container_width=True)

# ==============================================================================
# TAB 4: FABRIC COMPARISON (RADAR & METRICS)
# ==============================================================================
with tabs[3]:
    st.markdown("### ⚖️ Side-by-Side Fabric Comparison")
    st.caption("Compare trade-offs across multiple textiles with interactive radar charts.")

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
        colors = ["#2d6a4f", "#0077b6", "#e07a5f", "#9d4edd"]

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
            # Close the polygon
            values.append(values[0])

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=radar_categories + [radar_categories[0]],
                    fill="toself",
                    name=row["fabric_name"],
                    line=dict(color=colors[idx % len(colors)]),
                    opacity=0.6,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=40, b=40),
            height=480,
        )

        col_rad, col_tbl = st.columns([1, 1])
        with col_rad:
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_tbl:
            st.markdown("#### Comparison Metrics Table")
            display_cols = [
                "fabric_name", "category", "origin_type", "sustainability_score",
                "water_efficiency", "carbon_score", "comfort", "durability", "affordability_score"
            ]
            st.dataframe(
                compare_df[display_cols].rename(columns={
                    "fabric_name": "Fabric",
                    "category": "Category",
                    "origin_type": "Origin",
                    "sustainability_score": "Sust",
                    "water_efficiency": "Water Eff",
                    "carbon_score": "Carbon",
                    "comfort": "Comfort",
                    "durability": "Durability",
                    "affordability_score": "Afford",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Advantages & Limitations")
            for _, r in compare_df.iterrows():
                st.markdown(f"**{r['fabric_name']}:**")
                st.markdown(f"- *Advantages:* {r['advantages']}")
                st.markdown(f"- *Limitations:* {r['limitations']}")

# ==============================================================================
# TAB 5: SUSTAINABILITY HUB
# ==============================================================================
with tabs[4]:
    st.markdown("### 🌱 Multi-Dimensional Sustainability Hub")
    st.markdown(
        """
        > [!IMPORTANT]
        > **Sustainability is Never a Single Metric:** A fabric that scores high in biodegradability (like conventional cotton) might severely deplete fresh water supplies and require heavy agrochemicals. Synthetics like recycled polyester conserve water, but shed non-biodegradable microfibers. Our system captures this complete lifecycle footprint.
        """
    )

    fab_df = engine.get_all_fabrics()

    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        st.markdown("#### 🔬 Sustainability vs. Affordability Trade-off")
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
            title="Textile Landscape: Eco-Impact vs. Accessibility (Bubble size = Durability)",
            color_discrete_map={"Natural": "#2d6a4f", "Regenerated": "#0077b6", "Synthetic": "#e07a5f"},
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_s2:
        st.markdown("#### 🌍 Environmental Impact Dimensions")
        metric_choice = st.selectbox(
            "Select Environmental Dimension to Rank",
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
            height=480,
        )
        fig_dim.update_layout(margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
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
        - **Initial State ($\alpha = 0.0$):** Pure general recommendation engine.
        - **Ramp-Up ($\alpha = 0.15$ after 1 interaction):** Begins capturing user affinity.
        - **Full Adaptation ($\alpha = 0.30$ after 2+ interactions):** 70% General + 30% Personalized Preference score.
        """
    )

    prof = engine.personalization_engine.get_user_profile(active_user)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Past Interactions", prof["interaction_count"])
    m2.metric("Personalization Weight (α)", f"{int(prof['learning_weight'] * 100)}%")
    m3.metric("Avg Given Rating", f"{prof['avg_rating']} ⭐" if prof['avg_rating'] > 0 else "N/A")
    m4.metric("Learning Status", "Active" if prof["has_sufficient_history"] else "Collecting Data")

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
            fig_aff.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
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
            fig_ratings.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_ratings, use_container_width=True)

        with col_fb2:
            st.markdown("#### Most Chosen Fabrics")
            chosen_counts = all_fb["selected_fabric"].value_counts().reset_index()
            chosen_counts.columns = ["Fabric", "Selections"]
            fig_chosen = px.pie(chosen_counts, values="Selections", names="Fabric", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_chosen.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_chosen, use_container_width=True)

# ==============================================================================
# TAB 8: METHODOLOGY & ABOUT
# ==============================================================================
with tabs[7]:
    st.markdown("### ℹ️ About & Scientific Methodology")
    st.markdown(
        """
        #### Project Overview
        - **Project Title:** Smart Fabric Recommendation System
        - **Track:** Track 1 — FUTURE FABRIC
        - **Objective:** An intelligent, explainable, personalized decision-support engine assisting designers, brands, and conscious consumers to choose the most sustainable and functionally appropriate fabric for any garment.

        #### Multi-Criteria Decision Analysis (MCDA) Scoring Model
        Recommendations are computed via a dynamically weighted utility function:

        $$\\text{Score}_{\\text{general}} = \\sum_{i} w_i \\cdot S_i$$

        Where criteria $S_i$ include:
        1. **Sustainability Score ($w_1 = 30\\%$ baseline):** Multi-dimensional composite of water efficiency, low carbon footprint, recyclability, and natural biodegradability.
        2. **Climate Suitability ($w_2 = 20\\%$ baseline):** Empirical temperature and humidity regulation index for the target climate.
        3. **Comfort & Breathability ($w_3 = 15\\%$ baseline):** Tactile skin feel, softness, and air permeability.
        4. **Durability ($w_4 = 15\\%$ baseline):** Tensile strength, friction resistance, and laundering longevity.
        5. **Affordability / Cost Friendliness ($w_5 = 10\\%$ baseline):** Normalized cost tier.
        6. **Performance Requirement ($w_6 = 10\\%$ baseline):** Specific functional goals (e.g. moisture wicking for activewear, thermal insulation for winter).

        Dynamic user priority multipliers adjust individual weights, which are strictly re-normalized such that $\\sum w_i = 1.0$.

        #### Personalization & Adaptive Feedback Loop
        User feedback captured in SQLite is used to compute historical fabric and category affinity. When $\\ge 2$ interactions exist, the score blends according to:

        $$\\text{Score}_{\\text{final}} = 0.70 \\times \\text{Score}_{\\text{general}} + 0.30 \\times \\text{Score}_{\\text{user preference}}$$

        #### Data Governance & Disclaimer
        The numerical scores (0–100) are normalized demonstration benchmarks derived from synthesized life cycle assessments (LCAs), the Higg Materials Sustainability Index (MSI), and Textile Exchange material summaries. They are intended for demonstration and comparative decision-support.
        """
    )
