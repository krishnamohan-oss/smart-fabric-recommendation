# Smart Fabric Recommendation System 🌿

> **Track 1 — FUTURE FABRIC | Hackathon Ready AI Decision-Support Platform**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/krishnamohan-oss/smart-fabric-recommendation)

An intelligent, multi-criteria, and explainable AI decision-support platform that recommends sustainable and functionally optimal fabrics for diverse garments and climates. By balancing ecological footprint, thermal comfort, physical durability, athletic performance, affordability, and adaptive user personalization, the system bridges the gap between textile sustainability and garment practicality.

---

## 📖 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Technology Stack](#-technology-stack)
6. [Fabric Dataset & Benchmark Metrics](#-fabric-dataset--benchmark-metrics)
7. [Recommendation Algorithm (MCDA)](#-recommendation-algorithm-mcda)
8. [Personalization & Adaptive Learning](#-personalization--adaptive-learning)
9. [Multi-Dimensional Sustainability Methodology](#-multi-dimensional-sustainability-methodology)
10. [Installation & Setup](#-installation--setup)
11. [How to Run Locally](#-how-to-run-locally)
12. [Walkthrough & Screenshots](#-walkthrough--screenshots)
13. [Testing & Quality Assurance](#-testing--quality-assurance)
14. [Future Roadmap](#-future-roadmap)
15. [Contributors & License](#-contributors--license)

---

## 🎯 Problem Statement

Fashion and textile manufacturing is one of the most resource-intensive industries on the planet, responsible for significant global freshwater consumption, wastewater discharge, and greenhouse gas emissions.

However, naive sustainability approaches simply choose the fabric with the single lowest carbon or water metric. In practical fashion design and manufacturing, this naive selection fails:
- **Functional Incompatibility:** Selecting stiff raw hemp for fine sarees or fluid silk for heavy winter jackets causes structural and aesthetic failure.
- **Climate Mismatch:** Utilizing heavy virgin wool in humid tropical climates causes overheating; deploying ultralight linen in sub-zero winters causes hypothermia.
- **Hidden Ecological Trade-offs:** Conventional cotton is biodegradable but depletes freshwater through massive irrigation and pesticides; recycled polyester conserves water and resists tearing but sheds non-biodegradable microplastics; viscose can involve toxic carbon disulfide processing.

Sustainable textile selection is fundamentally a **Multi-Criteria Decision Problem with non-negotiable trade-offs**.

---

## 💡 Solution Overview

The **Smart Fabric Recommendation System** acts as an AI co-pilot for fashion designers, conscious brands, and consumers:
1. **Garment Compatibility Rule-Engine:** Pre-filters fabrics based on structural, drape, and mechanical suitability across 13+ garment categories.
2. **Dynamic Weighting MCDA:** Re-weights criteria dynamically based on the user's priorities (sustainability, durability, comfort, budget, and performance) and strictly normalizes weights to 100%.
3. **Personalization Learning Loop:** Backed by SQLite, the system captures user feedback, ratings (1–5 stars), and acceptance decisions, dynamically tuning recommendations using a **70% General + 30% User Preference** model.
4. **Explainable AI (XAI):** Unpacks the rationale with explicit *“Why this fabric?”* bullet points, material advantages, and explicit *Trade-off* comparisons against runner-up fabrics.
5. **Interactive Data Visualizations:** Real-time multi-dimensional Plotly radar charts, criteria breakdowns, and sustainability landscape scatter plots.

---

## ✨ Key Features

- **🏆 BEST MATCH & Top 3 Alternatives:** Instant generation of top-performing textiles with overall score (0–100) and criteria badges.
- **🌤️ Climate-Adaptive Intelligence:** Optimized scores across 5 climate conditions (*Hot, Hot & Humid, Moderate, Cold, Rainy*).
- **⚖️ Dynamic Weight Vector:** Sliders dynamically shift utility weights (e.g. tight budget elevates affordability importance; high sustainability elevates eco-metrics).
- **🔍 Multi-Dimensional Radar Comparison:** Interactive Plotly radar charts comparing 2 to 4 textiles across 8 essential dimensions simultaneously.
- **🌱 Sustainability Hub:** Unpacks the lifecycle assessment (LCA) paradoxes (water efficiency, carbon footprint, recyclability, and biodegradability).
- **👤 Adaptive Personalization Engine:** SQLite-powered learning profile tracking fabric affinities and updating rankings as users rate fabrics.
- **💬 Direct User Feedback System:** In-app rating and selection logger feeding directly into model training.
- **🤖 Dual-Mode Explainability:** Complete deterministic offline rule-based explanations, with optional Google Gemini LLM synthesis.

---

## 🏗️ System Architecture

```
smart-fabric-recommendation/
│
├── app.py                         # Streamlit multi-tab dashboard & UI
├── requirements.txt               # Pinned dependencies
├── README.md                      # Comprehensive documentation
├── .gitignore                     # Git ignore rules for caches & secrets
├── .env.example                   # Environment variable template
│
├── data/
│   ├── fabrics.csv                # 22 curated fabrics with 14+ standardized criteria
│   └── garment_compatibility.csv  # Compatibility rules matrix across 13 garment types
│
├── models/
│   ├── __init__.py
│   ├── recommendation.py          # Multi-criteria recommendation coordinator
│   └── personalization.py         # Bayesian-inspired user preference & learning model
│
├── utils/
│   ├── __init__.py
│   ├── scoring.py                 # Dynamic weight normalization & MCDA scoring math
│   ├── explanations.py            # Rule-based & LLM explainable AI engine
│   └── validation.py              # Input validation and fallback handlers
│
├── database/
│   ├── __init__.py
│   └── feedback.py                # SQLite database persistence layer
│
├── assets/
│   └── images/                    # UI branding and visual assets
│
└── tests/
    ├── __init__.py
    └── test_recommendation.py     # Comprehensive automated test suite
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | Streamlit (Python) | Reactive, modern eco-themed web dashboard |
| **Data Processing** | Pandas, NumPy | Data cleaning, vector math, and matrix operations |
| **Machine Learning / MCDA** | Scikit-learn, SciPy | Multi-criteria utility scoring, normalization, similarity |
| **Data Visualization** | Plotly (Graph Objects & Express) | Interactive radar spider charts, bar charts, scatter plots |
| **Database Persistence** | SQLite 3 | Zero-configuration local database for user feedback & profiles |
| **Optional Generative AI** | Google GenAI SDK (Gemini API) | Optional narrative synthesis for textile rationale |
| **Testing** | Unittest / Pytest | Automated test coverage for scoring, ranking, and fallbacks |

---

## 📊 Fabric Dataset & Benchmark Metrics

The dataset (`data/fabrics.csv`) includes **22 realistic demonstration fabrics** across 6 major fiber families:
- **Natural Plant:** Organic Cotton, Conventional Cotton, Linen, Hemp, Ramie, Jute, Banana Fiber.
- **Regenerated Cellulose:** Tencel / Lyocell, Modal, Bamboo Viscose, Seaweed Fiber (SeaCell).
- **Natural Animal:** Virgin Wool, Recycled Wool, Mulberry Silk, Organic Peace Silk (Ahimsa).
- **Synthetics & Recycled:** Recycled Polyester, Conventional Polyester, Recycled Nylon, Conventional Nylon, Recycled Cotton, Recycled Denim.
- **Next-Gen Bio-Material:** Piñatex (Pineapple Leaf Fiber).

### Evaluated Criteria (Scores 0–100)
- `sustainability_score`: Holistic ecological rating
- `water_efficiency`: Freshwater conservation score (100 = minimal irrigation/processing water)
- `carbon_score`: Greenhouse gas mitigation score (100 = carbon-neutral or negative crop)
- `recyclability`: Ease of closed-loop mechanical or chemical circular recycling
- `biodegradability`: Rate and non-toxicity of natural soil breakdown
- `breathability`: Air permeability and thermal ventilation
- `comfort`: Skin feel, softness, and tactile drape
- `durability`: Tensile strength, abrasion tolerance, and tear resistance
- `stretchability`: Natural mechanical flexibility
- `moisture_management`: Moisture wicking and evaporation speed
- `thermal_insulation`: Heat retention capacity in colder environments
- `water_resistance`: Repellency against external precipitation
- `affordability_score`: Price-friendliness (100 = low budget, 20 = high luxury)
- `suitability_hot`, `suitability_hot_humid`, `suitability_moderate`, `suitability_cold`, `suitability_rainy`

*Disclaimer: Scores are normalized demonstration values derived from textile lifecycle assessment (LCA) literature and Higg Index benchmarks.*

---

## 🧮 Recommendation Algorithm (MCDA)

The core engine implements a **Multi-Criteria Decision Analysis (MCDA)** framework:

### 1. Garment Compatibility Filtering
Before scoring, fabrics are filtered through `data/garment_compatibility.csv`:
$$\mathcal{F}_{\text{candidate}} = \{ f \in \mathcal{F} \mid \text{Compatibility}(f, \text{garment}) = 1 \}$$

### 2. Dynamic Weight Vector Normalization
Baseline weights:
- Sustainability: $w_1 = 0.30$
- Climate Suitability: $w_2 = 0.20$
- Comfort & Breathability: $w_3 = 0.15$
- Durability: $w_4 = 0.15$
- Cost / Affordability: $w_5 = 0.10$
- Performance: $w_6 = 0.10$

Priority sliders apply multipliers $m_i \in [0.6, 1.6]$. All weights are strictly normalized:
$$w_i' = \frac{w_i \cdot m_i}{\sum_{j=1}^6 w_j \cdot m_j}, \quad \text{such that} \quad \sum_{i=1}^6 w_i' = 1.0$$

### 3. Multi-Attribute Score
$$\text{Score}_{\text{general}}(f) = \sum_{i=1}^6 w_i' \cdot S_i(f)$$

---

## 🧠 Personalization & Adaptive Learning

The system implements a continuous feedback loop:
1. When a user rates a fabric (1–5 stars) or indicates acceptance (*Would you choose this fabric?*), the interaction is logged in SQLite (`database/fabrics_feedback.db`).
2. The `PersonalizationEngine` computes a user affinity score $A(u, f) \in [0, 100]$:
   - High ratings (4–5 stars) and positive selection boost affinity.
   - Low ratings (1–2 stars) penalize affinity.
   - Frequency of interaction increases confidence.
3. Recommendations are computed using adaptive score blending:
   $$\text{Score}_{\text{final}} = (1 - \alpha) \times \text{Score}_{\text{general}} + \alpha \times \text{Score}_{\text{user preference}}$$
   - $\alpha = 0.0$ for new users ($< 2$ reviews) $\rightarrow$ Clean fallback to general recommendation.
   - $\alpha = 0.30$ when sufficient interaction history is recorded ($70\% / 30\%$ blending).

---

## 🔬 Multi-Dimensional Sustainability Methodology

Sustainability cannot be reduced to a single carbon score:
```
                                ┌────────────────────────────────┐
                                │   Composite Sustainability     │
                                └───────────────┬────────────────┘
                ┌───────────────────┬───────────┴───────────┬───────────────────┐
                ▼                   ▼                       ▼                   ▼
       💧 Water Efficiency   📉 Carbon Impact        🔄 Recyclability    🍂 Biodegradability
       (Rain-fed vs Irrig)   (Fossil vs Crop Sink)   (Closed-loop circ)  (Microplastic risk)
```

- **Linen & Hemp:** Score exceptionally high in water conservation (rain-fed) and carbon sequestration, but require break-in washing to achieve softness.
- **Recycled Polyester:** Saves post-consumer PET bottles from landfills and excels in sportswear durability, but sheds non-biodegradable microfibers.
- **Tencel / Lyocell:** Achieves high sustainability via a 99.5% closed-loop amine oxide solvent recovery process.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.9+ or Python 3.11+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/krish/smart-fabric-recommendation.git
cd smart-fabric-recommendation
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional: Configure Environment Variables
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY if desired
```

---

## 🚀 How to Run Locally

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## 🖼️ Walkthrough & Screenshots

### 1. 🎯 Recommendation Engine & Best Match Card
Select your garment (e.g. T-shirt), climate (e.g. Hot & Humid), and adjust priority sliders. The system renders the **🏆 BEST MATCH** card with score breakdown, positive highlights, care instructions, and trade-offs.

### 2. ⚖️ Interactive Multi-Fabric Radar Comparison
Select up to 4 fabrics to generate interactive Plotly polar radar charts comparing 8 simultaneous criteria.

### 3. 👤 Personalization Learning Dashboard
Inspect your user profile, view active learning weights ($\alpha$), review past ratings, and observe how your recommendations adapt over time.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite:
```bash
python -m unittest tests/test_recommendation.py -v
```

The test suite validates:
- ✅ Garment compatibility filtering (e.g., T-shirt vs Winter Jacket)
- ✅ Dynamic weight normalization ($\sum w_i = 1.0$)
- ✅ Priority shifts (sustainability and budget multipliers)
- ✅ Monotonic score ranking and value bounds $[0, 100]$
- ✅ Personalization learning and 70/30 blending formula
- ✅ Insufficient history graceful fallback
- ✅ Malformed input sanitization and fallback defaults
- ✅ Explainable AI and trade-off generation

---

## 🔮 Future Roadmap

- [ ] **LCA API Integration:** Live integration with EcoInvent or Higg MSI API for real-time supply chain data.
- [ ] **Image-Based Garment Upload:** Computer vision model to detect garment silhouette from user photos.
- [ ] **Certified Supplier Marketplace:** Direct connection to GOTS and OEKO-TEX certified textile mills.
- [ ] **Color Dyeing Sustainability:** Factoring in natural vs azo dyes and waterless supercritical $CO_2$ dyeing.

---

## 👥 Contributors & License

- **Project:** Smart Fabric Recommendation System
- **Track:** Track 1 — FUTURE FABRIC
- **License:** MIT Open Source License

Developed with ❤️ for a sustainable textile future.
