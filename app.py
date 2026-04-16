# ====================================================================
# Insurance Charge Predictor  —  Minimalist & Elegant Dashboard
# Dataset : insurance.csv  |  Model : RandomForestRegressor (joblib)
# ====================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

# --------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------
st.set_page_config(
    page_title="InsurePredict · Épure",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------
# DESIGN SYSTEM — COULEURS CLAIRES & ÉPURÉES
# --------------------------------------------------------------------
COLORS = {
    "bg_page": "#F9F6F0",
    "bg_card": "#FFFFFF",
    "text_primary": "#2C2C2C",
    "text_secondary": "#6B6B6B",
    "accent": "#8BA989",
    "accent_light": "#D9E0D5",
    "accent_warm": "#D4BBA5",
    "divider": "#E6E2D8",
    "positive": "#8BA989",
    "negative": "#C28B8B",
    "chart_blue": "#A3B8C6",
    "chart_violet": "#B8A9C4",
}

st.markdown(f"""
<style>
    /* Reset & base */
    .stApp, .stApp > header, .stApp > div {{
        background-color: {COLORS["bg_page"]};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLORS["bg_card"]};
        border-right: 1px solid {COLORS["divider"]};
    }}
    /* Typographie */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {COLORS["text_primary"]};
    }}
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{
        font-weight: 500;
        letter-spacing: -0.02em;
        color: {COLORS["text_primary"]};
    }}
    /* Cartes & conteneurs */
    .card {{
        background: {COLORS["bg_card"]};
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02), 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid {COLORS["divider"]};
        transition: all 0.2s ease;
    }}
    .card-accent {{
        background: {COLORS["bg_card"]};
        border-radius: 20px;
        padding: 1.75rem;
        border-left: 4px solid {COLORS["accent"]};
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }}
    /* KPI principal */
    .kpi-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS["text_secondary"]};
        margin-bottom: 0.25rem;
    }}
    .kpi-value {{
        font-size: 3.5rem;
        font-weight: 500;
        line-height: 1.1;
        color: {COLORS["text_primary"]};
        margin: 0.25rem 0 0.25rem;
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: {COLORS["text_secondary"]};
        margin-top: 0.5rem;
    }}
    /* Badges de risque */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-top: 0.75rem;
    }}
    .badge-low {{
        background: #EFF5EF;
        color: #5A7D5A;
        border: 1px solid #D4E0D2;
    }}
    .badge-medium {{
        background: #F7F0E9;
        color: #BC8F6B;
        border: 1px solid #EBDBCE;
    }}
    .badge-high {{
        background: #F0E9F2;
        color: #8F7A9E;
        border: 1px solid #DFD4E4;
    }}
    .insight {{
        background: {COLORS["bg_page"]};
        border-radius: 14px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: {COLORS["text_secondary"]};
        margin-top: 1rem;
        line-height: 1.5;
        border: 1px solid {COLORS["divider"]};
    }}
    /* Métriques miniatures */
    .mini-metric {{
        background: {COLORS["bg_page"]};
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid {COLORS["divider"]};
    }}
    .mini-metric-val {{
        font-size: 1.6rem;
        font-weight: 500;
        color: {COLORS["text_primary"]};
        line-height: 1.2;
    }}
    .mini-metric-lbl {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLORS["text_secondary"]};
        margin-top: 0.25rem;
    }}
    /* Sidebar styles */
    .sidebar-section {{
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["accent"]};
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid {COLORS["divider"]};
    }}
    hr.divider {{
        margin: 1rem 0;
        border: none;
        border-top: 1px solid {COLORS["divider"]};
    }}
    /* Boutons */
    .stButton > button {{
        background-color: {COLORS["text_primary"]};
        color: white;
        border: none;
        border-radius: 40px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: 0.1s linear;
    }}
    .stButton > button:hover {{
        background-color: {COLORS["accent"]};
        color: white;
    }}
    /* Inputs & sliders */
    .stSlider > div > div {{
        background-color: {COLORS["accent_light"]};
    }}
    .stRadio > div, .stSelectbox > div {{
        color: {COLORS["text_primary"]};
    }}
    /* Masquer éléments par défaut */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# HELPERS & CONFIG
# --------------------------------------------------------------------
def light_figure(w=6, h=3.6):
    """Créer une figure Matplotlib aux couleurs claires."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(COLORS["bg_card"])
    ax.set_facecolor(COLORS["bg_card"])
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["divider"])
    ax.xaxis.label.set_color(COLORS["text_secondary"])
    ax.yaxis.label.set_color(COLORS["text_secondary"])
    ax.title.set_color(COLORS["text_primary"])
    return fig, ax

def risk_badge(amount):
    """Retourne la classe CSS, le texte et le message d'insight selon le montant."""
    if amount < 6000:
        return "badge-low", "🟢 Risque faible", "Profil favorable — charges inférieures à la moyenne."
    elif amount < 16000:
        return "badge-medium", "🟠 Risque modéré", "Charges dans la moyenne — certains facteurs à surveiller."
    else:
        return "badge-high", "🔴 Risque élevé", "Charges significatives — tabagisme ou IMC élevé possible."

@st.cache_resource(show_spinner=False)
def load_model():
    """Charger le modèle pré-entraîné."""
    return joblib.load("rf_model.joblib")

def build_input(age, bmi, children, sex, smoker, region):
    """Construire le DataFrame d'entrée pour la prédiction."""
    return pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }])

def predict(model, age, bmi, children, sex, smoker, region):
    """Effectuer la prédiction."""
    return model.predict(build_input(age, bmi, children, sex, smoker, region))[0]

# --------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------
# CHARGEMENT MODÈLE
# --------------------------------------------------------------------
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ `model.pkl` introuvable. Placez le fichier dans le même dossier que `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# --------------------------------------------------------------------
# SIDEBAR — FORMULAIRE UTILISATEUR
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="font-size:1.8rem; font-weight:500; margin-bottom:0;">InsurePredict</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem; color:#6B6B6B;">estimation de charges santé</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">👤 Profil</div>', unsafe_allow_html=True)
    age = st.slider("Âge", 18, 100, 32, key="age")
    sex = st.radio("Sexe", ["male", "female"], horizontal=True, format_func=lambda x: "♂ Homme" if x == "male" else "♀ Femme", key="sex")
    children = st.selectbox("Enfants à charge", list(range(0, 11)), key="children")

    st.markdown('<div class="sidebar-section">⚕️ Santé</div>', unsafe_allow_html=True)
    bmi = st.number_input("IMC (Body Mass Index)", 10.0, 60.0, 26.5, step=0.5, help="Poids (kg) / Taille² (m)", key="bmi")
    smoker = st.radio("Tabagisme", ["no", "yes"], horizontal=True, format_func=lambda x: "✅ Non-fumeur" if x == "no" else "🚬 Fumeur", key="smoker")

    st.markdown('<div class="sidebar-section">📍 Région</div>', unsafe_allow_html=True)
    region = st.selectbox(
        "Région (US)",
        ["northeast", "northwest", "southeast", "southwest"],
        format_func=lambda x: {
            "northeast": "Nord-Est",
            "northwest": "Nord-Ouest",
            "southeast": "Sud-Est",
            "southwest": "Sud-Ouest",
        }[x],
        key="region"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    predict_btn = st.button("Estimer mes charges", use_container_width=True, type="primary")
    st.caption(f"Modèle : Random Forest · {datetime.now().strftime('%d/%m/%Y')}")

# --------------------------------------------------------------------
# PAGE PRINCIPALE — TITRE
# --------------------------------------------------------------------
st.markdown('<div style="margin-top:-1rem;"></div>', unsafe_allow_html=True)
st.markdown('<h1 style="font-weight:500; letter-spacing:-0.02rem;">Estimation des frais de santé</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#6B6B6B; margin-top:-0.5rem;">Prédiction personnalisée basée sur votre profil</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --------------------------------------------------------------------
# PRÉDICTION ET AFFICHAGE DES RÉSULTATS
# --------------------------------------------------------------------
if predict_btn:
    with st.spinner("Calcul en cours..."):
        pred = predict(model, age, bmi, children, sex, smoker, region)
        pred_smoker = predict(model, age, bmi, children, sex, "yes", region)
        pred_nosmoker = predict(model, age, bmi, children, sex, "no", region)
        pred_younger = predict(model, max(18, age - 5), max(18.5, bmi - 3), children, sex, smoker, region)

    badge_cls, badge_txt, badge_insight = risk_badge(pred)

    # Ajout à l'historique
    st.session_state.history.append({
        "Heure": datetime.now().strftime("%H:%M:%S"),
        "Âge": age,
        "Sexe": sex,
        "BMI": bmi,
        "Enfants": children,
        "Fumeur": smoker,
        "Région": region,
        "Prédiction (USD)": round(pred, 2),
    })

    # --- LIGNE 1 : KPI principal + Importance features ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f"""
        <div class="card-accent">
            <div class="kpi-label">Charges annuelles estimées</div>
            <div class="kpi-value">${pred:,.0f}</div>
            <div class="kpi-sub">USD · assurance santé individuelle</div>
            <span class="badge {badge_cls}">{badge_txt}</span>
            <div class="insight">{badge_insight}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Poids des facteurs**")
        features = list(build_input(age, bmi, children, sex, smoker, region).columns)
        importances = model.feature_importances_
        fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")

        fig, ax = light_figure(5.5, 3.8)
        bars = ax.barh(fi["Feature"], fi["Importance"], color=COLORS["accent_light"], edgecolor=COLORS["accent"], height=0.55)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Importance relative", fontsize=8)
        ax.bar_label(bars, fmt=lambda x: f"{x:.1%}", padding=4, color=COLORS["text_secondary"], fontsize=7)
        ax.set_xlim(0, fi["Importance"].max() * 1.2)
        plt.tight_layout(pad=1)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- LIGNE 2 : Mini métriques comparatives ---
    st.markdown("**Analyse de scénarios**")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metrics = [
        ("Profil actuel", f"${pred:,.0f}", "référence"),
        ("Si fumeur", f"${pred_smoker:,.0f}", f"+${pred_smoker - pred:,.0f}"),
        ("Si non-fumeur", f"${pred_nosmoker:,.0f}", f"-${pred - pred_nosmoker:,.0f}" if smoker == "yes" else "identique"),
        ("-5 ans / -3 IMC", f"${pred_younger:,.0f}", f"-${pred - pred_younger:,.0f}"),
    ]
    for col, (label, value, subtitle) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
        col.markdown(f"""
        <div class="mini-metric">
            <div class="mini-metric-lbl">{label}</div>
            <div class="mini-metric-val">{value}</div>
            <div style="font-size:0.7rem; color:#8C8C8C; margin-top:0.25rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- LIGNE 3 : Graphique comparatif + Sensibilité BMI ---
    col3, col4 = st.columns([1, 1], gap="large")

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Comparaison des profils**")
        scenarios = ["Actuel", "Fumeur", "Non-fumeur", "Âge/IMC réduits"]
        values = [pred, pred_smoker, pred_nosmoker, pred_younger]
        fig2, ax2 = light_figure(5, 3.4)
        bars = ax2.bar(scenarios, values, color=COLORS["accent_light"], width=0.5, edgecolor=COLORS["accent"])
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.bar_label(bars, fmt=lambda x: f"${x:,.0f}", padding=5, color=COLORS["text_secondary"], fontsize=8)
        ax2.set_ylabel("Charges (USD)", fontsize=8)
        plt.tight_layout(pad=1)
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Sensibilité à l'IMC**")
        bmi_range = np.arange(15, 55, 1)
        bmi_preds = [predict(model, age, b, children, sex, smoker, region) for b in bmi_range]

        fig3, ax3 = light_figure(5, 3.4)
        ax3.plot(bmi_range, bmi_preds, color=COLORS["chart_blue"], linewidth=2)
        ax3.axvline(bmi, color=COLORS["accent_warm"], linestyle="--", linewidth=1.2, label=f"Votre IMC = {bmi}")
        ax3.fill_between(bmi_range, bmi_preds, alpha=0.1, color=COLORS["chart_blue"])
        ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax3.set_xlabel("IMC", fontsize=8)
        ax3.set_ylabel("Charges (USD)", fontsize=8)
        ax3.legend(fontsize=8, frameon=False)
        plt.tight_layout(pad=1)
        st.pyplot(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- LIGNE 4 : Évolution avec l'âge (fumeur vs non-fumeur) ---
    st.markdown("**Impact de l'âge et du tabagisme**")
    age_range = np.arange(18, 70, 1)
    age_preds_smoker = [predict(model, a, bmi, children, sex, "yes", region) for a in age_range]
    age_preds_nonsmoker = [predict(model, a, bmi, children, sex, "no", region) for a in age_range]

    fig4, ax4 = light_figure(11, 3.6)
    ax4.plot(age_range, age_preds_smoker, color=COLORS["negative"], linewidth=2, label="Fumeur")
    ax4.plot(age_range, age_preds_nonsmoker, color=COLORS["positive"], linewidth=2, label="Non-fumeur")
    ax4.fill_between(age_range, age_preds_nonsmoker, age_preds_smoker, alpha=0.1, color=COLORS["negative"])
    ax4.axvline(age, color=COLORS["accent_warm"], linestyle="--", linewidth=1.2, label=f"Votre âge ({age})")
    ax4.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.set_xlabel("Âge", fontsize=8)
    ax4.set_ylabel("Charges annuelles (USD)", fontsize=8)
    ax4.legend(fontsize=8, frameon=False)
    plt.tight_layout(pad=1)
    st.pyplot(fig4, use_container_width=True)

# --------------------------------------------------------------------
# HISTORIQUE ET EXPORT
# --------------------------------------------------------------------
if st.session_state.history:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**📋 Historique des estimations**")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        hist_df.style.format({"Prédiction (USD)": "${:,.2f}", "BMI": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
    col_dl1, col_dl2, _ = st.columns([1, 1, 3])
    with col_dl1:
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv, "historique_charges.csv", "text/csv", use_container_width=True)
    with col_dl2:
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# --------------------------------------------------------------------
# ÉTAT INITIAL (aucune prédiction)
# --------------------------------------------------------------------
if not predict_btn and not st.session_state.history:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <div style="font-size:3rem;">⚖️</div>
        <h3 style="font-weight:400; color:#6B6B6B;">Complétez votre profil</h3>
        <p style="color:#8C8C8C; max-width:400px; margin:0.5rem auto;">
            Renseignez les informations dans la barre latérale, puis cliquez sur <strong>Estimer mes charges</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)