"""
Crop Recommendation — Streamlit UI
Enter soil & weather details, get top crop recommendations from the FT-Transformer model.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project folder (same dir as app.py) so it works regardless of cwd
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import io
import csv
from datetime import datetime

import streamlit as st
import numpy as np

from inference import (
    load_checkpoint,
    predict_top_k,
    build_features_and_weather,
)
from crop_metadata import (
    get_metadata,
    get_current_season,
    npk_gaps,
    month_names,
    month_name,
    CROP_TYPES,
)

# Paths
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"

# Custom CSS: modern, high-contrast black text
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-main: #f0f2f5;
        --bg-card: #ffffff;
        --accent: #166534;
        --accent-hover: #15803d;
        --text: #000000;
        --text-body: #0f0f0f;
        --text-muted: #374151;
        --border: #e5e7eb;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        --radius: 14px;
        --radius-lg: 18px;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div {
        background: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    .main .block-container {
        max-width: 720px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    
    /* All text black / dark for visibility */
    p, span, label, .stMarkdown, .stNumberInput label, .stTextInput label, .stRadio label {
        color: var(--text-body) !important;
    }
    
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        letter-spacing: -0.03em;
        font-size: 1.85rem !important;
        margin-bottom: 0.35rem !important;
    }
    
    .subtitle {
        color: var(--text-muted) !important;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-weight: 600;
        color: var(--text) !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 1rem;
    }
    
    .section-box {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    
    .result-card:hover { box-shadow: var(--shadow); }
    
    .result-card.rank-1 { border-left: 4px solid #166534; }
    .result-card.rank-2 { border-left: 4px solid #0d9488; }
    .result-card.rank-3 { border-left: 4px solid #7c3aed; }
    
    .result-rank {
        font-weight: 700;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-right: 0.75rem;
        min-width: 1.5rem;
    }
    
    .result-name {
        font-weight: 600;
        color: var(--text) !important;
        font-size: 1.05rem;
    }
    
    .result-score {
        font-weight: 700;
        color: var(--text) !important;
        font-size: 1rem;
    }
    
    .result-bar {
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.4rem;
        max-width: 140px;
    }
    
    .result-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #166534, #22c55e);
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    
    button[kind="primary"] {
        background: var(--accent) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.6rem !important;
        border: none !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: var(--shadow-sm) !important;
        transition: background 0.2s, transform 0.1s !important;
    }
    
    button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }
    
    .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    
    .stNumberInput label, .stTextInput label, .stRadio label {
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    
    .success-msg {
        padding: 1rem 1.25rem;
        background: #ecfdf5;
        border-radius: var(--radius);
        border: 1px solid #a7f3d0;
        color: #065f46 !important;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .error-msg {
        padding: 1rem 1.25rem;
        background: #fef2f2;
        border-radius: var(--radius);
        border: 1px solid #fecaca;
        color: #991b1b !important;
        margin-bottom: 1rem;
    }
    
    .hero-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }
    
    .detail-row {
        font-size: 0.85rem;
        color: var(--text-body) !important;
        margin: 0.35rem 0;
    }
    .detail-label { font-weight: 600; color: var(--text-muted) !important; }
    .season-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .weather-summary {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        color: var(--text-body) !important;
        margin-bottom: 1rem;
    }
    .suitable { color: #166534 !important; font-weight: 600; }
    .not-suitable { color: #b91c1c !important; font-weight: 500; }
    
    /* Reference-inspired: gradient banner, hero card, detail grid, timing bar */
    .header-banner {
        background: linear-gradient(135deg, #15803d 0%, #166534 50%, #14532d 100%);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(22, 101, 52, 0.25);
    }
    .header-banner h1 { color: #ffffff !important; margin: 0 !important; font-size: 1.75rem !important; }
    .header-banner .subtitle { color: rgba(255,255,255,0.9) !important; margin: 0.35rem 0 0 0 !important; font-size: 0.9rem !important; }
    
    .weather-card-blue {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: #fff !important;
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3);
    }
    .weather-card-blue strong { color: #fff !important; }
    
    .hero-recommended-card {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        color: #fff !important;
        border-radius: var(--radius);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(21, 83, 45, 0.35);
    }
    .hero-recommended-card .rec-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9; margin-bottom: 0.25rem; }
    .hero-recommended-card .rec-name { font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0; }
    .hero-recommended-card .rec-meta { font-size: 0.9rem; opacity: 0.95; }
    
    .detail-card {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .detail-card .dc-label { font-size: 0.7rem; font-weight: 600; color: var(--accent) !important; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem; }
    .detail-card .dc-value { font-size: 0.95rem; font-weight: 600; color: var(--text) !important; }
    
    .npk-guide-box {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .npk-guide-box .npk-title { font-weight: 600; color: var(--text) !important; margin-bottom: 0.5rem; font-size: 0.9rem; }
    .npk-line { font-size: 0.85rem; color: var(--text-body) !important; margin: 0.35rem 0; }
    .npk-ideal { color: #166534 !important; font-weight: 500; }
    .npk-action-reduce { color: #b91c1c !important; font-weight: 500; }
    .npk-action-add { color: #1d4ed8 !important; font-weight: 500; }
    
    .timing-alert-bar {
        background: #ecfdf5;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: var(--text-body) !important;
    }
    .timing-alert-bar.good { background: #dcfce7; border-color: #22c55e; }
    .timing-alert-bar.warn { background: #fef9c3; border-color: #eab308; }
    
    .sowing-advisory-bar {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        color: var(--text-body) !important;
    }
    .sowing-advisory-bar .highlight { color: #166534; font-weight: 700; }
    
    .top-confidence-title { font-weight: 600; color: var(--text) !important; margin: 1rem 0 0.5rem 0; font-size: 0.95rem; }
    .confidence-row { display: flex; align-items: center; margin: 0.4rem 0; gap: 0.5rem; }
    .confidence-row .cr-name { min-width: 100px; font-weight: 500; color: var(--text) !important; font-size: 0.9rem; }
    .confidence-row .cr-bar-wrap { flex: 1; height: 20px; background: var(--border); border-radius: 4px; overflow: hidden; }
    .confidence-row .cr-bar { height: 100%; background: linear-gradient(90deg, #166534, #22c55e); border-radius: 4px; }
    .confidence-row .cr-pct { min-width: 42px; font-weight: 600; color: var(--text) !important; font-size: 0.85rem; text-align: right; }
    
    .stSlider [data-baseweb="slider"] { margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load model once and cache."""
    if not CHECKPOINT_PATH.is_file():
        return None, None, None, None
    device = __import__("torch").device(
        "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    model, scaler, label_encoder, dev = load_checkpoint(str(CHECKPOINT_PATH), device)
    return model, scaler, label_encoder, dev


def main():
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    today_month = datetime.now().month

    # --- Gradient header banner (reference-style) ---
    st.markdown(
        '<div class="header-banner">'
        '<h1>🌾 Crop Recommendation</h1>'
        '<p class="subtitle">AI-powered crop recommendation using soil analysis + real-time or manual weather.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Auto Season + optional Planned start month ---
    col_season, col_plan = st.columns([1, 1])
    with col_season:
        planned_month = st.selectbox(
            "📅 Planned start month (for season & timing)",
            options=list(range(1, 13)),
            format_func=lambda x: month_name(x),
            index=today_month - 1,
            key="planned_month",
            help="Use current month or when you plan to sow. Affects season suitability and advisory.",
        )
    current_season = get_current_season(planned_month)
    with col_plan:
        st.markdown(f'<p style="margin-top: 0.5rem;"><span class="detail-label">Current season:</span> <span class="season-badge">{current_season}</span></p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    model, scaler, label_encoder, device = get_model()
    if model is None:
        st.markdown(
            '<div class="error-msg">Model not found. Train first: <code>python train.py</code> then ensure <code>checkpoints/best_model.pt</code> exists.</div>',
            unsafe_allow_html=True,
        )
        return

    # --- Crop Type Filter ---
    with st.expander("🌾 Crop type filter (optional)", expanded=False):
        crop_type_filter = st.multiselect(
            "Show only these crop types",
            options=CROP_TYPES,
            default=[],
            help="Leave empty to see all recommended crops.",
        )

    # --- Soil section (sliders like reference UI) ---
    with st.expander("🌱 Soil parameters", expanded=True):
        st.caption("Typical ranges: N 0–140, P 0–145, K 0–205, pH 3.5–9.5 (ideal 5.5–7.5)")
        N = st.slider("Nitrogen (N) kg/ha", min_value=0.0, max_value=150.0, value=90.0, step=1.0, key="N")
        P = st.slider("Phosphorus (P) kg/ha", min_value=0.0, max_value=150.0, value=42.0, step=1.0, key="P")
        K = st.slider("Potassium (K) kg/ha", min_value=0.0, max_value=210.0, value=43.0, step=1.0, key="K")
        ph = st.slider("Soil pH", min_value=3.5, max_value=9.5, value=6.5, step=0.1, key="ph")

    # --- Weather section (card) ---
    with st.expander("🌤️ Weather", expanded=True):
        use_city = st.radio(
            "How do you want to provide weather?",
            ["Use my city (fetch live data)", "Enter values manually"],
            horizontal=True,
            key="weather_radio",
        )
        api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        temp = humidity = rainfall = None
        if use_city == "Use my city (fetch live data)":
            city = st.text_input("City name", placeholder="e.g. London, Mumbai, Chennai", key="city")
            if (not api_key or api_key.strip() == "your_openweathermap_api_key_here") and city:
                st.caption("⚠️ Add a **real** OpenWeatherMap API key to your `.env` file. Get one at openweathermap.org/api")
        else:
            city = None
            st.caption("Temperature 0–45°C typical; humidity 20–100%; rainfall 0–300 mm typical.")
            col1, col2, col3 = st.columns(3)
            with col1:
                temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5, format="%.1f", key="temp")
            with col2:
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0, format="%.1f", key="hum")
            with col3:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=350.0, value=100.0, step=1.0, format="%.1f", key="rain")

    # Empty state hint
    if st.session_state.last_result is None:
        st.info("👆 Enter values above and click **Get detailed crop recommendation** to see results.")

    if st.button("🌿 Get detailed crop recommendation", type="primary", use_container_width=True):
        if use_city == "Use my city (fetch live data)" and not city:
            st.error("Enter a city name or switch to manual weather.")
        elif use_city == "Use my city (fetch live data)" and (not api_key or api_key.strip() == "your_openweathermap_api_key_here"):
            st.error("Add a valid OpenWeatherMap API key to your `.env` file, or switch to **Enter values manually**.")
        else:
            with st.spinner("Getting recommendation..."):
                try:
                    features, weather_used = build_features_and_weather(
                        N=N, P=P, K=K, ph=ph,
                        temperature=temp, humidity=humidity, rainfall=rainfall,
                        api_key=api_key if city else None,
                        city=city if city else None,
                    )
                    k = 10 if crop_type_filter else 3
                    top_list = predict_top_k(model, scaler, label_encoder, features, device, k=k)
                    if crop_type_filter:
                        filtered = []
                        for name, score in top_list:
                            meta = get_metadata(name)
                            if meta["crop_type"] in crop_type_filter:
                                filtered.append((name, score))
                            if len(filtered) >= 3:
                                break
                        top_list = filtered if filtered else top_list[:3]
                    else:
                        top_list = top_list[:3]

                    # Weather summary — blue card (reference-style)
                    if weather_used:
                        w = weather_used
                        loc = w.get("city") or "Manual entry"
                        st.markdown(
                            f'<div class="weather-card-blue">'
                            f'<strong>📍 Weather used</strong> — {loc}<br>'
                            f'{w["temperature"]:.1f}°C &nbsp;|&nbsp; {w["humidity"]:.0f}% humidity &nbsp;|&nbsp; {w["rainfall"]:.1f} mm rainfall'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # First crop: hero "RECOMMENDED CROP" card + detail grid + NPK + timing
                    name1, score1 = top_list[0]
                    meta1 = get_metadata(name1)
                    pct1 = score1 * 100
                    suitable_seasons = meta1.get("suitable_seasons", [])
                    is_suitable = current_season in suitable_seasons
                    npk_msgs1 = npk_gaps(N, P, K, meta1)
                    n_lo, n_hi = meta1["ideal_N"]
                    p_lo, p_hi = meta1["ideal_P"]
                    k_lo, k_hi = meta1["ideal_K"]

                    st.markdown(
                        f'<div class="hero-recommended-card">'
                        f'<div class="rec-label">🏆 Recommended crop</div>'
                        f'<div class="rec-name">{name1.title()}</div>'
                        f'<div class="rec-meta">{meta1["crop_type"]} &nbsp;|&nbsp; {pct1:.1f}% confidence</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # 6 detail cards grid (reference-style)
                    cards = [
                        ("Type", meta1["crop_type"], "🌾"),
                        ("Sow in", month_names(meta1["sowing_months"]), "🌱"),
                        ("Duration", f"~{meta1['duration_days']} days", "⏱️"),
                        ("Season", current_season, "📅"),
                        ("Harvest in", month_names(meta1["harvest_months"]), "🫘"),
                        ("Water source", meta1["water_source"], "💧"),
                    ]
                    row1 = st.columns(3)
                    for idx in range(3):
                        with row1[idx]:
                            label, value, icon = cards[idx]
                            st.markdown(
                                f'<div class="detail-card"><div class="dc-label">{icon} {label}</div><div class="dc-value">{value}</div></div>',
                                unsafe_allow_html=True,
                            )
                    row2 = st.columns(3)
                    for idx in range(3, 6):
                        with row2[idx - 3]:
                            label, value, icon = cards[idx]
                            st.markdown(
                                f'<div class="detail-card"><div class="dc-label">{icon} {label}</div><div class="dc-value">{value}</div></div>',
                                unsafe_allow_html=True,
                            )

                    # NPK Fertilizer Guide (ideal + action)
                    def npk_line(lo, hi, msg):
                        ideal = f'Ideal: {lo:.0f}–{hi:.0f} kg/ha'
                        if "reduce" in msg.lower():
                            return f'<span class="npk-ideal">{ideal}</span> → <span class="npk-action-reduce">{msg}</span>'
                        if "add" in msg.lower():
                            return f'<span class="npk-ideal">{ideal}</span> → <span class="npk-action-add">{msg}</span>'
                        return f'<span class="npk-ideal">{ideal}</span> → <span class="npk-ideal">{msg}</span>'
                    st.markdown('<p class="npk-title">🧪 NPK fertilizer guide</p>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="npk-guide-box">'
                        f'<div class="npk-line"><strong>N:</strong> {npk_line(n_lo, n_hi, npk_msgs1[0])}</div>'
                        f'<div class="npk-line"><strong>P:</strong> {npk_line(p_lo, p_hi, npk_msgs1[1])}</div>'
                        f'<div class="npk-line"><strong>K:</strong> {npk_line(k_lo, k_hi, npk_msgs1[2])}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # Timing alert bar
                    if is_suitable:
                        st.markdown(
                            f'<div class="timing-alert-bar good">✅ Great timing! — {month_name(planned_month)} is ideal for growing this crop.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="timing-alert-bar warn">⚠️ Consider sowing in {month_names(meta1["sowing_months"])} for best results.</div>',
                            unsafe_allow_html=True,
                        )

                    # Sowing advisory bar
                    st.markdown(
                        f'<div class="sowing-advisory-bar">'
                        f'📅 If you sow in <span class="highlight">{month_name(planned_month)}</span>, estimated harvest: <span class="highlight">{month_names(meta1["harvest_months"])}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Top 3 Crop Confidence (bar list)
                    st.markdown('<p class="top-confidence-title">📊 Top 3 crop confidence</p>', unsafe_allow_html=True)
                    for name, score in top_list:
                        pct = score * 100
                        st.markdown(
                            f'<div class="confidence-row">'
                            f'<span class="cr-name">{name.title()}</span>'
                            f'<div class="cr-bar-wrap"><div class="cr-bar" style="width:{pct:.0f}%"></div></div>'
                            f'<span class="cr-pct">{pct:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Details for 2nd and 3rd crop (in expanders so nothing is lost)
                    for i, (name, score) in enumerate(top_list[1:], 2):
                        meta = get_metadata(name)
                        npk_msgs = npk_gaps(N, P, K, meta)
                        is_suit = current_season in meta.get("suitable_seasons", [])
                        suit_txt = "Suitable for current season" if is_suit else "Not ideal for current season"
                        with st.expander(f"Details for #{i}: {name.title()} ({score*100:.1f}%)", expanded=False):
                            st.markdown(f'**Type:** {meta["crop_type"]} • **Water:** ~{meta["water_requirement_mm"]} mm ({meta["water_category"]}) • **Source:** {meta["water_source"]}')
                            st.markdown(f'**Sowing:** {month_names(meta["sowing_months"])} → **Harvest:** {month_names(meta["harvest_months"])} • **Duration:** ~{meta["duration_days"]} days')
                            st.markdown(f'**NPK:** {" • ".join(npk_msgs)} • **Season:** {suit_txt}')

                    # Store for history and export
                    st.session_state.last_result = {
                        "top_list": top_list,
                        "N": N, "P": P, "K": K, "ph": ph,
                        "weather_used": weather_used,
                        "season": current_season,
                    }

                except Exception as e:
                    err = str(e)
                    if "401" in err and "Unauthorized" in err:
                        err = "Invalid or missing OpenWeatherMap API key. Use **Enter values manually** or add a valid key to `.env`."
                    st.markdown(f'<div class="error-msg">{err}</div>', unsafe_allow_html=True)

    # --- Export ---
    if st.session_state.last_result:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Rank", "Crop", "Score", "Type", "Water (mm)", "Duration (days)", "Sowing", "Harvest", "NPK note"])
        for i, (name, score) in enumerate(st.session_state.last_result["top_list"], 1):
            meta = get_metadata(name)
            w.writerow([
                i, name, f"{score:.2%}", meta["crop_type"], meta["water_requirement_mm"], meta["duration_days"],
                month_names(meta["sowing_months"]), month_names(meta["harvest_months"]),
                " • ".join(npk_gaps(st.session_state.last_result["N"], st.session_state.last_result["P"], st.session_state.last_result["K"], meta)),
            ])
        st.download_button("📥 Download results as CSV", data=buf.getvalue(), file_name="crop_recommendation.csv", mime="text/csv", use_container_width=True)

    # --- Previous recommendation (history) ---
    if st.session_state.last_result:
        with st.expander("📋 View last recommendation summary", expanded=False):
            r = st.session_state.last_result
            st.write("**Top crops:**", ", ".join(n for n, _ in r["top_list"]))
            st.write("**Season:**", r["season"])
            if r.get("weather_used"):
                w = r["weather_used"]
                st.write("**Weather:**", f"{w.get('city', 'Manual')} — {w['temperature']:.1f}°C, {w['humidity']:.0f}%, {w['rainfall']:.1f} mm")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #6b7280 !important; font-size: 0.8rem;">FT-Transformer • Kaggle Crop Recommendation Dataset • Soil (N, P, K, pH) + Weather</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
