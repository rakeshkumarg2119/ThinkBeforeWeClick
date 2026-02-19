"""
Streamlit frontend for the URL Risk Analyzer.

Start with:
    streamlit run frontend/app.py
"""
import streamlit as st
import requests
import json
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000/api/v1"
ANALYZE_ENDPOINT = f"{API_BASE}/analyze"
HEALTH_ENDPOINT  = f"{API_BASE}/health"
REQUEST_TIMEOUT  = 15   # seconds — generous for redirect checking

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="URL Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── hero header ─────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #00e5ff 0%, #00bfa5 50%, #69f0ae 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #8888aa;
    font-size: 1.05rem;
    font-family: 'Space Mono', monospace;
}

/* ── score cards ─────────────────────────────────────────────── */
.score-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.score-card .label {
    font-size: 0.72rem;
    color: #5555aa;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.score-card .value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 0.2rem;
}

/* ── risk badge ──────────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.risk-LOW    { background: #003d2e; color: #00e676; border: 1px solid #00e676; }
.risk-MEDIUM { background: #3d2900; color: #ffab40; border: 1px solid #ffab40; }
.risk-HIGH   { background: #3d0000; color: #ff5252; border: 1px solid #ff5252; }

/* ── score bar ───────────────────────────────────────────────── */
.bar-wrap { background: #1a1a2e; border-radius: 8px; height: 8px; overflow: hidden; margin-top: 4px; }
.bar-fill  { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

/* ── info block ──────────────────────────────────────────────── */
.info-block {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-left: 4px solid #00bfa5;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color:white;
    margin-bottom: 1rem;
}
.warn-block {
    background: #1a1000;
    border: 1px solid #ffab40;
    color:white;
    border-left: 4px solid #ffab40;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
.anomaly-block {
    background: #2a0030;
    border: 1px solid #ce93d8;
    border-left: 4px solid #ce93d8;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ── input override ──────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
    background: #13131f !important;
    border: 1px solid #2a2a45 !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #00bfa5 !important;
    box-shadow: 0 0 0 2px #00bfa52a !important;
}

/* ── button override ─────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00bfa5, #00e5ff) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2.2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── divider ─────────────────────────────────────────────────── */
hr { border-color: #1e1e30 !important; }

/* ── cached tag ──────────────────────────────────────────────── */
.cached-tag {
    display: inline-block;
    background: #1a1a35;
    border: 1px solid #3333aa;
    color: #7777ff;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 99px;
    font-family: 'Space Mono', monospace;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def colour_for_score(score: int, max_score: int) -> str:
    """Return a CSS colour string based on ratio of score/max."""
    ratio = score / max_score if max_score else 0
    if ratio < 0.35:
        return "#00e676"
    if ratio < 0.65:
        return "#ffab40"
    return "#ff5252"


def score_bar(score: int, max_score: int) -> str:
    pct = round(score / max_score * 100) if max_score else 0
    colour = colour_for_score(score, max_score)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill" style="width:{pct}%;background:{colour}"></div>'
        f'</div>'
    )


def call_analyze_api(url: str) -> dict:
    """
    POST to /api/v1/analyze and return parsed JSON.
    Raises requests.HTTPError for 4xx/5xx responses.
    """
    payload = {"url": url}
    resp = requests.post(
        ANALYZE_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# UI — Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ URL Risk Analyzer</h1>
    <p>AI-powered threat detection · Phishing · Malware · Scams · Piracy · Gambling</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend health banner
# ---------------------------------------------------------------------------
backend_ok = check_backend_health()
if not backend_ok:
    st.error(
        "⚠️  Cannot reach the FastAPI backend at `http://localhost:8000`.  "
        "Start it with: `uvicorn backend.main:app --reload --port 8000`",
        icon="🔴",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_input:
    url_input = st.text_input(
        label="URL to analyze",
        placeholder="https://example.com",
        label_visibility="collapsed",
    )

with col_btn:
    analyze_clicked = st.button("Analyze", use_container_width=True, disabled=not backend_ok)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL before clicking Analyze.")
    else:
        with st.spinner("Analyzing URL — checking domain, keywords, redirects…"):
            try:
                start = time.time()
                data = call_analyze_api(url_input.strip())
                elapsed = time.time() - start

            except requests.exceptions.Timeout:
                st.error("⏱️  Request timed out. The URL may be unreachable or slow to respond.")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"❌  API error: {detail}")
                st.stop()
            except requests.exceptions.RequestException as exc:
                st.error(f"🔌  Connection error: {exc}")
                st.stop()

        # ── Result Header ──────────────────────────────────────────────────
        risk   = data.get("risk_level", "Unknown").upper()
        cached = data.get("cached", False)

        cached_html = '<span class="cached-tag">⚡ CACHED</span>' if cached else ""
        st.markdown(
            f'<h3 style="margin-top:1.5rem">Result {cached_html}</h3>',
            unsafe_allow_html=True,
        )

        # ── Domain + Risk Badge ────────────────────────────────────────────
        badge_col, domain_col, time_col = st.columns([2, 4, 2])
        with badge_col:
            st.markdown(
                f'<span class="risk-badge risk-{risk}">{risk} RISK</span>',
                unsafe_allow_html=True,
            )
        with domain_col:
            st.markdown(
                f'<span style="font-family:\'Space Mono\',monospace;color:#8888aa;font-size:0.85rem">'
                f'🌐 {data.get("domain","")}</span>',
                unsafe_allow_html=True,
            )
        with time_col:
            if not cached:
                st.markdown(
                    f'<span style="color:#5555aa;font-family:\'Space Mono\',monospace;'
                    f'font-size:0.78rem">⏱ {elapsed:.2f}s</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # ── Top KPI Cards ──────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        total   = data.get("total_score", 0)
        conf    = data.get("confidence_percent", 0)
        sev     = data.get("risk_severity_index", 0)
        rtype   = data.get("risk_type", "Unknown")

        total_colour = colour_for_score(total, 100)
        sev_colour   = colour_for_score(sev, 100)
        conf_colour  = "#00e5ff"

        with kpi1:
            st.markdown(f"""
            <div class="score-card">
                <div class="label">Total Score</div>
                <div class="value" style="color:{total_colour}">{total}<span style="font-size:1rem;color:#555">/100</span></div>
            </div>""", unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="score-card">
                <div class="label">Confidence</div>
                <div class="value" style="color:{conf_colour}">{conf:.0f}<span style="font-size:1rem;color:#555">%</span></div>
            </div>""", unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="score-card">
                <div class="label">Severity</div>
                <div class="value" style="color:{sev_colour}">{sev}<span style="font-size:1rem;color:#555">/100</span></div>
            </div>""", unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="score-card">
                <div class="label">Threat Type</div>
                <div class="value" style="font-size:1.1rem;color:#e8e8f0;padding-top:0.35rem">{rtype}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Component Score Breakdown ──────────────────────────────────────
        st.markdown("#### 📊 Score Breakdown")

        components = [
            ("Domain",   data.get("domain_score", 0),   25),
            ("URL",      data.get("url_score", 0),      25),
            ("Keywords", data.get("keyword_score", 0),  25),
            ("Security", data.get("security_score", 0), 15),
            ("Redirect", data.get("redirect_score", 0), 10),
        ]

        for label, score, max_s in components:
            c1, c2, c3 = st.columns([2, 6, 1])
            with c1:
                st.markdown(
                    f'<span style="font-size:0.85rem;color:#8888aa;font-family:\'Space Mono\',monospace">{label}</span>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(score_bar(score, max_s), unsafe_allow_html=True)
            with c3:
                colour = colour_for_score(score, max_s)
                st.markdown(
                    f'<span style="color:{colour};font-family:\'Space Mono\',monospace;font-size:0.85rem">'
                    f'{score}<span style="color:#444">/{max_s}</span></span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Why Risk Explanation ───────────────────────────────────────────
        why = data.get("why_risk", "")
        if why:
            st.markdown(
                f'<div class="info-block">💡 {why}</div>',
                unsafe_allow_html=True,
            )

        # ── Anomaly Flag ───────────────────────────────────────────────────
        if data.get("anomaly_detected"):
            st.markdown(
                '<div class="anomaly-block">🔮 <strong>Anomaly Detected</strong> — '
                'This URL has an unusual feature pattern not seen in normal traffic. '
                'Treat with extra caution.</div>',
                unsafe_allow_html=True,
            )

        # ── Gambling Warning ───────────────────────────────────────────────
        gambling_warning = data.get("gambling_warning")
        if gambling_warning:
            st.markdown(
                f'<div class="warn-block">{gambling_warning.strip()}</div>',
                unsafe_allow_html=True,
            )

        # ── Raw JSON (collapsible) ─────────────────────────────────────────
        with st.expander("🔍 View raw API response"):
            st.code(json.dumps(data, indent=2), language="json")

# ---------------------------------------------------------------------------
# Empty state hint
# ---------------------------------------------------------------------------
elif not analyze_clicked:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#3a3a5a">
        <div style="font-size:3rem">🔗</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.85rem;margin-top:0.5rem">
            Enter a URL above and click Analyze
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#333355;font-family:'Space Mono',monospace;font-size:0.72rem;padding-bottom:1rem">
    URL Risk Analyzer v3.0 · Powered by Scikit-learn · FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)
