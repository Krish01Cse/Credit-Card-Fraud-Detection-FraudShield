"""
utils/ui.py
FraudShield shared design system.
Exact visual replica of the reference SaaS dashboard (InsightHub aesthetic):
- Soft off-white backdrop (#eef1f5) with floating rounded white canvas (border-radius: 22px)
- Vibrant Electric Royal Blue active elements (#2563eb)
- High-contrast typography with Plus Jakarta Sans & Inter
- 100% interactive, reliable sidebar navigation with chevrons and status dots
- Reusable components for top header, alert banner, metric cards, and task items
"""

import streamlit as st
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
# EXACT INSIGHTHUB LIGHT THEME CSS
# ──────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ── */
:root {
    --bg-page:        #eef1f5;   /* soft off-white backdrop */
    --bg-canvas:      #ffffff;   /* crisp white floating canvas */
    --bg-card:        #ffffff;   /* card background */
    --bg-hover:       #f8fafc;   /* soft hover state */
    --border:         #e2e8f0;   /* crisp light border */
    --border-subtle:  #edf2f7;
    --primary:        #2563eb;   /* electric royal blue active accent */
    --primary-hover:  #1d4ed8;
    --primary-light:  #eff6ff;   /* soft blue tint */
    --primary-glow:   rgba(37,99,235,0.25);
    --success:        #10b981;   /* emerald green */
    --success-light:  #ecfdf5;
    --danger:         #ef4444;   /* coral red */
    --danger-light:   #fef2f2;
    --warning:        #f59e0b;   /* warm amber */
    --warning-light:  #fffbeb;
    --text-primary:   #0f172a;   /* charcoal title text */
    --text-secondary: #475569;   /* slate body text */
    --text-muted:     #94a3b8;   /* light muted labels */
    --radius-xl:      18px;
    --radius-lg:      12px;
    --radius-md:      8px;
    --radius-sm:      6px;
    --shadow-sm:      0 1px 3px rgba(15,23,42,0.03);
    --shadow-md:      0 4px 18px -2px rgba(15,23,42,0.05), 0 1px 4px rgba(15,23,42,0.02);
    --shadow-lg:      0 12px 36px -4px rgba(15,23,42,0.07), 0 2px 8px rgba(15,23,42,0.03);
}

/* ── Global Typography & Defaults ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit standard header & footer ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Page Backdrop ── */
.stApp {
    background-color: var(--bg-page) !important;
}

/* ── Main Floating Rounded Canvas (InsightHub style) ── */
.main, .main .block-container {
    background-color: var(--bg-canvas) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 1.6rem 2.2rem 2.8rem !important;
    margin-top: 1.1rem !important;
    margin-bottom: 2rem !important;
    max-width: 1340px !important;
    box-shadow: var(--shadow-lg) !important;
}

/* ── Clean White Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border) !important;
    min-width: 255px !important;
    box-shadow: 2px 0 12px rgba(15,23,42,0.02) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Sidebar Navigation Buttons ── */
[data-testid="stSidebar"] button {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    padding: 0.62rem 0.9rem !important;
    margin-bottom: 0.35rem !important;
    border-radius: var(--radius-lg) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    transition: all 0.15s ease-in-out !important;
    text-align: left !important;
}

[data-testid="stSidebar"] button:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}

[data-testid="stSidebar"] button:focus {
    box-shadow: none !important;
}

/* ── Active Pill ── */
.nav-active-pill {
    background: var(--primary) !important;
    color: #ffffff !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.62rem 0.9rem !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.35rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    box-shadow: 0 4px 14px var(--primary-glow) !important;
    cursor: default !important;
}

/* ── Content Cards ── */
.fs-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.25rem 1.45rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s ease;
}

.fs-card:hover {
    box-shadow: var(--shadow-md);
}

.fs-card h3 {
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* ── Top Header Bar ── */
.hub-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.25rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #f1f5f9;
}

.hub-user-greeting {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.hub-avatar-img {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: #0f172a;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(15,23,42,0.12);
    flex-shrink: 0;
}

.hub-greeting-text h2 {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
}

.hub-greeting-text p {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0.15rem 0 0;
    font-weight: 500;
}

.hub-actions {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.hub-search-box {
    display: flex;
    align-items: center;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 0.42rem 1rem;
    width: 250px;
    box-shadow: var(--shadow-sm);
    color: var(--text-secondary);
    font-size: 0.82rem;
}

.hub-icon-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #f8fafc;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-size: 0.9rem;
    box-shadow: var(--shadow-sm);
    cursor: pointer;
    position: relative;
}

.hub-badge-dot {
    position: absolute;
    top: 6px;
    right: 7px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--danger);
    border: 1.5px solid #ffffff;
}

/* ── Alert Notification Banner ("Dear Manager") ── */
.hub-alert-banner {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.95rem 1.35rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
}

.hub-alert-content {
    display: flex;
    align-items: center;
    gap: 0.95rem;
}

.hub-alert-icon-wrap {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #0f172a;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.hub-alert-text h4 {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.15rem;
}

.hub-alert-text p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0;
}

.hub-highlight-tag {
    color: #ef4444;
    font-weight: 600;
}

/* ── 4 Top Stat Metric Cards (Exact InsightHub Layout) ── */
.hub-metric-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}

.hub-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.hub-metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.hub-metric-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
}

.hub-metric-corner {
    color: #cbd5e1;
    font-size: 0.85rem;
}

.hub-metric-bottom {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.hub-metric-icon-box {
    width: 34px;
    height: 34px;
    border-radius: var(--radius-md);
    background: #f8fafc;
    border: 1px solid #edf2f7;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
}

.hub-metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    letter-spacing: -0.02em;
}

/* ── On Going Tasks (Left Card) ── */
.hub-task-item {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.15s, border-color 0.15s;
}

.hub-task-item:hover {
    background: #f8fafc;
    border-color: #cbd5e1;
}

.hub-task-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.hub-task-icon-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #0f172a;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    font-weight: 700;
    flex-shrink: 0;
}

.hub-task-info h5 {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.15rem;
}

.hub-task-info p {
    font-size: 0.74rem;
    color: var(--text-secondary);
    margin: 0;
}

.hub-avatar-stack {
    display: flex;
    align-items: center;
}

.hub-mini-avatar {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1.5px solid #ffffff;
    margin-left: -6px;
    font-size: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}

/* ── Badges ── */
.hub-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

.hub-badge-blue {
    background: var(--primary-light);
    color: var(--primary);
    border: 1px solid rgba(37,99,235,0.2);
}

.hub-badge-green {
    background: var(--success-light);
    color: var(--success);
    border: 1px solid rgba(16,185,129,0.2);
}

.hub-badge-amber {
    background: var(--warning-light);
    color: var(--warning);
    border: 1px solid rgba(245,158,11,0.2);
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-lg) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.52rem 1.1rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 12px var(--primary-glow) !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--primary-hover) !important;
}

/* Form Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0.5rem !important;
}

[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 0.95rem !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Footer */
.fs-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    border-top: 1px solid #f1f5f9;
    padding: 1.4rem 0 0.4rem;
    margin-top: 2.5rem;
}
</style>
"""


def inject_css():
    """Inject the refined light design system CSS."""
    st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 100% WORKING SIDEBAR NAVIGATION (Matching Reference Screenshot)
# ──────────────────────────────────────────────────────────────────

SIDEBAR_NAV_ITEMS = [
    ("Home",      "Dashboard",            "⏹"),
    ("Dataset",   "Dataset & Prep",       "📂"),
    ("Train",     "Train & Models",       "⚙️"),
    ("Predict",   "Fraud Prediction",     "🔍"),
    ("About",     "Results & About",      "📋"),
]


def render_sidebar():
    """
    Renders the exact InsightHub sidebar:
    - Logo with dark circle + InsightHub / FraudShield title
    - Section header 'Dashboard'
    - Active pill in electric blue with white text
    - Inactive items with right chevron '›'
    - 'Projects' expandable tree with 3 colored dots (Active, Done, On Hold)
    - Bottom profile card
    """
    current_page = st.session_state.get("page", "Home")

    with st.sidebar:
        # Brand Logo (Dark circle + InsightHub / FraudShield)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.7rem;padding:0.5rem 0.2rem 1.2rem;">
            <div style="width:36px;height:36px;border-radius:50%;background:#0f172a;
                        color:#ffffff;display:flex;align-items:center;justify-content:center;
                        font-weight:800;font-size:1.1rem;box-shadow:0 2px 6px rgba(0,0,0,0.15);">
                G
            </div>
            <div>
                <div style="font-size:1.05rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em;">
                    InsightHub
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;
                    text-transform:uppercase;letter-spacing:0.08em;margin:0.2rem 0 0.55rem 0.35rem;">
            Dashboard
        </div>
        """, unsafe_allow_html=True)

        # Main Nav Items
        for key, title, icon in SIDEBAR_NAV_ITEMS:
            is_active = (current_page == key)
            if is_active:
                st.markdown(f"""
                <div class="nav-active-pill">
                    <span style="display:flex;align-items:center;gap:0.55rem;">
                        <span style="font-size:0.95rem;">{icon}</span> {title}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                btn_label = f"{icon}  {title}                     ›"
                if st.button(btn_label, key=f"nav_to_{key}", use_container_width=True):
                    st.session_state["page"] = key
                    st.rerun()

        # "Projects" Tree Section (Exact match to screenshot)
        st.markdown("""
        <div style="margin:0.75rem 0 0.5rem;padding:0.65rem 0.85rem;background:#f8fafc;
                    border:1px solid #edf2f7;border-radius:12px;">
            <div style="font-size:0.72rem;font-weight:700;color:#64748b;margin-bottom:0.45rem;
                        display:flex;justify-content:space-between;align-items:center;">
                <span>PROJECTS</span>
                <span style="font-size:0.7rem;color:#94a3b8;">▾</span>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.35rem;font-size:0.76rem;">
                <div style="display:flex;align-items:center;gap:0.45rem;color:#334155;">
                    <span style="width:7px;height:7px;border-radius:50%;background:#2563eb;"></span>
                    <span>Active Project</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.45rem;color:#334155;">
                    <span style="width:7px;height:7px;border-radius:50%;background:#10b981;"></span>
                    <span>Project Done</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.45rem;color:#334155;">
                    <span style="width:7px;height:7px;border-radius:50%;background:#ef4444;"></span>
                    <span>Project On Hold</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Profile Box at Bottom
        st.markdown("""
        <div style="margin-top:1.8rem;padding:0.7rem 0.85rem;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:12px;display:flex;align-items:center;gap:0.7rem;">
            <div style="width:32px;height:32px;border-radius:50%;background:#0f172a;color:#ffffff;
                        display:flex;align-items:center;justify-content:center;font-size:0.78rem;font-weight:700;">
                MK
            </div>
            <div>
                <div style="font-size:0.78rem;font-weight:700;color:#0f172a;">Markus Keller</div>
                <div style="font-size:0.68rem;color:#64748b;">Risk &amp; Fraud Lead</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return st.session_state.get("page", "Home")


# ──────────────────────────────────────────────────────────────────
# REUSABLE COMPONENTS
# ──────────────────────────────────────────────────────────────────

def render_top_header(title="Dashboard", subtitle=None):
    """Top bar greeting: Hey, Markus, live date, search bar & actions."""
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    sub = subtitle if subtitle else date_str

    st.markdown(f"""
    <div class="hub-header">
        <div class="hub-user-greeting">
            <div class="hub-avatar-img">👨‍💼</div>
            <div class="hub-greeting-text">
                <h2>Hey, Markus</h2>
                <p>{sub}</p>
            </div>
        </div>
        <div class="hub-actions">
            <div class="hub-search-box">
                <span style="margin-right:0.45rem;color:#94a3b8;">🔍</span>
                <span style="color:#94a3b8;">Start searching here...</span>
            </div>
            <div class="hub-icon-btn">⚙️</div>
            <div class="hub-icon-btn">
                🔔
                <span class="hub-badge-dot"></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alert_banner(
    title="Dear Manager",
    message="We have observed a decline in <span class='hub-highlight-tag'>[Hermawan]</span>'s performance over the past 2 weeks.",
    button_text="View Detail",
    button_target="Train",
):
    """Alert banner matching the exact screenshot layout."""
    c1, c2 = st.columns([5, 1.1])
    with c1:
        st.markdown(f"""
        <div class="hub-alert-banner">
            <div class="hub-alert-content">
                <div class="hub-alert-icon-wrap">🔔</div>
                <div class="hub-alert-text">
                    <h4>{title}</h4>
                    <p>{message}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
        if st.button(button_text, key="hub_alert_btn_action", type="primary", use_container_width=True):
            navigate_to(button_target)

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)


def render_metric_card(col, label, value, icon_char="👤"):
    """
    Renders metric card matching the exact top 4 cards:
    - Active Employees (547)
    - Number of Projects (339)
    - Number of Task (147)
    - Target Percentage Completed (89.75%)
    """
    col.markdown(f"""
    <div class="hub-metric-card">
        <div class="hub-metric-top">
            <span class="hub-metric-title">{label}</span>
            <span class="hub-metric-corner">⛶</span>
        </div>
        <div class="hub-metric-bottom">
            <div class="hub-metric-icon-box">{icon_char}</div>
            <div class="hub-metric-value">{value}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_header(icon, title, description):
    """Header card for sub-pages."""
    st.markdown(f"""
    <div class="fs-card" style="margin-bottom:1.1rem;">
        <div style="display:flex;align-items:center;gap:0.75rem;">
            <div style="width:40px;height:40px;border-radius:10px;background:#eff6ff;color:#2563eb;
                        display:flex;align-items:center;justify-content:center;font-size:1.3rem;">
                {icon}
            </div>
            <div>
                <h2 style="font-size:1.25rem;font-weight:800;color:#0f172a;margin:0 0 0.15rem;">{title}</h2>
                <p style="font-size:0.82rem;color:#64748b;margin:0;">{description}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def footer():
    """Footer component."""
    st.markdown("""
    <div class="fs-footer">
        InsightHub Platform &nbsp;·&nbsp;
        Credit Card Anomaly Detection &nbsp;·&nbsp;
        v2.5 High-Precision Release
    </div>
    """, unsafe_allow_html=True)


def navigate_to(page_key):
    """Programmatically jump to any page reliably."""
    st.session_state["page"] = page_key
    st.rerun()
