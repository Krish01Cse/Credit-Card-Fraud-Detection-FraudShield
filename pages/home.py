"""
pages/home.py — Home / Dashboard
Exact visual recreation of the reference dashboard screenshot:
- Header bar: Hey, Markus, Sunday, June 25, 2024, search pill, action icons
- Notification banner: "Dear Manager: We have observed a decline in [Hermawan]'s performance..."
- 4 Top Cards:
  * Active Employees (547)
  * Number of Projects (339)
  * Number of Task (147)
  * Target Percentage Completed (89.75%)
- Two-Column Layout:
  * Left: "On Going Task" (Journey Scarves, Edfier, ML Model tasks with avatars & status)
  * Right: "Graphs and Analysis" (Month dropdown, rounded bar chart with floating tooltip card)
- Quick navigation shortcuts to Dataset, Training, Prediction
"""

import streamlit as st
import plotly.graph_objects as go
from utils.ui import (
    render_top_header,
    render_alert_banner,
    render_metric_card,
    footer,
    navigate_to,
)


def _render_monthly_trends_chart():
    """
    Renders the exact rounded bar chart from the reference screenshot:
    - Months: Jan, Feb, Mar, Apr, Mei, Jun, Jul, Aug, Sep
    - Soft cyan/blue bars for normal months, tall Electric Royal Blue for peak month (Mei)
    - Floating dark tooltip card above peak bar:
        ● Project Done: 137
        ● Project New: 123
        ● Project Cost: 84
    """
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep"]
    # Heights matching the visual bars in the screenshot
    values = [45, 52, 48, 62, 137, 38, 70, 65, 72]

    # Colors: Mei is bright electric blue (#2563eb), others are soft gradient light blue (#bfdbfe)
    colors = ["#dbeafe", "#dbeafe", "#dbeafe", "#93c5fd", "#2563eb", "#dbeafe", "#bfdbfe", "#bfdbfe", "#bfdbfe"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=months,
        y=values,
        marker=dict(
            color=colors,
            cornerradius=8,
            line=dict(width=0),
        ),
        text=["", "", "", "", "137", "", "", "", ""],
        textposition="outside",
        textfont=dict(size=11, color="#2563eb", family="Plus Jakarta Sans"),
    ))

    # Add the floating dark tooltip badge above Mei (matching the screenshot)
    fig.add_annotation(
        x="Mei",
        y=115,
        text="<b>● Project Done:</b> 137<br><b>● Project New:</b> 123<br><b>● Project Cost:</b> 84",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#1e293b",
        ax=-55,
        ay=-35,
        bgcolor="#1e293b",
        bordercolor="#334155",
        borderwidth=1,
        borderpad=6,
        font=dict(size=9.5, color="#f8fafc", family="Plus Jakarta Sans"),
        align="left",
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#64748b", size=11),
        margin=dict(l=10, r=10, t=25, b=10),
        height=320,
        bargap=0.35,
        showlegend=False,
        xaxis=dict(
            gridcolor="#f8fafc",
            linecolor="#f1f5f9",
            tickfont=dict(size=11, color="#64748b"),
        ),
        yaxis=dict(
            range=[0, 155],
            gridcolor="#f8fafc",
            linecolor="#f1f5f9",
            showticklabels=False,
        ),
    )
    return fig


def render():
    # ── 1. Top Header (Hey, Markus | Sunday, June 25, 2024 | Search | Actions) ──
    render_top_header(title="Dashboard", subtitle="Sunday, June 25, 2024")

    # ── 2. Alert Notification Banner ("Dear Manager") ──
    render_alert_banner(
        title="Dear Manager",
        message="We have observed a decline in <span class='hub-highlight-tag'>[Hermawan]</span>'s performance over the past 2 weeks.",
        button_text="View Detail",
        button_target="Train",
    )

    # ── 3. Four Top Stat Metric Cards (Exact match to screenshot) ──
    df = st.session_state.get("df")
    total_tx = f"{len(df):,}" if df is not None else "547"

    c1, c2, c3, c4 = st.columns(4)
    render_metric_card(c1, "Active Employees", "547", icon_char="👤")
    render_metric_card(c2, "Number of Projects", "339", icon_char="📁")
    render_metric_card(c3, "Number of Task", "147", icon_char="📎")
    render_metric_card(c4, "Target Percentage Completed", "89.75%", icon_char="%")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── 4. Main Two-Column Cards: On Going Task & Graphs and Analysis ──
    col_left, col_right = st.columns([1.1, 1.3])

    # Left Column: "On Going Task"
    with col_left:
        st.markdown("""
        <div class="fs-card" style="height:100%;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">
                <div style="display:flex;align-items:center;gap:0.45rem;">
                    <span style="color:#2563eb;font-size:1.1rem;">🎯</span>
                    <h3 style="margin:0;font-size:1.02rem;">On Going Task</h3>
                </div>
                <div style="display:flex;gap:0.55rem;color:#94a3b8;font-size:0.88rem;">
                    <span style="cursor:pointer;">🔍</span>
                    <span style="cursor:pointer;">🝖</span>
                </div>
            </div>
            <p style="font-size:0.76rem;color:#64748b;margin:0 0 0.95rem;">
                Best performing employee ranking.
            </p>

            <!-- Task 1: Journey Scarves -->
            <div class="hub-task-item">
                <div class="hub-task-left">
                    <div class="hub-task-icon-circle" style="background:#0f172a;">JS</div>
                    <div class="hub-task-info">
                        <h5>Journey Scarves</h5>
                        <p>Rebranding and Website Design</p>
                    </div>
                </div>
                <div class="hub-avatar-stack">
                    <div class="hub-mini-avatar" style="background:#3b82f6;color:#ffffff;">👩‍💻</div>
                    <div class="hub-mini-avatar" style="background:#10b981;color:#ffffff;">👨‍🎨</div>
                    <div class="hub-mini-avatar" style="background:#f59e0b;color:#ffffff;">👩‍💼</div>
                </div>
            </div>

            <!-- Task 2: Edfier (Detailed with Progress) -->
            <div class="hub-task-item" style="flex-direction:column;align-items:stretch;gap:0.75rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div class="hub-task-left">
                        <div class="hub-task-icon-circle" style="background:#38bdf8;">ED</div>
                        <div class="hub-task-info">
                            <h5>Edfier</h5>
                            <p>Web Design &amp; Development</p>
                        </div>
                    </div>
                    <div class="hub-avatar-stack">
                        <div class="hub-mini-avatar" style="background:#8b5cf6;color:#ffffff;">👩‍💻</div>
                        <div class="hub-mini-avatar" style="background:#ec4899;color:#ffffff;">👨‍💻</div>
                    </div>
                </div>
                <!-- Progress & Due Date Row -->
                <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#64748b;
                            padding-top:0.45rem;border-top:1px solid #f1f5f9;">
                    <div>Status: <strong style="color:#0f172a;">On Going</strong></div>
                    <div>Progression: <strong style="color:#2563eb;">51%</strong></div>
                    <div>Due Date: <strong style="color:#0f172a;">Aug, 17 2024</strong></div>
                </div>
            </div>

            <!-- Task 3: FraudShield Active Model Benchmark -->
            <div class="hub-task-item">
                <div class="hub-task-left">
                    <div class="hub-task-icon-circle" style="background:#2563eb;">FS</div>
                    <div class="hub-task-info">
                        <h5>Random Forest Pipeline</h5>
                        <p>Real-time Credit Card Anomaly Detection</p>
                    </div>
                </div>
                <div>
                    <span class="hub-badge hub-badge-green">99.8% Acc</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Right Column: "Graphs and Analysis"
    with col_right:
        st.markdown("""
        <div class="fs-card" style="height:100%;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">
                <div style="display:flex;align-items:center;gap:0.45rem;">
                    <span style="color:#2563eb;font-size:1.1rem;">📊</span>
                    <h3 style="margin:0;font-size:1.02rem;">Graphs and Analysis</h3>
                </div>
                <div style="display:flex;align-items:center;gap:0.65rem;">
                    <span style="font-size:0.76rem;font-weight:600;color:#0f172a;background:#f8fafc;
                                 border:1px solid #e2e8f0;padding:0.22rem 0.65rem;border-radius:6px;">
                        Month ▾
                    </span>
                    <span style="color:#94a3b8;font-size:0.85rem;cursor:pointer;">↑</span>
                </div>
            </div>
            <p style="font-size:0.76rem;color:#64748b;margin:0 0 0.5rem;">
                Projects completed per month based on trends.
            </p>
        """, unsafe_allow_html=True)

        # Plotly Rounded Bar Chart
        fig = _render_monthly_trends_chart()
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 5. Quick Navigation Row ──
    st.markdown("<h4 style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:0.75rem;'>⚡ Platform Quick Actions</h4>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown("""
        <div class="hub-task-item" style="margin-bottom:0.4rem;">
            <div class="hub-task-left">
                <div class="hub-task-icon-circle" style="background:#2563eb;">📂</div>
                <div class="hub-task-info">
                    <h5>Dataset &amp; Preprocessing</h5>
                    <p>Load 2,500 demo rows or Kaggle CSV</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Dataset →", key="home_nav_ds", use_container_width=True):
            navigate_to("Dataset")

    with q2:
        st.markdown("""
        <div class="hub-task-item" style="margin-bottom:0.4rem;">
            <div class="hub-task-left">
                <div class="hub-task-icon-circle" style="background:#10b981;">⚙️</div>
                <div class="hub-task-info">
                    <h5>Train All Models</h5>
                    <p>One-click 4 classifier benchmarking</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Train All Models →", key="home_nav_tr", use_container_width=True, type="primary"):
            navigate_to("Train")

    with q3:
        st.markdown("""
        <div class="hub-task-item" style="margin-bottom:0.4rem;">
            <div class="hub-task-left">
                <div class="hub-task-icon-circle" style="background:#8b5cf6;">🔍</div>
                <div class="hub-task-info">
                    <h5>Fraud Prediction</h5>
                    <p>Live swipe testing &amp; risk gauge</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Prediction →", key="home_nav_pr", use_container_width=True):
            navigate_to("Predict")

    footer()
