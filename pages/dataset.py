"""
pages/dataset.py — Dataset & Preprocessing
Modern Light SaaS Theme (InsightHub aesthetic):
- One-click "Load Demo Dataset (2,500 Transactions)" button
- Kaggle creditcard.csv uploader with full schema validation
- Dataset overview metrics & data preview table
- Class distribution chart
- Train/Test split configuration & Preprocessing execution
"""

import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backend
from utils.ui import page_header, footer, navigate_to


def _class_distribution_chart(df):
    """Clean light theme chart showing legitimate vs fraudulent counts."""
    counts = df["Class"].value_counts()
    legit = int(counts.get(0, counts.get(0.0, 0)))
    fraud = int(counts.get(1, counts.get(1.0, 0)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Legitimate (0)", "Fraudulent (1)"],
        y=[legit, fraud],
        marker=dict(
            color=["#2563eb", "#ef4444"],
            cornerradius=6,
        ),
        text=[f"{legit:,} ({legit/len(df)*100:.2f}%)", f"{fraud:,} ({fraud/len(df)*100:.2f}%)"],
        textposition="outside",
        textfont=dict(size=11, color="#334155", family="Plus Jakarta Sans"),
    ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#475569", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title="Transaction Count"),
        xaxis=dict(linecolor="#e2e8f0"),
    )
    return fig


def render():
    page_header(
        "📂", "Dataset & Preprocessing",
        "Inspect, clean, and prepare credit card transaction features for model training.",
    )

    # ── Upload or Load Demo Data Card ──
    st.markdown('<div class="fs-card"><h3>📁 Transaction Dataset Source</h3>', unsafe_allow_html=True)

    c_demo, c_upload = st.columns([1, 1.4])

    with c_demo:
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;height:100%;">
            <div style="font-size:0.9rem;font-weight:700;color:#0f172a;margin-bottom:0.25rem;">
                ⚡ One-Click Demo Dataset
            </div>
            <p style="font-size:0.78rem;color:#64748b;margin:0 0 0.85rem;line-height:1.45;">
                Load 2,500 realistic synthetic transactions matching the Kaggle PCA schema (V1–V28, Time, Amount).
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Load Demo Dataset", key="btn_load_demo_data", type="primary", use_container_width=True):
            with st.spinner("Generating demo transaction records..."):
                demo_df = backend.generate_sample_dataset()
                st.session_state["df"] = demo_df
                st.session_state["dataset_ready"] = True
                # Auto-split
                X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(demo_df)
                st.session_state["X_train"] = X_tr
                st.session_state["X_test"] = X_te
                st.session_state["y_train"] = y_tr
                st.session_state["y_test"] = y_te
                st.session_state["scaler"] = scl
                st.session_state["feature_names"] = f_names
                st.success("✅ Demo dataset (2,500 rows) loaded and preprocessed!")
                st.rerun()

    with c_upload:
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;height:100%;">
            <div style="font-size:0.9rem;font-weight:700;color:#0f172a;margin-bottom:0.25rem;">
                📤 Upload Kaggle creditcard.csv
            </div>
            <p style="font-size:0.78rem;color:#64748b;margin:0 0 0.5rem;line-height:1.45;">
                Upload the full 150MB European cardholders dataset with Time, Amount, and Class columns.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload creditcard.csv",
            type=["csv"],
            key="csv_uploader_dataset",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_file_id") != file_id:
                try:
                    df_raw = pd.read_csv(io.BytesIO(uploaded.read()))
                    df_raw, err = backend.validate_dataframe(df_raw)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["df"] = df_raw
                        st.session_state["dataset_ready"] = True
                        st.session_state["_file_id"] = file_id
                        # Auto-preprocess
                        X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(df_raw)
                        st.session_state["X_train"] = X_tr
                        st.session_state["X_test"] = X_te
                        st.session_state["y_train"] = y_tr
                        st.session_state["y_test"] = y_te
                        st.session_state["scaler"] = scl
                        st.session_state["feature_names"] = f_names
                        st.success(f"✅ Loaded and preprocessed {len(df_raw):,} rows from {uploaded.name}!")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not read uploaded file: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None:
        return

    # ── Dataset Overview Stats ──
    summary = backend.get_dataset_summary(df)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="hub-metric-card">
            <span class="hub-metric-title">Total Records</span>
            <div class="hub-metric-value" style="font-size:1.5rem;margin:0.4rem 0;">{summary['total_transactions']:,}</div>
            <div class="hub-metric-badge"><span>●</span> {df.shape[1]} total columns</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="hub-metric-card">
            <span class="hub-metric-title">Legitimate Transactions</span>
            <div class="hub-metric-value" style="font-size:1.5rem;margin:0.4rem 0;color:#2563eb;">{summary['legitimate_transactions']:,}</div>
            <div class="hub-metric-badge"><span>●</span> {summary['legit_percentage']}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="hub-metric-card">
            <span class="hub-metric-title">Fraudulent Transactions</span>
            <div class="hub-metric-value" style="font-size:1.5rem;margin:0.4rem 0;color:#ef4444;">{summary['fraud_transactions']:,}</div>
            <div class="hub-metric-badge" style="color:#ef4444;"><span>●</span> {summary['fraud_percentage']}% anomaly rate</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="hub-metric-card">
            <span class="hub-metric-title">PCA Feature Vectors</span>
            <div class="hub-metric-value" style="font-size:1.5rem;margin:0.4rem 0;">{summary['num_features']}</div>
            <div class="hub-metric-badge"><span>●</span> V1–V28 + Time, Amount</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # ── Data Preview & Class Balance ──
    p_left, p_right = st.columns([1.3, 1])

    with p_left:
        st.markdown('<div class="fs-card"><h3>📋 Transaction Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=260)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_right:
        st.markdown('<div class="fs-card"><h3>⚖️ Class Distribution</h3>', unsafe_allow_html=True)
        st.plotly_chart(_class_distribution_chart(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Preprocessing Configuration & Next Step ──
    st.markdown("""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:1.1rem 1.4rem;display:flex;align-items:center;justify-content:space-between;margin-top:0.75rem;">
        <div>
            <h4 style="font-size:0.95rem;font-weight:700;color:#1e3a8a;margin:0 0 0.15rem;">
                ✅ Dataset is Preprocessed & Ready!
            </h4>
            <p style="font-size:0.8rem;color:#3b82f6;margin:0;">
                Features are standardized with StandardScaler (no data leakage). Proceed to train all 4 classifiers.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("Go to Model Training →", key="btn_goto_train_from_ds", type="primary"):
        navigate_to("Train")

    footer()
