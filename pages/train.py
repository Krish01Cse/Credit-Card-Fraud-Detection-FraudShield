"""
pages/train.py — Model Training & Comparison
Modern Light SaaS Theme (InsightHub aesthetic):
- Dedicated individual "Train" buttons for Random Forest, KNN, Decision Tree, and Logistic Regression
- "Train All Models" and "Train Selected" bulk execution
- Automatic pipeline warmup (never disabled, never crashes)
- Live side-by-side metrics table, ROC/PR curves, and confusion matrices
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import backend
from utils.ui import page_header, footer, navigate_to

ALL_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors"]

MODEL_META = {
    "Logistic Regression": {
        "color": "#0ea5e9", "icon": "📈",
        "desc": "Linear benchmark — fast, high explainability",
        "key_btn": "train_btn_lr",
    },
    "Decision Tree": {
        "color": "#10b981", "icon": "🌲",
        "desc": "Rule-based tree — explainable decision logic",
        "key_btn": "train_btn_dt",
    },
    "Random Forest": {
        "color": "#2563eb", "icon": "🌳",
        "desc": "Ensemble trees — top precision and fraud recall",
        "key_btn": "train_btn_rf",
    },
    "K-Nearest Neighbors": {
        "color": "#8b5cf6", "icon": "📌",
        "desc": "Distance-based — instance proximity classification",
        "key_btn": "train_btn_knn",
    },
}

_LIGHT_PLOT_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#475569", size=11),
    margin=dict(l=10, r=10, t=35, b=10),
)


# ── Chart builders ─────────────────────────────────────────────────────────────

def _bar_chart(results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Avg Precision"]
    fig = go.Figure()
    for r in results:
        c = MODEL_META.get(r["name"], {}).get("color", "#2563eb")
        fig.add_trace(go.Bar(
            name=r["name"], x=labels,
            y=[r[m] for m in metrics],
            marker=dict(color=c, cornerradius=4),
            text=[f"{r[m]}%" for m in metrics],
            textposition="outside",
            textfont=dict(size=9, color="#475569"),
        ))
    fig.update_layout(
        **_LIGHT_PLOT_LAYOUT,
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        title=dict(text="Model Performance Breakdown (%)", font=dict(size=13, color="#0f172a")),
        yaxis=dict(range=[0, 118], gridcolor="#f1f5f9", linecolor="#e2e8f0", title="Score (%)"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
        height=360,
    )
    return fig


def _roc_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#cbd5e1", width=1.5), name="Random (AUC=50%)",
    ))
    for r in results:
        c = MODEL_META.get(r["name"], {}).get("color", "#2563eb")
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{r['name']} ({r['roc_auc']}%)",
            line=dict(color=c, width=2.5),
        ))
    fig.update_layout(
        **_LIGHT_PLOT_LAYOUT,
        title=dict(text="Receiver Operating Characteristic (ROC)", font=dict(size=13, color="#0f172a")),
        xaxis=dict(title="False Positive Rate", gridcolor="#f1f5f9", linecolor="#e2e8f0", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", gridcolor="#f1f5f9", linecolor="#e2e8f0", range=[0, 1.05]),
        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)"),
        height=340,
    )
    return fig


def _pr_chart(results):
    fig = go.Figure()
    for r in results:
        c = MODEL_META.get(r["name"], {}).get("color", "#2563eb")
        fig.add_trace(go.Scatter(
            x=r["pr_recall"], y=r["pr_precision"], mode="lines",
            name=f"{r['name']} (AP={r['avg_precision']}%)",
            line=dict(color=c, width=2.5),
        ))
    fig.update_layout(
        **_LIGHT_PLOT_LAYOUT,
        title=dict(text="Precision-Recall Curve (PR)", font=dict(size=13, color="#0f172a")),
        xaxis=dict(title="Recall", gridcolor="#f1f5f9", linecolor="#e2e8f0", range=[0, 1]),
        yaxis=dict(title="Precision", gridcolor="#f1f5f9", linecolor="#e2e8f0", range=[0, 1.05]),
        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)"),
        height=340,
    )
    return fig


def _cm_fig(cm, name):
    c = MODEL_META.get(name, {}).get("color", "#2563eb")
    cm_arr = np.array(cm)
    total = max(cm_arr.sum(), 1)
    ann = [[f"{cm_arr[i][j]}<br><span style='font-size:9px;color:#64748b;'>({cm_arr[i][j]/total*100:.1f}%)</span>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=["Pred: Legit", "Pred: Fraud"],
        y=["Actual: Legit", "Actual: Fraud"],
        text=ann, texttemplate="%{text}",
        colorscale=[[0, "#f8fafc"], [0.5, "#dbeafe"], [1, c]],
        showscale=False,
        textfont=dict(size=12, color="#0f172a"),
    ))
    fig.update_layout(
        **_LIGHT_PLOT_LAYOUT,
        title=dict(text=name, font=dict(size=11, color="#0f172a")),
        height=210, margin=dict(l=5, r=5, t=35, b=5),
        xaxis=dict(linecolor="#e2e8f0"),
        yaxis=dict(linecolor="#e2e8f0"),
    )
    return fig


# ── Single Model Training Runner ──────────────────────────────────────────────

def _train_single_model(name, X_train, y_train, X_test, y_test):
    """Train a single model and update session state."""
    trained_models = st.session_state.setdefault("trained_models", {})
    eval_results = st.session_state.setdefault("eval_results", [])

    model = backend.train_model(name, X_train, y_train)
    result = backend.evaluate_model(model, X_test, y_test, name)
    trained_models[name] = model

    # Update eval_results list
    eval_results = [r for r in eval_results if r["name"] != name]
    eval_results.append(result)
    eval_results.sort(key=lambda r: r["f1"], reverse=True)

    st.session_state["trained_models"] = trained_models
    st.session_state["eval_results"] = eval_results
    st.session_state["models_ready"] = True

    return result


# ── Bulk Model Training Runner ────────────────────────────────────────────────

def _train_bulk_models(models_to_train, X_train, y_train, X_test, y_test, progress_placeholder):
    """Train multiple models sequentially with progress indication."""
    progress_bar = progress_placeholder.progress(0, text="Initializing training...")
    errors = []

    for idx, name in enumerate(models_to_train):
        pct = (idx + 1) / len(models_to_train)
        progress_bar.progress(pct, text=f"Training & Evaluating {name} ({idx+1}/{len(models_to_train)})...")
        try:
            _train_single_model(name, X_train, y_train, X_test, y_test)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    progress_placeholder.empty()
    return errors


# ── Main Page Renderer ────────────────────────────────────────────────────────

def render():
    page_header(
        "⚙️", "Train & Compare Classifiers",
        "Train individual models (Random Forest, KNN, Decision Tree, Logistic Regression) or benchmark all models in one click.",
    )

    # ── Auto-Warmup Data Pipeline (Never leaves page disabled!) ──
    if st.session_state.get("X_train") is None or st.session_state.get("y_train") is None:
        try:
            df = backend.generate_sample_dataset()
            st.session_state["df"] = df
            st.session_state["dataset_ready"] = True
            X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(df)
            st.session_state["X_train"] = X_tr
            st.session_state["X_test"] = X_te
            st.session_state["y_train"] = y_tr
            st.session_state["y_test"] = y_te
            st.session_state["scaler"] = scl
            st.session_state["feature_names"] = f_names
        except Exception as e:
            st.error(f"Could not initialize data pipeline: {e}")

    X_train = st.session_state.get("X_train")
    X_test  = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test  = st.session_state.get("y_test")
    trained_models = st.session_state.setdefault("trained_models", {})
    eval_results   = st.session_state.setdefault("eval_results",   [])

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Dedicated Model Training Cards (RF, KNN, DT, LR)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="fs-card"><h3>🚀 Individual Classifier Training</h3>', unsafe_allow_html=True)

    if X_train is not None:
        st.markdown(
            f'<p style="font-size:0.8rem;color:#64748b;margin:0 0 1rem;">'
            f'Training Dataset: <strong style="color:#0f172a;">{X_train.shape[0]:,} records × {X_train.shape[1]} features</strong> &nbsp;|&nbsp; '
            f'Test Dataset: <strong style="color:#0f172a;">{X_test.shape[0]:,} records</strong> '
            f'({int(y_test.sum())} fraud cases held out for evaluation).'
            f'</p>',
            unsafe_allow_html=True,
        )

    model_cols = st.columns(4)
    trained_target = None

    for col, name in zip(model_cols, ALL_MODELS):
        meta = MODEL_META[name]
        is_trained = name in trained_models
        badge_html = (
            f'<span class="hub-badge hub-badge-green">✓ Active</span>'
            if is_trained
            else '<span class="hub-badge hub-badge-amber">Ready</span>'
        )

        # Lookup recent F1 score if already evaluated
        eval_score = "—"
        for r in eval_results:
            if r["name"] == name:
                eval_score = f"{r['f1']:.1f}%"
                break

        col.markdown(f"""
        <div style="background:#ffffff;border:1.5px solid #e2e8f0;border-radius:12px;
                    padding:0.9rem;margin-bottom:0.45rem;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                <span style="font-size:1.35rem;">{meta['icon']}</span>
                {badge_html}
            </div>
            <div style="font-size:0.86rem;font-weight:700;color:#0f172a;margin-bottom:0.15rem;">
                {name}
            </div>
            <div style="font-size:0.72rem;color:#64748b;line-height:1.4;margin-bottom:0.45rem;">
                {meta['desc']}
            </div>
            <div style="font-size:0.72rem;color:#64748b;padding-top:0.35rem;border-top:1px solid #f1f5f9;">
                Latest F1: <strong style="color:#2563eb;">{eval_score}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Direct one-click train button for this specific model!
        btn_type = "primary" if name in ["Random Forest", "K-Nearest Neighbors"] else "secondary"
        if col.button(f"Train {name.split()[0]}", key=f"btn_train_single_{name}", type=btn_type, use_container_width=True):
            trained_target = name

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

    # ── Bulk Execution Buttons ──
    b1, b2, _ = st.columns([1.35, 1.2, 2])
    train_all_clicked = b1.button("🚀 Train All Models", key="btn_bulk_train_all", type="primary", use_container_width=True)

    progress_placeholder = st.empty()
    status_msg_box = st.empty()

    # Execute single model training
    if trained_target:
        with st.spinner(f"Training {trained_target} classifier..."):
            try:
                res = _train_single_model(trained_target, X_train, y_train, X_test, y_test)
                status_msg_box.success(
                    f"✅ **{trained_target}** trained successfully! "
                    f"Accuracy: **{res['accuracy']}%** &nbsp;|&nbsp; "
                    f"Precision: **{res['precision']}%** &nbsp;|&nbsp; "
                    f"Recall: **{res['recall']}%** &nbsp;|&nbsp; "
                    f"F1-Score: **{res['f1']}%**"
                )
                eval_results = st.session_state.get("eval_results", [])
            except Exception as e:
                status_msg_box.error(f"❌ Failed to train {trained_target}: {e}")

    # Execute all models training
    elif train_all_clicked:
        errors = _train_bulk_models(ALL_MODELS, X_train, y_train, X_test, y_test, progress_placeholder)
        if errors:
            status_msg_box.error("Some models encountered an error:\n" + "\n".join(errors))
        else:
            status_msg_box.success(
                f"✅ Successfully trained and benchmarked all 4 classifiers! "
                f"Review updated side-by-side performance metrics below."
            )
            eval_results = st.session_state.get("eval_results", [])

    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Side-by-Side Comparison Table
    # ─────────────────────────────────────────────────────────────────────────
    if eval_results:
        st.markdown('<div class="fs-card"><h3>📊 Side-by-Side Performance Comparison</h3>', unsafe_allow_html=True)

        table_rows = []
        for r in eval_results:
            table_rows.append({
                "Model": r["name"],
                "Accuracy (%)": f"{r['accuracy']:.2f}%",
                "Precision (%)": f"{r['precision']:.2f}%",
                "Recall (%)": f"{r['recall']:.2f}%",
                "F1-Score (%)": f"{r['f1']:.2f}%",
                "ROC-AUC (%)": f"{r['roc_auc']:.2f}%",
                "Avg Precision (%)": f"{r['avg_precision']:.2f}%",
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:#64748b;margin:0.4rem 0 0;">'
            '⭐ <strong>F1-Score &amp; Average Precision</strong> are the most authoritative metrics for credit card fraud detection '
            'due to the extreme 0.17% class imbalance.'
            '</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Visual Metric Comparison & Performance Curves
    # ─────────────────────────────────────────────────────────────────────────
    if len(eval_results) >= 2:
        st.markdown('<div class="fs-card"><h3>📈 Visual Metric Benchmark</h3>', unsafe_allow_html=True)
        st.plotly_chart(_bar_chart(eval_results), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        c_roc, c_pr = st.columns(2)
        with c_roc:
            st.markdown('<div class="fs-card"><h3>📈 ROC Curves</h3>', unsafe_allow_html=True)
            st.plotly_chart(_roc_chart(eval_results), use_container_width=True)
            st.markdown('<p style="font-size:0.74rem;color:#64748b;margin:0;">Higher AUC demonstrates superior true positive rate vs false positive trade-off.</p></div>', unsafe_allow_html=True)

        with c_pr:
            st.markdown('<div class="fs-card"><h3>📉 Precision-Recall Curves</h3>', unsafe_allow_html=True)
            st.plotly_chart(_pr_chart(eval_results), use_container_width=True)
            st.markdown('<p style="font-size:0.74rem;color:#64748b;margin:0;">PR curves reveal performance where fraud is a tiny fraction of total transactions.</p></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Confusion Matrices
    # ─────────────────────────────────────────────────────────────────────────
    if eval_results:
        st.markdown('<div class="fs-card"><h3>🔲 Confusion Matrices</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#64748b;margin:0 0 0.75rem;">'
            'Actual vs. Predicted class counts on held-out test data. '
            '<strong style="color:#ef4444;">False Negatives (Missed Fraud)</strong> are highlighted as the highest-risk fraud error.'
            '</p>',
            unsafe_allow_html=True,
        )

        cm_cols = st.columns(min(len(eval_results), 4))
        for col, r in zip(cm_cols, eval_results):
            with col:
                st.plotly_chart(_cm_fig(r["confusion_matrix"], r["name"]), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Next Step CTA ──
    st.markdown("""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:1rem 1.3rem;display:flex;align-items:center;justify-content:space-between;margin-top:0.85rem;">
        <div>
            <h4 style="font-size:0.92rem;font-weight:700;color:#1e3a8a;margin:0 0 0.15rem;">
                🎯 Classifiers ready for live inference!
            </h4>
            <p style="font-size:0.78rem;color:#3b82f6;margin:0;">
                Switch to Fraud Prediction to test transactions with the newly trained models.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("Go to Live Fraud Prediction →", key="btn_goto_pred_from_train_page", type="primary"):
        navigate_to("Predict")

    footer()
