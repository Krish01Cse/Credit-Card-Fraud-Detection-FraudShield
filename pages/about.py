"""
pages/about.py — Results & About
Modern Light SaaS Theme (InsightHub aesthetic):
- Project Specifications & Technology Stack
- Kaggle Dataset Documentation
- Academic Mini-Project Disclaimers
"""

import streamlit as st
from utils.ui import page_header, footer


def render():
    page_header(
        "📋", "System Overview & About",
        "Project architecture, machine learning models, and dataset provenance.",
    )

    st.markdown("""
    <div class="fs-card">
        <h3>Architecture & Specifications</h3>
        <table style="width:100%;border-collapse:collapse;font-size:0.875rem;">
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:0.6rem 0.75rem;color:#64748b;width:180px;font-weight:600;">System Platform</td>
                <td style="padding:0.6rem 0.75rem;color:#0f172a;font-weight:700;">FraudShield — AI Credit Card Anomaly Detection</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:0.6rem 0.75rem;color:#64748b;font-weight:600;">Design Philosophy</td>
                <td style="padding:0.6rem 0.75rem;color:#2563eb;font-weight:600;">
                    Modern SaaS Dashboard (InsightHub Edition)
                </td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:0.6rem 0.75rem;color:#64748b;font-weight:600;">Classifiers Supported</td>
                <td style="padding:0.6rem 0.75rem;color:#0f172a;">
                    Logistic Regression · Decision Tree · Random Forest · K-Nearest Neighbors
                </td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:0.6rem 0.75rem;color:#64748b;font-weight:600;">Evaluation Strategy</td>
                <td style="padding:0.6rem 0.75rem;color:#0f172a;">
                    Stratified Train/Test Split (80/20), StandardScaler (Fit on Train only)
                </td>
            </tr>
            <tr>
                <td style="padding:0.6rem 0.75rem;color:#64748b;font-weight:600;">Technology Stack</td>
                <td style="padding:0.6rem 0.75rem;color:#0f172a;">
                    Python 3.9+ · Streamlit · Scikit-learn · Pandas · NumPy · Plotly · Joblib
                </td>
            </tr>
        </table>
    </div>

    <div class="fs-card">
        <h3>Kaggle Dataset Provenance</h3>
        <p style="font-size:0.875rem;line-height:1.75;color:#475569;margin:0 0 0.85rem;">
            FraudShield benchmarks against the standard <strong style="color:#0f172a;">Kaggle Credit Card Fraud Detection</strong>
            dataset. It represents <strong style="color:#0f172a;">284,807 transactions</strong> made by European cardholders in September 2013,
            with <strong style="color:#ef4444;">492 fraudulent swipes (0.172% fraud rate)</strong>.
            Features V1 through V28 are numerical components obtained via Principal Component Analysis (PCA) to preserve user confidentiality.
        </p>
        <div style="font-size:0.82rem;font-weight:600;">
            <span style="color:#64748b;">Kaggle Reference: </span>
            <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" style="color:#2563eb;text-decoration:none;">
                kaggle.com/datasets/mlg-ulb/creditcardfraud ↗
            </a>
        </div>
    </div>

    <div style="background:#fffbeb;border:1px solid #fef3c7;border-radius:12px;padding:1.1rem 1.4rem;">
        <div style="font-size:0.86rem;font-weight:700;color:#b45309;margin-bottom:0.25rem;">
            ⚠️ Academic Demonstration Notice
        </div>
        <p style="font-size:0.82rem;line-height:1.65;color:#92400e;margin:0;">
            This system is built as an academic engineering mini-project demonstrating supervised machine learning
            on highly imbalanced tabular financial data. It is for research and educational demonstrations only.
        </p>
    </div>
    """, unsafe_allow_html=True)

    footer()
