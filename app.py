"""
app.py  ←  Run this file:  streamlit run app.py

FraudShield — Credit Card Fraud Detection System
Entry point: applies CSS, renders sidebar, and routes to the correct page.
"""

import streamlit as st
import backend

# ── Page config (must be the VERY FIRST Streamlit call) ──────────────────────
st.set_page_config(
    page_title="FraudShield — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared design system ─────────────────────────────────────────────────────
from utils.ui import inject_css, render_sidebar

# ── Page modules ─────────────────────────────────────────────────────────────
from pages import home, dataset, train, predict, about

# Map page key → render function
PAGES = {
    "Home":    home.render,
    "Dataset": dataset.render,
    "Train":   train.render,
    "Predict": predict.render,
    "About":   about.render,
}


def init_session():
    """Set all session-state defaults and ensure data and models are active immediately."""
    defaults = {
        "page":           "Home",   # currently active page
        "dataset_ready":  False,
        "models_ready":   False,
        "df":             None,     # loaded DataFrame
        "feature_names":  None,     # column names used as features
        "X_train":        None,
        "X_test":         None,
        "y_train":        None,
        "y_test":         None,
        "scaler":         None,
        "trained_models": {},       # {model_name: fitted_model}
        "eval_results":   [],       # list of dicts from backend.evaluate_model()
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Automatically load dataset and models so the user has an active, working system out of the box
    if not st.session_state.get("dataset_ready") or st.session_state.get("df") is None:
        try:
            demo_df = backend.generate_sample_dataset()
            st.session_state["df"] = demo_df
            st.session_state["dataset_ready"] = True
            X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(demo_df)
            st.session_state["X_train"] = X_tr
            st.session_state["X_test"] = X_te
            st.session_state["y_train"] = y_tr
            st.session_state["y_test"] = y_te
            st.session_state["scaler"] = scl
            st.session_state["feature_names"] = f_names

            # Load any saved models from disk
            saved_names = backend.get_saved_model_names()
            loaded = {}
            for name in saved_names:
                m = backend.load_model(name)
                if m is not None:
                    loaded[name] = m

            if loaded:
                st.session_state["trained_models"] = loaded
                st.session_state["eval_results"] = backend.evaluate_all_models(loaded, X_te, y_te)
                st.session_state["models_ready"] = True
        except Exception:
            pass


def main():
    inject_css()                         # inject modern SaaS CSS design system
    init_session()                       # set defaults & warm up components
    page = render_sidebar()              # draw sidebar, returns selected page key
    PAGES.get(page, home.render)()       # call the matching page renderer


if __name__ == "__main__":
    main()
