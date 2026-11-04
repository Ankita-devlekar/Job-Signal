"""Streamlit interface for saved fake-job-posting model inference."""

from __future__ import annotations

import logging
import hashlib
import os
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for module_dir in (SRC_DIR, APP_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import auth as auth_store
from auth import (
    AuthError,
    AuthStorageError,
    LoginRateLimitError,
    authenticate,
    create_account,
    create_session,
    get_session_user,
    initialize_database,
    revoke_session,
    session_is_active,
)
from model import load_model, predict_job_posting
from job_extraction import assess_posting_completeness, extract_job_fields
from ocr import (
    MAX_IMAGE_BYTES,
    OcrError,
    create_ocr_engine,
    decode_image,
    extract_text_from_image,
)
from preprocessing import load_preprocessing_artifacts
from risk_engine import risk_band, rule_risk_score
from web_verification import (
    extract_requisition_id,
    has_official_company_candidate,
    has_vacancy_candidate,
    search_web,
    verify_company_and_vacancy,
)
from xai import analyze_risk_factors
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    return create_ocr_engine()


@st.cache_data(ttl=900, max_entries=100, show_spinner=False)
def _cached_search(
    query: str,
    max_results: int,
    api_key_fingerprint: str,
    _api_key: str,
) -> list:
    return search_web(query, api_key=_api_key, max_results=max_results)


def build_verification_conclusion(
    company_check: dict,
    vacancy_check: dict,
) -> str:
    company_status = company_check["status"]
    vacancy_status = vacancy_check["status"]
    company_failed = company_status == "UNAVAILABLE"
    vacancy_failed = vacancy_status == "UNAVAILABLE"
    if company_failed and vacancy_failed:
        return "Verification could not be completed because the web search failed."
    if company_failed:
        if vacancy_status == "MATCH_FOUND":
            return (
                "The company search failed, but a relevant listing was located "
                "on an employer-controlled careers page."
            )
        return "The company search failed; the vacancy outcome is shown separately."
    if vacancy_failed:
        if company_status == "VERIFIED":
            return (
                "The company's identity was supported by its official website. "
                "The vacancy search failed."
            )
        return "The vacancy search failed; the company outcome is shown separately."
    if company_status == "VERIFIED" and vacancy_status == "MATCH_FOUND":
        return (
            "The company's identity was supported by its official website, "
            "and the role was located on an employer-controlled careers page."
        )
    if vacancy_status == "MATCH_FOUND":
        return (
            "The role was located on an employer-controlled careers page, but "
            "the company's identity was not independently supported by the "
            "company search."
        )
    if company_status == "VERIFIED":
        if vacancy_status == "POSSIBLE_MATCH":
            return (
                "The company's identity was supported by its official website. "
                "A possible third-party listing was found, but its relationship "
                "to the employer could not be confirmed."
            )
        if vacancy_status == "NO_INPUT":
            return (
                "The company's identity was supported by its official website. "
                "The vacancy could not be searched because the job title was "
                "not identified."
            )
        if vacancy_status == "EXPIRED":
            return (
                "The company's identity was supported by its official website. "
                "A relevant listing was found, but it appears to be closed."
            )
        return (
            "The company's identity was supported by its official website. "
            "The specific vacancy was not located on an official careers page."
        )
    if company_status == "INCONCLUSIVE":
        if vacancy_status == "POSSIBLE_MATCH":
            return (
                "The company could not be confirmed from a credible company "
                "source. A possible third-party vacancy listing was found, but "
                "its relationship to the employer could not be confirmed."
            )
        return (
            "The company could not be confirmed from a credible company source, "
            "and the specific vacancy was not confirmed."
        )
    if company_status in {"NOT_ESTABLISHED", "NO_INPUT"}:
        if vacancy_status == "POSSIBLE_MATCH":
            return (
                "The company's identity was not established. A possible "
                "third-party vacancy listing was found, but its relationship "
                "to the employer could not be confirmed."
            )
        return (
            "The company's identity could not be supported by the available "
            "search results, and no official vacancy listing was confirmed."
        )
    return (
        "The company and vacancy outcomes are shown separately; review each "
        "search result and its source before deciding what to check next."
    )


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_artifacts():
    return load_preprocessing_artifacts()


def _get_model_coefficients(model):
    """Return the LinearSVC coefficient vector."""
    if not hasattr(model, "coef_"):
        raise ValueError(
            "Instance-level XAI currently requires a linear model with coef_. "
            "The saved model does not expose model coefficients."
        )

    coef = model.coef_
    if getattr(coef, "ndim", 1) == 2:
        coef = coef[0]

    return coef


def _get_feature_names(artifacts):
    """Reconstruct feature names in the exact order used by transform_job_posting()."""
    names = []

    # 1. Binary block
    names.extend([str(x) for x in artifacts["binary_features"]])

    # 2. Numerical block
    numerical = artifacts["numerical_pipeline"]
    numerical_cols = [str(x) for x in artifacts["numerical_features"]]
    try:
        numerical_names = list(numerical.get_feature_names_out(numerical_cols))
    except (AttributeError, TypeError, ValueError):
        numerical_names = numerical_cols
    names.extend([str(x) for x in numerical_names])

    # 3. Missingness block
    names.extend(["company_profile_missing", "salary_missing"])

    # 4. Categorical block
    categorical = artifacts["categorical_pipeline"]
    categorical_cols = [str(x) for x in artifacts["categorical_features"]]
    try:
        categorical_names = list(categorical.get_feature_names_out(categorical_cols))
    except (AttributeError, TypeError, ValueError):
        categorical_names = categorical_cols
    names.extend([str(x) for x in categorical_names])

    # 5. TF-IDF block
    tfidf = artifacts["tfidf_vectorizer"]
    names.extend([f"text: {x}" for x in tfidf.get_feature_names_out()])

    # 6. Engineered block
    engineered = artifacts["engineered_features"]
    names.extend([f"engineered: {x}" for x in engineered])

    return names


def get_instance_xai(posting_text, model, artifacts, matrix, top_n=8):
    """
    Explain one prediction using the exact LinearSVC feature contributions.

    contribution = transformed_feature_value * model_coefficient

    Positive contribution -> pushes toward class 1 (fraudulent).
    Negative contribution -> pushes toward class 0 (legitimate).
    """
    coefficients = np.asarray(_get_model_coefficients(model)).reshape(-1)
    feature_names = _get_feature_names(artifacts)

    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Local explanations require a binary model with classes 0 and 1.")
    fraud_direction = 1 if classes[1] == 1 else -1

    if matrix.shape[1] != len(coefficients):
        raise ValueError(
            f"XAI mismatch: transformed matrix has {matrix.shape[1]} features, "
            f"but the model has {len(coefficients)} coefficients."
        )

    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch: transformed matrix has {matrix.shape[1]} "
            f"features, but {len(feature_names)} feature names were reconstructed."
        )

    row = matrix.getrow(0)
    indices = row.indices
    values = row.data
    contributions = values * coefficients[indices]

    rows = []
    for idx, value, contribution in zip(indices, values, contributions):
        rows.append(
            {
                "feature": feature_names[idx],
                "value": float(value),
                "contribution": float(contribution),
                "abs_contribution": float(abs(contribution)),
            }
        )

    xai_df = pd.DataFrame(rows)

    if xai_df.empty:
        return xai_df, xai_df

    fraud_side = xai_df["contribution"] * fraud_direction
    fraud_df = (
        xai_df[fraud_side > 0]
        .assign(_fraud_direction=fraud_side[fraud_side > 0])
        .sort_values("_fraud_direction", ascending=False)
        .head(top_n)
        .copy()
        .drop(columns="_fraud_direction")
    )

    legitimate_df = (
        xai_df[fraud_side < 0]
        .assign(_legitimate_direction=-fraud_side[fraud_side < 0])
        .sort_values("_legitimate_direction", ascending=False)
        .head(top_n)
        .copy()
        .drop(columns="_legitimate_direction")
    )

    expected_margin = float(model.decision_function(matrix)[0])
    reconstructed_margin = float(
        np.dot(values, coefficients[indices]) + model.intercept_[0]
    )
    if not np.isclose(expected_margin, reconstructed_margin, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "Local feature contributions do not reconcile with the model decision margin."
        )

    return fraud_df, legitimate_df


def _friendly_feature_name(feature):
    """Make technical feature names easier to understand in the UI."""
    feature = str(feature)

    if feature.startswith("text: "):
        return f'"{feature[6:]}"'

    if feature.startswith("engineered: "):
        raw = feature.replace("engineered: ", "", 1)
        readable = raw.replace("_", " ")
        return readable.capitalize()

    if feature.startswith("x0_"):
        return feature[3:].replace("_", " ").strip().capitalize()

    return feature.replace("_", " ").strip().capitalize()


def _explanation_for_feature(feature, contribution, prediction):
    """Generate a cautious, model-based explanation for one feature."""
    readable = _friendly_feature_name(feature)

    if contribution > 0:
        direction = "increased the model's tendency toward a potentially fraudulent classification"
    else:
        direction = "pushed the model toward a legitimate classification"

    return f'{readable} {direction}.'


def _clear_user_workspace() -> None:
    for key in (
        "auth_token",
        "authenticated_user",
        "posting_input",
        "text_posting_input",
        "text_company_override",
        "text_search_requested",
        "image_posting_input",
        "image_company_override",
        "image_search_requested",
        "job_posting_image",
        "analysis_history",
        "image_upload_digest",
        "image_ocr_digest",
        "login_email",
        "login_password",
        "signup_full_name",
        "signup_email",
        "signup_password",
        "signup_password_confirm",
        "_auth_saved_login_email",
        "_auth_saved_signup_full_name",
        "_auth_saved_signup_email",
    ):
        st.session_state.pop(key, None)
    st.session_state["jobsignal_page"] = "Home"
    st.session_state["auth_mode"] = "signin"


def _brand_mark_svg() -> str:
    return (
        '<span class="brand-mark" aria-hidden="true">'
        '<svg viewBox="0 0 40 40" focusable="false">'
        '<path d="M20 3.5 33 8v10.3c0 8.4-5.2 14.3-13 18.2C12.2 32.6 7 26.7 7 18.3V8L20 3.5Z" '
        'fill="none" stroke="#263B50" stroke-width="2.3" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        '<path d="m13.5 19.3 4.2 4.2 8.8-9" fill="none" stroke="#7D70A8" '
        'stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg></span>"
    )


def _set_page(page_name: str) -> None:
    st.session_state["jobsignal_page"] = page_name


def _set_auth_mode(mode: str) -> None:
    for key in ("login_email", "signup_full_name", "signup_email"):
        if key in st.session_state:
            st.session_state[f"_auth_saved_{key}"] = st.session_state[key]
    st.session_state["auth_mode"] = mode


def _restore_auth_input(key: str) -> None:
    saved_key = f"_auth_saved_{key}"
    if key not in st.session_state and saved_key in st.session_state:
        st.session_state[key] = st.session_state.pop(saved_key)


def _logout(token: str) -> None:
    try:
        revoke_session(token, db_path=auth_store.AUTH_DB_PATH)
    except AuthStorageError as exc:
        LOGGER.exception("The session could not be revoked.")
        st.error(f"Logout could not be completed. {exc}")
        return
    _clear_user_workspace()
    st.session_state["auth_notice"] = "You have been signed out."
    st.rerun()


@st.fragment(run_every="30s")
def _monitor_session_expiry(token: str) -> None:
    try:
        if not session_is_active(token, db_path=auth_store.AUTH_DB_PATH):
            st.rerun()
    except AuthStorageError:
        LOGGER.exception("Unable to check the authenticated session expiry.")
        st.rerun()
    st.empty()


def _render_auth_screen() -> None:
    auth_mode = st.session_state.setdefault("auth_mode", "signin")
    with st.container(key="auth_layout"):
        auth_columns = st.columns(
            [1.08, 0.92],
            gap="large",
            vertical_alignment="center",
        )
        with auth_columns[0]:
            st.markdown(
                f"""
                <section class="auth-panel" aria-labelledby="auth-welcome-title">
                    <div class="auth-brand">
                    {_brand_mark_svg()}
                        <span>JobSignal</span>
                    </div>
                    <div class="auth-panel-content">
                        <h1 id="auth-welcome-title">A more considered way to assess every opportunity.</h1>
                        <p class="auth-panel-copy">Bring clarity to your next career move with thoughtful, evidence-led job screening.</p>
                        <div class="auth-feature-list">
                            <div class="auth-feature">
                                <span class="auth-feature-icon" aria-hidden="true">&#10003;</span>
                                <div><strong>Screen job postings for suspicious signs</strong><p>Paste a job description. The saved model assesses postings when enough information is available.</p></div>
                            </div>
                            <div class="auth-feature">
                                <span class="auth-feature-icon" aria-hidden="true">&#9776;</span>
                                <div><strong>Understand the warning signs</strong><p>Review suspicious phrases, positive indicators, missing details, and practical safety recommendations. Model and rule-based signals stay separate.</p></div>
                            </div>
                            <div class="auth-feature">
                                <span class="auth-feature-icon" aria-hidden="true">&#8599;</span>
                                <div><strong>Check the employer and vacancy</strong><p>Optionally search public web results for company and vacancy evidence. Review the links yourself; search results do not prove a job is genuine.</p></div>
                            </div>
                        </div>
                    </div>
                </section>
                """,
                unsafe_allow_html=True,
            )
        with auth_columns[1]:
            with st.container(key="auth_form_card"):
                if auth_mode == "signup":
                    st.markdown(
                        '<div class="form-eyebrow">YOUR PRIVATE WORKSPACE</div><h2 class="auth-form-title">Create your account</h2><p class="auth-form-copy">Set up your account to begin reviewing job postings.</p>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Use a unique email and a password of at least 12 characters."
                    )
                    with st.form("signup_form"):
                        _restore_auth_input("signup_full_name")
                        full_name = st.text_input(
                            "Full name",
                            key="signup_full_name",
                            placeholder="Your name",
                        )
                        _restore_auth_input("signup_email")
                        signup_email = st.text_input(
                            "Email address",
                            key="signup_email",
                            placeholder="you@example.com",
                        )
                        signup_password = st.text_input(
                            "Create password",
                            type="password",
                            key="signup_password",
                            placeholder="At least 12 characters",
                        )
                        confirm_password = st.text_input(
                            "Confirm password",
                            type="password",
                            key="signup_password_confirm",
                            placeholder="Re-enter your password",
                        )
                        submitted = st.form_submit_button(
                            "Create account",
                            type="primary",
                            use_container_width=True,
                        )
                        st.form_submit_button(
                            "Sign in",
                            key="auth_switch_signin",
                            type="secondary",
                            on_click=_set_auth_mode,
                            args=("signin",),
                        )
                    if submitted:
                        if signup_password != confirm_password:
                            st.error("The password confirmation does not match.")
                        else:
                            try:
                                create_account(
                                    full_name,
                                    signup_email,
                                    signup_password,
                                    db_path=auth_store.AUTH_DB_PATH,
                                )
                            except AuthError as exc:
                                st.error(str(exc))
                            except AuthStorageError as exc:
                                LOGGER.exception("Account registration failed.")
                                st.error(
                                    f"Account creation is temporarily unavailable. {exc}"
                                )
                            else:
                                st.success(
                                    "Account created. Sign in with your new credentials."
                                )
                    st.markdown(
                        '<div class="auth-switch-prompt">Already have an account?</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="form-eyebrow">SECURE ACCOUNT ACCESS</div><h2 class="auth-form-title">Welcome back</h2><p class="auth-form-copy">Sign in to continue to your JobSignal workspace.</p>',
                        unsafe_allow_html=True,
                    )
                    with st.form("login_form"):
                        _restore_auth_input("login_email")
                        email = st.text_input(
                            "Email address",
                            key="login_email",
                            placeholder="you@example.com",
                        )
                        password = st.text_input(
                            "Password",
                            type="password",
                            key="login_password",
                            placeholder="Enter your password",
                        )
                        submitted = st.form_submit_button(
                            "Sign in",
                            type="primary",
                            use_container_width=True,
                        )
                        st.form_submit_button(
                            "Create an account",
                            key="auth_switch_signup",
                            type="secondary",
                            on_click=_set_auth_mode,
                            args=("signup",),
                        )
                    if submitted:
                        try:
                            user = authenticate(
                                email,
                                password,
                                db_path=auth_store.AUTH_DB_PATH,
                            )
                            if user is None:
                                st.error("Email or password is incorrect.")
                            else:
                                token = create_session(
                                    user["user_id"],
                                    db_path=auth_store.AUTH_DB_PATH,
                                )
                                st.session_state["auth_token"] = token
                                st.session_state["authenticated_user"] = user
                                st.session_state["jobsignal_page"] = "Home"
                                st.rerun()
                        except LoginRateLimitError as exc:
                            st.error(str(exc))
                        except AuthStorageError as exc:
                            LOGGER.exception(
                                "Authentication database operation failed."
                            )
                            st.error(f"Sign in is temporarily unavailable. {exc}")
                    st.markdown(
                        '<div class="auth-switch-prompt">New to JobSignal?</div>',
                        unsafe_allow_html=True,
                    )

    auth_notice = st.session_state.pop("auth_notice", None)
    if auth_notice:
        st.info(auth_notice)
    st.caption(
        "Protected with salted password hashing · Sessions expire after "
        "30 minutes of inactivity or 12 hours."
    )


_SCREENING_DISCLAIMER = (
    "JobSignal helps you spot possible warning signs, but it cannot confirm "
    "whether a job or company is genuine. Always check important details "
    "through trusted sources before applying, paying money, or sharing "
    "personal information."
)


def _render_screening_disclaimer() -> None:
    st.markdown(
        f"""
        <aside class="screening-disclaimer" aria-labelledby="screening-disclaimer-title">
            <span class="screening-disclaimer-icon" aria-hidden="true">i</span>
            <div><h2 id="screening-disclaimer-title">Helpful signals, not a final verdict</h2><p>{_SCREENING_DISCLAIMER}</p></div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def _render_home_page(user: dict) -> None:
    st.markdown(
        f"""
        <div class="home-page">
            <header class="home-hero" aria-labelledby="home-title">
                <div class="home-hero-copy">
                    <div class="hero-eyebrow">JOBSIGNAL · YOUR SCREENING WORKSPACE</div>
                    <h1 class="hero-title" id="home-title">A clearer view before you apply.</h1>
                    <p class="hero-subtitle">Review a job posting, understand potential warning signs, and explore available evidence before deciding what to verify.</p>
                </div>
                <div class="home-visual" aria-labelledby="home-review-title">
                    <h2 id="home-review-title">What your review includes</h2>
                    <div class="home-visual-row"><span class="home-visual-icon" aria-hidden="true">ML</span><span><strong>Model assessment</strong><small>A prediction from the saved machine-learning model.</small></span></div>
                    <div class="home-visual-row"><span class="home-visual-icon" aria-hidden="true">✓</span><span><strong>Posting completeness</strong><small>Important details that are present or missing.</small></span></div>
                    <div class="home-visual-row"><span class="home-visual-icon" aria-hidden="true">!</span><span><strong>Warning signals</strong><small>Suspicious phrases, positive indicators, and practical recommendations.</small></span></div>
                    <div class="home-visual-row"><span class="home-visual-icon" aria-hidden="true">↗</span><span><strong>Employer verification</strong><small>Optional company and vacancy search results for independent review.</small></span></div>
                </div>
            </header>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader(f"Welcome, {escape(str(user.get('full_name') or 'Job seeker'))}")
    cta_columns = st.columns([1.1, 2.2])
    with cta_columns[0]:
        st.button(
            "Analyze a Job Posting",
            type="primary",
            use_container_width=True,
            key="home_analyze",
            on_click=_set_page,
            args=("Job Posting Analysis",),
        )
    st.markdown(
        """
        <div class="home-section-heading">
                <span class="home-eyebrow">YOUR WORKSPACE</span>
                <h2 id="recent-analyses-title">Recent analyses</h2>
                <p>Summaries from this session. Past results cannot be reopened from this list.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    history = st.session_state.get("analysis_history", [])
    if not history:
        st.markdown(
            '<div class="home-empty-state"><span class="home-card-icon" aria-hidden="true">↗</span><div><h3>Your analysis history will appear here</h3><p>Start with a job posting to see its model assessment and separate text-based warning signals.</p></div></div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Analyze Your First Job",
            type="primary",
            key="home_first_analysis",
            on_click=_set_page,
            args=("Job Posting Analysis",),
        )
    else:
        for item in history[:5]:
            role = escape(str(item.get("Role") or "Untitled posting"))
            employer = escape(str(item.get("Employer") or "Not identified"))
            analyzed_at = escape(str(item.get("Time") or "Time unavailable"))
            model_assessment = escape(
                str(item.get("Model assessment") or "Not available")
            )
            warning_score = escape(str(item.get("Warning score") or "Not available"))
            risk_level = escape(str(item.get("Risk level") or "Not available"))
            st.markdown(
                f"""
                <article class="home-history-card">
                    <div class="home-history-title">
                        <span class="home-card-icon" aria-hidden="true">▤</span>
                        <div><h3>{role}</h3><p>{employer} · {analyzed_at}</p></div>
                    </div>
                    <div class="home-history-signal"><span>Model assessment</span><strong>{model_assessment}</strong></div>
                    <div class="home-history-signal"><span>Rule-based warning score</span><strong>{warning_score}</strong></div>
                    <div class="home-history-signal"><span>Warning band</span><strong>{risk_level}</strong></div>
                </article>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        """
        <section class="home-section" id="how-it-works" aria-labelledby="how-it-works-title">
            <div class="home-section-heading">
                <span class="home-eyebrow">A SIMPLE WORKFLOW</span>
                <h2 id="how-it-works-title">How JobSignal works</h2>
                <p>Move from a posting to details you can review and verify independently.</p>
            </div>
            <ol class="home-workflow">
                <li><span class="home-workflow-number">01</span><h3>Submit a job posting</h3><p>Paste a job description or upload a supported image containing job-posting text.</p></li>
                <li><span class="home-workflow-number">02</span><h3>Analyze the posting</h3><p>JobSignal processes the available information and examines the posting using its saved machine-learning model and text-based checks.</p></li>
                <li><span class="home-workflow-number">03</span><h3>Review the results</h3><p>Understand the model prediction, posting completeness, warning and positive indicators, and recommendations. Optional online search results may provide additional information for independent verification.</p></li>
                <li><span class="home-workflow-number">04</span><h3>Verify important details</h3><p>Check the employer, vacancy, contact details, payment requests, and other important claims through trusted independent sources before applying or sharing sensitive information.</p></li>
            </ol>
        </section>
        """,
        unsafe_allow_html=True,
    )
   
    st.markdown(
        """
        <section class="home-section" aria-labelledby="provides-title">
            <div class="home-section-heading">
                <span class="home-eyebrow">WHAT YOU CAN REVIEW</span>
                <h2 id="provides-title">Separate signals, clearer context</h2>
                <p>These outputs support further review; they are not a guaranteed verdict or calibrated fraud probability.</p>
            </div>
            <div class="home-output-grid">
                <article class="home-info-card"><span class="home-card-icon" aria-hidden="true">ML</span><h3>Model assessment</h3><p>The saved machine-learning model's assessment of the submitted job posting.</p></article>
                <article class="home-info-card"><span class="home-card-icon" aria-hidden="true">✓</span><h3>Posting completeness</h3><p>An indication of which important job details are present or missing.</p></article>
                <article class="home-info-card"><span class="home-card-icon" aria-hidden="true">!</span><h3>Explainable warning signals</h3><p>Text-matched warning and positive indicators, relevant snippets, and recommendations.</p></article>
                <article class="home-info-card"><span class="home-card-icon" aria-hidden="true">↗</span><h3>Company and vacancy search</h3><p>Optional online search results and links for independent review.</p></article>
            </div>
        </section>
        <section class="home-about-preview" aria-labelledby="home-about-title">
            <div><span class="home-eyebrow">ABOUT THE PROJECT</span><h2 id="home-about-title">Built to help you review job postings more thoughtfully.</h2><p>JobSignal combines machine-learning assessment, explainable text-based signals, and optional online search to support more informed job-posting reviews.</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    about_column, _ = st.columns([1.1, 2.2])
    with about_column:
        st.button(
            "About JobSignal",
            type="primary",
            use_container_width=True,
            key="home_about",
            on_click=_set_page,
            args=("About",),
        )
    _render_screening_disclaimer()


def _render_about_page() -> None:
    st.markdown(
        """
        <div class="about-page" aria-labelledby="about-title">
            <header class="about-hero">
                <div class="about-hero-copy">
                    <div class="about-eyebrow">FAKE JOB DETECTION · JobSignal</div>
                    <h1 id="about-title">About JobSignal</h1>
                    <p>JobSignal helps job seekers review suspicious job postings using machine learning, text-based indicators, and optional verification tools. It highlights possible warning signs and supports independent checks before you apply.</p>
                </div>
                <p class="about-safety-context"><strong>Stay informed and protect your personal information.</strong> Fraudulent job advertisements may expose applicants to financial loss, identity theft, and misleading opportunities. JobSignal helps identify details to check independently.</p>
            </header>
            <section class="about-section" aria-labelledby="prediction-title">
                <div class="about-section-heading"><span class="about-eyebrow">01 · MODEL WORKFLOW</span><h2 id="prediction-title">How the assessment works</h2><p>Research and application inference are distinct: the app uses saved fitted artifacts and does not train the model.</p></div>
                <ol class="about-stepper">
                    <li><span class="about-step-number">01</span><h3>Dataset and preparation</h3><p>The research notebooks reference the Fake Job Postings dataset, with target column <code>fraudulent</code>: 0 represents non-fraudulent and 1 represents fraudulent.</p></li>
                    <li><span class="about-step-number">02</span><h3>EDA and feature engineering</h3><p>Exploratory Data Analysis examines dataset characteristics. Preprocessing handles text, categorical, binary, numerical, and missing-value information and creates engineered features.</p></li>
                    <li><span class="about-step-number">03</span><h3>Model comparison and evaluation</h3><p>The research notebooks compare Logistic Regression, Linear SVM, Random Forest, and XGBoost using recorded evaluation metrics.</p></li>
                    <li><span class="about-step-number">04</span><h3>Saved-model inference</h3><p>The application uses the saved LinearSVC model and fitted preprocessing artifacts to assess sufficiently detailed postings. It does not train or update the model during inference.</p></li>
                </ol>
                <div class="about-tech-row" aria-label="Technology summary"><span>Python</span><span>Streamlit</span><span>Pandas</span><span>NumPy</span><span>Scikit-learn</span><span>TF-IDF text features</span><span>Saved LinearSVC model</span></div>
            </section>
            <section class="about-section" aria-labelledby="xai-title">
                <div class="about-section-heading"><span class="about-eyebrow">02 · Explainable AI · results</span><h2 id="xai-title">Understanding the results</h2><p>Each output is a separate aid for review; they are not combined into a single score.</p></div>
                <div class="about-editorial-grid">
                    <article><span class="about-editorial-icon" aria-hidden="true"></span><h3>Text-based indicators</h3><p>Predefined rules identify warning and positive patterns such as payment requests, urgency, sensitive-information requests, and specific job responsibilities.</p></article>
                    <article><span class="about-editorial-icon" aria-hidden="true"></span><h3>Evidence and explanations</h3><p>Matched patterns appear with relevant text snippets and plain-language explanations to show why a signal was raised.</p></article>
                    <article><span class="about-editorial-icon" aria-hidden="true"></span><h3>Practical recommendations</h3><p>Review suggested next steps, including independently confirming the employer and vacancy and avoiding fees or unnecessary disclosure of sensitive information.</p></article>
                </div>
                <p class="about-callout"><strong>Keep the signals distinct.</strong> The saved-model prediction, rule-based warning score, posting-completeness check, and optional online search are separate aids. The warning score is not a calibrated fraud probability, and search results can be incomplete. The project also implements linear-model feature contributions, but the active interface does not display a detailed contribution breakdown.</p>
            </section>
            <section class="about-section" aria-labelledby="verification-title">
                <div class="about-section-heading"><span class="about-eyebrow">03 · OPTIONAL ONLINE SEARCH</span><h2 id="verification-title">Company and vacancy verification</h2><p>Online search uses the existing Tavily Search API integration and runs only when selected. Results can be incomplete and do not prove whether a company or posting is genuine.</p></div>
                <ol class="about-stepper">
                    <li><span class="about-step-number">01</span><h3>Optional search</h3><p>The user chooses whether to search for the employer and vacancy using the available online-search integration.</p></li>
                    <li><span class="about-step-number">02</span><h3>Company search</h3><p>JobSignal searches for employer information and prioritizes official employer sources where available.</p></li>
                    <li><span class="about-step-number">03</span><h3>Vacancy matching</h3><p>Available listings are examined for matching employer, role, and location details. Company information and the specific vacancy are evaluated separately.</p></li>
                    <li><span class="about-step-number">04</span><h3>Review the evidence</h3><p>Users can inspect returned links and search status to decide what needs further verification. An unsuccessful search or a snippet alone is not proof of fraud or legitimacy.</p></li>
                </ol>
            </section>
            <section class="about-section" aria-labelledby="research-title">
                <div class="about-section-heading"><span class="about-eyebrow">04 · MODEL EVALUATION</span></div>
                <h3 class="about-metrics-title">Held-out research metrics</h3>
                <p class="about-metrics-note">Recorded values from the saved model comparison output.</p>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_path = PROJECT_ROOT / "results" / "model_results" / "model_comparison.csv"
    try:
        metrics = pd.read_csv(metrics_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        LOGGER.exception("The saved model comparison results could not be loaded.")
        st.error(f"Recorded research metrics are unavailable. {exc}")
    else:
        required_columns = ("Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
        if metrics.empty or any(column not in metrics.columns for column in required_columns):
            LOGGER.error("The saved model comparison results have an unexpected format.")
            st.error(
                "Recorded research metrics are unavailable because the results "
                "file has an unexpected format."
            )
        else:
            metric_rows = []
            for _, row in metrics.iterrows():
                cells = "".join(
                    f"<td>{float(row[column]):.4f}</td>"
                    for column in required_columns[1:]
                )
                metric_rows.append(
                    f"<tr><th scope=\"row\">{escape(str(row['Model']))}</th>{cells}</tr>"
                )
            st.markdown(
                f"""
                <div class="about-metrics-wrap">
                    <table class="about-metrics-table">
                        <caption>Saved model comparison · held-out evaluation</caption>
                        <thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">ROC-AUC</th></tr></thead>
                        <tbody>{''.join(metric_rows)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_screening_disclaimer()

def main() -> None:
    st.set_page_config(
        page_title="JobSignal | Job-posting analysis",
        page_icon="🔎",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root {
            --ink: #263B50;
            --muted: #4b6075;
            --panel: #eaf0f5;
            --panel-raised: #CEC5E8;
            --line: #b8c7d8;
            --navy: #263B50;
            --accent: #7D70A8;
            --accent-soft: #CEC5E8;
            --accent-hover: #b8afd7;
            --focus: #7D70A8;
            --page-bg: #DCEAF2;
            --input-bg: #ffffff;
            --auth-surface: #ffffff;
            --teal: #CEC5E8;
            --teal-soft: #CEC5E8;
            --peach: #E8E4F5;
            --ivory: #eaf0f5;
            --radius-control: 10px;
            --radius-card: 14px;
            --shadow-card: 0 8px 22px rgba(36, 52, 71, 0.05);
            --space-sm: 0.5rem;
            --space-md: 0.75rem;
            --space-lg: 1rem;
        }
        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            color: var(--ink);
            background: var(--page-bg);
        }
        [data-testid="stHeader"] { background: var(--page-bg); }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 1260px; padding: 1.25rem 2.25rem 2.5rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.025em; }
        p, [data-testid="stCaptionContainer"] { color: var(--muted); }
        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.35rem 0 0.8rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.65rem;
        }
        .brand-lockup { display: flex; align-items: center; gap: 0.65rem; color: var(--navy); font-weight: 780; letter-spacing: -0.02em; }
        .brand-mark {
            display: grid;
            width: 2rem;
            height: 2rem;
            place-items: center;
            border-radius: 0.65rem;
            background: linear-gradient(145deg, #173654, #087f83);
            color: #fff;
            font-size: 0.9rem;
            box-shadow: 0 5px 14px rgba(8, 127, 131, 0.18);
        }
        .nav-links { display: flex; align-items: center; gap: 1.25rem; }
        .nav-links a { color: #50657b; font-size: 0.86rem; font-weight: 650; text-decoration: none; }
        .nav-links a:hover, .nav-links a:focus-visible { color: var(--teal); text-decoration: underline; text-underline-offset: 4px; }
        .hero {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.8fr);
            align-items: center;
            gap: 2rem;
            padding: 0.8rem 0 1rem;
        }
        .hero-eyebrow, .section-eyebrow {
            color: var(--teal);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0.65rem 0 0.7rem;
            color: var(--ink);
            font-size: clamp(2rem, 3.5vw, 2.7rem);
            line-height: 1.08;
            font-weight: 780;
        }
        .hero-subtitle { max-width: 650px; margin: 0; font-size: 1rem; line-height: 1.7; }
        .hero-tags { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.2rem; }
        .home-hero, .auth-intro {
            border: 1px solid #d4e4eb;
            border-radius: 18px;
            background: linear-gradient(120deg, #ffffff 0%, #edf7f7 100%);
            padding: clamp(1.35rem, 3vw, 2.5rem);
            margin: 0.8rem 0 1.35rem;
        }
        .auth-brand { display: flex; align-items: center; gap: 0.7rem; color: var(--navy); font-size: 1.1rem; font-weight: 780; }
        .auth-intro { margin-top: 1.5rem; }
        .auth-benefits { padding: 1rem 0 0.5rem; }
        .auth-benefits h2 { font-size: 1.65rem; }
        .account-label { color: var(--muted); font-size: 0.76rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
        .page-title { margin: 0.55rem 0 0.45rem; }
        .page-subtitle { color: var(--muted); max-width: 760px; line-height: 1.65; }
        .hero-tag {
            border: 1px solid #c8e8e4;
            border-radius: 999px;
            background: #edf8f7;
            color: #17696c;
            padding: 0.38rem 0.72rem;
            font-size: 0.7rem;
            font-weight: 650;
            letter-spacing: 0.035em;
        }
        .section-heading { margin: 1.7rem 0 1rem; scroll-margin-top: 1.5rem; }
        .section-heading h2 { margin: 0.35rem 0 0.25rem; font-size: 1.6rem; }
        .section-heading p { margin: 0; font-size: 0.91rem; line-height: 1.6; }
        div[data-testid="stForm"] {
            border: 1px solid #dce6ee;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.94);
            padding: 1.35rem 1.45rem 1.15rem;
            box-shadow: 0 12px 30px rgba(23, 54, 84, 0.055);
        }
        div[data-testid="stForm"] h3 { margin-top: 0; }
        label, div[data-testid="stTextArea"] label { color: #29445f; font-weight: 650; }
        [data-testid="stWidgetLabel"] p { color: #29445f; }
        input, textarea, [data-baseweb="select"] > div {
            border-radius: 10px !important;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 270px;
            border: 1px solid #cfdae4;
            border-radius: 11px;
            background: #fbfdff;
            color: var(--ink);
            line-height: 1.65;
            padding: 0.95rem 1rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--teal);
            background: #fff;
            box-shadow: 0 0 0 3px rgba(8, 127, 131, 0.13);
        }
        div[data-testid="stTextArea"] textarea::placeholder { color: #73869a; }
        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 3.15rem;
            border: 1px solid #087f83;
            border-radius: var(--radius-control);
            background: linear-gradient(105deg, #173654, #087f83);
            color: white;
            font-weight: 750;
            box-shadow: 0 8px 18px rgba(8, 94, 112, 0.16);
            transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px);
            filter: brightness(1.08);
            box-shadow: 0 11px 24px rgba(8, 94, 112, 0.23);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line);
            border-radius: 14px;
            background: #ffffff;
        }
        .results-heading {
            margin: 1.5rem 0 0.8rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
            scroll-margin-top: 1rem;
        }
        .results-heading h2 { margin: 0.25rem 0; font-size: 1.5rem; }
        .results-heading p { margin: 0; font-size: 0.9rem; }
        .history-empty {
            border: 1px dashed #cbd8e2;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            background: rgba(255,255,255,0.68);
            color: #52667a;
        }
        .result-card {
            border: 1px solid;
            border-radius: 15px;
            padding: 1.3rem 1.4rem;
            min-height: 126px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 10px 25px rgba(23, 54, 84, 0.05);
        }
        .result-card.legitimate { border-color: #a9dcd5; background: linear-gradient(120deg, #f0faf8, #ffffff); }
        .result-card.fraudulent { border-color: #efc0b4; background: linear-gradient(120deg, #fff5f1, #ffffff); }
        .result-card.insufficient { border-color: var(--line); background: var(--accent-soft); }
        .risk-indicator {
            margin-top: 0.8rem;
            border: 1px solid var(--line);
            border-radius: 11px;
            padding: 0.78rem 0.95rem;
            font-size: 0.85rem;
            font-weight: 700;
        }
        .risk-low { color: #176b5c; border-color: #b6ded5; background: #edf8f4; }
        .risk-medium { color: var(--navy); border-color: var(--line); background: var(--accent-soft); }
        .risk-high { color: #8c3d2c; border-color: #edc2b8; background: #fff2ee; }
        .risk-insufficient { color: #4e5e76; border-color: #cbd5e1; background: #f1f5f9; }
        .result-kicker { color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
        .result-title { margin-top: 0.42rem; font-size: 1.4rem; line-height: 1.25; font-weight: 760; }
        .legitimate .result-title { color: #176b5c; }
        .fraudulent .result-title { color: #8c3d2c; }
        .insufficient .result-title { color: var(--navy); }
        .stButton > button {
            min-height: 2.65rem;
            border: 1px solid #cfdae4;
            border-radius: var(--radius-control);
            background: #ffffff;
            color: var(--navy);
            font-weight: 650;
            transition: border-color 150ms ease, color 150ms ease, background 150ms ease;
        }
        .stButton > button:hover, .stButton > button:focus-visible {
            border-color: var(--teal);
            color: #075d60;
            background: #f1faf9;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #ffffff;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 22px rgba(23, 54, 84, 0.045);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; }
        [data-testid="stSpinner"] { color: var(--teal); }
        :focus-visible { outline: 3px solid rgba(8, 127, 131, 0.45); outline-offset: 2px; }
        hr { border-color: var(--line); }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
        }
        @media (max-width: 850px) {
            .block-container { padding: 1rem 1.15rem 2.25rem; }
            .top-nav { padding-bottom: 0.9rem; }
            .nav-links { gap: 0.9rem; }
            .hero { grid-template-columns: 1fr; gap: 1rem; padding-top: 0.35rem; }
            .hero-title { font-size: clamp(2rem, 8vw, 2.6rem); }
            .home-hero, .auth-intro { padding: 1.25rem; }
            div[data-testid="stForm"] { padding: 1.05rem 0.95rem 0.85rem; }
            div[data-testid="stTextArea"] textarea { min-height: 235px; }
        }
        @media (max-width: 520px) {
            .block-container { padding-right: 0.85rem; padding-left: 0.85rem; }
            .top-nav { align-items: center; flex-wrap: wrap; gap: 0.6rem; }
            .nav-links { width: 100%; justify-content: space-between; gap: 0.5rem; padding-top: 0; }
            .nav-links a { font-size: 0.78rem; }
            .section-heading h2 { font-size: 1.4rem; }
        }
        html, body, [class*="css"] {
            font-family: Inter, "Aptos", "Segoe UI", sans-serif;
            color: var(--ink);
        }
        .stApp, [data-testid="stAppViewContainer"] {
            color: var(--ink);
            background: var(--page-bg);
        }
        [data-testid="stHeader"] { background: var(--page-bg); }
        .block-container { max-width: 1320px; padding: 1.5rem 2.25rem 3rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.03em; }
        p, [data-testid="stCaptionContainer"] { color: var(--muted); }
        .brand-lockup, .auth-brand {
            color: var(--navy);
            font-weight: 760;
            letter-spacing: -0.025em;
        }
        .brand-mark {
            background: linear-gradient(145deg, #204c3d, #17392f);
            box-shadow: 0 6px 16px rgba(23, 61, 49, 0.2);
        }
        .hero-eyebrow, .section-eyebrow, .form-eyebrow {
            color: var(--accent);
            font-weight: 750;
            letter-spacing: 0.12em;
        }
        .home-hero {
            border-color: var(--line);
            background: var(--panel);
            box-shadow: 0 12px 34px rgba(44, 54, 44, 0.045);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line);
            border-radius: 16px;
            background: var(--panel);
            box-shadow: 0 12px 30px rgba(38, 49, 39, 0.055);
        }
        div[data-testid="stForm"] {
            border-color: var(--line);
            border-radius: 14px;
            background: var(--input-bg);
            box-shadow: none;
        }
        label, div[data-testid="stTextArea"] label,
        [data-testid="stWidgetLabel"] p { color: var(--ink); font-weight: 650; }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
            background: var(--input-bg) !important;
            color: var(--ink) !important;
            box-shadow: none;
        }
        div[data-testid="stTextInput"] input {
            border: 0 !important;
            border-radius: inherit !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            min-height: 2.8rem;
            padding: 0.72rem 0.85rem;
        }
        div[data-testid="stTextInput"] input[type="password"] {
            padding-right: 3.25rem !important;
        }
        div[data-testid="stTextArea"] textarea { min-height: 270px; line-height: 1.65; }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder,
        input::placeholder, textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1;
        }
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInputRootElement"]:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(125, 112, 168, 0.18) !important;
            background: #ffffff !important;
        }
        div[data-testid="stCheckbox"] label { color: #47574c; }
        .form-eyebrow {
            margin-top: 0.4rem;
            font-size: 0.68rem;
            text-transform: uppercase;
        }
        .auth-form-title {
            margin: 0.4rem 0 0.3rem;
            font-size: clamp(1.55rem, 2.4vw, 2rem);
        }
        .auth-form-copy { margin: 0 0 1rem; font-size: 0.92rem; line-height: 1.55; }
        .auth-panel {
            position: relative;
            display: flex;
            min-height: 635px;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            border: 1px solid rgba(235, 226, 199, 0.26);
            border-radius: 22px;
            padding: clamp(1.7rem, 4vw, 3rem);
            color: #f7f3e8;
            background:
                radial-gradient(ellipse at 84% 14%, rgba(194, 166, 104, 0.17), transparent 18rem),
                radial-gradient(ellipse at 12% 100%, rgba(89, 126, 101, 0.2), transparent 22rem),
                linear-gradient(145deg, #1b4335 0%, #15362d 56%, #102d25 100%);
            box-shadow: 0 24px 55px rgba(22, 48, 38, 0.18);
        }
        .auth-panel::after {
            position: absolute;
            right: -6rem;
            bottom: 6rem;
            width: 17rem;
            height: 17rem;
            border: 1px solid rgba(224, 201, 145, 0.16);
            border-radius: 50%;
            box-shadow: 0 0 0 2.1rem rgba(224, 201, 145, 0.035), 0 0 0 4.2rem rgba(224, 201, 145, 0.025);
            content: "";
            pointer-events: none;
        }
        .auth-panel .auth-brand { color: #fffdf6; font-size: 1.16rem; }
        .auth-panel .brand-mark {
            border: 1px solid rgba(231, 211, 164, 0.48);
            background: var(--accent-soft);
            color: #17392f;
            box-shadow: none;
        }
        .auth-panel-content { position: relative; z-index: 1; margin: 2.3rem 0; }
        .auth-kicker {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--accent-soft);
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.15em;
        }
        .auth-kicker > span, .auth-footer-dot {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: var(--accent);
        }
        .auth-panel h1 {
            max-width: 540px;
            margin: 1.2rem 0 0.9rem;
            color: #fffdf6;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2.35rem, 4vw, 3.5rem);
            font-weight: 500;
            letter-spacing: -0.045em;
            line-height: 1.05;
        }
        .auth-panel-copy { max-width: 430px; color: #d5ded5; font-size: 1rem; line-height: 1.7; }
        .auth-feature-list { display: grid; gap: 1.1rem; margin-top: 2rem; }
        .auth-feature { display: flex; align-items: flex-start; gap: 0.9rem; }
        .auth-feature-icon {
            display: grid;
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            place-items: center;
            border: 1px solid rgba(218, 196, 143, 0.4);
            border-radius: 0.65rem;
            color: #dbc68f;
            font-size: 0.68rem;
            font-weight: 700;
        }
        .auth-feature strong { color: #f8f4e9; font-size: 0.92rem; }
        .auth-feature p { margin: 0.2rem 0 0; color: #bdcbbf; font-size: 0.8rem; line-height: 1.5; }
        .auth-panel-footer {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: #c0cbbf;
            font-size: 0.78rem;
        }
        .auth-columns { align-items: center; }
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            min-height: 3rem;
            border: 1px solid #173d31;
            border-radius: 10px;
            background: linear-gradient(115deg, #1c4938, #15382e);
            color: #fffdf6;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(23, 61, 49, 0.15);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover {
            transform: translateY(-1px);
            border-color: var(--accent);
            background: linear-gradient(115deg, #245440, #193f33);
            box-shadow: 0 11px 22px rgba(23, 61, 49, 0.21);
        }
        div[data-testid="stFormSubmitButton"] button:active,
        div[data-testid="stButton"] button:active { transform: translateY(0); }
        div[data-testid="stButton"] button {
            min-height: 2.65rem;
            border-color: var(--line);
            border-radius: 10px;
            background: var(--input-bg);
            color: var(--ink);
            font-weight: 650;
            transition: border-color 150ms ease, color 150ms ease, background 150ms ease;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--accent);
            color: var(--navy);
            background: var(--accent-soft);
        }
        div[data-testid="stButton"] button:disabled,
        div[data-testid="stFormSubmitButton"] button:disabled {
            opacity: 0.56;
            box-shadow: none;
            cursor: not-allowed;
        }
        .top-nav { border-color: var(--line); }
        div[data-testid="stMetric"] {
            border-color: var(--line);
            border-radius: 14px;
            background: var(--panel);
            box-shadow: 0 8px 22px rgba(38, 49, 39, 0.05);
        }
        .result-card.legitimate { border-color: var(--line); background: var(--panel); }
        .result-card.fraudulent { border-color: var(--line); background: var(--accent-soft); }
        .result-card.insufficient { border-color: var(--line); background: var(--accent-soft); }
        .risk-low { color: #315e44; border-color: #b8ceb9; background: #eef5ec; }
        .risk-medium { color: var(--navy); border-color: var(--line); background: var(--accent-soft); }
        .risk-high { color: #823f33; border-color: #e8c0b6; background: #fff1ed; }
        .risk-insufficient { color: #58665e; border-color: #d4d7cb; background: #f2f3ec; }
        [data-testid="stAlert"] { border-radius: 11px; }
        [data-testid="stExpander"] { border-color: var(--line); border-radius: 12px; background: var(--panel); }
        :focus-visible { outline: 3px solid rgba(125, 112, 168, 0.48); outline-offset: 2px; }
        hr { border-color: var(--line); }
        .verification-badge { display: inline-flex; align-items: center; width: fit-content; padding: 0.32rem 0.65rem; border: 1px solid; border-radius: 999px; font-size: 0.73rem; font-weight: 760; letter-spacing: 0.025em; }
        .verification-found { border-color: #b8ceb9; color: #315e44; background: #eef5ec; }
        .verification-not-found { border-color: var(--line); color: var(--navy); background: var(--accent-soft); }
        .verification-unable { border-color: #d4d7cb; color: #58665e; background: #f2f3ec; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) { height: 100%; border-color: var(--line); background: var(--input-bg); }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) h4 { margin-bottom: 0.55rem; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) [data-testid="stCaptionContainer"] p { font-size: 0.72rem; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) a { color: #234a39; }
        @media (max-width: 1050px) {
            .block-container { padding: 1.1rem 1.35rem 2.5rem; }
            .auth-panel { min-height: 600px; padding: 1.7rem; }
            .auth-panel h1 { font-size: clamp(2.1rem, 4vw, 2.8rem); }
            .auth-feature-list { gap: 0.9rem; margin-top: 1.5rem; }
        }
        @media (max-width: 760px) {
            .block-container { padding: 0.9rem 1rem 2rem; }
            .auth-panel { min-height: auto; padding: 1.5rem; }
            .auth-panel-content { margin: 1.5rem 0; }
            .auth-panel h1 { max-width: none; font-size: clamp(2rem, 7vw, 2.65rem); }
            .auth-feature-list { grid-template-columns: 1fr; }
            .st-key-auth_form_card { padding: 1.2rem; }
            .st-key-auth_layout > [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            .st-key-auth_layout > [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            .about-hero { grid-template-columns: 1fr; gap: 1.25rem; }
        }
        @media (max-width: 520px) {
            .block-container { padding-right: 0.75rem; padding-left: 0.75rem; }
            .auth-panel { border-radius: 16px; padding: 1.25rem; }
            .auth-panel-content { margin: 1.15rem 0; }
            .auth-panel h1 { font-size: 2.05rem; }
            .auth-feature-list { gap: 0.8rem; }
            .auth-feature p { font-size: 0.76rem; }
            .auth-form-title { font-size: 1.55rem; }
            .st-key-auth_form_card { padding: 1rem; }
            .about-hero { padding: 1.35rem; border-radius: 16px; }
            .about-section { margin: 1.7rem 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
        }
        .stApp, [data-testid="stAppViewContainer"] {
            color: var(--ink);
            background: var(--page-bg);
        }
        [data-testid="stHeader"] { background: var(--page-bg); }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup) {
            align-items: center;
            padding: 0.25rem 0 0.75rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
            background: transparent !important;
        }
        h1, h2, h3, h4, p, label, [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"] p { color: var(--ink); }
        .brand-lockup, .auth-brand, .nav-links a, a,
        .hero-eyebrow, .section-eyebrow, .form-eyebrow,
        .about-eyebrow {
            color: var(--ink);
        }
        .nav-links a:hover, .nav-links a:focus-visible, a:hover, a:focus-visible {
            color: #34586b;
        }
        .brand-mark, .auth-panel .brand-mark {
            display: grid;
            width: 2.5rem;
            height: 2.5rem;
            flex: 0 0 2.5rem;
            place-items: center;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }
        .brand-mark svg { display: block; width: 100%; height: 100%; }
        .brand-lockup, .auth-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .home-hero, .auth-intro, .about-hero {
            border-color: var(--line);
            color: var(--ink);
            background: linear-gradient(125deg, var(--panel), var(--page-bg));
            box-shadow: 0 12px 30px rgba(36, 52, 71, 0.06);
        }
        .auth-panel h1, .auth-panel .auth-brand, .auth-panel-copy,
        .auth-panel-footer, .auth-feature strong, .auth-feature p,
        .auth-kicker, .auth-feature-icon, .about-hero h1,
        .about-hero-copy > p {
            color: var(--ink);
        }
        .auth-panel {
            border-color: var(--line);
            color: var(--ink);
            background: linear-gradient(
                135deg,
                var(--peach),
                var(--page-bg) 62%,
                var(--accent-soft)
            );
            box-shadow: 0 18px 42px rgba(36, 52, 71, 0.08);
        }
        .st-key-auth_layout > [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        .st-key-auth_form_card {
            box-sizing: border-box;
            height: 100%;
            padding: clamp(1.2rem, 2.3vw, 2rem);
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--auth-surface);
            box-shadow: var(--shadow-card);
        }
        .st-key-auth_form_card [data-testid="stForm"] {
            margin-top: var(--space-md);
            padding: 0;
            border: 0;
            background: transparent;
            box-shadow: none;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stForm"]),
        div[data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-posting_controls [data-testid="stVerticalBlockBorderWrapper"] {
            margin: 0;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .st-key-auth_form_card [data-testid="stWidgetLabel"] {
            margin-bottom: 0.3rem;
        }
        .st-key-auth_form_card [data-testid="stCheckbox"] {
            margin-top: var(--space-sm);
        }
        .auth-switch-prompt {
            margin: var(--space-lg) 0 var(--space-sm);
            color: var(--muted);
            font-size: 0.86rem;
        }
        div.st-key-auth_switch_signup button,
        div.st-key-auth_switch_signin button {
            width: auto !important;
            min-height: 2rem;
            padding: 0.1rem 0 !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            color: var(--accent) !important;
            font-weight: 700;
        }
        div.st-key-auth_switch_signup button p,
        div.st-key-auth_switch_signin button p {
            color: inherit !important;
        }
        div.st-key-auth_switch_signup button:hover,
        div.st-key-auth_switch_signin button:hover {
            background: transparent !important;
            color: var(--navy) !important;
            text-decoration: underline;
            text-underline-offset: 3px;
        }
        .auth-panel::after { border-color: rgba(36, 52, 71, 0.12); }
        .auth-kicker > span, .auth-footer-dot { background: var(--ink); }
        .auth-feature-icon {
            border-color: var(--line);
            background: rgba(234, 240, 245, 0.9);
        }
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stForm"],
        div[data-testid="stMetric"],
        [data-testid="stExpander"] {
            border-color: var(--line);
            background: var(--ivory);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-home_about) {
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        label, div[data-testid="stTextArea"] label,
        [data-testid="stWidgetLabel"] p,
        div[data-testid="stCheckbox"] label { color: var(--ink); }
        div[data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
            border-color: var(--line) !important;
            background: var(--input-bg) !important;
            color: var(--ink) !important;
        }
        div[data-testid="stTextInput"] input {
            border: 0 !important;
            background: transparent !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextInputRootElement"] {
            min-height: 2.8rem;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-control) !important;
            background: var(--input-bg) !important;
            box-shadow: none !important;
        }
        [role="listbox"], [role="option"], [data-baseweb="popover"] {
            border-color: var(--line) !important;
            background: #eaf0f5 !important;
            color: var(--ink) !important;
        }
        [role="dialog"], [data-testid="stPopover"] {
            border-color: var(--line) !important;
            background: #eaf0f5 !important;
            color: var(--ink) !important;
        }
        [role="option"]:hover, [role="option"][aria-selected="true"] {
            background: #CEC5E8 !important;
            color: var(--ink) !important;
        }
        input::placeholder, textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1;
        }
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInputRootElement"]:focus-within {
            border-color: var(--focus) !important;
            box-shadow: 0 0 0 3px rgba(125, 112, 168, 0.24) !important;
        }
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            border-color: #9bbfc6;
            background: var(--accent-soft) !important;
            color: var(--ink) !important;
            box-shadow: 0 6px 16px rgba(36, 52, 71, 0.1);
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover {
            border-color: #b8afd7 !important;
            background: var(--peach) !important;
            color: var(--ink) !important;
            box-shadow: 0 9px 20px rgba(36, 52, 71, 0.13);
        }
        div[data-testid="stButton"] button {
            border-color: var(--line);
            background: var(--input-bg) !important;
            color: var(--ink) !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #b8afd7;
            background: var(--peach) !important;
            color: var(--ink) !important;
        }
        div[class*="st-key-nav_link_"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-nav_link_"]) {
            padding: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button {
            position: relative;
            width: auto;
            min-height: 2.5rem;
            padding: 0.35rem 0.6rem 0.5rem;
            border: 1px solid transparent !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: #263B50 !important;
            box-shadow: none !important;
            font-weight: 600;
            white-space: nowrap;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button p {
            color: inherit !important;
            font-size: 0.87rem;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button:hover {
            border-color: transparent !important;
            background: transparent !important;
            color: #716397 !important;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button[kind="primary"] {
            color: #263B50 !important;
            font-weight: 700;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button[kind="primary"]::after {
            position: absolute;
            right: 0.6rem;
            bottom: 0.22rem;
            left: 0.6rem;
            height: 2px;
            border-radius: 2px;
            background: #9183bd;
            content: "";
        }
        div[data-testid="stForm"] {
            border: 0;
            border-radius: 0;
            padding: 0.65rem 0 0;
            background: transparent;
            box-shadow: none;
        }
        div[data-testid="stForm"] [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }
        div[data-testid="stTextArea"] textarea { min-height: 250px; }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 2.8rem;
        }
        div.st-key-clear_job_input button {
            min-height: 2.2rem !important;
            padding: 0.2rem 0.35rem !important;
            font-size: 0.78rem;
            white-space: nowrap !important;
        }
        .hero-tag {
            border-color: var(--line);
            background: var(--teal);
            color: var(--ink);
        }
        .history-empty {
            border-color: var(--line);
            background: #eaf0f5;
            color: var(--ink);
        }
        .result-card.legitimate, .result-card.insufficient {
            border-color: var(--line);
            background: var(--teal);
        }
        .result-card.fraudulent { border-color: #e6bfa8; background: var(--peach); }
        .result-title, .legitimate .result-title, .fraudulent .result-title,
        .insufficient .result-title { color: var(--ink); }
        .risk-low, .risk-medium, .risk-insufficient,
        .verification-found, .verification-not-found, .verification-unable {
            border-color: var(--line);
            background: var(--teal);
            color: var(--ink);
        }
        .risk-high { border-color: #e6bfa8; background: var(--peach); color: var(--ink); }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) {
            border-color: var(--line);
            background: var(--ivory);
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) a {
            color: var(--ink);
        }
        div.st-key-profile_menu {
            display: flex;
            justify-content: flex-end;
        }
        div.st-key-profile_menu button {
            max-width: 100%;
            border: 1px solid transparent !important;
            border-radius: 999px;
            background: transparent !important;
            color: var(--ink) !important;
            font-weight: 650;
        }
        div.st-key-profile_menu button p,
        div.st-key-profile_menu button span,
        div.st-key-profile_menu button svg {
            color: var(--ink) !important;
            fill: currentColor;
        }
        div.st-key-profile_menu button:hover {
            border-color: #7d70a8 !important;
            background: #CEC5E8 !important;
            color: var(--ink) !important;
        }
        div.st-key-profile_menu,
        div.st-key-profile_menu [data-testid="stPopover"],
        div.st-key-profile_menu [data-testid="stVerticalBlockBorderWrapper"] {
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        @media (max-width: 760px) {
            div.st-key-profile_menu { justify-content: flex-start; }
            div.st-key-profile_menu button {
                min-height: 2.45rem;
            }
        }
        :focus-visible { outline-color: #7d70a8; }
        [data-testid="stCaptionContainer"] p {
            color: #4b6075;
            line-height: 1.55;
        }
        .summary-card {
            box-sizing: border-box;
            min-height: 205px;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            border: 1px solid #b8c7d8;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            background: #eaf0f5;
        }
        .summary-card h3 {
            margin: 0;
            color: #263B50;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .assessment-card { justify-content: center; }
        .assessment-card.legitimate {
            border-color: #b8d4c2;
            border-left: 4px solid #78a98a;
            background: #eef6f0;
        }
        .assessment-card.fraudulent {
            border-color: #dfb8b8;
            border-left: 4px solid #bf7777;
            background: #f9eeee;
        }
        .assessment-card.insufficient {
            border-color: #c8c1dd;
            border-left: 4px solid #9183bd;
            background: #f1eff8;
        }
        .assessment-status {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
        }
        .assessment-icon {
            display: inline-flex;
            width: 1.4rem;
            height: 1.4rem;
            flex: 0 0 1.4rem;
            margin-top: 0.1rem;
        }
        .assessment-icon svg { display: block; width: 100%; height: 100%; }
        .assessment-card.legitimate .assessment-icon { color: #49795b; }
        .assessment-card.fraudulent .assessment-icon { color: #995858; }
        .assessment-card.insufficient .assessment-icon { color: #716397; }
        .assessment-title {
            color: #263B50;
            font-size: 1.2rem;
            font-weight: 740;
            line-height: 1.3;
        }
        .assessment-copy {
            color: #4b6075;
            font-size: 0.86rem;
            line-height: 1.55;
        }
        .coverage-copy {
            margin: 0;
            color: #4b6075;
            font-size: 0.8rem;
            line-height: 1.5;
        }
        .summary-value {
            color: #263B50;
            font-size: 1.8rem;
            font-weight: 740;
            line-height: 1.15;
        }
        .coverage-track {
            height: 0.55rem;
            overflow: hidden;
            border-radius: 999px;
            background: #d6dfeb;
        }
        .coverage-track span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: #9183bd;
        }
        .warning-card .risk-indicator { margin: 0; }
        .warning-card p { margin: 0; font-size: 0.82rem; line-height: 1.5; }
        .warning-card .warning-score { font-weight: 680; color: #263B50; }
        .positive-indicator, .warning-indicator {
            margin: 0.45rem 0 0.75rem;
            border-left: 3px solid;
            border-radius: 0 9px 9px 0;
            padding: 0.65rem 0.85rem;
        }
        .positive-indicator {
            border-color: #78a98a;
            background: #f2f7f2;
        }
        .warning-medium {
            border-color: var(--line);
            background: var(--accent-soft);
        }
        .warning-high {
            border-color: #bf7777;
            background: #faf1f0;
        }
        .positive-indicator > strong, .warning-indicator > strong {
            display: inline-block;
            margin-right: 0.45rem;
            color: #263B50;
        }
        .positive-indicator p, .warning-indicator p {
            margin: 0.35rem 0 0;
            color: #3c5167;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .warning-indicator .indicator-level {
            color: #53657b;
            font-size: 0.75rem;
            font-weight: 650;
        }
        .positive-indicator .indicator-evidence,
        .warning-indicator .indicator-evidence {
            color: #4b6075;
            font-size: 0.8rem;
        }
        .risk-indicator {
            display: inline-flex;
            width: fit-content;
            border: 1px solid;
            border-radius: 999px;
            padding: 0.3rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .risk-low { border-color: #b8d4c2; background: #eef6f0; color: #284d39; }
        .risk-medium { border-color: var(--line); background: var(--accent-soft); color: var(--navy); }
        .risk-high { border-color: #dfb8b8; background: #f9eeee; color: #733d3d; }
        .warning-card:has(.risk-low) { border-left: 4px solid #78a98a; }
        .warning-card:has(.risk-medium) { border-left: 4px solid var(--accent); }
        .warning-card:has(.risk-high) { border-left: 4px solid #bf7777; }
        .results-disclaimer {
            margin: 0.2rem 0 0;
            color: #4b6075;
            font-size: 0.78rem;
            line-height: 1.5;
        }
        .results-guidance {
            margin: 0.75rem 0 0;
            color: #64758a;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        @media (max-width: 760px) {
            [data-testid="stHorizontalBlock"]:has(.assessment-card) {
                flex-wrap: wrap;
            }
            [data-testid="stHorizontalBlock"]:has(.assessment-card) > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            .summary-card { min-height: 0; }
            div[class*="st-key-nav_link_"] { justify-content: flex-start; }
            div[class*="st-key-nav_link_"] [data-testid="stButton"] button {
                padding-right: 0.35rem;
                padding-left: 0.35rem;
            }
        }
        .block-container {
            max-width: 1200px;
            padding: 1.1rem clamp(1rem, 3vw, 2.25rem) 2.75rem;
        }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup) {
            gap: clamp(0.4rem, 1.5vw, 1.25rem);
            align-items: center;
        }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup) > [data-testid="stColumn"] {
            min-width: 0;
        }
        .brand-lockup { gap: 0.7rem; }
        div.st-key-profile_menu {
            justify-content: flex-end;
            min-width: 0;
        }
        div.st-key-profile_menu button {
            width: auto !important;
            max-width: 13rem;
            min-height: 2.5rem;
            overflow: hidden;
            padding: 0.4rem 0.75rem;
        }
        div.st-key-profile_menu button p {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .section-heading#analysis {
            max-width: 900px;
            margin: 0.65rem auto 1.15rem;
            scroll-margin-top: 1rem;
        }
        .section-heading#analysis .section-eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.12em;
        }
        .section-heading#analysis .page-title {
            margin: 0.45rem 0 0.4rem;
            font-size: clamp(2rem, 4vw, 2.65rem);
            font-weight: 760;
            line-height: 1.12;
        }
        .section-heading#analysis .page-subtitle {
            max-width: 680px;
            margin: 0;
            line-height: 1.6;
        }
        .st-key-posting_controls [data-testid="stHorizontalBlock"] {
            align-items: end;
        }
        .st-key-posting_controls > [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        div[data-testid="stRadio"] [role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }
        div[data-testid="stRadio"] input[type="radio"] {
            accent-color: var(--accent);
        }
        div[data-testid="stRadio"] [role="radiogroup"] label {
            box-sizing: border-box;
            min-height: 3rem;
            margin: 0;
            padding: 0.6rem 0.8rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-control);
            background: var(--panel);
            transition: border-color 150ms ease, background 150ms ease;
        }
        div[data-testid="stRadio"] [role="radiogroup"] label:has(
            input[type="radio"]:checked,
            [role="radio"][aria-checked="true"]
        ) {
            border-color: var(--accent);
            background: var(--accent-soft);
        }
        div[data-testid="stRadio"] [role="radiogroup"] label:hover {
            border-color: var(--accent-hover);
        }
        div[data-testid="stForm"]:has([data-testid="stTextArea"]) {
            margin-top: 0.3rem;
            padding: 0.2rem 0 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }
        div[data-testid="stForm"]:has([data-testid="stTextArea"])
        [data-testid="stVerticalBlock"] {
            gap: 0.95rem;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 250px;
            padding: 0.85rem 0.95rem !important;
            border-radius: var(--radius-control) !important;
            line-height: 1.6;
        }
        div[data-testid="stForm"]:has([data-testid="stTextArea"])
        [data-testid="stWidgetLabel"] {
            margin-bottom: 0.35rem;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border-radius: 10px;
            padding: 0.65rem 1.1rem;
            font-weight: 700;
        }
        div.st-key-clear_job_input button {
            min-height: 2.55rem !important;
            padding: 0.45rem 0.65rem !important;
            border: 1px solid transparent !important;
            border-radius: var(--radius-control);
            background: transparent !important;
            box-shadow: none !important;
            color: var(--accent) !important;
        }
        div.st-key-clear_job_input button:hover {
            border-color: var(--accent-hover) !important;
            background: var(--accent-soft) !important;
            color: var(--navy) !important;
        }
        .summary-card {
            min-height: 0;
            height: 100%;
            gap: 0.8rem;
            padding: 1.1rem 1.15rem;
            border-radius: 14px;
        }
        .results-heading {
            margin-top: 1.75rem;
            padding-top: 1.15rem;
        }
        .positive-indicator, .warning-indicator {
            margin: 0.55rem 0 0.85rem;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.verification-badge) {
            height: 100%;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(36, 52, 71, 0.045);
        }
        @media (max-width: 760px) {
            .block-container { padding: 0.9rem 1rem 2rem; }
            .st-key-posting_controls [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            .st-key-posting_controls [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"]:first-child {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            .st-key-posting_controls [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"]:last-child {
                flex: 0 0 auto !important;
                width: auto !important;
                margin-left: auto;
            }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup) {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 0.25rem 0.75rem;
            }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup)
            > [data-testid="stColumn"]:first-child {
                grid-column: 1;
                grid-row: 1;
                width: auto !important;
            }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup)
            > [data-testid="stColumn"]:nth-child(2) {
                grid-column: 1 / -1;
                grid-row: 2;
                width: 100% !important;
            }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup)
            > [data-testid="stColumn"]:nth-child(3) {
                grid-column: 2;
                grid-row: 1;
                width: auto !important;
            }
            div.st-key-profile_menu { justify-content: flex-end; }
            .section-heading#analysis { margin-bottom: 0.9rem; }
            div[data-testid="stForm"]:has([data-testid="stTextArea"]) {
                padding: 0.85rem 0.75rem 0.7rem;
            }
            div[data-testid="stTextArea"] textarea { min-height: 225px; }
        }
        @media (max-width: 520px) {
            .block-container { padding-right: 0.75rem; padding-left: 0.75rem; }
            .st-key-posting_controls [data-testid="stHorizontalBlock"] {
                row-gap: 0.35rem;
            }
            [data-testid="stRadio"] [role="radiogroup"] {
                grid-template-columns: 1fr;
                gap: 0.45rem;
            }
            .section-heading#analysis .page-title { font-size: 2rem; }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup)
            > [data-testid="stColumn"]:nth-child(3) {
                justify-self: end;
            }
            div[class*="st-key-nav_link_"] [data-testid="stButton"] button {
                padding-right: 0.25rem;
                padding-left: 0.25rem;
                white-space: normal;
            }
        }
        html { scroll-behavior: smooth; }
        .home-page { max-width: 1120px; margin: 0 auto; }
        .home-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(250px, 0.8fr);
            align-items: center;
            gap: clamp(1.25rem, 4vw, 3.25rem);
            padding: clamp(1.4rem, 3.5vw, 2.5rem);
        }
        .home-hero-copy { max-width: 650px; }
        .home-hero .hero-eyebrow, .home-eyebrow {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .home-hero .hero-title {
            margin: 0.6rem 0 0.75rem;
            color: var(--navy);
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.08;
        }
        .home-hero .hero-subtitle {
            max-width: 620px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
        }
        .home-visual {
            display: grid;
            gap: 0.25rem;
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--panel);
            box-shadow: var(--shadow-card);
        }
        .home-visual h2 {
            margin: 0 0 0.4rem;
            color: var(--navy);
            font-size: 1.1rem;
            letter-spacing: -0.015em;
        }
        .home-visual-row, .home-history-title {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .home-visual-row {
            min-width: 0;
            padding: 0.5rem 0;
        }
        .home-visual-row > span:last-child { display: grid; gap: 0.12rem; }
        .home-visual-row strong, .home-history-title h3 { color: var(--navy); font-size: 0.86rem; }
        .home-visual-row small { color: var(--muted); font-size: 0.76rem; line-height: 1.4; }
        .home-visual-icon {
            display: inline-grid;
            width: 1.9rem;
            height: 1.9rem;
            flex: 0 0 1.9rem;
            place-items: center;
            border-radius: 50%;
            background: var(--accent-soft);
            color: var(--navy);
            font-size: 0.7rem;
            font-weight: 750;
        }
        .home-card-icon {
            display: inline-grid;
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            place-items: center;
            border: 1px solid var(--line);
            border-radius: 0.65rem;
            background: var(--accent-soft);
            color: var(--navy);
            font-size: 0.72rem;
            font-weight: 750;
        }
        .home-section {
            max-width: 1120px;
            margin: 2.2rem auto 0;
            scroll-margin-top: 1rem;
        }
        .home-section-heading { margin-bottom: 0.9rem; }
        .home-section-heading h2, .home-about-preview h2, .screening-disclaimer h2 {
            margin: 0.3rem 0 0.35rem;
            color: var(--navy);
            font-size: clamp(1.3rem, 2.5vw, 1.7rem);
        }
        .home-section-heading p, .home-about-preview p, .screening-disclaimer p {
            margin: 0;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.65;
        }
        .home-secondary-link {
            display: inline-flex;
            min-height: 2.5rem;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.7rem;
            border: 1px solid transparent;
            border-radius: var(--radius-control);
            color: var(--navy) !important;
            font-size: 0.88rem;
            font-weight: 650;
            text-decoration: none;
            transition: background 150ms ease, border-color 150ms ease;
        }
        .home-secondary-link:hover {
            border-color: var(--line);
            background: var(--accent-soft);
        }
        .home-secondary-link:focus-visible {
            outline: 3px solid var(--focus);
            outline-offset: 2px;
        }
        .home-workflow, .home-output-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.1rem;
        }
        .home-workflow {
            position: relative;
            margin: 0;
            padding: 0;
            list-style: none;
        }
        .home-workflow::before {
            position: absolute;
            top: 0.9rem;
            right: 0;
            left: 0;
            height: 1px;
            background: var(--line);
            content: "";
        }
        .home-workflow li {
            position: relative;
            min-width: 0;
            padding-top: 2.45rem;
        }
        .home-workflow-number {
            position: absolute;
            top: 0;
            left: 0;
            display: grid;
            width: 1.9rem;
            height: 1.9rem;
            place-items: center;
            border: 1px solid var(--accent);
            border-radius: 50%;
            background: var(--page-bg);
            color: var(--navy);
            font-size: 0.68rem;
            font-weight: 750;
        }
        .home-workflow h3 {
            margin: 0 0 0.4rem;
            color: var(--navy);
            font-size: 0.95rem;
        }
        .home-workflow p {
            margin: 0;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .home-info-card, .home-history-card {
            position: relative;
            min-width: 0;
            padding: 1rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--ivory);
            box-shadow: var(--shadow-card);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }
        .home-info-card:hover, .home-history-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: 0 11px 24px rgba(36, 52, 71, 0.09);
        }
        .home-info-card h3, .home-empty-state h3 {
            margin: 0.65rem 0 0.35rem;
            color: var(--navy);
            font-size: 0.96rem;
        }
        .home-info-card p, .home-empty-state p {
            margin: 0;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .home-step-number {
            position: absolute;
            top: 0.8rem;
            right: 0.85rem;
            color: var(--accent);
            font-size: 0.66rem;
            font-weight: 750;
            letter-spacing: 0.05em;
        }
        .home-empty-state {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 1rem;
            border: 1px dashed var(--line);
            border-radius: var(--radius-card);
            background: var(--panel);
        }
        .home-empty-state h3 { margin-top: 0; }
        .home-history-card {
            display: grid;
            grid-template-columns: minmax(1.25fr, 2fr) repeat(3, minmax(0, 1fr));
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.65rem;
        }
        .home-history-title { min-width: 0; }
        .home-history-title > div { min-width: 0; }
        .home-history-title h3 {
            overflow-wrap: anywhere;
            margin: 0 0 0.15rem;
            font-size: 0.9rem;
        }
        .home-history-title p { margin: 0; color: var(--muted); font-size: 0.74rem; }
        .home-history-signal {
            display: grid;
            gap: 0.25rem;
            min-width: 0;
            padding-left: 0.7rem;
            border-left: 1px solid var(--line);
        }
        .home-history-signal span { color: var(--muted); font-size: 0.68rem; }
        .home-history-signal strong { color: var(--navy); font-size: 0.78rem; overflow-wrap: anywhere; }
        .home-about-preview {
            max-width: 1120px;
            margin: 2rem auto 0.3rem;
            padding: clamp(1.1rem, 2.5vw, 1.7rem);
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--panel);
        }
        .home-about-preview h2 { max-width: 720px; }
        .home-about-preview p { max-width: 780px; }
        .screening-disclaimer {
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
            max-width: 1120px;
            margin: 1.7rem auto 0;
            padding: 1.1rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--accent-soft);
        }
        .screening-disclaimer h2 { font-size: 1.05rem; }
        .screening-disclaimer p { max-width: 850px; }
        .screening-disclaimer-icon {
            display: inline-grid;
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            place-items: center;
            border: 1px solid rgba(38, 59, 80, 0.2);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.35);
            color: var(--navy);
            font-weight: 750;
        }
        .about-page { max-width: 1080px; margin: 0 auto; color: var(--ink); }
        .about-eyebrow {
            color: var(--accent);
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .about-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(220px, 0.75fr);
            align-items: center;
            gap: 1.5rem;
            padding: clamp(1.4rem, 3vw, 2.2rem);
            border: 1px solid var(--line);
            border-radius: 18px;
            background: linear-gradient(125deg, var(--panel), var(--page-bg));
            box-shadow: var(--shadow-card);
        }
        .about-hero h1 { margin: 0.45rem 0; color: var(--navy); font-size: clamp(2rem, 4vw, 2.7rem); }
        .about-hero .about-eyebrow { color: var(--accent); }
        .about-hero-copy > p, .about-safety-context { color: var(--muted); font-size: 0.9rem; line-height: 1.7; }
        .about-safety-context { margin: 0; padding-left: 1rem; border-left: 2px solid var(--accent); }
        .about-safety-context strong { display: block; margin-bottom: 0.3rem; color: var(--navy); }
        .about-section { margin: 2rem 0; scroll-margin-top: 1rem; }
        .about-section-heading { margin-bottom: 0.85rem; }
        .about-section-heading h2 { margin: 0.25rem 0; color: var(--navy); font-size: clamp(1.35rem, 2.5vw, 1.75rem); }
        .about-section-heading > p, .about-metrics-note { margin: 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6; }
        .about-stepper {
            position: relative;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 0;
            padding: 0;
            list-style: none;
        }
        .about-stepper::before {
            position: absolute;
            top: 0.9rem;
            right: 0;
            left: 0;
            height: 1px;
            background: var(--line);
            content: "";
        }
        .about-stepper li {
            position: relative;
            min-width: 0;
            padding-top: 2.5rem;
        }
        .about-step-number {
            display: inline-grid;
            width: 1.9rem;
            height: 1.9rem;
            place-items: center;
            border: 1px solid var(--accent);
            border-radius: 50%;
            background: var(--page-bg);
            color: var(--navy);
            font-size: 0.68rem;
            font-weight: 750;
        }
        .about-stepper h3 { margin: 0 0 0.35rem; color: var(--navy); font-size: 0.92rem; }
        .about-stepper p { margin: 0; color: var(--muted); font-size: 0.79rem; line-height: 1.62; }
        .about-stepper code { color: var(--navy); }
        .about-tech-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
        .about-tech-row span {
            padding: 0.35rem 0.65rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--navy);
            font-size: 0.72rem;
            font-weight: 650;
        }
        .about-callout {
            margin: 0.8rem 0 0;
            padding: 0.85rem 1rem;
            border-left: 3px solid var(--accent);
            border-radius: 0 var(--radius-control) var(--radius-control) 0;
            background: var(--panel);
            color: var(--muted);
            font-size: 0.83rem;
            line-height: 1.65;
        }
        .about-callout strong { color: var(--navy); }
        .about-editorial-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.25rem;
        }
        .about-editorial-grid article { min-width: 0; padding-right: 1.1rem; border-right: 1px solid var(--line); }
        .about-editorial-grid article:last-child { padding-right: 0; border-right: 0; }
        .about-editorial-icon { color: var(--accent); font-size: 1rem; font-weight: 750; }
        .about-editorial-grid h3 { margin: 0.35rem 0; color: var(--navy); font-size: 0.95rem; }
        .about-editorial-grid p { margin: 0; color: var(--muted); font-size: 0.83rem; line-height: 1.65; }
        .about-metrics-title { margin: 1.15rem 0 0.25rem; color: var(--navy); font-size: 1rem; }
        .about-metrics-wrap { max-width: 100%; margin: 0.75rem 0 1.8rem; overflow-x: auto; border-bottom: 1px solid var(--line); }
        .about-metrics-table { width: 100%; border-collapse: collapse; color: var(--ink); font-size: 0.8rem; }
        .about-metrics-table caption { padding: 0.8rem; color: var(--navy); font-weight: 700; text-align: left; }
        .about-metrics-table th, .about-metrics-table td { padding: 0.75rem 0.9rem; border-top: 1px solid var(--line); text-align: left; white-space: nowrap; }
        .about-metrics-table thead { background: rgba(206, 197, 232, 0.32); }
        .about-metrics-table thead th { color: var(--navy); font-weight: 700; }
        .about-metrics-table tbody th { color: var(--navy); font-weight: 650; }
        .about-metrics-table th:nth-child(n + 2),
        .about-metrics-table td:nth-child(n + 2) {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        .screening-disclaimer h2 { margin: 0.2rem 0 0.35rem; color: var(--navy); font-size: 1.1rem; }
        .screening-disclaimer p { margin: 0; color: var(--muted); font-size: 0.84rem; line-height: 1.65; }
        .home-section .stButton button, .home-page ~ [data-testid="stHorizontalBlock"] button {
            transition: transform 150ms ease, box-shadow 150ms ease;
        }
        .home-section .stButton button:hover, .home-page ~ [data-testid="stHorizontalBlock"] button:hover {
            transform: translateY(-1px);
        }
        @media (max-width: 900px) {
            .home-hero { grid-template-columns: minmax(0, 1.1fr) minmax(220px, 0.9fr); }
            .home-output-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .home-workflow, .about-stepper {
                grid-template-columns: 1fr;
                gap: 1.2rem;
                padding-left: 2.8rem;
            }
            .home-workflow::before, .about-stepper::before {
                top: 0.9rem;
                right: auto;
                bottom: 0.9rem;
                left: 0.9rem;
                width: 1px;
                height: auto;
            }
            .home-workflow li, .about-stepper li { padding-top: 0; }
            .home-workflow-number, .about-step-number { left: -2.8rem; }
            .about-stepper li .about-step-number { position: absolute; top: 0; }
            .home-history-card { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .home-history-signal { min-height: 2.4rem; }
        }
        @media (max-width: 640px) {
            html { scroll-behavior: auto; }
            .home-hero, .about-hero { grid-template-columns: 1fr; }
            .home-output-grid, .about-editorial-grid { grid-template-columns: 1fr; }
            .home-history-card { grid-template-columns: 1fr; }
            .home-history-signal { padding: 0.45rem 0 0 0.65rem; }
            .home-secondary-link { padding-left: 0.25rem; }
            .about-editorial-grid article {
                padding: 0 0 1rem;
                border-right: 0;
                border-bottom: 1px solid var(--line);
            }
            .about-editorial-grid article:last-child { padding-bottom: 0; border-bottom: 0; }
            .about-section { margin: 1.6rem 0; }
            .screening-disclaimer { padding: 0.9rem; }
        }
        .block-container {
            box-sizing: border-box;
            width: 100%;
            max-width: 1120px;
            padding: 0.7rem clamp(0.85rem, 2.5vw, 1.75rem) 2.5rem;
        }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup) {
            gap: 0.65rem;
            align-items: center;
            padding: 0.15rem 0 0.4rem;
            margin-bottom: 0.45rem;
        }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup) [data-testid="stVerticalBlock"] {
            gap: 0.15rem;
        }
        .brand-lockup {
            gap: 0.45rem;
            font-size: 1.02rem;
            background: transparent !important;
            border: 0;
            box-shadow: none;
        }
        [data-testid="stHorizontalBlock"]:has(.brand-lockup)
        > [data-testid="stColumn"]:first-child,
        [data-testid="stHorizontalBlock"]:has(.brand-lockup)
        > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"]:has(.brand-lockup)
        > [data-testid="stColumn"]:first-child [data-testid="stMarkdown"],
        [data-testid="stHorizontalBlock"]:has(.brand-lockup)
        > [data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"] {
            background: transparent !important;
            border: 0;
            box-shadow: none;
        }
        .st-key-navigation_links [data-testid="stHorizontalBlock"] {
            justify-content: center;
            gap: 0.75rem !important;
        }
        .st-key-navigation_links [data-testid="stColumn"] {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        .brand-mark {
            width: 2.15rem;
            height: 2.15rem;
            flex-basis: 2.15rem;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button {
            min-height: 2.65rem;
            padding: 0.35rem 0.55rem 0.45rem;
            font-size: 0.9rem;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button p {
            font-size: 0.9rem;
        }
        div[class*="st-key-nav_link_"] [data-testid="stButton"] button[kind="primary"]::after {
            right: 0.55rem;
            bottom: 0.15rem;
            left: 0.55rem;
        }
        div.st-key-profile_menu button {
            min-height: 2.65rem;
            padding: 0.35rem 0.75rem;
            font-size: 0.9rem;
            white-space: nowrap;
        }
        .st-key-posting_controls > [data-testid="stVerticalBlock"] {
            gap: 0.25rem;
        }
        .st-key-posting_controls [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.45rem;
        }
        .st-key-posting_input_header [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.75rem;
        }
        .st-key-posting_input_header [data-testid="stMarkdownContainer"] p {
            margin: 0;
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 650;
        }
        .st-key-clear_job_input [data-testid="stButton"] > button {
            width: auto !important;
            min-width: 8.25rem;
            min-height: 2.45rem !important;
            padding: 0.35rem 0.75rem !important;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-control) !important;
            background: #ffffff !important;
            color: var(--navy) !important;
            box-shadow: none !important;
            font-size: 0.875rem;
            font-weight: 600;
            white-space: nowrap !important;
        }
        .st-key-clear_job_input [data-testid="stButton"] > button:hover {
            border-color: var(--accent-hover) !important;
            background: var(--panel) !important;
            color: var(--navy) !important;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.45rem;
            align-items: center;
        }
        .section-heading#analysis {
            max-width: 1080px;
            margin: 0.3rem auto 0.75rem;
        }
        .section-heading#analysis .page-title {
            margin: 0.35rem 0 0.3rem;
            font-size: clamp(1.9rem, 3.4vw, 2.4rem);
        }
        .section-heading#analysis .page-subtitle {
            line-height: 1.5;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            min-height: 2.65rem;
            padding: 0.45rem 0.8rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-control) var(--radius-control) 0 0;
            background: var(--panel);
            color: var(--muted);
            font-size: 0.875rem;
            font-weight: 650;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--navy);
        }
        div[data-testid="stForm"]:has([data-testid="stTextArea"]) {
            margin-top: 0.2rem;
            padding: 0.85rem 0.95rem 0.75rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-card);
            background: var(--input-bg);
            box-shadow: 0 5px 16px rgba(36, 52, 71, 0.045);
        }
        div[data-testid="stForm"]:has([data-testid="stTextArea"])
        [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 240px;
            padding: 0.8rem 0.9rem !important;
            border-radius: var(--radius-control) !important;
            background: #ffffff !important;
            line-height: 1.55;
        }
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInputRootElement"]:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(125, 112, 168, 0.18) !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 2.75rem;
            border-radius: var(--radius-control);
            box-shadow: 0 4px 12px rgba(36, 52, 71, 0.08);
        }
        @media (max-width: 760px) {
            .block-container { padding: 0.55rem 1rem 2rem; }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup) {
                gap: 0.25rem 0.65rem;
                padding-bottom: 0.35rem;
                margin-bottom: 0.4rem;
            }
            [data-testid="stHorizontalBlock"]:has(.brand-lockup)
            > [data-testid="stColumn"]:nth-child(2) {
                min-width: 0 !important;
            }
            div[data-testid="stForm"]:has([data-testid="stTextArea"]) {
                padding: 0.75rem 0.8rem 0.65rem;
            }
            div[data-testid="stTextArea"] textarea { min-height: 220px; }
        }
        @media (max-width: 520px) {
            .block-container { padding: 0.45rem 0.75rem 1.75rem; }
            .brand-lockup { font-size: 0.96rem; }
            .brand-mark {
                width: 1.95rem;
                height: 1.95rem;
                flex-basis: 1.95rem;
            }
            div[class*="st-key-nav_link_"] [data-testid="stButton"] button {
                min-height: 2.65rem;
                padding: 0.25rem 0.35rem 0.4rem;
                white-space: normal;
            }
            div[class*="st-key-nav_link_"] [data-testid="stButton"] button p {
                font-size: 0.875rem;
                line-height: 1.2;
            }
            .st-key-navigation_links [data-testid="stHorizontalBlock"] {
                gap: 0.2rem !important;
                flex-wrap: wrap;
                row-gap: 0.1rem;
            }
            div[data-testid="stTabs"] [role="tablist"] {
                gap: 0.25rem;
            }
            div[data-testid="stTabs"] button[role="tab"] {
                padding-right: 0.55rem;
                padding-left: 0.55rem;
                font-size: 0.84rem;
            }
            .section-heading#analysis { margin-top: 0.2rem; }
            .section-heading#analysis .page-title { font-size: 1.9rem; }
            div[data-testid="stForm"]:has([data-testid="stTextArea"]) {
                padding: 0.7rem 0.65rem 0.6rem;
            }
            div[data-testid="stTextArea"] textarea { min-height: 205px; }
        }
        @media (prefers-reduced-motion: reduce) {
            html { scroll-behavior: auto; }
            .home-info-card, .home-history-card,
            .home-secondary-link { transition: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        initialize_database(auth_store.AUTH_DB_PATH)
    except AuthStorageError as exc:
        LOGGER.exception("The authentication database is unavailable.")
        st.error(f"JobSignal sign-in is unavailable. {exc}")
        return

    user = None
    token = st.session_state.get("auth_token")
    if token:
        try:
            user = get_session_user(token, db_path=auth_store.AUTH_DB_PATH)
        except AuthStorageError as exc:
            LOGGER.exception("The authenticated session could not be loaded.")
            _clear_user_workspace()
            st.error(f"Your session could not be verified. {exc}")
            return
        if user is None:
            _clear_user_workspace()
            st.session_state["auth_notice"] = (
                "Your session expired or ended. Sign in to continue."
            )
    if user is None:
        _render_auth_screen()
        return

    st.session_state["authenticated_user"] = user
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    st.session_state.setdefault("text_posting_input", "")
    st.session_state.setdefault("image_posting_input", "")
    st.session_state.setdefault("image_upload_digest", "")
    st.session_state.setdefault("image_ocr_digest", "")
    _monitor_session_expiry(token)

    if "jobsignal_page" not in st.session_state:
        st.session_state["jobsignal_page"] = "Home"
    navigation = st.columns(
        [1.15, 2.25, 1.15],
        gap="small",
        vertical_alignment="center",
    )
    with navigation[0]:
        st.markdown(
            f'<div class="brand-lockup">{_brand_mark_svg()}<span>JobSignal</span></div>',
            unsafe_allow_html=True,
        )
    page_options = (
        ("Home", "Home"),
        ("Job Posting Analysis", "Job Posting Analysis"),
        ("About", "About"),
    )
    with navigation[1]:
        with st.container(key="navigation_links"):
            nav_buttons = st.columns([0.9, 1.7, 0.8], gap="small")
            for column, (label, page_name) in zip(nav_buttons, page_options):
                with column:
                    page_key = page_name.lower().replace(" ", "_")
                    with st.container(key=f"nav_link_{page_key}"):
                        st.button(
                            label,
                            type=(
                                "primary"
                                if st.session_state["jobsignal_page"] == page_name
                                else "secondary"
                            ),
                            use_container_width=False,
                            key=f"nav_{page_key}",
                            on_click=_set_page,
                            args=(page_name,),
                        )
    with navigation[2]:
        with st.container(key="profile_menu"):
            with st.popover(
                f"👤 {user['full_name']}",
                use_container_width=False,
            ):
                st.markdown("#### Account")
                st.caption(user["email"])
                if st.button(
                    "Profile",
                    use_container_width=True,
                    key="profile_menu_profile",
                ):
                    _set_page("Profile")
                    st.rerun()
                if st.button(
                    "Log out",
                    use_container_width=True,
                    key="profile_menu_logout",
                ):
                    _logout(token)
    page = st.session_state["jobsignal_page"]
    if page == "Home":
        _render_home_page(user)
        return
    if page == "About":
        _render_about_page()
        return
    if page == "Profile":
        st.markdown(
            '<div class="section-heading"><h1>Profile</h1>'
            "<p>Your JobSignal account details.</p></div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(f"**Name**  \n{escape(user['full_name'])}")
            st.markdown(f"**Email address**  \n{escape(user['email'])}")
        st.button(
            "Back to Home",
            on_click=_set_page,
            args=("Home",),
            key="profile_back_home",
        )
        return

    st.markdown(
        """
        <div class="section-heading" id="analysis">
            <div class="section-eyebrow">JOBSIGNAL · POSTING ANALYSIS</div>
            <h1 class="page-title">Review a job posting</h1>
            <p class="page-subtitle">Paste or enter the full job posting before analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="posting_controls"):
        text_tab, image_tab = st.tabs(("Input Text", "Upload Image"))

    text_submitted = False
    image_submitted = False
    with text_tab:
        with st.container(key="posting_input_header"):
            label_column, clear_column = st.columns(
                [5, 1.2],
                gap="small",
                vertical_alignment="center",
            )
            with label_column:
                st.markdown("**Job posting text**")
            with clear_column:
                if st.button("Clear posting", key="clear_job_input"):
                    st.session_state["text_posting_input"] = ""
                    st.session_state["image_posting_input"] = ""
                    st.session_state["image_ocr_digest"] = ""
        with st.form("text_job_posting_form"):
            text_posting = st.text_area(
                "Job posting text",
                height=280,
                max_chars=20000,
                key="text_posting_input",
                label_visibility="collapsed",
                placeholder=(
                    "Include the job title, company, responsibilities, "
                    "qualifications, salary, location, and contact information."
                ),
            )
            text_company_override = st.text_input(
                "Company name (optional — used if none is extracted)",
                max_chars=200,
                key="text_company_override",
                help=(
                    "This is used only when the posting does not provide a company "
                    "name. It does not change the model assessment."
                ),
            )
            text_search_requested = st.checkbox(
                "Check the company and job posting online",
                key="text_search_requested",
                value=False,
                help=(
                    "If enabled, relevant job details, including the employer, job "
                    "title, and available location, may be sent to Tavily, an "
                    "external search service, to find matching public web results."
                ),
            )
            text_submitted = st.form_submit_button(
                "Analyze Posting",
                type="primary",
            )

    with image_tab:
        uploaded_image = st.file_uploader(
            "Upload a job advertisement image (PNG, JPG, or JPEG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="job_posting_image",
            help=(
                "PNG and JPEG images up to "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB are processed locally "
                "on this server."
            ),
        )
        image_bytes = uploaded_image.getvalue() if uploaded_image else b""
        image_digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
        image_is_valid = False
        if image_bytes:
            if image_digest != st.session_state["image_upload_digest"]:
                st.session_state["image_upload_digest"] = image_digest
                st.session_state["image_ocr_digest"] = ""
                st.session_state["image_posting_input"] = ""
            try:
                image_preview = decode_image(image_bytes, uploaded_image.name)
                image_is_valid = True
                st.image(image_preview, caption=uploaded_image.name, use_container_width=True)
            except OcrError as exc:
                st.error(str(exc))

        if st.button(
            "Extract text with PaddleOCR",
            key="extract_image_text",
            disabled=not image_is_valid,
        ):
            if st.session_state["image_ocr_digest"] == image_digest:
                st.info("Text has already been extracted from this image.")
            else:
                try:
                    with st.spinner("Extracting text locally with PaddleOCR..."):
                        extracted_text = extract_text_from_image(
                            image_bytes,
                            uploaded_image.name,
                            engine=get_ocr_engine(),
                        )
                    st.session_state["image_ocr_digest"] = image_digest
                    st.session_state["image_posting_input"] = extracted_text
                    if extracted_text:
                        st.success("Text extracted. Review and edit it before analysis.")
                    else:
                        st.warning(
                            "No readable text was detected. You can still enter the "
                            "job-posting text below."
                        )
                except OcrError as exc:
                    LOGGER.exception("Local PaddleOCR text extraction failed.")
                    st.error(str(exc))

        with st.form("image_job_posting_form"):
            image_posting = st.text_area(
                "Extracted job text (editable)",
                height=320,
                max_chars=20000,
                key="image_posting_input",
                placeholder=(
                    "Extract text from an image, then review and correct it here "
                    "before analysis."
                ),
            )
            image_company_override = st.text_input(
                "Company name (optional — used if none is extracted)",
                max_chars=200,
                key="image_company_override",
            )
            image_search_requested = st.checkbox(
                "Check the company and job posting online",
                key="image_search_requested",
                value=False,
                help=(
                    "If enabled, relevant job details may be sent to Tavily, an "
                    "external search service, to find matching public web results."
                ),
            )
            image_submitted = st.form_submit_button(
                "Analyze Posting",
                type="primary",
            )

    submitted = text_submitted or image_submitted
    if text_submitted:
        posting_text = text_posting
        company_override = text_company_override
        search_requested = text_search_requested
    else:
        posting_text = image_posting
        company_override = image_company_override
        search_requested = image_search_requested

    if submitted:
        if not posting_text.strip():
            st.warning("Enter a complete job posting before analyzing it.")
        elif len(posting_text) > 20000:
            st.warning("The posting is too long. Limit the input to 20,000 characters.")
        else:
            try:
                job_data = extract_job_fields(posting_text)
                completeness = assess_posting_completeness(job_data)
            except (ValueError, TypeError) as exc:
                st.error(f"Posting input is invalid: {exc}")
                return

            result = None
            if not completeness["is_insufficient"]:
                try:
                    with st.spinner("Running the saved model and preprocessing pipeline..."):
                        result = predict_job_posting(
                            job_data,
                            model=get_model(),
                            preprocessing_artifacts=get_artifacts(),
                        )
                except (
                    FileNotFoundError,
                    ValueError,
                    TypeError,
                    OSError,
                    ImportError,
                    EOFError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    st.error(
                        "The ML model could not analyze this posting. Check that "
                        f"the saved model and preprocessing artifacts are intact. Details: {exc}"
                    )

            analysis = analyze_risk_factors(
                posting_text,
                prediction=int(result["prediction"]) if result is not None else None,
            )
            risk_score = rule_risk_score(analysis["risks"])
            level = risk_band(risk_score)
            if completeness["is_insufficient"]:
                prediction_label = "More information needed"
            elif result is None:
                prediction_label = "Model unavailable"
            elif result["prediction"] == 1:
                prediction_label = "Potential fraud warning"
            else:
                prediction_label = "Appears legitimate"
            st.session_state["analysis_history"].insert(
                0,
                {
                    "Time": datetime.now().strftime("%b %d, %Y · %H:%M"),
                    "Role": str(job_data.get("title") or "Untitled posting"),
                    "Employer": str(job_data.get("company_name") or "Not identified"),
                    "Warning score": f"{risk_score}/100",
                    "Risk level": level,
                    "Model assessment": prediction_label,
                },
            )
            st.session_state["analysis_history"] = st.session_state[
                "analysis_history"
            ][:10]

            st.markdown(
                """
                <div class="results-heading" id="results">
                    <div class="section-eyebrow">ANALYSIS COMPLETE</div>
                    <h2>Your posting assessment</h2>
                    <p>The model assessment, posting details, and text-based warning level are separate signals.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            assessment_column, details_column, warning_column = st.columns(
                [1.45, 1, 1],
                gap="medium",
            )
            if completeness["is_insufficient"]:
                result_class = "insufficient"
                assessment_copy = (
                    "The posting does not contain enough recognizable job "
                    "information for the model to assess it."
                )
                assessment_icon = (
                    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
                    '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>'
                    '<path d="M12 11v5m0-8h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                    "</svg>"
                )
            elif result is None:
                result_class = "insufficient"
                assessment_copy = (
                    "The model could not assess this posting. Please try again later."
                )
                assessment_icon = (
                    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
                    '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>'
                    '<path d="M12 11v5m0-8h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                    "</svg>"
                )
            elif result["prediction"] == 1:
                result_class = "fraudulent"
                assessment_copy = (
                    "Based on patterns identified by the machine-learning model "
                    "in the submitted job posting."
                )
                assessment_icon = (
                    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
                    '<path d="M10.3 3.9 2.6 17.3A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.7L13.7 3.9a2 2 0 0 0-3.4 0Z" '
                    'stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>'
                    '<path d="M12 9v4m0 3h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                    "</svg>"
                )
            else:
                result_class = "legitimate"
                assessment_copy = (
                    "Based on patterns identified by the machine-learning model "
                    "in the submitted job posting."
                )
                assessment_icon = (
                    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
                    '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>'
                    '<path d="m8 12 2.5 2.5L16.5 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                    "</svg>"
                )
            coverage_score = max(0, min(100, int(completeness["score"])))
            risk_label = level.replace(" RISK", "").title()
            risk_class = {
                "LOW RISK": "risk-low",
                "MEDIUM RISK": "risk-medium",
                "HIGH RISK": "risk-high",
            }[level]
            with assessment_column:
                st.markdown(
                    f"""
                    <div class="summary-card assessment-card {result_class}">
                        <h3>Job assessment</h3>
                        <div class="assessment-status">
                            <span class="assessment-icon" aria-hidden="true">{assessment_icon}</span>
                            <div>
                                <div class="assessment-title">{escape(prediction_label)}</div>
                                <div class="assessment-copy">{escape(assessment_copy)}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with details_column:
                st.markdown(
                    f"""
                    <div class="summary-card details-card">
                        <h3>Posting details</h3>
                        <div class="summary-value">{coverage_score}%</div>
                        <p class="coverage-copy">Shows how many of the key job details were identified in the submitted posting, such as the job title, employer, description, and role details.</p>
                        <div class="coverage-track" role="progressbar"
                             aria-label="Posting detail coverage"
                             aria-valuemin="0" aria-valuemax="100"
                             aria-valuenow="{coverage_score}">
                            <span style="width: {coverage_score}%"></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with warning_column:
                st.markdown(
                    f"""
                    <div class="summary-card warning-card">
                        <h3>Warning level</h3>
                        <div class="risk-indicator {risk_class}">{escape(risk_label)}</div>
                        <p class="warning-score">Rule-based warning score: {risk_score}/100</p>
                        <p>Based on warning patterns detected in the submitted job text.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### Positive signs in this posting")
            if analysis["positives"]:
                for item in analysis["positives"][:5]:
                    evidence = (
                        escape(str(item["evidence"][0]))
                        if item.get("evidence")
                        else ""
                    )
                    evidence_html = (
                        f'<p class="indicator-evidence">Text evidence: “{evidence}”</p>'
                        if evidence
                        else ""
                    )
                    st.markdown(
                        f"""
                        <div class="positive-indicator">
                            <strong>{escape(str(item['title']))}</strong>
                            <p>{escape(str(item['text']))}</p>
                            {evidence_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No positive indicators were identified in the submitted text.")

            st.markdown("### Warning signs to review")
            if analysis["risks"]:
                for item in analysis["risks"]:
                    severity = int(item.get("severity", 1))
                    level_label = (
                        "Higher concern"
                        if severity >= 5
                        else "Worth checking"
                        if severity >= 3
                        else "Caution"
                    )
                    indicator_class = (
                        "warning-high"
                        if severity >= 5
                        else "warning-medium"
                    )
                    evidence = (
                        escape(str(item["evidence"][0]))
                        if item.get("evidence")
                        else ""
                    )
                    evidence_html = (
                        f'<p class="indicator-evidence">Text evidence: “{evidence}”</p>'
                        if evidence
                        else ""
                    )
                    st.markdown(
                        f"""
                        <div class="warning-indicator {indicator_class}">
                            <strong>{escape(str(item['title']))}</strong>
                            <span class="indicator-level">{level_label}</span>
                            <p>{escape(str(item['why']))}</p>
                            {evidence_html}
                            <p><strong>Suggested action:</strong> {escape(str(item['action']))}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No major warning patterns were identified in the submitted text."
                )

            company_check = None
            vacancy_check = None
            verification_conclusion = None
            if search_requested:
                api_key = os.getenv("TAVILY_API_KEY", "").strip()
                if not api_key:
                    api_key = str(
                        dotenv_values(PROJECT_ROOT / ".env").get("TAVILY_API_KEY")
                        or ""
                    ).strip()
                extracted_company_name = str(job_data.get("company_name") or "")
                user_provided_employer = company_override.strip()
                verification_employer = (
                    extracted_company_name or user_provided_employer
                )
                job_title = str(job_data.get("title") or "")
                location = str(job_data.get("location") or "")
                requisition_id = extract_requisition_id(posting_text)
                company_error = ""
                vacancy_error = ""
                company_search_unavailable = False
                vacancy_search_unavailable = False
                company_results = []
                vacancy_results = []
                if not api_key:
                    company_error = (
                        "Configure TAVILY_API_KEY in the project .env file or "
                        "environment to run Tavily Search API."
                    )
                    vacancy_error = company_error
                    company_search_unavailable = True
                    vacancy_search_unavailable = True
                else:
                    company_queries = [
                        (f'"{verification_employer}"', "exact"),
                        (
                            f"{verification_employer} official website",
                            "fallback",
                        ),
                        (f"{verification_employer} company profile", "fallback"),
                    ]
                    vacancy_queries = [
                        " ".join(
                            term
                            for term in (
                                f'"{verification_employer}"',
                                f'"{job_title}"' if job_title else "",
                                f'"{requisition_id}"' if requisition_id else "",
                                f'"{location}"' if location else "",
                                "careers jobs",
                            )
                            if term
                        ),
                        f'"{verification_employer}" "{job_title}"',
                        " ".join(
                            term for term in (
                                verification_employer, job_title, location,
                                "job opening",
                            ) if term
                        ),
                    ]
                    api_key_fingerprint = hashlib.sha256(
                        api_key.encode("utf-8")
                    ).hexdigest()

                    def collect_search_results(queries, *, kind):
                        collected = []
                        seen_links = set()
                        for query_entry in queries:
                            if isinstance(query_entry, tuple):
                                query, search_stage = query_entry
                            else:
                                query, search_stage = query_entry, "exact"
                            try:
                                found = _cached_search(
                                    query, 8, api_key_fingerprint, api_key
                                )
                            except (ValueError, requests.RequestException, RuntimeError) as exc:
                                message = (
                                    f"{kind} search unavailable: {exc}"
                                )
                                return collected, message, True
                            for item in found:
                                link = str(item.get("link") or "").strip().casefold()
                                if link and link not in seen_links:
                                    seen_links.add(link)
                                    collected.append({
                                        **item,
                                        "search_stage": search_stage,
                                    })
                            if kind == "Company":
                                if has_official_company_candidate(
                                    verification_employer, collected
                                ):
                                    break
                            elif has_vacancy_candidate(
                                verification_employer,
                                job_title,
                                location,
                                collected,
                                requisition_id,
                            ):
                                break
                        return collected, "", False

                    with st.spinner("Searching company and vacancy sources…"):
                        if verification_employer:
                            company_results, company_error, company_search_unavailable = (
                                collect_search_results(company_queries, kind="Company")
                            )
                        else:
                            company_error = (
                                "No employer was identified. Enter a company name "
                                "above to search."
                            )

                        if verification_employer and job_title:
                            vacancy_results, vacancy_error, vacancy_search_unavailable = (
                                collect_search_results(vacancy_queries, kind="Vacancy")
                            )
                        else:
                            vacancy_error = (
                                "A company name and extracted job title are "
                                "required to search for a vacancy."
                            )

                verification = verify_company_and_vacancy(
                    verification_employer,
                    job_title,
                    location,
                    company_results,
                    vacancy_results,
                    requisition_id=requisition_id,
                )
                company_check = verification["company"]
                vacancy_check = verification["vacancy"]
                if company_search_unavailable:
                    company_check.update({
                        "status": "UNAVAILABLE",
                        "summary": (
                            "Unable to verify the exact company name because the "
                            "search service was unavailable."
                        ),
                        "source": None,
                        "error": company_error,
                    })
                elif company_error:
                    company_check.update({
                        "status": "NO_INPUT",
                        "summary": "Unable to verify the exact company name; no employer was identified.",
                        "source": None,
                    })
                if vacancy_search_unavailable:
                    vacancy_check.update({
                        "status": "UNAVAILABLE",
                        "summary": "The vacancy could not be checked because the search service was unavailable.",
                        "source": None,
                        "error": vacancy_error,
                    })
                elif vacancy_error:
                    vacancy_check.update({
                        "status": "NO_INPUT",
                        "summary": "A company name and extracted job title are required to search for a vacancy.",
                        "source": None,
                    })

                st.markdown("### Online verification results")
                company_column, vacancy_column = st.columns(2)
                if (
                    user_provided_employer
                    and extracted_company_name
                    and " ".join(user_provided_employer.split()).casefold()
                    != " ".join(extracted_company_name.split()).casefold()
                ):
                    st.caption(
                        "The company name extracted from the posting is retained "
                        "for verification; the separately entered name is not used."
                    )
                label_map = {
                    "VERIFIED": "Matching information found",
                    "MATCH_FOUND": "Matching information found",
                    "NOT_ESTABLISHED": "No matching information found",
                    "NOT_FOUND": "No matching information found",
                    "INCONCLUSIVE": "Possible match found",
                    "POSSIBLE_MATCH": "Possible match found",
                    "EXPIRED": "Closed listing found",
                    "UNAVAILABLE": "Search unavailable",
                    "NO_INPUT": "Not checked",
                }
                displayed_links = set()
                for column, title, check, kind in (
                    (
                        company_column,
                        "Company verification",
                        company_check,
                        "company",
                    ),
                    (
                        vacancy_column,
                        "Job posting verification",
                        vacancy_check,
                        "vacancy",
                    ),
                ):
                    with column:
                        with st.container(border=True):
                            st.markdown(f"#### {title}")
                            badge_class = {
                                "VERIFIED": "verification-found",
                                "MATCH_FOUND": "verification-found",
                                "UNAVAILABLE": "verification-unable",
                                "NO_INPUT": "verification-unable",
                            }.get(check["status"], "verification-not-found")
                            st.markdown(
                                f'<span class="verification-badge {badge_class}">'
                                f'{label_map.get(check["status"], "Not Found")}</span>',
                                unsafe_allow_html=True,
                            )
                            if kind == "company":
                                st.write(
                                    check.get("company_name")
                                    or "Employer not identified."
                                )
                                source = check.get("source")
                                if source:
                                    source_title = (
                                        source.get("search_title")
                                        or source.get("title")
                                        or "Search result"
                                    )
                                    source_kind = (
                                        "Broader fallback result"
                                        if source.get("search_stage") == "fallback"
                                        else "Exact-name search result"
                                    )
                                    st.caption(f"{source_kind}: {source_title}")
                                if source and source.get("profile_available"):
                                    st.markdown("**About the company**")
                                    st.write(source["summary"])
                                else:
                                    st.caption(
                                        "A reliable company profile could not be "
                                        "retrieved from the available page."
                                    )
                            else:
                                role_details = [
                                    check.get("job_title")
                                    or "Job title not identified."
                                ]
                                if check.get("location"):
                                    role_details.append(check["location"])
                                st.write(" · ".join(role_details))
                            st.write(check["summary"])
                            source = check.get("source")
                            if (
                                source
                                and source.get("link")
                                and source["link"] not in displayed_links
                                and len(displayed_links) < 3
                            ):
                                st.link_button(
                                    (
                                        f"Open evidence: "
                                        f"{source.get('search_title') or source['title']}"
                                        if kind == "company"
                                        else f"View source: {source['title']}"
                                    ),
                                    source["link"],
                                    use_container_width=True,
                                )
                                displayed_links.add(source["link"])
                verification_conclusion = build_verification_conclusion(
                    company_check,
                    vacancy_check,
                )

            st.markdown("### Recommended next steps")
            for recommendation in analysis["verification_recommendations"][:5]:
                st.markdown(f"- {recommendation}")

            st.markdown("### Overall conclusion")
            if completeness["is_insufficient"]:
                conclusion_parts = [
                    "More information is needed before the model can assess this "
                    "posting. Add a recognizable job title and a fuller description."
                ]
            elif result is None:
                conclusion_parts = [
                    "The model could not assess this posting, so no model result is available."
                ]
            elif int(result["prediction"]) == 1:
                conclusion_parts = [
                    "The model found patterns associated with potentially fraudulent postings."
                ]
            else:
                conclusion_parts = [
                    "The model found patterns associated with legitimate postings."
                ]
            if analysis["risks"]:
                risk_titles = ", ".join(
                    item["title"] for item in analysis["risks"][:3]
                )
                conclusion_parts.append(
                    f"Warning patterns to review: {risk_titles}."
                )
            elif analysis["positives"]:
                positive_titles = ", ".join(
                    item["title"] for item in analysis["positives"][:3]
                )
                conclusion_parts.append(
                    f"Positive details identified: {positive_titles}."
                )
            if (
                not completeness["is_insufficient"]
                and result is not None
                and (
                    (int(result["prediction"]) == 0 and analysis["risks"])
                    or (int(result["prediction"]) == 1 and not analysis["risks"])
                )
            ):
                conclusion_parts.append(
                    "The model assessment and rule-based warning signs do not fully "
                    "align; treat them as separate signals and review the evidence."
                )
            if search_requested and verification_conclusion:
                conclusion_parts.append(verification_conclusion)
            elif not search_requested:
                conclusion_parts.append(
                    "Online verification was not selected, so the company and vacancy were not checked."
                )
            conclusion_parts.append(
                "Review the highlighted evidence and recommended next steps."
            )
            st.write(" ".join(conclusion_parts))
            st.markdown(
                f"""
                <p class="results-guidance">Use the model assessment and available verification results together to make a more informed decision.</p>
                <p class="results-disclaimer">{_SCREENING_DISCLAIMER}</p>
                """,
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
