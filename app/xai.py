"""
Explainable AI utilities for the Fake Job Detection project.

This module has two layers:

1. explain_linear_svm()
   Research/internal local explanation using feature_value × coefficient.

2. analyze_risk_factors() + render_user_facing_xai()
   Beginner-friendly, instance-specific explanations based on the actual
   submitted job posting. These explain why a detected pattern is a warning
   sign or a positive signal without exposing ML terminology.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Existing research XAI
# ---------------------------------------------------------------------------

def explain_linear_svm(model, feature_vector, feature_names, top_n=5):
    """
    Generate a local explanation for a Linear SVM.

    contribution = feature_value × SVM_coefficient

    Positive contribution -> pushes toward class 1 (fraudulent)
    Negative contribution -> pushes toward class 0 (legitimate)
    """
    if hasattr(feature_vector, "toarray"):
        values = feature_vector.toarray().ravel()
    else:
        values = np.asarray(feature_vector).ravel()

    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Linear SVM explanation requires binary classes 0 and 1.")
    coefficients = np.asarray(model.coef_[0]).reshape(-1)
    fraud_direction = 1 if classes[1] == 1 else -1

    if len(values) != len(coefficients):
        raise ValueError(
            f"Feature vector has {len(values)} features, "
            f"but model expects {len(coefficients)}."
        )

    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Feature-name list has {len(feature_names)} names, "
            f"but model has {len(coefficients)}."
        )

    contributions = values * coefficients

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": values,
        "coefficient": coefficients,
        "contribution": contributions,
    })

    explanation = explanation[
        np.abs(explanation["value"]) > 1e-12
    ].copy()

    fraud_side = explanation["contribution"] * fraud_direction
    fraud_factors = (
        explanation[fraud_side > 0]
        .assign(_fraud_direction=fraud_side[fraud_side > 0])
        .sort_values("_fraud_direction", ascending=False)
        .head(top_n)
        .copy()
        .drop(columns="_fraud_direction")
    )

    legitimate_factors = (
        explanation[fraud_side < 0]
        .assign(_legitimate_direction=-fraud_side[fraud_side < 0])
        .sort_values("_legitimate_direction", ascending=False)
        .head(top_n)
        .copy()
        .drop(columns="_legitimate_direction")
    )

    decision_score = float(model.decision_function(feature_vector)[0])
    contribution_sum = float(contributions.sum())
    intercept = float(model.intercept_[0])

    return {
        "fraud_factors": fraud_factors,
        "legitimate_factors": legitimate_factors,
        "decision_score": decision_score,
        "contribution_sum": contribution_sum,
        "intercept": intercept,
        "reconstructed_score": contribution_sum + intercept,
    }


# ---------------------------------------------------------------------------
# User-facing, posting-specific fraud-risk explanation
# ---------------------------------------------------------------------------

RISK_RULES: Sequence[Dict[str, Any]] = [
    {
        "id": "payment_request",
        "title": "Payment or fee request",
        "patterns": [
            r"\bregistration\s+fee\b",
            r"\bapplication\s+fee\b",
            r"\bprocessing\s+fee\b",
            r"\btraining\s+fee\b",
            r"\bsecurity\s+deposit\b",
            r"\bdeposit\s+(?:money|amount|funds)\b",
            r"\bpay\s+(?:a|the|an)?\s*(?:fee|charge|deposit|amount)\b",
            r"\bpay\s+(?:a|the|an)?\s*[$£€₹]?\s*[\d,]+(?:\.\d+)?\s*(?:registration|application|processing|security|training)\s+fee\b",
            r"\b(?:send|transfer|deposit)\s+[$£€₹]?\s*[\d,]+(?:\.\d+)?\s+(?:to\s+)?(?:apply|secure|get|start)\b",
            r"\bpayment\s+(?:is\s+)?required\b",
            r"\bpay\s+to\s+(?:apply|start|join|get\s+the\s+job)\b",
            r"\b(?:send|transfer|deposit)\s+(?:money|funds)\b",
        ],
        "why": (
            "A request for money in order to apply, get hired, start work, "
            "or receive training is a strong warning sign. Job seekers should "
            "be especially careful before paying any fee or deposit."
        ),
        "action": (
            "Do not pay. Verify the employer independently using its official "
            "website or known contact details."
        ),
        "severity": 5,
    },
    {
        "id": "easy_high_income",
        "title": "Unusually attractive or easy-income promise",
        "patterns": [
            r"\beasy\s+money\b",
            r"\bmake\s+money\s+(?:fast|quickly)\b",
            r"\bguaranteed\s+(?:income|earnings|profit)\b",
            r"\bunlimited\s+(?:income|earnings)\b",
            r"\bget\s+rich\b",
            r"\bquick\s+cash\b",
            r"\bhigh\s+income\s+(?:with|without)\s+experience\b",
            r"\bearn\s+(?:up\s+to\s+)?[\$£€₹]?\s*\d[\d,]*(?:\.\d+)?\s*(?:per|a)\s*(?:day|week|month)\b",
            r"\b[\$£€₹]\s*\d[\d,]*(?:\.\d+)?\s*(?:per|a)\s*(?:day|week|month)\b",
        ],
        "why": (
            "Promises of unusually easy, guaranteed, or very attractive earnings "
            "can be used to attract applicants to fraudulent schemes. The earning "
            "claim should be checked against the employer and job role."
        ),
        "action": (
            "Verify the salary, employer, role, and payment terms before sharing "
            "personal information."
        ),
        "severity": 4,
    },
    {
        "id": "no_experience",
        "title": "No experience or qualifications required",
        "patterns": [
            r"\bno\s+experience\s+(?:required|necessary|needed)\b",
            r"\bwithout\s+(?:any\s+)?experience\b",
            r"\bexperience\s+not\s+(?:required|necessary|needed)\b",
            r"\banyone\s+can\s+apply\b",
            r"\bno\s+qualification(?:s)?\s+(?:required|necessary|needed)\b",
            r"\bno\s+skills?\s+(?:required|necessary|needed)\b",
        ],
        "why": (
            "Not requiring experience is not automatically fraudulent. It becomes "
            "more concerning when combined with easy-money promises, urgency, or "
            "requests for payment or sensitive information."
        ),
        "action": (
            "Check whether the responsibilities, employer identity, and pay are "
            "clearly described."
        ),
        "severity": 2,
    },
    {
        "id": "urgency",
        "title": "Urgent or pressure-based hiring language",
        "patterns": [
            r"\bstart\s+(?:today|immediately|now)\b",
            r"\bimmediate(?:ly)?\s+(?:start|joining|join|hire|hiring)\b",
            r"\burgent(?:ly)?\s+(?:hiring|hire|needed|required)\b",
            r"\blimited\s+(?:positions?|spots?|openings?)\b",
            r"\bact\s+now\b",
            r"\bonly\s+\d+\s+(?:positions?|spots?|openings?)\b",
        ],
        "why": (
            "Pressure to act immediately can reduce the chance that an applicant "
            "checks the employer, recruiter, website, or payment terms carefully."
        ),
        "action": (
            "Slow down and independently verify the employer before applying or "
            "sending documents."
        ),
        "severity": 3,
    },
    {
        "id": "sensitive_information",
        "title": "Request for sensitive personal or financial information",
        "patterns": [
            r"\bbank\s+account\b",
            r"\bbank\s+details?\b",
            r"\bcredit\s+card\b",
            r"\bdebit\s+card\b",
            r"\botp\b",
            r"\bpassword\b",
            r"\bpin\b",
            r"\bssn\b",
            r"\bsocial\s+security\b",
            r"\bgovernment\s+id\b",
            r"\bidentity\s+(?:document|proof)\b",
            r"\bpassport\s+(?:copy|number|details?)\b",
            r"\baadhaar\b",
            r"\bpan\s+card\b",
        ],
        "why": (
            "A job posting that asks for sensitive financial, account, or identity "
            "information before a trustworthy hiring process is established deserves "
            "extra caution."
        ),
        "action": (
            "Do not share passwords, OTPs, card details, or unnecessary identity "
            "documents."
        ),
        "severity": 5,
    },
    {
        "id": "off_platform_contact",
        "title": "Unusual or informal application/contact channel",
        "patterns": [
            r"\bcontact\s+(?:me|us)?\s*(?:on|via|through)\s+(?:telegram|whatsapp)\b",
            r"\b(?:telegram|whatsapp)\s+(?:only|number|contact)\b",
            r"\bmessage\s+(?:me|us)\s+on\s+(?:telegram|whatsapp)\b",
        ],
        "why": (
            "Moving recruitment to informal messaging channels is not proof of fraud, "
            "but it can make it harder to verify who is actually offering the job."
        ),
        "action": (
            "Confirm that the recruiter and contact method are connected to the real employer."
        ),
        "severity": 2,
    },
    {
        "id": "remote_plus_risk",
        "title": "Remote-work claim combined with other warning signs",
        "patterns": [
            r"\bwork\s+from\s+home\b",
            r"\bwork\s+from\s+anywhere\b",
            r"\bremote\s+work\b",
        ],
        "why": (
            "Remote work itself is completely legitimate. It becomes more important "
            "to verify the employer when it appears together with easy-money, no-experience, "
            "urgency, or payment-related claims."
        ),
        "action": (
            "Verify the employer and job responsibilities rather than relying on the "
            "remote-work claim alone."
        ),
        "severity": 1,
        "conditional": "remote_context",
    },
]


POSITIVE_RULES: Sequence[Dict[str, Any]] = [
    {
        "id": "clear_responsibilities",
        "title": "Clear job responsibilities",
        "patterns": [
            r"\bresponsibilit(?:y|ies)\b",
            r"\byou\s+will\b",
            r"\bresponsible\s+for\b",
            r"\bmanage\b",
            r"\bdevelop\b",
            r"\banaly[sz]e\b",
            r"\bcoordinate\b",
        ],
        "text": (
            "The posting gives concrete information about the work instead of "
            "only making broad promises."
        ),
    },
    {
        "id": "specific_qualifications",
        "title": "Specific skills or qualifications",
        "patterns": [
            r"\brequirements?\b",
            r"\bqualification(?:s)?\b",
            r"\bskills?\b",
            r"\bexperience\b",
            r"\bdegree\b",
            r"\bbachelor(?:'s)?\b",
            r"\bmaster(?:'s)?\b",
        ],
        "text": (
            "The posting describes skills, qualifications, or experience expected for the role."
        ),
    },
    {
        "id": "professional_role_language",
        "title": "Professional role-specific language",
        "patterns": [
            r"\bjob\s+title\b",
            r"\bteam\b",
            r"\bdepartment\b",
            r"\bproject\b",
            r"\breport(?:ing)?\s+to\b",
            r"\bclient(?:s)?\b",
        ],
        "text": (
            "The posting contains role-specific details that make the opportunity "
            "easier to understand and verify."
        ),
    },
    {
        "id": "structured_compensation",
        "title": "Specific compensation information",
        "patterns": [
            r"\bsalary\b",
            r"\bcompensation\b",
            r"\bper\s+(?:hour|day|week|month|year)\b",
            r"(?:\$|£|€|₹)\s*\d",
        ],
        "text": (
            "The posting provides compensation information rather than only making "
            "vague earning promises."
        ),
    },
]


def _normalize(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matched_snippets(
    text: str,
    patterns: Iterable[str],
    limit: int = 2,
    ignore_negated: bool = False,
) -> List[str]:
    found: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            if ignore_negated:
                context_start = max(0, match.start() - 55)
                context_end = min(len(text), match.end() + 35)
                context = text[context_start:context_end]
                before = text[context_start:match.start()]
                if re.search(
                    r"\b(?:no|not|never|without|don't|doesn't|do not|does not|will not)"
                    r"\b[^.!?]{0,45}$",
                    before,
                    flags=re.I,
                ) or re.search(
                    r"\b(?:not required|not requested|never requested|not charged)\b",
                    context,
                    flags=re.I,
                ):
                    continue
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 55)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if snippet and snippet not in found:
                found.append(snippet)
            if len(found) >= limit:
                return found
    return found


def analyze_risk_factors(
    posting_text: str,
    prediction: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Produce explanations from the exact posting submitted by the user.

    The rules do not override the ML prediction. They provide human-readable
    evidence for the user and deliberately distinguish warning signs from proof.
    """
    text = _normalize(posting_text)
    lower = text.lower()
    risks: List[Dict[str, Any]] = []

    for rule in RISK_RULES:
        conditional = rule.get("conditional")

        if conditional == "remote_context":
            remote_found = any(
                re.search(pattern, lower, re.I)
                for pattern in rule["patterns"]
            )
            companion_found = any(
                item["id"] in {
                    "easy_high_income",
                    "no_experience",
                    "payment_request",
                    "urgency",
                }
                for item in risks
            )
            if not (remote_found and companion_found):
                continue

        matched = _matched_snippets(
            text,
            rule["patterns"],
            ignore_negated=True,
        )

        if matched:
            risks.append({
                "id": rule["id"],
                "title": rule["title"],
                "why": rule["why"],
                "action": rule["action"],
                "severity": rule["severity"],
                "evidence": matched,
            })

    risk_ids = {item["id"] for item in risks}

    # Context matters: two individually weaker signals together deserve
    # a stronger explanation.
    if {"no_experience", "easy_high_income"}.issubset(risk_ids):
        risks.append({
            "id": "combined_easy_entry",
            "title": "Easy-entry claim combined with attractive earnings",
            "why": (
                "The posting combines little or no experience requirements with "
                "attractive or easy-income language. Either point alone is not proof "
                "of fraud, but together they deserve stronger verification."
            ),
            "action": (
                "Verify the employer, role, salary, and hiring process before applying."
            ),
            "severity": 4,
            "evidence": [],
        })

    # Avoid duplicate categories.
    unique: Dict[str, Dict[str, Any]] = {}
    for item in risks:
        unique[item["id"]] = item
    risks = sorted(
        unique.values(),
        key=lambda item: (-item["severity"], item["title"]),
    )

    positives: List[Dict[str, Any]] = []
    for rule in POSITIVE_RULES:
        matched = _matched_snippets(text, rule["patterns"], limit=1)
        if matched:
            positives.append({
                "id": rule["id"],
                "title": rule["title"],
                "text": rule["text"],
                "evidence": matched,
            })

    if prediction is None:
        overall = (
            "No ML classification was made because the posting did not provide "
            "enough recognizable role information. The warning rules below are "
            "not a substitute for a model prediction."
        )
    elif prediction == 1:
        if risks:
            overall = (
                "This posting was classified as potentially fraudulent and "
                "contains specific warning signs that should be checked before you apply."
            )
        else:
            overall = (
                "This posting was classified as potentially fraudulent, but the "
                "common warning patterns used by this explanation layer were not "
                "clearly present. Review the employer, contact details, and application "
                "process carefully."
            )
    else:
        if risks:
            overall = (
                "This posting received a legitimate-looking model classification, "
                "but some warning signs are still present. That classification does not remove "
                "the need to verify the employer."
            )
        else:
            overall = (
                "This posting received a legitimate-looking model classification "
                "and no major warning pattern was detected by this explanation layer. You should still "
                "verify the employer before sharing sensitive information."
            )

    recommendations = [
        "Confirm the employer and vacancy through contact details found independently on the employer's official website.",
        "Do not pay fees or share passwords, one-time codes, or unnecessary financial or identity information.",
    ]
    recommendations.extend(item["action"] for item in risks)
    recommendations = list(dict.fromkeys(recommendations))

    return {
        "risks": risks,
        "positives": positives,
        "overall": overall,
        "risk_count": len(risks),
        "positive_count": len(positives),
        "verification_recommendations": recommendations,
    }


def render_user_facing_xai(
    st,
    analysis: Dict[str, Any],
    prediction: Optional[int],
) -> None:
    """Render plain-language XAI for a beginner user."""
    risks = analysis["risks"]
    positives = analysis["positives"]

    st.markdown("### 🔎 What should you look out for?")

    if risks:
        st.markdown("#### 🚨 Fraud-risk alerts found in this posting")

        for index, item in enumerate(risks, start=1):
            if item["severity"] >= 5:
                level = "High concern"
            elif item["severity"] >= 3:
                level = "Important to check"
            else:
                level = "Caution"

            with st.container(border=True):
                st.markdown(
                    f"**{index}. {item['title']}** · `{level}`"
                )
                st.write(item["why"])

                if item.get("evidence"):
                    st.caption(
                        f'Found in your posting: “{item["evidence"][0]}”'
                    )

                st.markdown(
                    f"**What you should do:** {item['action']}"
                )
    else:
        st.success(
            "No major common fraud-warning pattern was found in the text you submitted."
        )

    if positives:
        st.markdown("#### ✅ Positive / reassuring details")

        for item in positives[:5]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.write(item["text"])

                if item.get("evidence"):
                    st.caption(
                        f'Found in your posting: “{item["evidence"][0]}”'
                    )

    st.markdown("#### 🛡️ Overall assessment")
    st.info(analysis["overall"])

    st.markdown("#### ✅ Verification recommendations")
    for recommendation in analysis["verification_recommendations"]:
        st.markdown(f"- {recommendation}")

    st.caption(
        "This is a risk assessment, not proof that a job is fraudulent or genuine. "
        "Always verify the employer independently before sharing money, passwords, "
        "OTPs, bank details, or unnecessary identity documents."
    )
