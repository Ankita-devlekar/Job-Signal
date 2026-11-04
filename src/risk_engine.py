from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def risk_band(score: float) -> str:
    if score < 40:
        return "LOW RISK"
    if score < 70:
        return "MEDIUM RISK"
    return "HIGH RISK"


def web_risk_score(signals: Dict[str, Any]) -> Optional[float]:
    """Score only evidence explicitly marked as authoritatively verified."""
    if (
        signals.get("search_unavailable")
        or signals.get("search_error")
        or not signals.get("authoritative_verification")
    ):
        return None

    score = 0.0
    if signals.get("company_found"):
        score += 10
    if signals.get("official_domain_found"):
        score += 5
    if not signals.get("job_found_officially"):
        score += 25
    if signals.get("job_found_third_party"):
        score += 8
    if signals.get("email_domain_mismatch"):
        score += 20
    if signals.get("application_domain_mismatch"):
        score += 15
    if signals.get("conflicting_information"):
        score += 20
    if signals.get("company_not_found"):
        score += 10
    return clamp(score)


def rule_risk_score(risks: Any) -> int:
    """Return a transparent warning-signal score, not a probability of fraud."""
    unique_risks = {item["id"]: item for item in risks}
    return int(clamp(sum(int(item["severity"]) * 10 for item in unique_risks.values())))


def overall_risk_score(ml_score: float, rule_score: float, web_score: Optional[float] = None) -> Tuple[float, str]:
    """Project-defined heuristic scores are intentionally transparent and not calibrated probabilities."""
    if web_score is None:
        overall = 0.7 * ml_score + 0.3 * rule_score
    else:
        overall = 0.5 * ml_score + 0.3 * web_score + 0.2 * rule_score
    overall = clamp(overall)
    return round(overall, 1), risk_band(overall)


__all__ = [
    "clamp",
    "overall_risk_score",
    "risk_band",
    "rule_risk_score",
    "web_risk_score",
]
