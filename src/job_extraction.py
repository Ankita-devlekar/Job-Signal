from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Optional


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Job posting text must be a string.")
    if len(value) > 20_000:
        raise ValueError("Job posting text must be 20,000 characters or fewer.")
    if not value.strip():
        return ""
    text = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?i)<br\s*/?>|</(?:p|div|li|h[1-6])\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    headings = (
        r"job\s+title|title|position|role|company(?:\s+(?:profile|name))?|employer|"
        r"employer\s+name|organization|"
        r"location|office|work\s+location|country|city|salary|compensation|pay|"
        r"description|requirements|qualifications|benefits|perks|about(?:\s+(?:company|us))?|"
        r"responsibilities|application|department|employment\s+type|job\s+type|"
        r"experience|education|industry|function"
    )
    text = re.sub(
        rf"(?i)(?<!^)[ \t|]+(?=(?:{headings})\s*[:=—–-])",
        "\n",
        text,
    )
    return text.strip()


def _extract_company_name(text: str) -> str:
    patterns = [
        r"(?im)^company(?:\s+name)?\s*[:=—–-]\s*(.+)",
        r"(?im)^employer(?:\s+name)?\s*[:=—–-]\s*(.+)",
        r"(?im)^organization\s*[:=—–-]\s*(.+)",
        r"(?im)^\s*(?:careers?|jobs?)\s+at\s+(.+)",
        r"(?im)^\s*join\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.split("\n")[0].split("|")[0].strip(" -•,;")
            candidate = re.sub(r"(?i)\s+careers?$", "", candidate).strip()
            if 2 <= len(candidate) <= 200:
                return candidate

    career_url = re.search(
        r"(?i)(?:https?://)?(?:[a-z0-9-]+\.)*"
        r"(?:careers?|jobs?)\.([a-z0-9-]+)\.[a-z]{2,}(?:\.[a-z]{2,})?",
        text,
    )
    if career_url:
        domain_name = career_url.group(1)
        if domain_name.casefold() not in {"www", "co", "com", "org", "net"}:
            return domain_name.replace("-", " ").title()

    return ""


def _extract_title(text: str) -> str:
    patterns = [
        r"(?im)^job\s+title\s*[:=—–-]\s*(.+)",
        r"(?im)^position\s*[:=—–-]\s*(.+)",
        r"(?im)^role\s*[:=—–-]\s*(.+)",
        r"(?im)^title\s*[:=—–-]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.split("\n")[0].strip(" -•")
            if candidate:
                return candidate

    # fallback: use the first line that looks like a job role and not a sentence.
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 4 or len(line) > 120:
            continue
        if re.search(r"(?i)\b(analyst|engineer|manager|specialist|developer|designer|consultant|associate|coordinator|agent|assistant|executive|intern|architect|scientist|director|lead)\b", line):
            return line
    return ""


def _extract_location(text: str) -> str:
    patterns = [
        r"(?im)^location\s*[:=—–-]\s*(.+)",
        r"(?im)^office\s*[:=—–-]\s*(.+)",
        r"(?im)^work\s+location\s*[:=—–-]\s*(.+)",
        r"(?im)^country\s*[:=—–-]\s*(.+)",
        r"(?im)^city\s*[:=—–-]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.split("\n")[0].strip(" -•")
            if candidate:
                return candidate
    return ""


def _extract_salary_fields(text: str) -> Dict[str, Any]:
    salary_lines = [
        line for line in text.splitlines()
        if re.match(r"(?i)^\s*(?:salary|compensation|pay)\s*[:=—–-]", line)
    ]
    if not salary_lines:
        salary_lines = [
            line for line in text.splitlines()
            if re.search(r"[$£€₹]|\b(?:USD|GBP|EUR|INR)\b", line, flags=re.I)
            and re.search(r"\b(?:per|hourly|annually|yearly|annual)\b", line, flags=re.I)
        ]

    number_pattern = re.compile(
        r"(?<![\w.])(?:[$£€₹]\s*)?"
        r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
        r"\s*(k|m|thousand|million)?\b",
        flags=re.I,
    )
    numbers = []
    for match in number_pattern.finditer(" ".join(salary_lines)):
        number = float(match.group(1).replace(",", ""))
        scale = (match.group(2) or "").lower()
        if scale in {"k", "thousand"}:
            number *= 1_000
        elif scale in {"m", "million"}:
            number *= 1_000_000
        if number > 0:
            numbers.append(number)

    if numbers:
        min_salary = min(numbers)
        max_salary = max(numbers)
        return {
            "salary_range": f"{min_salary:g}-{max_salary:g}",
            "min_salary": float(min_salary),
            "max_salary": float(max_salary),
            "mid_salary": float((min_salary + max_salary) / 2),
        }

    return {"salary_range": None, "min_salary": None, "max_salary": None, "mid_salary": None}


def _extract_section(text: str, heading: str) -> str:
    section_headings = (
        r"job\s+title|title|position|role|company(?:\s+(?:profile|name))?|employer|"
        r"employer\s+name|organization|"
        r"location|office|work\s+location|country|city|salary|compensation|pay|"
        r"description|requirements|qualifications|benefits|perks|about(?:\s+(?:company|us))?|"
        r"responsibilities|application|department|employment\s+type|job\s+type|"
        r"experience|education|industry|function"
    )
    pattern = (
        rf"(?ims)^\s*(?:{heading})\s*[:=—–-]\s*(.*?)"
        rf"(?=^\s*(?:{section_headings})\s*[:=—–-]|\Z)"
    )
    match = re.search(pattern, text)
    if not match:
        return ""
    section = match.group(1)
    return re.sub(r"\s+", " ", section).strip()


def _is_remote_posting(text: str) -> int:
    remote_patterns = [
        r"\bwork\s+from\s+home\b",
        r"\bremote\b",
        r"\bhybrid\b",
        r"\bremote\s+work\b",
    ]
    return int(any(re.search(pattern, text, flags=re.I) for pattern in remote_patterns))


def extract_job_fields(text: str) -> Dict[str, Any]:
    text = _normalize_text(text)
    if not text:
        return {
            "title": "",
            "company_profile": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "location": "",
            "department": "",
            "employment_type": "",
            "required_experience": "",
            "required_education": "",
            "industry": "",
            "function": "",
            "salary_range": None,
            "min_salary": None,
            "max_salary": None,
            "mid_salary": None,
            "telecommuting": 0,
            "has_company_logo": None,
            "has_questions": None,
            "company_name": "",
        }

    company_name = _extract_company_name(text)
    title = _extract_title(text)
    location = _extract_location(text)
    salary = _extract_salary_fields(text)
    requirements = _extract_section(text, "requirements") or _extract_section(text, "qualifications")
    benefits = _extract_section(text, "benefits") or _extract_section(text, "perks")
    company_profile = (
        _extract_section(text, r"about(?:\s+(?:company|us))?")
        or _extract_section(text, "company profile")
    )
    description = " ".join(
        part
        for part in (
            _extract_section(text, "description"),
            _extract_section(text, "responsibilities"),
        )
        if part
    )
    if not description:
        description = text
    department = _extract_section(text, "department")
    employment_type = _extract_section(text, "employment type") or _extract_section(text, "job type")
    required_experience = _extract_section(text, "experience")
    required_education = _extract_section(text, "education")

    if not employment_type:
        employment_type = "Full-time" if re.search(r"\bfull[- ]?time\b", text, flags=re.I) else ""
        if not employment_type:
            employment_type = "Contract" if re.search(r"\bcontract\b", text, flags=re.I) else ""

    if not required_experience:
        required_experience = "Not specified" if not re.search(r"\b(?:no\s+experience|experience\s+not\s+required)\b", text, flags=re.I) else "No experience required"

    if not required_education:
        required_education = "Not specified"

    industry = ""
    function = ""

    record = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "location": location,
        "department": department,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "salary_range": salary.get("salary_range"),
        "min_salary": salary.get("min_salary"),
        "max_salary": salary.get("max_salary"),
        "mid_salary": salary.get("mid_salary"),
        "telecommuting": _is_remote_posting(text),
        "has_company_logo": None,
        "has_questions": None,
        "company_name": company_name,
    }
    for feature, text_column in (
        ("title_word_count", "title"),
        ("company_profile_word_count", "company_profile"),
        ("description_word_count", "description"),
        ("requirements_word_count", "requirements"),
        ("benefits_word_count", "benefits"),
    ):
        record[feature] = len(record[text_column].split())
    return record


def assess_posting_completeness(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Measure visible posting detail; this is not a fraud or accuracy score."""
    has_title = bool(str(job_data.get("title") or "").strip())
    has_employer = bool(
        str(job_data.get("company_name") or "").strip()
        or str(job_data.get("company_profile") or "").strip()
    )
    body_word_count = sum(
        len(str(job_data.get(key) or "").split())
        for key in ("description", "requirements", "benefits")
    )
    has_substantive_body = body_word_count >= 20
    has_role_details = any(
        str(job_data.get(key) or "").strip()
        for key in (
            "requirements",
            "benefits",
            "location",
            "employment_type",
            "required_experience",
            "required_education",
        )
    )
    checks = (
        ("Job title", has_title),
        ("Employer identity", has_employer),
        ("Job description", has_substantive_body),
        ("Role details", has_role_details),
    )
    missing = [label for label, present in checks if not present]
    score = round(100 * (len(checks) - len(missing)) / len(checks))
    return {
        "score": score,
        "missing": missing,
        "is_incomplete": bool(missing),
        "is_insufficient": not has_title or not has_substantive_body,
    }


__all__ = ["assess_posting_completeness", "extract_job_fields"]
