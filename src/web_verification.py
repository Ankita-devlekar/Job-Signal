from __future__ import annotations

import os
import re
import socket
import ipaddress
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def is_valid_url(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    try:
        parsed = urlsplit(text)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(
        (".localhost", ".local")
    ):
        return False
    try:
        parsed.port
    except ValueError:
        return False
    try:
        if not ipaddress.ip_address(hostname).is_global:
            return False
    except ValueError:
        pass
    return True


def normalize_domain(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().lower().replace("https://", "").replace("http://", "").replace("www.", "")
    text = text.rstrip("/")
    return text.split("/")[0].split("?")[0]


def extract_domain_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    return normalize_domain(url)


def extract_email_addresses(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    matches = re.findall(pattern, text)
    deduped: List[str] = []
    for item in matches:
        if item not in deduped:
            deduped.append(item)
    return deduped


def extract_phone_numbers(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"\+?\d[\d\s().-]{7,}\d"
    matches = re.findall(pattern, text)
    deduped: List[str] = []
    for item in matches:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def extract_application_urls(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"https?://[^\s)\]>\"']+"
    matches = re.findall(pattern, text, flags=re.I)
    deduped: List[str] = []
    for item in matches:
        if item not in deduped:
            deduped.append(item)
    return deduped


def fetch_public_page(url: str, timeout: int = 15) -> Dict[str, Any]:
    """Fetch a public page while validating every redirect target.

    Redirects are common on legitimate company sites. Each hop is validated
    separately so following redirects does not bypass the public-host checks.
    """
    if not 1 <= timeout <= 30:
        return {"ok": False, "error": "Timeout must be between 1 and 30 seconds."}

    current_url = url
    max_redirects = 5
    for redirect_count in range(max_redirects + 1):
        hostname = urlsplit(current_url).hostname
        if not hostname:
            return {"ok": False, "error": "Invalid or non-public URL."}
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            return {"ok": False, "error": "Only publicly routable website hosts are allowed."}
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    hostname, None, type=socket.SOCK_STREAM
                )
            }
        except OSError:
            return {"ok": False, "error": "Could not resolve the public website host."}
        try:
            if not addresses or any(
                not ipaddress.ip_address(address).is_global
                for address in addresses
            ):
                return {
                    "ok": False,
                    "error": "Only publicly routable website hosts are allowed.",
                }
        except ValueError:
            return {
                "ok": False,
                "error": "The website host resolved to an invalid address.",
            }

        if not is_valid_url(current_url):
            return {"ok": False, "error": "Only publicly routable website hosts are allowed."}

        response = None
        try:
            response = requests.get(
                current_url,
                headers=REQUEST_HEADERS,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    return {
                        "ok": False,
                        "error": "The website returned a redirect without a destination.",
                        "status_code": response.status_code,
                    }
                if redirect_count >= max_redirects:
                    return {
                        "ok": False,
                        "error": "The website exceeded the redirect limit.",
                        "status_code": response.status_code,
                    }
                current_url = urljoin(current_url, location)
                continue
            if response.status_code >= 400:
                status_code = response.status_code
                response.close()
                return {
                    "ok": False,
                    "error": f"HTTP {status_code}",
                    "status_code": status_code,
                }
            content = response.raw.read(200_000, decode_content=True).decode(
                response.encoding or "utf-8",
                errors="replace",
            )
            final_url = current_url
            status_code = response.status_code
            response.close()
            return {
                "ok": True,
                "status_code": status_code,
                "content": content,
                "final_url": final_url,
            }
        except requests.Timeout:
            if response is not None:
                response.close()
            return {"ok": False, "error": "Request timed out."}
        except requests.RequestException:
            if response is not None:
                response.close()
            return {"ok": False, "error": "Request failed due to a network error."}

    return {"ok": False, "error": "The website could not be fetched."}


def extract_readable_text_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text_blocks = []
    seen = set()
    for tag in soup.find_all(["main", "article", "p", "li", "h1", "h2", "h3"]):
        snippet = tag.get_text(" ", strip=True)
        if snippet and snippet not in seen:
            seen.add(snippet)
            text_blocks.append(snippet)
    metadata = []
    desc = soup.find("meta", attrs={"name": "description"})
    if desc and desc.get("content"):
        metadata.append(desc.get("content", ""))
    combined = " ".join(metadata + [title] + text_blocks)
    return re.sub(r"\s+", " ", combined).strip()[:20_000]


def extract_requisition_id(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:job\s*(?:id|number|#)|requisition\s*(?:id|number|#)?|"
        r"req\s*(?:id|number|#)?)\s*[:=#-]\s*"
        r"([A-Z0-9][A-Z0-9-]{1,30})\b",
        text or "",
    )
    return match.group(1).strip() if match else ""


def _requisition_ids(text: str) -> set[str]:
    pattern = re.compile(
        r"\b(?:job\s*(?:id|number|#)|requisition\s*(?:id|number|#)?|"
        r"req\s*(?:id|number|#)?)\s*[:=#-]\s*"
        r"([A-Z0-9][A-Z0-9-]{1,30})\b",
        re.I,
    )
    return {
        re.sub(r"[^a-z0-9]", "", match.group(1).casefold())
        for match in pattern.finditer(text or "")
    }


def _extract_page_job_details(html: str) -> Dict[str, Any]:
    if not html:
        return {
            "title": "",
            "headings": [],
            "primary_heading": "",
            "heading_sections": [],
            "text": "",
            "canonical_url": "",
            "requisition_ids": set(),
        }
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(
        ["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]
    ):
        tag.decompose()
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    headings = list(dict.fromkeys(
        re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading.get_text(" ", strip=True)
    ))
    heading_sections = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_title = re.sub(
            r"\s+", " ", heading.get_text(" ", strip=True)
        ).strip()
        if not heading_title:
            continue
        container = heading.find_parent(["article", "li"])
        if container is None:
            for parent in heading.parents:
                if parent.name in {"main", "body", "html"}:
                    break
                marker = " ".join(
                    str(parent.get(attribute) or "")
                    for attribute in ("id", "class")
                )
                if re.search(
                    r"\b(job|vacancy|opening|position|role)[-_]"
                    r"(card|item|posting|listing|details?|content)\b",
                    marker,
                    re.I,
                ):
                    container = parent
                    break
        if container is not None:
            section_text = container.get_text(" ", strip=True)
        else:
            heading_level = int(heading.name[1])
            section_parts = [heading_title]
            for sibling in heading.find_next_siblings():
                if re.fullmatch(r"h[1-6]", sibling.name or "") and int(
                    sibling.name[1]
                ) <= heading_level:
                    break
                section_parts.append(sibling.get_text(" ", strip=True))
            section_text = " ".join(section_parts)
        heading_sections.append({
            "title": heading_title,
            "text": re.sub(r"\s+", " ", section_text).strip(),
        })
    primary_heading = next(
        (
            re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
            for heading in soup.find_all("h1")
            if heading.get_text(" ", strip=True)
        ),
        "",
    )
    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    page_text = re.sub(
        r"\s+",
        " ",
        content_root.get_text(" ", strip=True),
    ).strip()[:20_000]
    identified_ids = set(_requisition_ids(page_text))
    for element in soup.find_all(True):
        for attribute in ("data-job-id", "data-requisition-id"):
            value = element.get(attribute)
            if isinstance(value, str):
                identified_ids.update(_requisition_ids(f"Job ID: {value}"))
        if element.get("itemprop") == "identifier":
            value = element.get("content") or element.get_text(" ", strip=True)
            if isinstance(value, str):
                identified_ids.update(_requisition_ids(f"Job ID: {value}"))
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical_url = str(canonical.get("href") or "").strip() if canonical else ""
    return {
        "title": page_title,
        "headings": headings,
        "primary_heading": primary_heading,
        "heading_sections": heading_sections,
        "text": page_text,
        "canonical_url": canonical_url,
        "requisition_ids": identified_ids,
    }


def search_web(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict[str, str]]:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Web search requires TAVILY_API_KEY.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty search query is required.")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "query": query.strip(),
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=12,
        )
    except requests.Timeout as exc:
        raise RuntimeError("Tavily search timed out.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Tavily search request failed due to a network error.") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403}:
            raise RuntimeError(
                f"Tavily rejected the API key (HTTP {status_code}). Check TAVILY_API_KEY."
            ) from exc
        status_text = f" (HTTP {status_code})" if status_code else ""
        raise RuntimeError(f"Tavily search request failed{status_text}.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The search provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("The search provider returned an unexpected response.")
    if payload.get("error"):
        raise RuntimeError("The search provider reported an error.")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise RuntimeError("The search provider returned an unexpected results format.")

    items = payload["results"]
    results: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        link = item.get("url") or ""
        snippet = item.get("content") or ""
        if not all(isinstance(value, str) for value in (title, link, snippet)):
            continue
        title, link, snippet = title.strip(), link.strip(), snippet.strip()
        if link:
            results.append({"title": title, "link": link, "snippet": snippet})
    if items and not results:
        raise RuntimeError("The search provider returned malformed results.")
    return results


_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "limited",
    "ltd", "llc", "plc", "lp", "ag", "sa", "nv", "holdings", "group",
}
_GENERIC_COMPANY_WORDS = {
    "official", "website", "site", "jobs", "job", "careers", "career",
    "hiring", "vacancy", "vacancies", "opportunity", "opportunities",
    "company", "companies", "recruitment", "recruiting", "employer",
}
_GENERIC_JOB_WORDS = {
    "job", "jobs", "career", "careers", "position", "role", "opening",
    "opportunity", "vacancy", "hiring", "official", "website", "company",
}


def _tokens(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (value or "").casefold())


def _normalize_company_name(value: str) -> str:
    return " ".join(
        token for token in _tokens(value)
        if token not in _LEGAL_SUFFIXES
        and token not in {"the", "and", "of"}
        and token not in _GENERIC_COMPANY_WORDS
    ).strip()


def _company_tokens(company_name: str) -> List[str]:
    tokens = [
        token for token in _tokens(company_name)
        if token not in _LEGAL_SUFFIXES
        and token not in {"the", "and", "of", "at", "for", "in", "with", "career", "careers", "job", "jobs"}
    ]
    return [token for token in tokens if token not in _GENERIC_COMPANY_WORDS]


def _company_identity_matches(company_name: str, candidate_text: str) -> bool:
    """Match a complete company name without accepting reordered tokens."""
    expected = _company_tokens(company_name)
    candidate = _company_tokens(candidate_text)
    if not expected or not candidate:
        return False

    # Generic page wording (e.g. "official website" or "careers") is removed
    # by _company_tokens. Company-name word order remains significant.
    return expected == candidate


def normalize_company_name(value: str) -> str:
    return " ".join(
        token for token in _tokens(value)
        if token not in _LEGAL_SUFFIXES
        and token not in {"the", "and", "of", "company", "companies"}
    ).strip()


def _company_mentioned(company_name: str, text: str) -> bool:
    return _company_identity_matches(company_name, text)


def _company_mentioned_in_title(company_name: str, title: str) -> bool:
    """Match a full company name at the end of a result title.

    Search titles often use "Role at Company". Requiring the complete company
    token sequence at the end avoids accepting a shorter prefix in a longer,
    unrelated company name.
    """
    if _company_identity_matches(company_name, title):
        return True
    expected = _company_tokens(company_name)
    observed = _company_tokens(title)
    return bool(expected and len(observed) >= len(expected)
                and observed[-len(expected):] == expected)


def _explicit_alias_relationship(
    company_name: str, alternate_name: str, page_text: str
) -> bool:
    """Require an official page sentence to explicitly connect both names."""
    expected = _company_tokens(company_name)
    alternate = _company_tokens(alternate_name)
    if not expected or not alternate or expected == alternate:
        return False

    def name_pattern(tokens: Sequence[str]) -> str:
        return r"\b" + r"[\W_]+".join(re.escape(token) for token in tokens) + r"\b"

    expected_pattern = name_pattern(expected)
    alternate_pattern = name_pattern(alternate)
    relationship = re.compile(
        r"\b(?:also known as|formerly known as|formerly|previously known as|"
        r"trading as|doing business as|d/b/a|operating as|rebranded as|"
        r"alternate name|alternative name|known as)\b",
        re.I,
    )
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", page_text or ""):
        if (
            re.search(expected_pattern, sentence, re.I)
            and re.search(alternate_pattern, sentence, re.I)
            and relationship.search(sentence)
        ):
            return True
    return False


def _relevant_company_result(company_name: str, item: Dict[str, str]) -> bool:
    title = item["title"]
    context = f"{title} {item['snippet']}".casefold()
    if not _company_mentioned(company_name, title):
        return False
    return bool(
        re.search(
            r"\b(about|official|company|corporate|careers?|jobs?|website|"
            r"organization|profile|contact|investor)\b",
            context,
        )
    )


def _domain_matches_company(company_name: str, url: str) -> bool:
    """Accept a domain only when its label represents the full company name."""
    if not is_valid_url(url):
        return False
    hostname = (urlsplit(url).hostname or "").casefold().rstrip(".")
    if not hostname:
        return False

    expected = _company_tokens(company_name)
    if not expected:
        return False
    company_signature = "".join(expected)
    labels = [
        label for label in hostname.split(".")
        if label and label != "www"
    ]

    for label in labels:
        label_tokens = _tokens(label)
        label_signature = "".join(label_tokens)
        # Supports both bright-future.example and brightfuture.example, while
        # rejecting brightfutureglobal.example as a match for Bright Future.
        if label_signature == company_signature:
            return True
        if label_tokens == expected:
            return True

    # Some domains represent multiword names across subdomain labels.
    domain_tokens = [
        token for label in labels for token in _tokens(label)
    ]
    return _contains_ordered_tokens(expected, domain_tokens)


def _contains_ordered_tokens(expected: Sequence[str], observed: Sequence[str]) -> bool:
    if len(expected) > len(observed):
        return False
    return any(
        list(observed[index:index + len(expected)]) == list(expected)
        for index in range(len(observed) - len(expected) + 1)
    )


def _title_match(title: str, text: str) -> bool:
    expected = _tokens(title)
    if not expected:
        return False
    observed = _tokens(text)
    return any(
        observed[index:index + len(expected)] == expected
        for index in range(len(observed) - len(expected) + 1)
    )


def _is_generic_careers_heading(value: str) -> bool:
    normalized = " ".join(_tokens(value))
    return normalized in {
        "careers",
        "career",
        "jobs",
        "job openings",
        "open positions",
        "current openings",
        "career opportunities",
        "job opportunities",
        "join us",
        "working at",
    } or normalized.startswith(
        ("careers at ", "jobs at ", "open positions at ", "job openings at ")
    )


def _looks_like_job_title(value: str) -> bool:
    return bool(re.search(
        r"\b(architect|engineer|analyst|manager|specialist|developer|designer|"
        r"consultant|associate|coordinator|assistant|executive|intern|scientist|"
        r"director|recruiter|technician|researcher|advisor|adviser|representative|"
        r"nurse|cashier|accountant|sales|support|operator|producer|writer|"
        r"administrator|trainer|therapist|teacher|officer|lead)\b",
        value,
        re.I,
    ))


def _page_title_context(
    job_title: str,
    page_title: str,
    headings: Sequence[str],
    primary_heading: str = "",
    heading_sections: Sequence[Dict[str, str]] = (),
    page_text: str = "",
) -> str:
    if primary_heading and not _is_generic_careers_heading(primary_heading):
        if not _title_match(job_title, primary_heading):
            return ""
        return next(
            (
                section["text"]
                for section in heading_sections
                if section["title"] == primary_heading and section["text"]
            ),
            page_text,
        )
    for section in heading_sections:
        if _title_match(job_title, section["title"]):
            return section["text"] or page_text
    has_specific_heading = any(
        not _is_generic_careers_heading(heading)
        for heading in headings
    )
    if has_specific_heading:
        return ""
    candidates = [*headings, page_title, page_text]
    return next(
        (page_text for candidate in candidates if _title_match(job_title, candidate)),
        "",
    )


def _canonical_page_url(value: str, base_url: str) -> str:
    candidate = urljoin(base_url, value) if value else ""
    normalized = _normalized_result({"link": candidate}) if candidate else None
    return normalized["link"] if normalized else ""


def _is_low_quality_source(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").casefold().rstrip(".")
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in {
            "wikipedia.org", "google.com", "bing.com", "search.yahoo.com",
            "duckduckgo.com", "yandex.com", "jooble.org", "jobrapido.com",
            "careerjet.com", "jobsora.com", "jobisjob.com",
            "alljobopenings.com", "jobs2careers.com",
        }
    )


def _normalized_result(item: Dict[str, str]) -> Optional[Dict[str, str]]:
    link = str(item.get("link") or "").strip()
    if not is_valid_url(link):
        return None
    parsed = urlsplit(link)
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in {"ref", "source", "fbclid", "gclid"}
        ]
    )
    path = parsed.path.rstrip("/") or "/"
    link = urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, query, "")
    )
    result = {
        "title": str(item.get("title") or "").strip() or "Search result",
        "link": link,
        "snippet": re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip(),
    }
    search_stage = str(item.get("search_stage") or "exact").strip().casefold()
    if search_stage in {"exact", "fallback"}:
        result["search_stage"] = search_stage
    return result


def _ranked_results(
    company_name: str,
    search_results: Sequence[Dict[str, str]],
    job_title: str = "",
    location: str = "",
    requisition_id: str = "",
) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, str]] = {}
    for raw_item in search_results:
        item = _normalized_result(raw_item)
        if item and not _is_low_quality_source(item["link"]):
            unique.setdefault(item["link"], item)
    ranked = []
    for item in unique.values():
        text = f"{item['title']} {item['snippet']}"
        official_domain = _domain_matches_company(company_name, item["link"])
        company_match = (
            _company_mentioned_in_title(company_name, item["title"])
            or official_domain
        )
        title_match = not job_title or _title_match(job_title, text)
        normalized_requisition_id = re.sub(
            r"[^a-z0-9]", "", requisition_id.casefold()
        )
        requisition_match = bool(
            normalized_requisition_id
            and normalized_requisition_id
            in _requisition_ids(f"{item['title']} {item['snippet']}")
        )
        location_match = not location or _title_match(
            location, f"{text} {item['link']}"
        )
        careers_match = bool(re.search(
            r"\b(careers?|jobs?|openings?)\b",
            f"{item['title']} {item['link']}",
            re.I,
        ))
        score = (
            100 * int(official_domain)
            + 15 * int(
                official_domain
                and len(
                    [
                        part
                        for part in (urlsplit(item["link"]).hostname or "")
                        .casefold()
                        .split(".")
                        if part and part != "www"
                    ]
                )
                <= 2
            )
            + 35 * int(company_match)
            + 30 * int(title_match and bool(job_title))
            + 15 * int(location_match and bool(location))
            + 10 * int(careers_match)
        )
        ranked.append({
            **item,
            "relevance": score,
            "company_match": company_match,
            "official_domain": official_domain,
            "title_match": title_match,
            "requisition_match": requisition_match,
            "location_match": location_match,
        })
    return sorted(ranked, key=lambda item: (-item["relevance"], item["link"]))


def has_company_candidate(
    company_name: str, search_results: Sequence[Dict[str, str]]
) -> bool:
    """Return whether results contain a plausible company-name/domain candidate."""
    return any(
        item["company_match"]
        for item in _ranked_results(company_name, search_results)
    )


def has_official_company_candidate(
    company_name: str, search_results: Sequence[Dict[str, str]]
) -> bool:
    """Return whether results include a relevant company page on its domain."""
    return any(
        item["company_match"]
        and item["official_domain"]
        and _relevant_company_result(company_name, item)
        for item in _ranked_results(company_name, search_results)
    )


def has_vacancy_candidate(
    company_name: str,
    job_title: str,
    location: str,
    search_results: Sequence[Dict[str, str]],
    requisition_id: str = "",
) -> bool:
    """Return whether results contain an official-domain company/title candidate."""
    return any(
        item["official_domain"]
        and item["company_match"]
        and (item["title_match"] or item["requisition_match"])
        and item["location_match"]
        for item in _ranked_results(
            company_name,
            search_results,
            job_title,
            location,
            requisition_id,
        )
    )


def _sentence_summary(text: str, terms: Sequence[str], max_sentences: int = 4) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
        if len(sentence.strip()) > 25
    ]
    if not sentences:
        return clean[:600].rstrip()
    weighted_terms = {term.casefold() for term in terms if len(term) > 2}
    ranked = []
    for index, sentence in enumerate(sentences):
        words = set(_tokens(sentence))
        ranked.append((sum(term in words for term in weighted_terms), index, sentence))
    selected = sorted(
        sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_sentences],
        key=lambda item: item[1],
    )
    summary = " ".join(sentence for _, _, sentence in selected)
    if len(summary) > 900:
        summary = summary[:897].rsplit(" ", 1)[0] + "..."
    return summary


def summarize_source(
    item: Dict[str, Any],
    *,
    company_name: str,
    job_title: str = "",
    page_fetcher=fetch_public_page,
) -> Dict[str, Any]:
    """Summarize a page or source with a stricter fallback to facts instead of snippets."""
    url = str(item.get("link") or "")
    page = page_fetcher(url, timeout=5) if is_valid_url(url) else {"ok": False}
    page_text = ""
    page_title = ""
    page_headings: List[str] = []
    canonical_url = ""
    final_url = url
    if page.get("ok"):
        html = str(page.get("content") or "")
        details = _extract_page_job_details(html)
        page_text = details["text"]
        page_title = str(page.get("title") or details["title"])
        page_headings = details["headings"]
        final_url = str(page.get("final_url") or url)
        canonical_url = _canonical_page_url(details["canonical_url"], final_url)
    # Search snippets are useful discovery evidence, but they are not a company
    # description. Only summarize text extracted from the fetched page itself.
    source_text = page_text
    summary_terms = [
        token for token in re.findall(r"[A-Za-z0-9]+", f"{company_name} {job_title}")
        if token and len(token) > 2
    ]
    summary = _sentence_summary(source_text, summary_terms)
    if not summary:
        summary = (
            "Company information could not be extracted from an accessible page. "
            "The search result snippet is not presented as a verified description."
        )
    page_confirms_company = (
        _company_mentioned(company_name, page_text)
        or _domain_matches_company(company_name, final_url)
    )
    return {
        "title": page_title or str(item.get("title") or "Search result"),
        "search_title": str(item.get("title") or "Search result"),
        "search_stage": str(item.get("search_stage") or "exact"),
        "link": url,
        "summary": summary,
        "page_title": page_title,
        "content_source": "page" if page_text else "none",
        "retrieved": bool(page_text),
        "profile_available": bool(page_text and page_confirms_company),
        "final_url": final_url,
        "canonical_url": canonical_url,
        "page_headings": page_headings,
        "primary_heading": details["primary_heading"] if page.get("ok") else "",
        "heading_sections": details["heading_sections"] if page.get("ok") else [],
        "page_requisition_ids": (
            sorted(details["requisition_ids"]) if page.get("ok") else []
        ),
        "source_text": source_text,
    }


def verify_company_and_vacancy(
    company_name: str,
    job_title: str,
    location: str,
    company_results: Sequence[Dict[str, str]],
    vacancy_results: Sequence[Dict[str, str]],
    *,
    requisition_id: str = "",
    page_fetcher=fetch_public_page,
) -> Dict[str, Any]:
    """Evaluate company identity and a specific vacancy as separate outcomes."""
    company_name = (company_name or "").strip()
    normalized_company = normalize_company_name(company_name)
    company_ranked = _ranked_results(company_name, company_results)
    vacancy_ranked = _ranked_results(
        company_name, vacancy_results, job_title, location, requisition_id
    )

    company_evidence = [
        item for item in company_ranked
        if item["company_match"] and item["official_domain"]
    ]
    company_supporting = [
        item for item in company_ranked
        if item["company_match"]
        and not item["official_domain"]
        and _relevant_company_result(company_name, item)
    ]
    company_source_summary = None
    verified_alias = None
    for item in company_ranked:
        if item["search_stage"] != "fallback":
            continue
        alternate_name = " ".join(_company_tokens(item["title"]))
        if (
            not alternate_name
            or _company_identity_matches(company_name, alternate_name)
            or not _domain_matches_company(alternate_name, item["link"])
        ):
            continue
        candidate_summary = summarize_source(
            item,
            company_name=company_name,
            page_fetcher=page_fetcher,
        )
        page_text = str(candidate_summary.get("source_text") or "")
        if (
            candidate_summary.get("retrieved")
            and _domain_matches_company(
                alternate_name, str(candidate_summary.get("final_url") or "")
            )
            and _explicit_alias_relationship(
                company_name, alternate_name, page_text
            )
        ):
            candidate_summary["profile_available"] = True
            candidate_summary["source_company_name"] = item["title"]
            candidate_summary.pop("source_text", None)
            candidate_summary.pop("page_requisition_ids", None)
            candidate_summary.pop("heading_sections", None)
            company_source_summary = candidate_summary
            verified_alias = item["title"]
            break

    if company_evidence:
        company_status = "VERIFIED"
        company_source = company_evidence[0]
        company_summary = "Company identified and an official website or company-controlled source was found."
    elif verified_alias and company_source_summary:
        company_status = "VERIFIED"
        company_source = None
        company_summary = (
            f"The official website identifies {company_name} as an alternative "
            f"name for {verified_alias}."
        )
    elif company_supporting:
        company_status = "INCONCLUSIVE"
        company_source = company_supporting[0]
        company_summary = (
            "Unable to verify the exact company name. Search results mention a "
            "possible match, but no official identity was established."
        )
    else:
        company_status = "NOT_ESTABLISHED"
        company_source = None
        company_summary = "Unable to verify the exact company name."

    vacancy_matches = [
        item for item in vacancy_ranked
        if item["company_match"]
        and (item["title_match"] or item["requisition_match"])
        and item["location_match"]
    ]
    official_vacancies = [
        item for item in vacancy_matches if item["official_domain"]
    ]
    third_party_vacancies = [
        item for item in vacancy_matches if not item["official_domain"]
    ]
    related_official_vacancies = [
        item for item in vacancy_ranked
        if item["company_match"] and item["official_domain"]
    ]
    if official_vacancies:
        vacancy_status = "POSSIBLE_MATCH"
        vacancy_source = official_vacancies[0]
        vacancy_summary = (
            "A company careers result was found, but the accessible page must confirm "
            "the specific role and location."
        )
    elif third_party_vacancies:
        vacancy_status = "POSSIBLE_MATCH"
        vacancy_source = third_party_vacancies[0]
        vacancy_summary = (
            "A possible third-party listing was found, but its relationship to the "
            "employer could not be confirmed."
        )
    elif related_official_vacancies:
        vacancy_status = "POSSIBLE_MATCH"
        vacancy_source = related_official_vacancies[0]
        vacancy_summary = (
            "A page on the employer's domain was found, but the search result does "
            "not confirm the requested vacancy."
        )
    else:
        vacancy_status = "NOT_FOUND"
        vacancy_source = None
        vacancy_summary = (
            "The search completed, but no sufficiently relevant vacancy listing was found."
        )

    if company_source and company_source_summary is None:
        company_source_summary = summarize_source(
            company_source,
            company_name=company_name,
            page_fetcher=page_fetcher,
        )
        company_source_summary["evidence_role"] = "company"

    vacancy_source_summary = None
    if vacancy_source:
        vacancy_source_summary = summarize_source(
            vacancy_source,
            company_name=company_name,
            job_title=job_title,
            page_fetcher=page_fetcher,
        )
        page_text = (
            vacancy_source_summary["source_text"]
            if vacancy_source_summary["content_source"] == "page"
            else ""
        )
        page_headings = vacancy_source_summary["page_headings"]
        primary_heading = vacancy_source_summary["primary_heading"]
        heading_sections = vacancy_source_summary["heading_sections"]
        page_title = vacancy_source_summary["page_title"]
        normalized_requisition_id = re.sub(
            r"[^a-z0-9]", "", requisition_id.casefold()
        )
        page_confirms_requisition = bool(
            normalized_requisition_id
            and normalized_requisition_id
            in vacancy_source_summary["page_requisition_ids"]
        )
        expired = bool(re.search(
            r"\b(expired|position (?:is )?closed|no longer accepting|"
            r"no longer available|job has been filled|applications closed)\b",
            page_text,
            re.I,
        ))
        role_context = (
            next(
                (
                    section["text"]
                    for section in heading_sections
                    if normalized_requisition_id
                    in _requisition_ids(section["text"])
                ),
                page_text,
            )
            if page_confirms_requisition
            else _page_title_context(
                job_title,
                page_title,
                page_headings,
                primary_heading,
                heading_sections,
                page_text,
            )
        )
        page_confirms_role = bool(page_text and role_context)
        page_confirms_location = not location or _title_match(
            location,
            role_context,
        )
        page_confirms_company = (
            bool(normalized_company)
            and (
                _company_mentioned(company_name, page_text)
                or _domain_matches_company(
                    company_name,
                    vacancy_source_summary["final_url"],
                )
            )
        )
        if expired:
            vacancy_status = "EXPIRED"
            vacancy_summary = "The listing page suggests the role may no longer be active."
        elif vacancy_source["link"] and vacancy_source["link"].strip().lower().endswith((".pdf", ".doc", ".docx")):
            vacancy_status = "POSSIBLE_MATCH"
            vacancy_summary = "The result appears to be a document or non-standard page, not a current job listing."
        elif vacancy_source["official_domain"] and (
            page_confirms_company
            and page_confirms_role
            and page_confirms_location
        ):
            vacancy_status = "MATCH_FOUND"
            vacancy_summary = (
                "The employer-controlled page contains the requested job title and "
                "supports the company and location match."
            )
        elif page_text and not page_confirms_role:
            vacancy_status = "NOT_FOUND"
            related_title = next(
                (
                    heading
                    for heading in ([primary_heading] if primary_heading else [])
                    + page_headings
                    if not _is_generic_careers_heading(heading)
                ),
                "",
            )
            if vacancy_source["official_domain"] and related_title and not _title_match(
                job_title, related_title
            ):
                vacancy_summary = (
                    f"The employer-controlled page shows a different role "
                    f"({related_title[:160]}). The requested vacancy was not confirmed."
                )
            elif related_title and not _title_match(job_title, related_title):
                vacancy_summary = (
                    f"The retrieved page shows a different role "
                    f"({related_title[:160]}). The requested vacancy was not confirmed."
                )
            else:
                vacancy_summary = (
                    "The retrieved page did not contain the requested "
                    "job title. The specific vacancy was not confirmed."
                )
        elif vacancy_source["official_domain"]:
            vacancy_status = "POSSIBLE_MATCH"
            vacancy_summary = (
                "The company careers result could not confirm the requested role and "
                "location from its page content."
            )
        else:
            vacancy_status = "POSSIBLE_MATCH"
            vacancy_summary = (
                "A possible third-party listing was found, but the page does not confirm "
                "that the employer authorized it."
            )

        vacancy_source_summary.pop("source_text", None)
        vacancy_source_summary.pop("page_requisition_ids", None)
        vacancy_source_summary.pop("heading_sections", None)
    if company_source_summary:
        company_source_summary.setdefault(
            "source_company_name",
            company_source_summary.get("search_title") or company_source_summary["title"],
        )
        company_source_summary.pop("source_text", None)
        company_source_summary.pop("page_requisition_ids", None)
        company_source_summary.pop("heading_sections", None)

    source_links = {}
    source_keys = {}
    for role, source in (
        ("company", company_source_summary),
        ("vacancy", vacancy_source_summary),
    ):
        if source:
            key = (
                source.get("canonical_url")
                or source.get("final_url")
                or source["link"]
            )
            if key not in source_links:
                source_links[key] = {**source, "evidence_role": role}
            source_keys[source["link"]] = key
    sources = list(source_links.values())[:3]
    if company_source_summary:
        company_source_summary = source_links[
            source_keys[company_source_summary["link"]]
        ]
    if vacancy_source_summary:
        vacancy_source_summary = source_links[
            source_keys[vacancy_source_summary["link"]]
        ]

    return {
        "company": {
            "status": company_status,
            "company_name": company_name,
            "summary": company_summary,
            "source": company_source_summary,
            "profile_available": bool(
                company_source_summary
                and company_source_summary.get("profile_available")
            ),
        },
        "vacancy": {
            "status": vacancy_status,
            "job_title": job_title,
            "company_name": company_name,
            "location": location,
            "summary": vacancy_summary,
            "source": vacancy_source_summary,
            "related_vacancy_found": bool(
                vacancy_source_summary
                and vacancy_status == "NOT_FOUND"
                and any(
                    _looks_like_job_title(heading)
                    for heading in (
                        [vacancy_source_summary.get("primary_heading") or ""]
                        + vacancy_source_summary.get("page_headings", [])
                    )
                )
            ),
        },
        "sources": sources,
    }


def summarize_search_evidence(
    company_name: str,
    job_title: str,
    search_results: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """Summarize links without treating search snippets as verification."""
    evidence = []
    for item in search_results:
        link = (item.get("link") or "").strip()
        if not is_valid_url(link):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        combined_text = f"{title} {snippet}".casefold()
        evidence.append(
            {
                "title": title or "Search result",
                "link": link,
                "snippet": snippet,
                "mentions_company": bool(
                    company_name
                    and (
                        _company_mentioned_in_title(company_name, title)
                        or _company_identity_matches(company_name, snippet)
                    )
                ),
                "mentions_job_title": bool(
                    job_title and job_title.casefold() in combined_text
                ),
            }
        )

    return {
        "status": "unable_to_verify",
        "status_label": "Unable to verify",
        "evidence": evidence,
        "summary": (
            "Search results are pointers, not proof of employer identity or an active "
            "vacancy. No result is labeled verified from snippets alone."
            if evidence
            else "No usable search results were returned. The employer and vacancy "
            "could not be verified."
        ),
        "verified_evidence_available": False,
        "conflicting_evidence_available": False,
    }


def build_search_queries(job_data: Dict[str, Any]) -> List[str]:
    company = str(job_data.get("company_name") or job_data.get("company") or "").strip()
    title = str(job_data.get("title") or "").strip()
    location = str(job_data.get("location") or "").strip()
    user_text = str(job_data.get("description") or "").strip()
    queries: List[str] = []

    if company:
        queries.append(f'"{company}" official website')
        queries.append(f'"{company}" careers')
        if title:
            queries.append(f'"{company}" "{title}"')
            queries.append(f'"{company}" "{title}" {location}'.strip())

    if title:
        queries.append(f'"{title}" company careers')
        queries.append(f'"{title}" hiring')

    if not queries:
        keywords = re.findall(r"[A-Za-z][A-Za-z0-9 &/.-]{4,60}", user_text)[:6]
        if keywords:
            queries.append(" ".join(keywords[:4]))

    unique_queries: List[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized.lower() not in seen:
            unique_queries.append(normalized)
            seen.add(normalized.lower())
    return unique_queries[:6]


def classify_email_domain(email: str, company_domain: str, official_domain: str) -> str:
    if not email:
        return "NO EMAIL"
    domain = normalize_domain(email.split("@")[-1]) if "@" in email else ""
    if not domain:
        return "UNKNOWN"
    if domain == company_domain or domain == official_domain:
        return "MATCH"
    if domain.endswith("gmail.com") or domain.endswith("yahoo.com") or domain.endswith("outlook.com"):
        return "PERSONAL EMAIL"
    if company_domain and domain != company_domain:
        return "MISMATCH"
    return "UNKNOWN"


def company_verification(company_name: str, search_results: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    if not company_name:
        return {
            "company_found": False,
            "official_domain_found": False,
            "careers_page_found": None,
            "status": "UNABLE_TO_VERIFY",
            "status_label": "Unable to Verify",
            "evidence": [],
            "company_domain": "",
            "official_domain": "",
            "summary": "Unable to verify the exact company name.",
        }

    usable_results: List[Dict[str, str]] = []
    for item in search_results:
        link = (item.get("link") or "").strip()
        if not is_valid_url(link):
            continue
        usable_results.append(
            {
                "title": (item.get("title") or "").strip() or "Search result",
                "link": link,
                "snippet": (item.get("snippet") or "").strip(),
            }
        )

    matching_results = [
        item
        for item in usable_results
        if _company_mentioned_in_title(company_name, item["title"])
    ]
    status = "FOUND" if matching_results else "NOT_FOUND"
    summary = (
        "A search result mentions this company. The result is a candidate to inspect "
        "and does not independently confirm the company's identity or official website."
        if matching_results
        else "Unable to verify the exact company name. This does not prove that "
        "the employer is illegitimate."
    )

    return {
        "company_found": bool(matching_results),
        "official_domain_found": False,
        "careers_page_found": None,
        "status": status,
        "status_label": "Found" if matching_results else "Not Found",
        "evidence": matching_results[:5],
        "company_domain": "",
        "official_domain": "",
        "summary": summary,
        "official_website_candidate": (
            matching_results[0]["link"] if matching_results else ""
        ),
    }


def job_verification(job_data: Dict[str, Any], search_results: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    company = (job_data.get("company_name") or job_data.get("company") or "").strip()
    title = (job_data.get("title") or "").strip()
    location = (job_data.get("location") or "").strip()
    if not company or not title:
        return {
            "job_found_officially": False,
            "job_found_third_party": False,
            "status": "UNABLE_TO_VERIFY",
            "status_label": "Unable to Verify",
            "evidence": [],
            "summary": (
                "A company name and exact job title are needed to search for a "
                "matching vacancy."
            ),
            "conflicting_information": False,
        }

    usable_results = []
    for item in search_results:
        link = (item.get("link") or "").strip()
        if is_valid_url(link):
            usable_results.append(
                {
                    "title": (item.get("title") or "").strip() or "Search result",
                    "link": link,
                    "snippet": (item.get("snippet") or "").strip(),
                }
            )

    possible_hits = []
    for item in usable_results:
        searchable_text = " ".join(
            (item["title"], item["snippet"], item["link"])
        ).casefold()
        if company.casefold() not in searchable_text:
            continue
        if title.casefold() not in searchable_text:
            continue
        if location and location.casefold() not in searchable_text:
            continue
        possible_hits.append(item)

    status = "FOUND" if possible_hits else "NOT_FOUND"
    summary = (
        "A result contains the company, job title, and supplied location. It is a "
        "candidate vacancy match; confirm the listing and its current status on the "
        "employer's careers page or the source site."
        if possible_hits
        else "No search result matched the company, exact job title, and supplied "
        "location. This does not prove that the vacancy is fraudulent."
    )
    return {
        "job_found_officially": False,
        "job_found_third_party": False,
        "status": status,
        "status_label": "Found" if possible_hits else "Not Found",
        "evidence": possible_hits[:5],
        "summary": summary,
        "conflicting_information": False,
    }


def evaluate_application_domain(application_url: Optional[str], company_domain: str, official_domain: str) -> Dict[str, Any]:
    if not application_url:
        return {"status": "NO APPLICATION URL", "domain": "", "mismatch": False, "summary": "No application URL was provided in the posting."}
    domain = normalize_domain(application_url)
    if not domain:
        return {"status": "UNKNOWN", "domain": "", "mismatch": False, "summary": "Could not extract a usable application domain."}
    if company_domain and domain and domain != company_domain and domain != official_domain:
        status = "DOMAIN MISMATCH"
        summary = "The application domain does not match the identified company domain."
        mismatch = True
    elif official_domain and domain == official_domain:
        status = "MATCH"
        summary = "The application URL appears to align with the company domain."
        mismatch = False
    else:
        status = "UNKNOWN"
        summary = "Application URL was present, but the domain match could not be verified confidently."
        mismatch = False
    return {"status": status, "domain": domain, "mismatch": mismatch, "summary": summary}


def verify_contact_details(company_name: str, company_domain: str, official_domain: str, job_text: str) -> Dict[str, Any]:
    emails = extract_email_addresses(job_text)
    phone_numbers = extract_phone_numbers(job_text)
    if emails:
        email = emails[0]
        status = classify_email_domain(email, company_domain, official_domain)
        summary = f"Recruiter email domain status: {status}."
    else:
        email = ""
        status = "NO EMAIL"
        summary = "No recruiter or contact email was found in the posting."

    return {
        "email": email,
        "email_domain_status": status,
        "phone_numbers": phone_numbers,
        "summary": summary,
    }


def fetch_and_extract_url_data(url: str) -> Dict[str, Any]:
    page = fetch_public_page(url)
    if not page.get("ok"):
        return {"ok": False, "error": page.get("error", "Could not access the page."), "title": "", "text": "", "company_name": ""}
    html = page.get("content", "")
    text = extract_readable_text_from_html(html)
    title = re.findall(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    title_text = re.sub(r"\s+", " ", title[0]).strip() if title else ""
    company_name = ""
    company_match = re.search(r"(?i)(?:company|employer|organization)\s*[:\-]\s*([A-Za-z0-9&.,'\- ]{2,80})", text)
    if company_match:
        company_name = company_match.group(1).strip(" -•")
    return {"ok": True, "error": "", "title": title_text, "text": text, "company_name": company_name, "final_url": page.get("final_url", url)}


__all__ = [
    "build_search_queries",
    "company_verification",
    "classify_email_domain",
    "evaluate_application_domain",
    "extract_application_urls",
    "extract_domain_from_url",
    "extract_email_addresses",
    "extract_phone_numbers",
    "extract_requisition_id",
    "extract_readable_text_from_html",
    "fetch_and_extract_url_data",
    "fetch_public_page",
    "is_valid_url",
    "job_verification",
    "has_company_candidate",
    "has_official_company_candidate",
    "has_vacancy_candidate",
    "normalize_domain",
    "search_web",
    "summarize_search_evidence",
    "verify_contact_details",
]
