"""Production preprocessing utilities for the fake-job-detection pipeline.

This module mirrors the preprocessing and feature-engineering logic used during
training in the project notebooks, so a single new job posting can be converted
into the exact sparse matrix expected by the final Linear SVM model.

Important notes
---------------
- The notebook logic is reproduced here without changing the model or training
  methodology.
- The final model input is assembled in the exact order used in training:
  binary -> numerical -> missingness -> categorical -> TF-IDF -> engineered
  and then hstack(...) to a CSR matrix.
- The final model expects 5,457 features. If the transformed matrix does not have
  exactly 5,457 columns, a ValueError is raised.

The saved artifacts are loaded from the project-root-relative paths:
- models/engineered_features.pkl
- models/engineered_imputer.pkl
- data/processed/tfidf_vectorizer.pkl
- data/processed/categorical_pipeline.pkl
- data/processed/numerical_pipeline.pkl
- data/processed/binary_features.pkl
- data/processed/categorical_features.pkl
- data/processed/numerical_features.pkl
- data/processed/text_features.pkl
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPECTED_FEATURE_COUNT = 5457
BINARY_DEFAULTS = {
    "telecommuting": 0.0,
    "has_company_logo": 1.0,
    "has_questions": 0.0,
}


def load_preprocessing_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load the fitted preprocessing and feature-engineering artifacts.

    The file list is intentionally limited to the artifacts that are required to
    reproduce the training-time feature space for a single new job posting.
    """
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR

    return {
        "engineered_features": joblib.load(models_dir / "engineered_features.pkl"),
        "engineered_imputer": joblib.load(models_dir / "engineered_imputer.pkl"),
        "tfidf_vectorizer": joblib.load(processed_dir / "tfidf_vectorizer.pkl"),
        "categorical_pipeline": joblib.load(processed_dir / "categorical_pipeline.pkl"),
        "numerical_pipeline": joblib.load(processed_dir / "numerical_pipeline.pkl"),
        "binary_features": joblib.load(processed_dir / "binary_features.pkl"),
        "categorical_features": joblib.load(processed_dir / "categorical_features.pkl"),
        "numerical_features": joblib.load(processed_dir / "numerical_features.pkl"),
        "text_features": joblib.load(processed_dir / "text_features.pkl"),
    }


def clean_text(text: Any) -> str:
    """Mirror the text cleaning logic used in the preprocessing notebook.

    The notebook used:
      - lowercase conversion
      - HTML tag removal
      - URL removal
      - whitespace normalization
      - removal of non-word characters except punctuation that remains useful
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?$%+\-/]", " ", text)
    return text.strip()


def _coerce_job_record(job_posting: Union[Mapping[str, Any], pd.DataFrame, pd.Series]) -> pd.Series:
    """Normalize a single job record to a one-row pandas Series."""
    if isinstance(job_posting, pd.DataFrame):
        if job_posting.shape[0] != 1:
            raise ValueError("Expected a single job posting row, but a DataFrame with more than one row was provided.")
        record = job_posting.iloc[0].copy()
    elif isinstance(job_posting, pd.Series):
        record = job_posting.copy()
    elif isinstance(job_posting, Mapping):
        record = pd.Series(job_posting)
    else:
        raise TypeError("job_posting must be a dictionary, pandas Series, or single-row DataFrame.")

    for column, value in record.items():
        if isinstance(value, (list, tuple, dict, set, np.ndarray)):
            raise TypeError(f"Job posting field {column!r} must be a scalar value.")
    return record


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_binary_flag(value: Any, field_name: str) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Binary field {field_name!r} must be 0 or 1.")
    if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
        raise ValueError(f"Binary field {field_name!r} must be 0 or 1.")
    return int(numeric_value)


def _parse_salary(value: Any) -> tuple[float, float]:
    """Parse salary strings in the same way used by the preprocessing notebook."""
    if pd.isna(value):
        return (np.nan, np.nan)

    value = str(value)
    numbers = re.findall(r"\d+(?:\.\d+)?", value)

    if len(numbers) >= 2:
        min_salary = float(numbers[0])
        max_salary = float(numbers[1])
        if min_salary > max_salary:
            min_salary, max_salary = max_salary, min_salary
        return (min_salary, max_salary)

    if len(numbers) == 1:
        salary = float(numbers[0])
        return (salary, salary)

    return (np.nan, np.nan)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio matching the notebook helper function."""
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def create_engineered_features(
    job_posting: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Construct the engineered features exactly as in the training notebook.

    The final engineered feature list is the 9 columns saved in the model artifact:
    - total_word_count
    - text_section_count
    - salary_range_log
    - company_profile_present
    - requirements_present
    - benefits_present
    - requirements_description_ratio_log
    - benefits_description_ratio_log
    - company_description_ratio_log
    """
    if artifacts is None:
        artifacts = load_preprocessing_artifacts()

    record = _coerce_job_record(job_posting)
    text_columns = list(artifacts["text_features"])

    for column in text_columns:
        value = record.get(column)
        if not _is_missing(value) and not isinstance(value, str):
            raise TypeError(f"Job posting text field {column!r} must be a string.")

    cleaned_values = {col: clean_text(record.get(col)) for col in text_columns}
    combined_text = " ".join(cleaned_values[col] for col in text_columns)

    word_counts = {col: len(str(cleaned_values[col]).split()) for col in text_columns}
    total_word_count = sum(word_counts.values())
    text_section_count = int(sum(1 for col in text_columns if cleaned_values[col].strip()))

    # Salary range mirrors the notebook's salary parsing and log transform.
    salary_range_raw = record.get("salary_range")
    if pd.isna(salary_range_raw):
        min_salary = pd.to_numeric(record.get("min_salary"), errors="coerce")
        max_salary = pd.to_numeric(record.get("max_salary"), errors="coerce")
        if pd.notna(min_salary) and pd.notna(max_salary):
            salary_range = float(max_salary - min_salary)
        else:
            salary_range = np.nan
    else:
        min_salary, max_salary = _parse_salary(salary_range_raw)
        if pd.notna(min_salary) and pd.notna(max_salary):
            salary_range = float(max_salary - min_salary)
        else:
            salary_range = np.nan

    requirements_word_count = word_counts.get("requirements", 0)
    description_word_count = word_counts.get("description", 0)
    benefits_word_count = word_counts.get("benefits", 0)
    company_profile_word_count = word_counts.get("company_profile", 0)

    requirements_description_ratio = _safe_ratio(requirements_word_count, description_word_count)
    benefits_description_ratio = _safe_ratio(benefits_word_count, description_word_count)
    company_description_ratio = _safe_ratio(company_profile_word_count, description_word_count)

    engineered_dict = {
        "total_word_count": float(total_word_count),
        "text_section_count": float(text_section_count),
        "salary_range_log": float(np.log1p(salary_range)) if pd.notna(salary_range) else np.nan,
        "company_profile_present": int(bool(cleaned_values["company_profile"].strip())),
        "requirements_present": int(bool(cleaned_values["requirements"].strip())),
        "benefits_present": int(bool(cleaned_values["benefits"].strip())),
        "requirements_description_ratio_log": float(np.log1p(requirements_description_ratio)) if pd.notna(requirements_description_ratio) else np.nan,
        "benefits_description_ratio_log": float(np.log1p(benefits_description_ratio)) if pd.notna(benefits_description_ratio) else np.nan,
        "company_description_ratio_log": float(np.log1p(company_description_ratio)) if pd.notna(company_description_ratio) else np.nan,
    }

    engineered_df = pd.DataFrame([engineered_dict])
    engineered_order = list(artifacts["engineered_features"]) if isinstance(artifacts["engineered_features"], Iterable) else list(engineered_dict.keys())
    engineered_df = engineered_df[engineered_order]

    # Preserve the exact training-time field order for the final model.
    return engineered_df


def transform_job_posting(
    job_posting: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
    artifacts: Optional[Dict[str, Any]] = None,
) -> csr_matrix:
    """Transform one job posting into the sparse feature matrix expected by the model.

    The order matches the training notebook exactly:
      binary -> numerical -> missingness -> categorical -> TF-IDF -> engineered
    followed by hstack(..., format='csr').
    """
    if artifacts is None:
        artifacts = load_preprocessing_artifacts()

    record = _coerce_job_record(job_posting)

    # 1) Raw text preprocessing
    text_columns = list(artifacts["text_features"])
    input_columns = (
        text_columns
        + list(artifacts["categorical_features"])
        + list(artifacts["numerical_features"])
        + list(artifacts["binary_features"])
    )
    if not any(not _is_missing(record.get(column)) for column in input_columns):
        raise ValueError("Job posting has no usable model input fields.")

    for column in text_columns:
        value = record.get(column)
        if not _is_missing(value) and not isinstance(value, str):
            raise TypeError(f"Job posting text field {column!r} must be a string.")

    cleaned_text = {col: clean_text(record.get(col)) for col in artifacts["text_features"]}
    combined_job_text = " ".join(cleaned_text[col] for col in artifacts["text_features"])

    # 2) TF-IDF block
    tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    tfidf_matrix = tfidf_vectorizer.transform([combined_job_text])

    # 3) Categorical preprocessing block
    categorical_cols = list(artifacts["categorical_features"])
    categorical_df = pd.DataFrame(index=[0], columns=categorical_cols, dtype=object)
    for col in categorical_cols:
        raw_value = record.get(col)
        categorical_df.loc[0, col] = "Unknown" if _is_missing(raw_value) else str(raw_value)

    categorical_matrix = artifacts["categorical_pipeline"].transform(categorical_df[categorical_cols])

    # 4) Numerical preprocessing block
    numerical_cols = list(artifacts["numerical_features"])
    numerical_df = pd.DataFrame(index=[0], columns=numerical_cols, dtype=float)
    text_word_counts = {
        f"{col}_word_count": len(cleaned_text.get(col, "").split())
        for col in artifacts["text_features"]
    }
    for col in numerical_cols:
        value = record.get(col)
        if _is_missing(value) and col in text_word_counts:
            numerical_df.loc[0, col] = text_word_counts[col]
        elif _is_missing(value):
            numerical_df.loc[0, col] = np.nan
        else:
            numeric_value = pd.to_numeric(value, errors="raise")
            if not np.isfinite(float(numeric_value)):
                raise ValueError(f"Numeric field {col!r} must be finite.")
            numerical_df.loc[0, col] = numeric_value

    numerical_matrix = artifacts["numerical_pipeline"].transform(numerical_df[numerical_cols])

    # 5) Binary feature block. These defaults match the training-set modes.
    binary_cols = list(artifacts["binary_features"])
    binary_array = np.zeros((1, len(binary_cols)), dtype=float)
    for idx, col in enumerate(binary_cols):
        value = record.get(col)
        if _is_missing(value):
            binary_array[0, idx] = BINARY_DEFAULTS.get(col, 0.0)
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Binary field {col!r} must be 0 or 1.")
            if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
                raise ValueError(f"Binary field {col!r} must be 0 or 1.")
            binary_array[0, idx] = numeric_value

    binary_matrix = csr_matrix(binary_array)

    # 6) Separate missingness block from the preprocessing notebook.
    company_profile_missing = record.get("company_profile_missing")
    if _is_missing(company_profile_missing):
        company_profile_missing = int(_is_missing(record.get("company_profile")))
    else:
        company_profile_missing = _coerce_binary_flag(
            company_profile_missing,
            "company_profile_missing",
        )

    salary_missing = record.get("salary_missing")
    if _is_missing(salary_missing):
        salary_present = not _is_missing(record.get("salary_range"))
        if not salary_present:
            salary_present = any(
                not _is_missing(record.get(column))
                for column in ("min_salary", "max_salary", "mid_salary")
            )
        salary_missing = int(not salary_present)
    else:
        salary_missing = _coerce_binary_flag(salary_missing, "salary_missing")

    missingness_array = np.array(
        [[company_profile_missing, salary_missing]], dtype=float
    )
    missingness_matrix = csr_matrix(missingness_array)

    # 7) Engineered features block
    engineered_df = create_engineered_features(record, artifacts=artifacts)
    engineered_matrix = artifacts["engineered_imputer"].transform(engineered_df)
    engineered_sparse = csr_matrix(engineered_matrix)

    # 8) Final feature combination in the SAME order used in training.
    # The notebook stacked:
    #   binary + numerical + missingness + categorical + TF-IDF + engineered
    final_matrix = hstack(
        [
            binary_matrix,
            numerical_matrix,
            missingness_matrix,
            categorical_matrix,
            tfidf_matrix,
            engineered_sparse,
        ],
        format="csr",
    )

    if final_matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Feature mismatch: the transformed job posting produced "
            f"{final_matrix.shape[1]} features, but the trained Linear SVM expects exactly "
            f"{EXPECTED_FEATURE_COUNT} features. This usually means the preprocessing "
            "logic or feature ordering differs from the training notebook."
        )
    if not np.isfinite(final_matrix.data).all():
        raise ValueError("Preprocessing produced non-finite model features.")

    return final_matrix


__all__ = [
    "EXPECTED_FEATURE_COUNT",
    "clean_text",
    "create_engineered_features",
    "load_preprocessing_artifacts",
    "transform_job_posting",
]
