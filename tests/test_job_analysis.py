from __future__ import annotations

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import requests
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest
from scipy.sparse import csr_matrix, hstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import auth as auth_store
import app as app_module
from app import _get_feature_names, get_instance_xai
from auth import (
    AuthError,
    LoginRateLimitError,
    authenticate,
    create_account,
    create_session,
    get_session_user,
    initialize_database,
    revoke_session,
    session_is_active,
    SESSION_IDLE_TIMEOUT_SECONDS,
    SESSION_MAX_AGE_SECONDS,
)
from job_extraction import assess_posting_completeness, extract_job_fields
from model import load_model, predict_job_posting
from preprocessing import (
    EXPECTED_FEATURE_COUNT,
    create_engineered_features,
    load_preprocessing_artifacts,
    transform_job_posting,
)
from risk_engine import risk_band, rule_risk_score, web_risk_score
from web_verification import (
    _company_identity_matches,
    _domain_matches_company,
    company_verification,
    extract_requisition_id,
    is_valid_url,
    job_verification,
    search_web,
    summarize_search_evidence,
    summarize_source,
    verify_company_and_vacancy,
)
from xai import analyze_risk_factors


SUSPICIOUS_POSTING = """\
Job Title: Remote Data Entry Assistant
Company: Acme Data Inc
Location: Remote
Salary: $5,000 per week
Description: Easy money! No experience required. Pay a $99 application fee
today to start immediately. Send your bank details and OTP on WhatsApp.
Requirements: No experience required.
"""

LEGITIMATE_LOOKING_POSTING = """\
Job Title: Senior Data Analyst
Company: Northwind Analytics
Location: Seattle, WA
Employment Type: Full-time
Salary: $90,000-$120,000 per year
About Company: Northwind Analytics builds data tools for healthcare teams.
Description: You will develop dashboards, analyze data, and work with product
teams. Responsibilities include reporting to the Analytics Director.
Requirements: 5 years of SQL and Python experience, bachelor's degree preferred.
Benefits: Health insurance and paid leave.
"""

INCOMPLETE_POSTING = "Data entry job. Work from home. Apply now."
ALTERNATE_FORMAT_POSTING = (
    "Job Title — Product Analyst | Company Name = Example Research Ltd | "
    "Location: Boston, MA | Employment Type: Full-time | "
    "Description: Analyze customer research, prepare reports, and collaborate "
    "with product and design teams to improve product decisions. "
    "Requirements: SQL, spreadsheet modeling, and clear written communication."
)
MISLEADING_POSTING = (
    "Job Title: Immediate Hiring Assistant\n"
    "Company: Bright Future Group\nLocation: Remote\n"
    "Description: Urgently hiring. Start today and earn $1,000 per day. "
    "Pay a registration fee to secure this position. No interview required.\n"
    "Requirements: No experience necessary."
)


class AuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_directory.name) / "accounts.sqlite3"
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_account_registration_hashes_password_and_login_is_case_insensitive(self) -> None:
        user_id = create_account(
            "JobSignal User",
            "Person@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
            now=100,
        )
        connection = __import__("sqlite3").connect(self.db_path)
        try:
            password_salt, password_hash = connection.execute(
                "SELECT password_salt, password_hash FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        finally:
            connection.close()

        self.assertNotEqual(password_hash, b"A-long-unique-password-93")
        self.assertEqual(len(password_salt), 16)
        self.assertEqual(len(password_hash), 32)
        user = authenticate(
            "PERSON@EXAMPLE.COM",
            "A-long-unique-password-93",
            db_path=self.db_path,
            now=101,
        )
        self.assertEqual(user["user_id"], user_id)
        self.assertEqual(user["full_name"], "JobSignal User")
        self.assertNotIn("password_hash", user)

    def test_registration_validates_email_password_and_duplicate_accounts(self) -> None:
        with self.assertRaisesRegex(AuthError, "valid email"):
            create_account("User", "not-an-email", "A-long-unique-password-93", db_path=self.db_path)
        with self.assertRaisesRegex(AuthError, "12 characters"):
            create_account("User", "user@example.com", "short", db_path=self.db_path)
        with self.assertRaisesRegex(AuthError, "name"):
            create_account("", "user@example.com", "A-long-unique-password-93", db_path=self.db_path)

        create_account(
            "User",
            "user@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
        )
        with self.assertRaisesRegex(AuthError, "already exists"):
            create_account(
                "Another user",
                "USER@example.com",
                "A-different-password-93",
                db_path=self.db_path,
            )

    def test_login_failures_are_rate_limited_without_account_enumeration(self) -> None:
        create_account(
            "User",
            "user@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
            now=100,
        )
        self.assertIsNone(
            authenticate("user@example.com", "incorrect-password", db_path=self.db_path, now=100)
        )
        self.assertIsNone(
            authenticate("missing@example.com", "incorrect-password", db_path=self.db_path, now=100)
        )
        for _ in range(4):
            authenticate(
                "user@example.com",
                "incorrect-password",
                db_path=self.db_path,
                now=100,
            )
        with self.assertRaises(LoginRateLimitError):
            authenticate(
                "user@example.com",
                "A-long-unique-password-93",
                db_path=self.db_path,
                now=101,
            )
        user = authenticate(
            "user@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
            now=100 + 15 * 60,
        )
        self.assertIsNotNone(user)

    def test_server_side_sessions_expire_on_idle_and_absolute_limits(self) -> None:
        user_id = create_account(
            "User",
            "user@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
            now=100,
        )
        token = create_session(user_id, db_path=self.db_path, now=100)
        self.assertTrue(session_is_active(token, db_path=self.db_path, now=100))
        self.assertEqual(
            get_session_user(token, db_path=self.db_path, now=100)["user_id"],
            user_id,
        )
        self.assertTrue(
            session_is_active(
                token,
                db_path=self.db_path,
                now=100 + SESSION_IDLE_TIMEOUT_SECONDS - 1,
            )
        )
        self.assertFalse(
            session_is_active(
                token,
                db_path=self.db_path,
                now=100 + SESSION_IDLE_TIMEOUT_SECONDS,
            )
        )
        self.assertIsNone(
            get_session_user(
                token,
                db_path=self.db_path,
                now=100 + SESSION_IDLE_TIMEOUT_SECONDS,
            )
        )

        long_session = create_session(user_id, db_path=self.db_path, now=1000)
        for activity in range(
            1000 + SESSION_IDLE_TIMEOUT_SECONDS - 1,
            1000 + SESSION_MAX_AGE_SECONDS,
            SESSION_IDLE_TIMEOUT_SECONDS - 1,
        ):
            self.assertIsNotNone(
                get_session_user(long_session, db_path=self.db_path, now=activity)
            )
        self.assertFalse(
            session_is_active(
                long_session,
                db_path=self.db_path,
                now=1000 + SESSION_MAX_AGE_SECONDS,
            )
        )

    def test_logout_revokes_the_server_side_session(self) -> None:
        user_id = create_account(
            "User",
            "user@example.com",
            "A-long-unique-password-93",
            db_path=self.db_path,
        )
        token = create_session(user_id, db_path=self.db_path)
        revoke_session(token, db_path=self.db_path)
        self.assertIsNone(get_session_user(token, db_path=self.db_path))


class JobAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model()
        cls.artifacts = load_preprocessing_artifacts()
        cls.auth_temp_directory = tempfile.TemporaryDirectory()
        cls.auth_db_path = Path(cls.auth_temp_directory.name) / "app_accounts.sqlite3"
        cls.auth_db_patcher = patch.object(auth_store, "AUTH_DB_PATH", cls.auth_db_path)
        cls.auth_db_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.auth_db_patcher.stop()
        cls.auth_temp_directory.cleanup()

    def _open_app(self) -> AppTest:
        app_module._cached_search.clear()
        initialize_database(self.auth_db_path)
        email = f"user-{id(self)}@example.com"
        user_id = create_account(
            f"Test User {id(self)}",
            email,
            "A-long-unique-password-93",
            db_path=self.auth_db_path,
        )
        token = create_session(user_id, db_path=self.auth_db_path)
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        )
        app.session_state["auth_token"] = token
        app.run()
        self.assertFalse(app.exception)
        next(
            button
            for button in app.button
            if button.label == "Job Posting Analysis"
        ).click().run()
        self.assertFalse(app.exception)
        return app

    def test_suspicious_posting_is_structured_and_flags_posted_evidence(self) -> None:
        fields = extract_job_fields(SUSPICIOUS_POSTING)
        analysis = analyze_risk_factors(SUSPICIOUS_POSTING)

        self.assertEqual(fields["title"], "Remote Data Entry Assistant")
        self.assertEqual(fields["company_name"], "Acme Data Inc")
        self.assertEqual(fields["min_salary"], 5000.0)
        self.assertEqual(fields["max_salary"], 5000.0)
        self.assertEqual(
            {item["id"] for item in analysis["risks"]} & {
                "payment_request",
                "sensitive_information",
            },
            {"payment_request", "sensitive_information"},
        )
        self.assertGreater(rule_risk_score(analysis["risks"]), 0)
        prediction = predict_job_posting(fields, self.model, self.artifacts)
        self.assertEqual(prediction["feature_count"], EXPECTED_FEATURE_COUNT)
        self.assertIn(prediction["prediction"], (0, 1))

    def test_legitimate_looking_and_negated_warning_language(self) -> None:
        fields = extract_job_fields(LEGITIMATE_LOOKING_POSTING)
        analysis = analyze_risk_factors(LEGITIMATE_LOOKING_POSTING)
        no_fee_analysis = analyze_risk_factors(
            "We never charge an application fee."
        )

        self.assertEqual(fields["company_name"], "Northwind Analytics")
        self.assertEqual(fields["min_salary"], 90000.0)
        self.assertEqual(fields["max_salary"], 120000.0)
        self.assertFalse(analysis["risks"])
        self.assertTrue(analysis["positives"])
        prediction = predict_job_posting(fields, self.model, self.artifacts)
        self.assertEqual(prediction["feature_count"], EXPECTED_FEATURE_COUNT)
        self.assertIn(prediction["prediction"], (0, 1))
        self.assertEqual(
            risk_band(rule_risk_score(no_fee_analysis["risks"])),
            "LOW RISK",
        )

    def test_incomplete_posting_is_analyzed_but_marked_incomplete(self) -> None:
        fields = extract_job_fields(INCOMPLETE_POSTING)
        completeness = assess_posting_completeness(fields)

        self.assertTrue(completeness["is_incomplete"])
        self.assertTrue(completeness["is_insufficient"])
        self.assertLess(completeness["score"], 50)
        self.assertIn(
            "No ML classification",
            analyze_risk_factors(INCOMPLETE_POSTING)["overall"],
        )

    def test_alternate_heading_and_single_line_format_are_parsed(self) -> None:
        fields = extract_job_fields(ALTERNATE_FORMAT_POSTING)
        completeness = assess_posting_completeness(fields)

        self.assertEqual(fields["title"], "Product Analyst")
        self.assertEqual(fields["company_name"], "Example Research Ltd")
        self.assertEqual(fields["location"], "Boston, MA")
        self.assertIn("customer research", fields["description"])
        self.assertIn("SQL", fields["requirements"])
        self.assertFalse(completeness["is_insufficient"])

    def test_extracted_company_name_preserves_its_original_word_order(self) -> None:
        posting = (
            "Job Title: Data Analyst\n"
            "Company: Global Work Hub\n"
            "Location: Remote\n"
            "Description: Analyze business data and prepare reports for the team."
        )

        self.assertEqual(
            extract_job_fields(posting)["company_name"],
            "Global Work Hub",
        )

    def test_misleading_posting_has_rule_evidence_and_a_separate_ml_result(self) -> None:
        fields = extract_job_fields(MISLEADING_POSTING)
        analysis = analyze_risk_factors(MISLEADING_POSTING)
        result = predict_job_posting(fields, self.model, self.artifacts)

        risk_ids = {item["id"] for item in analysis["risks"]}
        self.assertTrue({"payment_request", "urgency"}.issubset(risk_ids))
        self.assertEqual(result["feature_count"], EXPECTED_FEATURE_COUNT)
        self.assertIn(result["prediction"], (0, 1))
        self.assertIsNone(result["probability"])

    def test_rule_risk_bands_cover_low_medium_and_high(self) -> None:
        self.assertEqual(risk_band(0), "LOW RISK")
        self.assertEqual(risk_band(40), "MEDIUM RISK")
        self.assertEqual(risk_band(70), "HIGH RISK")

    def test_invalid_inputs_and_invalid_numeric_features_fail_explicitly(self) -> None:
        with self.assertRaises(TypeError):
            extract_job_fields(None)
        with self.assertRaises(ValueError):
            extract_job_fields("x" * 20_001)
        with self.assertRaises(ValueError):
            predict_job_posting(
                {"description": "A detailed role description", "telecommuting": 2},
                self.model,
                self.artifacts,
            )
        with self.assertRaises(ValueError):
            predict_job_posting(
                {"description": "A detailed role description", "min_salary": "not-a-number"},
                self.model,
                self.artifacts,
            )
        with self.assertRaises(FileNotFoundError):
            load_model(PROJECT_ROOT / "models" / "missing_model.pkl")

    def test_live_transform_matches_saved_training_feature_order(self) -> None:
        test_rows = joblib.load(PROJECT_ROOT / "data" / "processed" / "X_test_cleaned.pkl")
        saved_features = joblib.load(PROJECT_ROOT / "data" / "processed" / "X_test_final.pkl")
        row = test_rows.iloc[0]

        actual = transform_job_posting(row, self.artifacts)
        engineered = self.artifacts["engineered_imputer"].transform(
            create_engineered_features(row, self.artifacts)
        )
        expected = hstack(
            [saved_features[0], csr_matrix(engineered)],
            format="csr",
        )

        self.assertEqual(actual.shape, (1, EXPECTED_FEATURE_COUNT))
        self.assertEqual((actual != expected).nnz, 0)

    def test_linear_explanation_reconciles_with_model_margin(self) -> None:
        fields = extract_job_fields(SUSPICIOUS_POSTING)
        result = predict_job_posting(fields, self.model, self.artifacts)
        fraud_factors, legitimate_factors = get_instance_xai(
            SUSPICIOUS_POSTING,
            self.model,
            self.artifacts,
            result["feature_vector"],
        )
        coefficients = self.model.coef_[0]
        reconstructed = float(
            result["feature_vector"].dot(coefficients)[0]
            + self.model.intercept_[0]
        )

        self.assertEqual(
            len(_get_feature_names(self.artifacts)),
            EXPECTED_FEATURE_COUNT,
        )
        self.assertLessEqual(len(fraud_factors), 8)
        self.assertLessEqual(len(legitimate_factors), 8)
        self.assertAlmostEqual(
            reconstructed,
            result["decision_score"],
            places=6,
        )
        self.assertTrue((fraud_factors["contribution"] > 0).all())
        self.assertTrue((legitimate_factors["contribution"] < 0).all())

    def test_search_configuration_and_provider_failures_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
            search_web("example employer")

        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "provider unavailable"
        )
        with patch("web_verification.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Tavily search request failed"):
                search_web("example employer", "test-key")

    def test_tavily_rejected_api_key_error_does_not_expose_key(self) -> None:
        provider_response = Mock()
        provider_response.status_code = 401
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "unauthorized",
            response=provider_response,
        )

        with patch("web_verification.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "rejected the API key") as error:
                search_web("example employer", "private-test-key")

        self.assertIn("HTTP 401", str(error.exception))
        self.assertNotIn("private-test-key", str(error.exception))

    def test_tavily_request_uses_backend_auth_header_and_normalizes_results(self) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "Company careers",
                    "url": "https://company.example/careers",
                    "content": "Open roles at Company",
                },
                {"title": "Invalid result", "url": None, "content": "ignored"},
            ]
        }
        with patch(
            "web_verification.requests.post",
            return_value=response,
        ) as request:
            results = search_web("Company careers", "test-secret", max_results=3)

        self.assertEqual(
            request.call_args.args[0],
            "https://api.tavily.com/search",
        )
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-secret",
        )
        self.assertNotIn("test-secret", request.call_args.args[0])
        self.assertEqual(
            request.call_args.kwargs["json"],
            {
                "query": "Company careers",
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        self.assertEqual(
            results,
            [{
                "title": "Company careers",
                "link": "https://company.example/careers",
                "snippet": "Open roles at Company",
            }],
        )

    def test_microsoft_employer_is_extracted_from_fields_and_careers_domain(self) -> None:
        postings = (
            (
                "Job Title: Senior Software Engineer\n"
                "Company: Microsoft Corporation\n"
                "Location: Redmond, WA\n"
                "Description: Build secure cloud services with a global engineering team.",
                "Microsoft Corporation",
            ),
            (
                "Senior Software Engineer\n"
                "https://jobs.careers.microsoft.com/global/en/job/123\n"
                "Location: Redmond, WA\n"
                "Description: Build secure cloud services with a global engineering team.",
                "Microsoft",
            ),
            (
                "Job Title: Senior Software Engineer\n"
                "Employer: Microsoft Careers\n"
                "Description: Build secure cloud services with a global engineering team.",
                "Microsoft",
            ),
            (
                "Job Title: Senior Software Engineer\n"
                "Posted by: Microsoft Careers\n"
                "Description: Build secure cloud services with a global engineering team.",
                "",
            ),
        )

        for posting, employer in postings:
            with self.subTest(posting=posting[:40]):
                self.assertEqual(
                    extract_job_fields(posting)["company_name"],
                    employer,
                )

    def test_missing_or_ambiguous_employer_is_not_guessed_from_posted_by(self) -> None:
        posting = (
            "Job Title: Data Analyst\nPosted by: Indeed\n"
            "Description: Analyze data and prepare reports for the team. "
            "The team uses Microsoft products."
        )

        self.assertEqual(extract_job_fields(posting)["company_name"], "")

    def test_official_company_can_be_supported_without_a_vacancy_match(self) -> None:
        result = verify_company_and_vacancy(
            "Microsoft Corporation",
            "Senior Software Engineer",
            "Redmond, WA",
            [{
                "title": "Microsoft official company website",
                "link": "https://www.microsoft.com/en-us/",
                "snippet": "About Microsoft and its products.",
            }],
            [],
            page_fetcher=lambda *args, **kwargs: {"ok": False},
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["vacancy"]["status"], "NOT_FOUND")
        self.assertEqual(result["company"]["company_name"], "Microsoft Corporation")
        self.assertIsNotNone(result["company"]["source"])
        self.assertFalse(result["company"]["profile_available"])
        self.assertFalse(result["company"]["source"]["profile_available"])
        self.assertIsNone(result["vacancy"]["source"])

    def test_matching_official_vacancy_requires_readable_page_evidence(self) -> None:
        result = verify_company_and_vacancy(
            "Microsoft",
            "Software Engineer",
            "Redmond, WA",
            [{
                "title": "Microsoft official company website",
                "link": "https://www.microsoft.com/",
                "snippet": "About Microsoft.",
            }],
            [{
                "title": "Software Engineer - Microsoft Careers",
                "link": "https://apply.careers.microsoft.com/careers/job/123",
                "snippet": "Software Engineer in Redmond, WA at Microsoft.",
            }],
            page_fetcher=lambda *args, **kwargs: {
                "ok": True,
                "title": "Software Engineer",
                "content": (
                    "<html><title>Software Engineer</title><body><main>"
                    "<p>Microsoft is hiring a Software Engineer in Redmond, WA.</p>"
                    "</main></body></html>"
                ),
            },
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["vacancy"]["status"], "MATCH_FOUND")
        self.assertLessEqual(len(result["sources"]), 3)

    def test_salesforce_related_role_is_not_reported_as_exact_match(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [{
                "title": "Salesforce official company website",
                "link": "https://www.salesforce.com/company/",
                "snippet": "Salesforce company information.",
            }],
            [{
                "title": "Salesforce Patient Services Architect at Salesforce",
                "link": "https://careers.salesforce.com/jobs/123",
                "snippet": "Salesforce Patient Services Architect, Remote.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "final_url": url,
                "content": (
                    "<html><title>Salesforce Life Sciences Functional Architect "
                    "Senior Manager</title><main><h1>Salesforce Life Sciences "
                    "Functional Architect Senior Manager</h1><p>Remote role "
                    "supporting Salesforce Life Sciences customers.</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["vacancy"]["status"], "NOT_FOUND")
        self.assertTrue(result["vacancy"]["related_vacancy_found"])
        self.assertIn("different role", result["vacancy"]["summary"])
        self.assertIsNotNone(result["vacancy"]["source"])

    def test_similar_title_with_shared_role_terms_is_not_an_exact_match(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "",
            [],
            [{
                "title": "Salesforce Life Sciences Functional Architect Senior Manager",
                "link": "https://careers.salesforce.com/jobs/456",
                "snippet": "Salesforce Life Sciences Functional Architect Senior Manager.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    "<html><main><h1>Salesforce Life Sciences Functional "
                    "Architect Senior Manager</h1><p>Salesforce role.</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "NOT_FOUND")
        self.assertTrue(result["vacancy"]["related_vacancy_found"])
        self.assertNotIn("requested job title", result["vacancy"]["summary"])

    def test_title_punctuation_and_case_are_normalized_for_exact_match(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [],
            [{
                "title": "SALESFORCE PATIENT SERVICES ARCHITECT",
                "link": "https://careers.salesforce.com/jobs/789",
                "snippet": "Remote position.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    "<html><main><h1>Salesforce Patient-Services Architect</h1>"
                    "<p>Location: Remote</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "MATCH_FOUND")

    def test_matching_requisition_id_can_confirm_a_revised_page_title(self) -> None:
        self.assertEqual(
            extract_requisition_id(
                "Job Title: Patient Services Architect\nRequisition ID: SF-1042"
            ),
            "SF-1042",
        )
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [],
            [{
                "title": "Salesforce Functional Architect",
                "link": "https://careers.salesforce.com/jobs/1042",
                "snippet": "Requisition ID: SF-1042 · Remote.",
            }],
            requisition_id="SF-1042",
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    '<html><main><h1>Patient Services Architect</h1>'
                    "<p>Requisition ID: SF-1042. Location: Remote.</p>"
                    "</main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "MATCH_FOUND")

    def test_multiple_vacancies_match_only_the_exact_heading(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [],
            [{
                "title": "Salesforce careers",
                "link": "https://careers.salesforce.com/jobs",
                "snippet": "Open roles at Salesforce.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    "<html><title>Careers at Salesforce</title><main>"
                    "<h1>Careers</h1><h2>Salesforce Life Sciences Functional "
                    "Architect Senior Manager</h2><p>San Francisco</p>"
                    "<h2>Salesforce Patient Services Architect</h2>"
                    "<p>Remote</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "MATCH_FOUND")

    def test_location_from_another_vacancy_does_not_confirm_the_requested_role(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [],
            [{
                "title": "Salesforce careers",
                "link": "https://careers.salesforce.com/jobs",
                "snippet": "Open roles at Salesforce.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    "<html><main><h1>Careers</h1><h2>Salesforce Patient "
                    "Services Architect</h2><p>Seattle, WA</p>"
                    "<h2>Salesforce Life Sciences Functional Architect "
                    "Senior Manager</h2><p>Remote</p></main></html>"
                ),
            },
        )

        self.assertNotEqual(result["vacancy"]["status"], "MATCH_FOUND")
        self.assertIn("location", result["vacancy"]["summary"])

    def test_canonical_urls_deduplicate_the_same_source_page(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "",
            [{
                "title": "Salesforce official website",
                "link": "https://careers.salesforce.com/company/",
                "snippet": "Company information.",
            }],
            [{
                "title": "Salesforce Patient Services Architect",
                "link": "https://careers.salesforce.com/jobs/123?source=search",
                "snippet": "Salesforce Patient Services Architect.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "final_url": "https://careers.salesforce.com/jobs/123",
                "content": (
                    '<html><head><link rel="canonical" '
                    'href="https://careers.salesforce.com/jobs/123"></head>'
                    "<main><h1>Salesforce Patient "
                    "Services Architect</h1></main></html>"
                ),
            },
        )

        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(
            sum(
                source["canonical_url"] == "https://careers.salesforce.com/jobs/123"
                for source in result["sources"]
            ),
            1,
        )

    def test_careers_redirect_with_unrelated_listing_does_not_confirm_vacancy(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "Remote",
            [],
            [{
                "title": "Salesforce Patient Services Architect",
                "link": "https://careers.salesforce.com/redirect/123",
                "snippet": "Salesforce Patient Services Architect, Remote.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "final_url": "https://careers.salesforce.com/jobs",
                "content": (
                    "<html><main><h1>Careers</h1><h2>Salesforce Life Sciences "
                    "Functional Architect Senior Manager</h2><p>Remote</p>"
                    "</main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "NOT_FOUND")
        self.assertIn("different role", result["vacancy"]["summary"])

    def test_matching_search_snippet_is_overruled_by_different_page_title(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Salesforce Patient Services Architect",
            "",
            [],
            [{
                "title": "Salesforce Patient Services Architect",
                "link": "https://careers.salesforce.com/jobs/123",
                "snippet": "Salesforce Patient Services Architect role.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "content": (
                    "<html><main><h1>Salesforce Life Sciences Functional "
                    "Architect Senior Manager</h1></main></html>"
                ),
            },
        )

        self.assertEqual(result["vacancy"]["status"], "NOT_FOUND")
        self.assertIn("different role", result["vacancy"]["summary"])

    def test_company_profile_is_summarized_from_page_with_source_link(self) -> None:
        result = verify_company_and_vacancy(
            "Salesforce",
            "Data Analyst",
            "",
            [{
                "title": "Salesforce about",
                "link": "https://www.salesforce.com/company/",
                "snippet": "Company profile.",
            }],
            [],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "final_url": url,
                "content": (
                    "<html><title>About Salesforce</title><main><h1>About "
                    "Salesforce</h1><p>Salesforce is a cloud software company "
                    "that provides customer relationship management products "
                    "and enterprise applications for sales, service, marketing, "
                    "analytics, and automation.</p></main></html>"
                ),
            },
        )

        company_source = result["company"]["source"]
        self.assertTrue(result["company"]["profile_available"])
        self.assertTrue(company_source["profile_available"])
        self.assertEqual(
            company_source["link"].rstrip("/"),
            "https://www.salesforce.com/company",
        )
        self.assertIn("cloud software company", company_source["summary"])

    def test_company_profile_renders_in_company_verification_card(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(
            LEGITIMATE_LOOKING_POSTING.replace(
                "Northwind Analytics", "Salesforce"
            )
        )
        app.text_input[0].set_value("Salesforce")
        app.checkbox[0].set_value(True)

        def search_response(url, headers, json, timeout):
            query = json["query"]
            response = Mock()
            if "official website" in query or "company profile" in query:
                response.json.return_value = {
                    "results": [{
                        "title": "Salesforce official company website",
                        "url": "https://www.salesforce.com/company/",
                        "content": "Salesforce company information.",
                    }]
                }
            else:
                response.json.return_value = {
                    "results": [{
                        "title": "Senior Data Analyst at Salesforce",
                        "url": "https://careers.salesforce.com/jobs/123",
                        "content": "Senior Data Analyst in Seattle, WA at Salesforce.",
                    }]
                }
            return response

        def page_response(url, headers, timeout, allow_redirects, stream):
            response = Mock()
            response.status_code = 200
            response.headers = {}
            response.encoding = "utf-8"
            if "/company" in url:
                page_html = (
                    "<html><title>About Salesforce</title><main><h1>About "
                    "Salesforce</h1><p>Salesforce provides cloud software and "
                    "customer relationship management products.</p></main></html>"
                )
            else:
                page_html = (
                    "<html><main><h1>Senior Data Analyst</h1><p>Salesforce "
                    "is hiring in Seattle, WA.</p></main></html>"
                )
            response.raw.read.return_value = page_html.encode("utf-8")
            return response

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-profile-key"}), patch(
            "web_verification.requests.post",
            side_effect=search_response,
        ), patch(
            "web_verification.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
        ), patch(
            "web_verification.requests.get",
            side_effect=page_response,
        ):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("About the company", page_markdown)
        self.assertIn("Salesforce provides cloud software", page_markdown)

    def test_verification_links_are_unique_across_company_and_vacancy(self) -> None:
        shared_result = {
            "title": "Senior Software Engineer - Microsoft Careers",
            "link": "https://careers.microsoft.com/job/123",
            "snippet": "Senior Software Engineer in Redmond, WA at Microsoft.",
        }
        result = verify_company_and_vacancy(
            "Microsoft",
            "Senior Software Engineer",
            "Redmond, WA",
            [shared_result],
            [shared_result],
            page_fetcher=lambda *args, **kwargs: {
                "ok": True,
                "title": "Senior Software Engineer",
                "content": (
                    "<html><main><p>Microsoft is hiring a Senior Software "
                    "Engineer in Redmond, WA.</p></main></html>"
                ),
            },
        )

        links = [source["link"] for source in result["sources"]]
        self.assertEqual(len(links), len(set(links)))
        self.assertEqual(len(links), 1)
        self.assertIsNotNone(result["vacancy"]["source"])

    def test_irrelevant_company_mention_does_not_verify_or_surface_a_source(self) -> None:
        result = verify_company_and_vacancy(
            "Microsoft",
            "Senior Software Engineer",
            "Redmond, WA",
            [{
                "title": "A story about unrelated software",
                "link": "https://news.example/articles/story",
                "snippet": "The author briefly compares a tool to Microsoft products.",
            }],
            [],
            page_fetcher=lambda *args, **kwargs: {"ok": False},
        )

        self.assertEqual(result["company"]["status"], "NOT_ESTABLISHED")
        self.assertIsNone(result["company"]["source"])
        self.assertEqual(result["sources"], [])

    def test_generic_careers_extraction_and_ambiguous_postings(self) -> None:
        posting = (
            "Senior Product Analyst\n"
            "https://careers.cambay.com/jobs/4512\n"
            "Location: Toronto, ON\n"
            "Description: Analyze product trends and report findings."
        )
        self.assertEqual(extract_job_fields(posting)["company_name"], "Cambay")
        self.assertEqual(
            extract_job_fields(
                "Job Title: Product Analyst\n"
                "Posted by: Northstar Recruiting\n"
                "Description: The client uses Oracle and Microsoft tools."
            )["company_name"],
            "",
        )

    def test_external_ats_listing_is_not_confirmed_as_official(self) -> None:
        result = verify_company_and_vacancy(
            "Acme Data LLC",
            "Senior Product Analyst",
            "Toronto, ON",
            [{
                "title": "Acme Data official company website",
                "link": "https://acmedata.com/",
                "snippet": "About Acme Data and its analytics products.",
            }],
            [{
                "title": "Senior Product Analyst at Acme Data",
                "link": "https://boards.greenhouse.io/acmedata/jobs/4512",
                "snippet": "Senior Product Analyst in Toronto, ON at Acme Data.",
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "title": "Senior Product Analyst",
                "content": (
                    "<html><main><p>Acme Data is hiring a Senior Product Analyst "
                    "in Toronto, ON.</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["vacancy"]["status"], "POSSIBLE_MATCH")
        self.assertIn("third-party listing", result["vacancy"]["summary"])
        self.assertEqual(
            result["vacancy"]["source"]["link"],
            "https://boards.greenhouse.io/acmedata/jobs/4512",
        )

    def test_equivalent_result_urls_are_deduplicated(self) -> None:
        result = verify_company_and_vacancy(
            "Acme Data",
            "Product Analyst",
            "",
            [
                {
                    "title": "Acme Data official company website",
                    "link": "https://acmedata.com/?utm_source=search",
                    "snippet": "About Acme Data.",
                },
                {
                    "title": "Acme Data official company website",
                    "link": "https://ACMEDATA.com/",
                    "snippet": "About Acme Data.",
                },
            ],
            [],
            page_fetcher=lambda *args, **kwargs: {"ok": False},
        )

        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["link"], "https://acmedata.com/")

    def test_conclusion_distinguishes_search_failure_and_evidence_outcomes(self) -> None:
        verified_company = {"status": "VERIFIED"}
        vacancy_missing = {"status": "NOT_FOUND"}
        self.assertIn(
            "specific vacancy was not located",
            app_module.build_verification_conclusion(
                verified_company,
                vacancy_missing,
            ),
        )
        self.assertIn(
            "web search failed",
            app_module.build_verification_conclusion(
                {"status": "UNAVAILABLE"},
                {"status": "UNAVAILABLE"},
            ),
        )
        self.assertIn(
            "possible third-party vacancy listing",
            app_module.build_verification_conclusion(
                {"status": "NOT_ESTABLISHED"},
                {"status": "POSSIBLE_MATCH"},
            ),
        )

    def test_unseen_company_and_exact_official_vacancy_can_be_verified(self) -> None:
        result = verify_company_and_vacancy(
            "Vela Robotics",
            "Staff Machine Learning Engineer",
            "Pune, India",
            [{
                "title": "Vela Robotics official website",
                "link": "https://velarobotics.example/about",
                "snippet": "Vela Robotics develops autonomous inspection systems.",
            }],
            [{
                "title": "Staff Machine Learning Engineer - Vela Robotics",
                "link": "https://velarobotics.example/careers/987",
                "snippet": (
                    "Vela Robotics is hiring a Staff Machine Learning Engineer "
                    "in Pune, India."
                ),
            }],
            page_fetcher=lambda url, **kwargs: {
                "ok": True,
                "title": "Vela Robotics Careers",
                "content": (
                    "<html><main><p>Vela Robotics is hiring a Staff Machine "
                    "Learning Engineer in Pune, India to build robotics systems."
                    "</p></main></html>"
                ),
            },
        )
        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["vacancy"]["status"], "MATCH_FOUND")
        self.assertEqual(
            result["vacancy"]["source"]["link"],
            "https://velarobotics.example/careers/987",
        )

    def test_company_matching_rejects_longer_unrelated_names(self) -> None:
        self.assertFalse(
            _company_identity_matches(
                "Bright Future", "Bright Future Global Careers"
            )
        )
        self.assertFalse(
            _company_identity_matches(
                "Global Online Services", "Global Online Services International"
            )
        )
        self.assertTrue(
            _company_identity_matches(
                "Northwind Analytics Ltd", "Northwind Analytics official website"
            )
        )
        self.assertFalse(
            _company_identity_matches("Global Work Hub", "Work Global")
        )
        self.assertFalse(
            _company_identity_matches("Global Work Hub", "Hub Global Work")
        )

    def test_company_verification_rejects_reordered_or_similar_company_names(self) -> None:
        for result_title in ("Work Global", "Global Work Hub International"):
            with self.subTest(result_title=result_title):
                result = verify_company_and_vacancy(
                    "Global Work Hub",
                    "Data Analyst",
                    "",
                    [{
                        "title": f"{result_title} official company website",
                        "link": "https://workglobal.example/about",
                        "snippet": f"About {result_title}.",
                    }],
                    [],
                    page_fetcher=lambda *args, **kwargs: {"ok": False},
                )
                self.assertEqual(result["company"]["status"], "NOT_ESTABLISHED")
                self.assertEqual(result["company"]["company_name"], "Global Work Hub")
                self.assertIsNone(result["company"]["source"])
                self.assertEqual(
                    result["company"]["summary"],
                    "Unable to verify the exact company name.",
                )

    def test_exact_company_name_match_ignores_case_and_extra_whitespace(self) -> None:
        result = verify_company_and_vacancy(
            "  Global   Work Hub  ",
            "Data Analyst",
            "",
            [{
                "title": "global work hub official company website",
                "link": "https://globalworkhub.example/about",
                "snippet": "About Global Work Hub.",
            }],
            [],
            page_fetcher=lambda *args, **kwargs: {"ok": False},
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["company"]["company_name"], "Global   Work Hub")

    def test_official_site_must_explicitly_support_an_alternative_company_name(self) -> None:
        search_result = [{
            "title": "Work Global official company website",
            "link": "https://workglobal.example/about",
            "snippet": "Work Global company information.",
            "search_stage": "fallback",
        }]

        result = verify_company_and_vacancy(
            "Global Work Hub",
            "Data Analyst",
            "",
            search_result,
            [],
            page_fetcher=lambda *args, **kwargs: {
                "ok": True,
                "title": "Work Global",
                "final_url": "https://workglobal.example/about",
                "content": (
                    "<html><main><p>Global Work Hub, also known as Work Global, "
                    "provides staffing services.</p></main></html>"
                ),
            },
        )

        self.assertEqual(result["company"]["status"], "VERIFIED")
        self.assertEqual(result["company"]["company_name"], "Global Work Hub")
        self.assertIn("alternative name", result["company"]["summary"])
        self.assertEqual(
            result["company"]["source"]["search_title"],
            "Work Global official company website",
        )
        self.assertEqual(
            result["company"]["source"]["link"],
            "https://workglobal.example/about",
        )
        self.assertEqual(
            result["company"]["source"]["search_stage"],
            "fallback",
        )

    def test_no_exact_company_result_reports_unable_to_verify(self) -> None:
        result = verify_company_and_vacancy(
            "Global Work Hub",
            "Data Analyst",
            "",
            [],
            [],
            page_fetcher=lambda *args, **kwargs: {"ok": False},
        )

        self.assertEqual(result["company"]["status"], "NOT_ESTABLISHED")
        self.assertEqual(
            result["company"]["summary"],
            "Unable to verify the exact company name.",
        )

    def test_domain_matching_requires_the_full_company_signature(self) -> None:
        self.assertTrue(
            _domain_matches_company("Bright Future", "https://brightfuture.example/")
        )
        self.assertFalse(
            _domain_matches_company(
                "Bright Future", "https://brightfutureglobal.example/"
            )
        )
        self.assertFalse(
            _domain_matches_company(
                "Global Online Services", "https://global.example/"
            )
        )
        self.assertTrue(
            _domain_matches_company(
                "Global Online Services", "https://globalonlineservices.example/"
            )
        )

    def test_unavailable_company_page_does_not_turn_search_snippet_into_description(self) -> None:
        summary = summarize_source(
            {
                "title": "Vela Robotics official website",
                "link": "https://velarobotics.example/about",
                "snippet": "Vela Robotics is a leading robotics company with many products.",
            },
            company_name="Vela Robotics",
            page_fetcher=lambda *args, **kwargs: {
                "ok": False,
                "error": "blocked",
            },
        )
        self.assertEqual(summary["content_source"], "none")
        self.assertFalse(summary["retrieved"])
        self.assertIn("could not be extracted", summary["summary"])
        self.assertNotIn("leading robotics company", summary["summary"])

    def test_public_page_fetch_follows_safe_redirects(self) -> None:
        from web_verification import fetch_public_page

        first = Mock()
        first.status_code = 301
        first.headers = {"Location": "https://example.com/about"}
        second = Mock()
        second.status_code = 200
        second.headers = {}
        second.encoding = "utf-8"
        second.raw.read.return_value = b"<html><title>About</title></html>"

        with patch(
            "web_verification.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
        ), patch(
            "web_verification.requests.get",
            side_effect=[first, second],
        ) as get:
            result = fetch_public_page("http://example.com", timeout=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["final_url"], "https://example.com/about")
        self.assertEqual(get.call_count, 2)
        self.assertFalse(get.call_args_list[0].kwargs["allow_redirects"])

    def test_tavily_timeout_authentication_and_empty_results_are_distinct(self) -> None:
        with patch(
            "web_verification.requests.post",
            side_effect=requests.Timeout("provider timed out"),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                search_web("Microsoft official company", "test-secret")

        provider_response = Mock()
        provider_response.status_code = 401
        auth_response = Mock()
        auth_response.raise_for_status.side_effect = requests.HTTPError(
            "unauthorized",
            response=provider_response,
        )
        with patch("web_verification.requests.post", return_value=auth_response):
            with self.assertRaisesRegex(RuntimeError, "rejected the API key"):
                search_web("Microsoft official company", "test-secret")

        empty_response = Mock()
        empty_response.json.return_value = {"results": []}
        with patch("web_verification.requests.post", return_value=empty_response):
            self.assertEqual(
                search_web("Microsoft official company", "test-secret"),
                [],
            )

        malformed_response = Mock()
        malformed_response.json.return_value = {"results": [{"url": None}]}
        with patch(
            "web_verification.requests.post",
            return_value=malformed_response,
        ):
            with self.assertRaisesRegex(RuntimeError, "malformed results"):
                search_web("Microsoft official company", "test-secret")

    def test_search_snippets_are_never_reported_as_verified(self) -> None:
        results = [
            {
                "title": "Acme Careers",
                "link": "https://search.example/jobs",
                "snippet": "Acme Data is hiring a data analyst.",
            }
        ]

        company = company_verification("Acme", results)
        job = job_verification(
            {"company_name": "Acme", "title": "Data Analyst"},
            results,
        )

        self.assertTrue(company["company_found"])
        self.assertFalse(company["official_domain_found"])
        self.assertEqual(company["status"], "FOUND")
        self.assertFalse(job["job_found_officially"])
        self.assertFalse(job["job_found_third_party"])
        self.assertEqual(job["status"], "FOUND")
        self.assertIn("candidate vacancy match", job["summary"])
        report = summarize_search_evidence("Acme", "Data Analyst", results)
        self.assertEqual(report["status"], "unable_to_verify")
        self.assertFalse(report["verified_evidence_available"])
        self.assertFalse(report["conflicting_evidence_available"])
        self.assertTrue(report["evidence"][0]["mentions_company"])
        self.assertTrue(report["evidence"][0]["mentions_job_title"])
        self.assertIsNone(
            web_risk_score(
                {
                    "company_found": company["company_found"],
                    "job_found_officially": job["job_found_officially"],
                }
            )
        )

    def test_verification_reports_found_and_not_found_without_claiming_proof(self) -> None:
        matching_company = [{
            "title": "Acme Data official website",
            "link": "https://acme.example/",
            "snippet": "Company information",
        }]
        company_found = company_verification("Acme Data", matching_company)
        company_missing = company_verification(
            "Acme Data",
            [{
                "title": "Unrelated organization",
                "link": "https://unrelated.example/",
                "snippet": "Other company",
            }],
        )
        invalid_company_url = company_verification(
            "Acme Data",
            [{
                "title": "Acme Data official",
                "link": "javascript:alert(1)",
                "snippet": "Company information",
            }],
        )
        reordered_company = company_verification(
            "Global Work Hub",
            [{
                "title": "Work Global official website",
                "link": "https://workglobal.example/",
                "snippet": "Work Global company information.",
            }],
        )

        self.assertEqual(company_found["status"], "FOUND")
        self.assertEqual(
            company_found["official_website_candidate"],
            "https://acme.example/",
        )
        self.assertFalse(company_found["official_domain_found"])
        self.assertEqual(company_missing["status"], "NOT_FOUND")
        self.assertEqual(reordered_company["status"], "NOT_FOUND")
        self.assertEqual(
            reordered_company["summary"],
            "Unable to verify the exact company name. This does not prove that "
            "the employer is illegitimate.",
        )
        self.assertEqual(invalid_company_url["evidence"], [])
        self.assertEqual(
            company_verification("", matching_company)["status"],
            "UNABLE_TO_VERIFY",
        )

        posting = {
            "company_name": "Acme Data",
            "title": "Senior Data Analyst",
            "location": "Seattle",
        }
        vacancy_found = job_verification(
            posting,
            [{
                "title": "Senior Data Analyst - Acme Data",
                "link": "https://jobs.example/acme/seattle/senior-data-analyst",
                "snippet": "Acme Data Senior Data Analyst in Seattle",
            }],
        )
        vacancy_not_found = job_verification(
            posting,
            [{
                "title": "Data Analyst - Acme Data",
                "link": "https://jobs.example/acme/data-analyst",
                "snippet": "Acme Data role in Seattle",
            }],
        )
        self.assertEqual(vacancy_found["status"], "FOUND")
        self.assertEqual(vacancy_not_found["status"], "NOT_FOUND")
        self.assertEqual(
            vacancy_found["evidence"][0]["link"],
            "https://jobs.example/acme/seattle/senior-data-analyst",
        )
        self.assertFalse(vacancy_found["job_found_officially"])
        self.assertFalse(vacancy_found["job_found_third_party"])

    def test_search_links_are_limited_to_valid_http_urls(self) -> None:
        self.assertTrue(is_valid_url("https://jobs.example/role"))
        self.assertFalse(is_valid_url("javascript:alert(1)"))
        self.assertFalse(is_valid_url("ftp://jobs.example/role"))
        self.assertFalse(is_valid_url("https://jobs.example:invalid/role"))
        self.assertFalse(is_valid_url("http://127.0.0.1/admin"))
        self.assertFalse(is_valid_url("http://localhost/admin"))

    def test_public_page_fetch_blocks_private_hosts_without_network_request(self) -> None:
        from web_verification import fetch_public_page

        with patch("web_verification.requests.get") as request:
            result = fetch_public_page("http://127.0.0.1/admin")

        self.assertFalse(result["ok"])
        self.assertIn("publicly routable", result["error"])
        request.assert_not_called()

    def test_streamlit_form_runs_the_prediction_flow(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(INCOMPLETE_POSTING)
        analyze_button = next(
            button for button in app.button if button.label == "Analyze Posting"
        )
        analyze_button.click().run()

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("Posting detail coverage", page_markdown)
        self.assertIn("Warning level", page_markdown)
        self.assertTrue(
            any("More information needed" in item.value for item in app.markdown)
        )
        assessment_html = next(
            item.value
            for item in app.markdown
            if 'class="summary-card assessment-card ' in item.value
        )
        self.assertIn('assessment-card insufficient', assessment_html)
        self.assertIn(
            "The posting does not contain enough recognizable job information for the model to assess it.",
            assessment_html,
        )
        self.assertEqual(len(app.session_state["analysis_history"]), 1)

    def test_streamlit_model_assessment_cards_are_independent_from_rule_warning(
        self,
    ) -> None:
        expected_assessments = (
            (
                0,
                "assessment-card legitimate",
                "Appears legitimate",
                "Based on patterns identified by the machine-learning model in the submitted job posting.",
            ),
            (
                1,
                "assessment-card fraudulent",
                "Potential fraud warning",
                "Based on patterns identified by the machine-learning model in the submitted job posting.",
            ),
        )
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        for prediction, card_class, label, supporting_text in expected_assessments:
            with self.subTest(prediction=prediction):
                with patch.object(
                    sys.modules["model"],
                    "predict_job_posting",
                    return_value={"prediction": prediction},
                ):
                    next(
                        button
                        for button in app.button
                        if button.label == "Analyze Posting"
                    ).click().run()

                self.assertFalse(app.exception)
                assessment_html = next(
                    item.value
                    for item in app.markdown
                    if 'class="summary-card assessment-card ' in item.value
                )
                warning_html = next(
                    item.value
                    for item in app.markdown
                    if 'class="summary-card warning-card"' in item.value
                )
                self.assertIn(card_class, assessment_html)
                self.assertIn(label, assessment_html)
                self.assertIn(supporting_text, assessment_html)
                self.assertIn("risk-high", warning_html)

    def test_streamlit_result_cards_explain_their_calculation_basis(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        with patch.object(
            sys.modules["model"],
            "predict_job_posting",
            return_value={"prediction": 0},
        ):
            next(
                button
                for button in app.button
                if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        summary_html = "\n".join(
            item.value
            for item in app.markdown
            if 'class="summary-card ' in item.value
        )
        self.assertIn(
            "Shows how many of the key job details were identified in the submitted posting, such as the job title, employer, description, and role details.",
            summary_html,
        )
        self.assertIn(
            "Based on warning patterns detected in the submitted job text.",
            summary_html,
        )
        self.assertIn("Rule-based warning score:", summary_html)
        self.assertNotIn("not a probability of fraud", summary_html.lower())
        self.assertNotIn("model confidence", summary_html.lower())

    def test_streamlit_hides_ml_contribution_breakdown(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        next(
            button for button in app.button if button.label == "Analyze Posting"
        ).click().run()

        self.assertFalse(app.exception)
        self.assertFalse(
            any(
                "What pushed the ML assessment?" in item.value
                for item in app.markdown
            )
        )

    def test_streamlit_search_uses_dynamic_employer_title_and_location(self) -> None:
        app = self._open_app()
        posting = (
            "Job Title: Senior Software Engineer\n"
            "Company: Microsoft Corporation\n"
            "Job ID: MS-1234\n"
            "Location: Redmond, WA\n"
            "Description: Build secure cloud services with a global engineering team."
        )
        app.text_area[0].set_value(posting)
        app.checkbox[0].set_value(True)
        queries = []

        def empty_search(url, headers, json, timeout):
            queries.append(json["query"])
            response = Mock()
            response.json.return_value = {"results": []}
            return response

        with patch.dict(
            "os.environ",
            {"TAVILY_API_KEY": "test-microsoft-query-key"},
        ), patch("web_verification.requests.post", side_effect=empty_search):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(queries), 2)
        self.assertEqual(queries[0], '"Microsoft Corporation"')
        self.assertIn("Microsoft Corporation official website", queries)
        self.assertIn("Microsoft Corporation company profile", queries)
        self.assertTrue(any('"MS-1234"' in query for query in queries))
        self.assertTrue(any(
            '"Microsoft Corporation"' in query
            and '"Senior Software Engineer"' in query
            and '"Redmond, WA"' in query
            for query in queries
        ))
        self.assertTrue(any("job opening" in query for query in queries))
        self.assertTrue(
            any(
                "Microsoft Corporation" in item.value
                for item in app.markdown
            )
        )
        self.assertIn("### Overall conclusion", "\n".join(item.value for item in app.markdown))
        self.assertNotIn("Extracted employer", "\n".join(item.value for item in app.caption))

    def test_streamlit_company_override_recovers_an_ambiguous_employer(self) -> None:
        app = self._open_app()
        posting = (
            "Job Title: Data Analyst\nPosted by: Indeed\n"
            "Description: Analyze data and prepare reports for the team."
        )
        app.text_area[0].set_value(posting)
        next(
            item
            for item in app.text_input
            if item.label.startswith("Company name (optional")
        ).set_value("Microsoft")
        app.checkbox[0].set_value(True)
        queries = []

        def empty_search(url, headers, json, timeout):
            queries.append(json["query"])
            response = Mock()
            response.json.return_value = {"results": []}
            return response

        with patch.dict(
            "os.environ",
            {"TAVILY_API_KEY": "test-company-override-key"},
        ), patch("web_verification.requests.post", side_effect=empty_search):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        self.assertEqual(queries[0], '"Microsoft"')
        self.assertIn("Microsoft official website", queries)
        self.assertIn("Microsoft company profile", queries)
        self.assertTrue(any("Microsoft" in query and "Data Analyst" in query for query in queries))
        self.assertTrue(
            any(
                "Microsoft" in item.value
                for item in app.markdown
            )
        )
        self.assertNotIn("Extracted employer", "\n".join(item.value for item in app.caption))

    def test_streamlit_search_keeps_extracted_company_ahead_of_manual_name(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(
            "Job Title: Data Analyst\n"
            "Company: Global Work Hub\n"
            "Location: Remote\n"
            "Description: Analyze business data and prepare reports for the team."
        )
        next(
            item
            for item in app.text_input
            if item.label.startswith("Company name (optional")
        ).set_value("Work Global")
        app.checkbox[0].set_value(True)
        queries = []

        def empty_search(url, headers, json, timeout):
            queries.append(json["query"])
            response = Mock()
            response.json.return_value = {"results": []}
            return response

        with patch.dict(
            "os.environ",
            {"TAVILY_API_KEY": "test-preserve-extracted-company"},
        ), patch("web_verification.requests.post", side_effect=empty_search):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        self.assertEqual(queries[0], '"Global Work Hub"')
        self.assertTrue(any("Global Work Hub" in query for query in queries))
        self.assertFalse(any("Work Global" in query for query in queries))

    def test_streamlit_company_and_vacancy_checks_show_search_links(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        app.checkbox[0].set_value(True)

        def search_response(url, headers, json, timeout):
            query = json["query"]
            if "official website" in query:
                items = [{
                        "title": "Acme Data Inc official website",
                        "url": "https://acmedata.example",
                        "snippet": "Acme Data Inc company information.",
                    }]
            else:
                items = [{
                    "title": "Remote Data Entry Assistant at Acme Data Inc",
                    "url": "https://acme.example/careers/remote-data-entry-assistant",
                    "content": "Acme Data Inc Remote Data Entry Assistant Remote",
                }]
            response = Mock()
            response.json.return_value = {"results": items}
            return response

        with patch.dict(
            "os.environ",
            {"TAVILY_API_KEY": "test-key"},
        ), patch("web_verification.requests.post", side_effect=search_response):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("Job posting verification", page_markdown)
        self.assertIn("Company verification", page_markdown)
        self.assertIn("Possible match found", page_markdown)
        self.assertIn("Matching information found", page_markdown)
        self.assertNotIn("Inconclusive", page_markdown)
        self.assertNotIn("Additional relevant source", page_markdown)
        self.assertEqual(page_markdown.count("### Overall conclusion"), 1)
        self.assertTrue(
            any(
                "Broader fallback result: Acme Data Inc official website" in item.value
                for item in app.caption
            )
        )
        self.assertTrue(
            "### Overall conclusion" in page_markdown
        )

    def test_streamlit_missing_company_result_is_not_proof_of_fraud(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        app.checkbox[0].set_value(True)
        response = Mock()
        response.json.return_value = {"results": []}

        with patch.dict(
            "os.environ",
            {"TAVILY_API_KEY": "test-empty-results-key"},
        ), patch("web_verification.requests.post", return_value=response):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("No matching information found", page_markdown)
        self.assertIn("no sufficiently relevant vacancy listing was found", page_markdown)
        self.assertIn("### Overall conclusion", page_markdown)

    def test_streamlit_tavily_failures_show_search_unavailable(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        app.checkbox[0].set_value(True)

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-timeout-key"}), patch(
            "web_verification.requests.post",
            side_effect=requests.Timeout("search timed out"),
        ):
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertGreaterEqual(page_markdown.count("Search unavailable"), 2)
        self.assertNotIn("Search Failed", page_markdown)
        self.assertIn(
            "Unable to verify the exact company name because the search service was unavailable.",
            page_markdown,
        )
        self.assertIn("### Overall conclusion", page_markdown)

    def test_streamlit_analysis_hides_sample_controls(self) -> None:
        app = self._open_app()
        self.assertFalse(any(button.label == "Load sample" for button in app.button))
        self.assertEqual(len(app.selectbox), 0)
        self.assertFalse(app.exception)
        app.text_area[0].set_value("Text to clear").run()
        next(button for button in app.button if button.label == "Clear posting").click().run()
        self.assertEqual(app.text_area[0].value, "")
        self.assertFalse(app.exception)

    def test_streamlit_text_analysis_does_not_initialize_ocr(self) -> None:
        with patch(
            "ocr.create_ocr_engine",
            side_effect=AssertionError("Text analysis must not initialize OCR."),
        ) as create_engine:
            app = self._open_app()
            app.text_area[0].set_value(LEGITIMATE_LOOKING_POSTING)
            next(
                button for button in app.button if button.label == "Analyze Posting"
            ).click().run()

        self.assertFalse(app.exception)
        create_engine.assert_not_called()

    def test_ocr_engine_is_cached_between_calls(self) -> None:
        st.cache_resource.clear()
        engine = object()
        with patch.object(
            app_module,
            "create_ocr_engine",
            return_value=engine,
        ) as create_engine:
            self.assertIs(app_module.get_ocr_engine(), engine)
            self.assertIs(app_module.get_ocr_engine(), engine)

        create_engine.assert_called_once_with()
        app_module.get_ocr_engine.clear()

    def test_streamlit_results_follow_readable_signal_order_without_model_internals(self) -> None:
        app = self._open_app()
        app.text_area[0].set_value(SUSPICIOUS_POSTING)
        next(
            button for button in app.button if button.label == "Analyze Posting"
        ).click().run()

        self.assertFalse(app.exception)
        page_markdown = "\n".join(item.value for item in app.markdown)
        positive_index = page_markdown.index("Positive signs in this posting")
        warning_index = page_markdown.index("Warning signs to review")
        recommendation_index = page_markdown.index("Recommended next steps")
        conclusion_index = page_markdown.index("Overall conclusion")
        self.assertLess(positive_index, warning_index)
        self.assertLess(warning_index, recommendation_index)
        self.assertLess(recommendation_index, conclusion_index)
        self.assertIn("Posting detail coverage", page_markdown)
        disclaimer = (
            "JobSignal helps you spot possible warning signs, but it cannot "
            "confirm whether a job or company is genuine. Always check important "
            "details through trusted sources before applying, paying money, or "
            "sharing personal information."
        )
        guidance = (
            "Use the model assessment and available verification results together "
            "to make a more informed decision."
        )
        self.assertEqual(page_markdown.count(disclaimer), 1)
        self.assertEqual(page_markdown.count(guidance), 1)
        self.assertLess(page_markdown.index(guidance), page_markdown.index(disclaimer))
        self.assertNotIn("this is not a guarantee", page_markdown.lower())
        self.assertNotIn("these do not prove", page_markdown.lower())
        self.assertNotIn("Decision margin", page_markdown)
        self.assertNotIn("Transformed features", page_markdown)
        self.assertIn("Online verification was not selected", page_markdown)

    def test_streamlit_analysis_exposes_separate_text_and_image_inputs(self) -> None:
        app = self._open_app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(app.text_area[0].label, "Job posting text")
        self.assertEqual(
            app.text_area[1].label,
            "Extracted job text (editable)",
        )

    def test_streamlit_image_analysis_uses_ocr_text_then_corrected_text(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (80, 40), "white").save(image_buffer, format="PNG")
        uploaded_image = Mock()
        uploaded_image.name = "job.png"
        uploaded_image.getvalue.return_value = image_buffer.getvalue()
        engine = Mock()
        engine.predict.return_value = [
            {
                "rec_texts": [LEGITIMATE_LOOKING_POSTING],
                "rec_boxes": [[0, 0, 80, 20]],
            }
        ]
        app_module.get_ocr_engine.clear()

        with patch(
            "streamlit.file_uploader",
            return_value=uploaded_image,
        ), patch("ocr.create_ocr_engine", return_value=engine):
            app = self._open_app()
            extract_button = next(
                button
                for button in app.button
                if button.label == "Extract text with PaddleOCR"
            )
            self.assertFalse(extract_button.disabled)
            extract_button.click().run()
            self.assertFalse(app.exception)
            self.assertEqual(
                app.text_area[1].value.strip(),
                LEGITIMATE_LOOKING_POSTING.strip(),
            )

            app.text_area[1].set_value(SUSPICIOUS_POSTING)
            analyze_buttons = [
                button for button in app.button if button.label == "Analyze Posting"
            ]
            self.assertEqual(len(analyze_buttons), 2)
            analyze_buttons[1].click().run()

        st.cache_resource.clear()
        self.assertFalse(app.exception)
        engine.predict.assert_called_once()
        page_markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("Potential fraud warning", page_markdown)
        self.assertIn("Warning signs to review", page_markdown)

    def test_streamlit_image_analysis_shows_friendly_ocr_failure(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (80, 40), "white").save(image_buffer, format="PNG")
        uploaded_image = Mock()
        uploaded_image.name = "job.png"
        uploaded_image.getvalue.return_value = image_buffer.getvalue()
        engine = Mock()
        engine.predict.side_effect = RuntimeError("internal engine detail")
        st.cache_resource.clear()

        with patch(
            "streamlit.file_uploader",
            return_value=uploaded_image,
        ), patch("ocr.create_ocr_engine", return_value=engine):
            app = self._open_app()
            next(
                button
                for button in app.button
                if button.label == "Extract text with PaddleOCR"
            ).click().run()

        st.cache_resource.clear()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "PaddleOCR could not read this image" in error.value
                for error in app.error
            )
        )

    def test_streamlit_dashboard_requires_a_real_login(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                'class="auth-panel"' in item.value
                and 'id="auth-welcome-title"' in item.value
                for item in app.markdown
            )
        )
        self.assertTrue(
            any(
                "--accent: #7D70A8" in item.value
                and "--page-bg: #DCEAF2" in item.value
                for item in app.markdown
            )
        )
        auth_copy = "\n".join(item.value for item in app.markdown)
        self.assertIn("Screen job postings for suspicious signs", auth_copy)
        self.assertIn("Understand the warning signs", auth_copy)
        self.assertIn("Check the employer and vacancy", auth_copy)
        self.assertEqual(app.session_state["auth_mode"], "signin")
        self.assertFalse(any(field.key == "signup_full_name" for field in app.text_input))
        self.assertEqual(
            next(field for field in app.text_input if field.key == "login_email").proto.placeholder,
            "you@example.com",
        )
        self.assertEqual(
            next(field for field in app.text_input if field.key == "login_password").proto.type,
            1,
        )

    def test_streamlit_signup_requires_matching_password_confirmation(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        next(
            button for button in app.button if button.label == "Create an account"
        ).click().run()
        self.assertEqual(app.session_state["auth_mode"], "signup")
        fields = {
            "signup_full_name": "Validation User",
            "signup_email": f"validation-{id(app)}@example.com",
            "signup_password": "A-long-unique-password-93",
            "signup_password_confirm": "A-different-password-93",
        }
        for key, value in fields.items():
            next(field for field in app.text_input if field.key == key).set_value(value)
        next(button for button in app.button if button.label == "Create account").click().run()
        self.assertTrue(
            any("confirmation does not match" in item.value for item in app.error)
        )
        self.assertNotIn("auth_token", app.session_state)

        next(
            field
            for field in app.text_input
            if field.key == "signup_password_confirm"
        ).set_value(fields["signup_password"])
        next(button for button in app.button if button.label == "Create account").click().run()
        self.assertTrue(any("Account created" in item.value for item in app.success))
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Sign in" for button in app.button))
        self.assertTrue(any(button.label == "Create account" for button in app.button))
        self.assertFalse(any(button.label == "Analyze Posting" for button in app.button))
        self.assertFalse(
            any(button.label == "Job Posting Analysis" for button in app.button)
        )

    def test_streamlit_auth_mode_switch_preserves_unrelated_session_state(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        app.session_state["analysis_history"] = [{"Role": "Retained posting"}]
        next(field for field in app.text_input if field.key == "login_email").set_value(
            "retained@example.com"
        )

        next(
            button for button in app.button if button.label == "Create an account"
        ).click().run()
        self.assertEqual(app.session_state["analysis_history"][0]["Role"], "Retained posting")
        self.assertEqual(
            app.session_state["_auth_saved_login_email"],
            "retained@example.com",
        )

        next(button for button in app.button if button.label == "Sign in").click().run()
        self.assertEqual(app.session_state["auth_mode"], "signin")
        self.assertEqual(app.session_state["analysis_history"][0]["Role"], "Retained posting")
        self.assertEqual(app.session_state["login_email"], "retained@example.com")

    def test_streamlit_login_form_creates_a_real_authenticated_session(self) -> None:
        email = f"login-{id(self)}@example.com"
        create_account(
            "Login Test User",
            email,
            "A-long-unique-password-93",
            db_path=self.auth_db_path,
        )
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        next(field for field in app.text_input if field.key == "login_email").set_value(email)
        next(field for field in app.text_input if field.key == "login_password").set_value(
            "A-long-unique-password-93"
        )
        next(button for button in app.button if button.label == "Sign in").click().run()
        self.assertFalse(app.exception)
        self.assertIn("auth_token", app.session_state)
        self.assertTrue(any(button.label == "Job Posting Analysis" for button in app.button))
        self.assertTrue(any("Welcome," in item.value for item in app.subheader))

    def test_streamlit_auth_forms_use_native_password_visibility_controls(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        login_password = next(
            field for field in app.text_input if field.key == "login_password"
        )
        self.assertEqual(login_password.proto.type, 1)
        self.assertFalse(any("password" in item.label.lower() for item in app.checkbox))
        next(
            button for button in app.button if button.label == "Create an account"
        ).click().run()
        signup_password = next(
            field for field in app.text_input if field.key == "signup_password"
        )
        self.assertEqual(signup_password.proto.type, 1)
        confirm_password = next(
            field for field in app.text_input if field.key == "signup_password_confirm"
        )
        self.assertEqual(confirm_password.proto.type, 1)
        self.assertFalse(any("password" in item.label.lower() for item in app.checkbox))
    def test_streamlit_logout_revokes_session_and_clears_posting_history(self) -> None:
        app = self._open_app()
        token = app.session_state["auth_token"]
        app.session_state["analysis_history"] = [
            {"Role": "Sensitive posting", "Employer": "Example"}
        ]
        next(button for button in app.button if button.label == "Log out").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Sign in" for button in app.button))
        self.assertFalse(any(button.label == "Analyze Posting" for button in app.button))
        self.assertNotIn("analysis_history", app.session_state)
        self.assertNotIn("posting_input", app.session_state)
        self.assertIsNone(
            get_session_user(token, db_path=self.auth_db_path)
        )

    def test_streamlit_home_about_and_analysis_are_separate_pages(self) -> None:
        app = self._open_app()
        next(button for button in app.button if button.label == "Home").click().run()
        self.assertFalse(app.exception)
        home_html = "\n".join(item.value for item in app.markdown)
        for section in (
            "A clearer view before you apply.",
            "Analyze a Job Posting",
            "Recent analyses",
            "Analyze Your First Job",
            "How JobSignal works",
            "What your review includes",
            "Posting completeness",
            "Optional company and vacancy search results",
            "Submit a job posting",
            "Verify important details",
            "Separate signals, clearer context",
            "Built to help you review job postings more thoughtfully.",
            "Helpful signals, not a final verdict",
            "JobSignal helps you spot possible warning signs",
        ):
            self.assertIn(section, home_html + "\n" + "\n".join(
                button.label for button in app.button
            ))
        self.assertNotIn("How It Works", home_html)
        self.assertFalse(any(button.label == "Analyze Posting" for button in app.button))
        analyze_button = next(
            button for button in app.button if button.key == "home_analyze"
        )
        about_button = next(
            button for button in app.button if button.key == "home_about"
        )
        self.assertEqual(analyze_button.proto.type, about_button.proto.type)
        next(button for button in app.button if button.label == "Analyze a Job Posting").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Analyze Posting" for button in app.button))

        next(button for button in app.button if button.label == "Home").click().run()
        next(button for button in app.button if button.label == "About").click().run()
        self.assertFalse(app.exception)
        about_html = "\n".join(item.value for item in app.markdown)
        for section in (
            "About JobSignal",
            "Research and application inference are distinct",
            "How the assessment works",
            "Dataset and preparation",
            "EDA and feature engineering",
            "Model comparison and evaluation",
            "Saved-model inference",
            "Python",
            "Streamlit",
            "Scikit-learn",
            "Pandas",
            "NumPy",
            "LinearSVC",
            "TF-IDF text features",
            "Explainable AI",
            "Text-based indicators",
            "Evidence and explanations",
            "Practical recommendations",
            "Tavily Search API",
            "Company and vacancy verification",
            "posting-completeness check",
            "optional online search are separate aids",
            "Held-out research metrics",
            "0.9829",
            "Helpful signals, not a final verdict",
            "JobSignal helps you spot possible warning signs",
            "Results can be incomplete",
        ):
            self.assertIn(section, about_html)
        self.assertNotIn("Limitations", about_html)
        self.assertNotIn("Future scope", about_html)
        self.assertIn('aria-labelledby="about-title"', about_html)
        self.assertIn("--accent: #7D70A8", "\n".join(item.value for item in app.markdown))
        self.assertFalse(any(button.label == "Analyze Posting" for button in app.button))

        next(
            button for button in app.button if button.label == "Job Posting Analysis"
        ).click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Analyze Posting" for button in app.button))

    def test_streamlit_home_history_shows_stored_signals_separately(self) -> None:
        app = self._open_app()
        app.session_state["analysis_history"] = [
            {
                "Role": "Data Analyst",
                "Employer": "Northwind",
                "Time": "Oct 25, 2026 · 12:30",
                "Model assessment": "Potential fraud warning",
                "Warning score": "40/100",
                "Risk level": "MEDIUM RISK",
            }
        ]
        app.session_state["jobsignal_page"] = "Home"
        app.run()

        self.assertFalse(app.exception)
        home_html = "\n".join(item.value for item in app.markdown)
        for signal in (
            "Data Analyst",
            "Northwind",
            "Potential fraud warning",
            "Rule-based warning score",
            "40/100",
            "MEDIUM RISK",
        ):
            self.assertIn(signal, home_html)
        self.assertNotIn("Analyze Your First Job", home_html)

if __name__ == "__main__":
    unittest.main()
