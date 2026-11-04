"""Persistent local account and server-side session helpers for JobSignal."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTH_DB_PATH = PROJECT_ROOT / "data" / "jobsignal_users.sqlite3"
PASSWORD_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
SESSION_IDLE_TIMEOUT_SECONDS = 30 * 60
SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
LOGIN_ATTEMPT_LIMIT = 5
LOGIN_WINDOW_SECONDS = 15 * 60
_EMAIL_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$",
    re.IGNORECASE,
)
_DUMMY_SALT = b"jobsignal-dummy!"
_DUMMY_HASH = hashlib.pbkdf2_hmac(
    "sha256", b"not-a-real-password", _DUMMY_SALT, PASSWORD_ITERATIONS
)


class AuthError(ValueError):
    """Raised when account input is invalid or registration cannot proceed."""


class AuthStorageError(RuntimeError):
    """Raised when the local account database cannot be used safely."""


class LoginRateLimitError(RuntimeError):
    """Raised when too many login attempts have occurred in the current window."""


def _connect(db_path: Path | str = AUTH_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise AuthStorageError(
            "The local account database could not be opened."
        ) from exc


def initialize_database(db_path: Path | str = AUTH_DB_PATH) -> None:
    """Create missing authentication tables without replacing existing data."""
    connection = _connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS auth_sessions_user_id
                ON auth_sessions(user_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_login_attempts (
                    email_hash TEXT PRIMARY KEY,
                    window_started_at INTEGER NOT NULL,
                    attempts INTEGER NOT NULL
                )
                """
            )
            now = int(time.time())
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ? OR created_at <= ?",
                (now, now - SESSION_MAX_AGE_SECONDS),
            )
    except sqlite3.Error as exc:
        raise AuthStorageError(
            "The local account database schema could not be initialized."
        ) from exc
    finally:
        connection.close()


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def _validate_email(email: str) -> str:
    if not isinstance(email, str):
        raise AuthError("Enter a valid email address.")
    normalized = normalize_email(email)
    if len(normalized) > 254 or not _EMAIL_PATTERN.fullmatch(normalized):
        raise AuthError("Enter a valid email address.")
    return normalized


def _validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise AuthError("Enter a password.")
    if len(password) > 128:
        raise AuthError("Use a password no longer than 128 UTF-8 bytes.")
    if len(password) < 12:
        raise AuthError("Use a password with at least 12 characters.")
    try:
        encoded = password.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuthError("The password contains unsupported characters.") from exc
    if len(encoded) > 128:
        raise AuthError("Use a password no longer than 128 UTF-8 bytes.")
    if not password.strip() or any(ord(character) < 32 for character in password):
        raise AuthError("The password cannot be blank or contain control characters.")


def _password_digest(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )


def _email_bucket(email: str) -> str:
    return hashlib.sha256(email.encode("utf-8", errors="replace")).hexdigest()


def _record_failed_attempt(
    connection: sqlite3.Connection, email_hash: str, now: int
) -> None:
    row = connection.execute(
        "SELECT window_started_at, attempts FROM auth_login_attempts WHERE email_hash = ?",
        (email_hash,),
    ).fetchone()
    if row is None or now - row["window_started_at"] >= LOGIN_WINDOW_SECONDS:
        connection.execute(
            """
            INSERT INTO auth_login_attempts (email_hash, window_started_at, attempts)
            VALUES (?, ?, 1)
            ON CONFLICT(email_hash) DO UPDATE SET
                window_started_at = excluded.window_started_at,
                attempts = excluded.attempts
            """,
            (email_hash, now),
        )
    else:
        connection.execute(
            "UPDATE auth_login_attempts SET attempts = attempts + 1 WHERE email_hash = ?",
            (email_hash,),
        )


def create_account(
    full_name: str,
    email: str,
    password: str,
    *,
    db_path: Path | str = AUTH_DB_PATH,
    now: int | None = None,
) -> int:
    """Create a user with a salted PBKDF2 password hash and return its ID."""
    name = full_name.strip() if isinstance(full_name, str) else ""
    if not name or len(name) > 100 or any(ord(character) < 32 for character in name):
        raise AuthError("Enter a name between 1 and 100 characters.")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuthError("The name contains unsupported characters.") from exc
    normalized_email = _validate_email(email)
    _validate_password(password)
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    password_hash = _password_digest(password, salt, PASSWORD_ITERATIONS)
    created_at = int(time.time()) if now is None else int(now)
    connection = _connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO users (
                    full_name, username, email, password_salt,
                    password_hash, password_iterations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    normalized_email,
                    normalized_email,
                    salt,
                    password_hash,
                    PASSWORD_ITERATIONS,
                    created_at,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise AuthError(
            "An account with that email address already exists."
        ) from exc
    except sqlite3.Error as exc:
        raise AuthStorageError("The account could not be created.") from exc
    finally:
        connection.close()


def authenticate(
    email: str,
    password: str,
    *,
    db_path: Path | str = AUTH_DB_PATH,
    now: int | None = None,
) -> dict[str, Any] | None:
    """Authenticate credentials without revealing whether an email is registered."""
    normalized_email = normalize_email(email) if isinstance(email, str) else ""
    current_time = int(time.time()) if now is None else int(now)
    valid_email = (
        len(normalized_email) <= 254
        and _EMAIL_PATTERN.fullmatch(normalized_email) is not None
    )
    email_hash = _email_bucket(normalized_email[:254])
    connection = _connect(db_path)
    try:
        with connection:
            attempts = connection.execute(
                "SELECT window_started_at, attempts FROM auth_login_attempts WHERE email_hash = ?",
                (email_hash,),
            ).fetchone()
            if (
                attempts is not None
                and current_time - attempts["window_started_at"] < LOGIN_WINDOW_SECONDS
                and attempts["attempts"] >= LOGIN_ATTEMPT_LIMIT
            ):
                raise LoginRateLimitError(
                    "Too many unsuccessful attempts. Wait 15 minutes and try again."
                )

            user = None
            if valid_email:
                user = connection.execute(
                    """
                    SELECT user_id, full_name, email, password_salt,
                           password_hash, password_iterations
                    FROM users WHERE email = ?
                    """,
                    (normalized_email,),
                ).fetchone()
            valid_password_input = (
                isinstance(password, str)
                and len(password) <= 128
                and not any(ord(character) < 32 for character in password)
            )
            if valid_password_input:
                try:
                    password.encode("utf-8")
                except UnicodeEncodeError:
                    valid_password_input = False
            if user is None or not valid_password_input:
                dummy_digest = _password_digest(
                    password if valid_password_input else "",
                    _DUMMY_SALT,
                    PASSWORD_ITERATIONS,
                )
                hmac.compare_digest(dummy_digest, _DUMMY_HASH)
                _record_failed_attempt(connection, email_hash, current_time)
                return None

            actual_hash = _password_digest(
                password,
                user["password_salt"],
                user["password_iterations"],
            )
            if not hmac.compare_digest(actual_hash, user["password_hash"]):
                _record_failed_attempt(connection, email_hash, current_time)
                return None

            if user["password_iterations"] < PASSWORD_ITERATIONS:
                new_salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
                connection.execute(
                    """
                    UPDATE users
                    SET password_salt = ?, password_hash = ?, password_iterations = ?
                    WHERE user_id = ?
                    """,
                    (
                        new_salt,
                        _password_digest(password, new_salt, PASSWORD_ITERATIONS),
                        PASSWORD_ITERATIONS,
                        user["user_id"],
                    ),
                )
            connection.execute(
                "DELETE FROM auth_login_attempts WHERE email_hash = ?",
                (email_hash,),
            )
            return {
                "user_id": user["user_id"],
                "full_name": user["full_name"],
                "email": user["email"],
            }
    except sqlite3.Error as exc:
        raise AuthStorageError("The account database could not complete login.") from exc
    finally:
        connection.close()


def create_session(
    user_id: int,
    *,
    db_path: Path | str = AUTH_DB_PATH,
    now: int | None = None,
) -> str:
    """Create a random session token while storing only its SHA-256 digest."""
    current_time = int(time.time()) if now is None else int(now)
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    connection = _connect(db_path)
    try:
        with connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ? OR created_at <= ?",
                (current_time, current_time - SESSION_MAX_AGE_SECONDS),
            )
            connection.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    token_hash,
                    user_id,
                    current_time,
                    min(
                        current_time + SESSION_IDLE_TIMEOUT_SECONDS,
                        current_time + SESSION_MAX_AGE_SECONDS,
                    ),
                ),
            )
        return token
    except sqlite3.Error as exc:
        raise AuthStorageError("A secure login session could not be created.") from exc
    finally:
        connection.close()


def get_session_user(
    token: str,
    *,
    db_path: Path | str = AUTH_DB_PATH,
    now: int | None = None,
) -> dict[str, Any] | None:
    """Validate a session, enforce idle and absolute expiry, and extend idle time."""
    if not isinstance(token, str) or not token:
        return None
    current_time = int(time.time()) if now is None else int(now)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    connection = _connect(db_path)
    try:
        with connection:
            row = connection.execute(
                """
                SELECT s.user_id, s.created_at, s.expires_at, u.full_name, u.email
                FROM auth_sessions AS s
                JOIN users AS u ON u.user_id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if (
                current_time >= row["expires_at"]
                or current_time - row["created_at"] >= SESSION_MAX_AGE_SECONDS
            ):
                connection.execute(
                    "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
                )
                return None

            connection.execute(
                "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
                (
                    min(
                        current_time + SESSION_IDLE_TIMEOUT_SECONDS,
                        row["created_at"] + SESSION_MAX_AGE_SECONDS,
                    ),
                    token_hash,
                ),
            )
            return {
                "user_id": row["user_id"],
                "full_name": row["full_name"],
                "email": row["email"],
            }
    except sqlite3.Error as exc:
        raise AuthStorageError("The login session could not be validated.") from exc
    finally:
        connection.close()


def session_is_active(
    token: str,
    *,
    db_path: Path | str = AUTH_DB_PATH,
    now: int | None = None,
) -> bool:
    """Check expiry without extending the inactivity timeout."""
    if not isinstance(token, str) or not token:
        return False
    current_time = int(time.time()) if now is None else int(now)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    connection = _connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT created_at, expires_at FROM auth_sessions
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        return bool(
            row is not None
            and current_time < row["expires_at"]
            and current_time - row["created_at"] < SESSION_MAX_AGE_SECONDS
        )
    except sqlite3.Error as exc:
        raise AuthStorageError("The login session could not be checked.") from exc
    finally:
        connection.close()


def revoke_session(
    token: str, *, db_path: Path | str = AUTH_DB_PATH
) -> None:
    """Revoke a server-side session by its original random token."""
    if not isinstance(token, str) or not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    connection = _connect(db_path)
    try:
        with connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            )
    except sqlite3.Error as exc:
        raise AuthStorageError("The login session could not be revoked.") from exc
    finally:
        connection.close()
