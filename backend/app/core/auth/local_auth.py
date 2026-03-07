"""
Email/password authentication endpoints backed by the relational database.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.core.auth.jwt_auth import create_access_token
from backend.app.core.db.relational import DBUser, get_relational_session


router = APIRouter(prefix="/auth", tags=["auth"])

PBKDF2_ITERATIONS = 120_000
PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _username_from_email(email: str) -> str:
    base = (email or "").split("@", 1)[0].strip().lower() or "user"
    base = re.sub(r"[^a-z0-9._-]+", "_", base).strip("._-") or "user"
    return base[:120]


def _unique_username(db, base: str) -> str:
    candidate = base
    idx = 1
    while db.query(DBUser.id).filter(DBUser.username == candidate).first():
        suffix = f"_{idx}"
        candidate = f"{base[: max(1, 120 - len(suffix))]}{suffix}"
        idx += 1
    return candidate


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, iterations, salt_hex, digest_hex = password_hash.split("$", 3)
        if algo != f"pbkdf2_{PBKDF2_ALGORITHM}":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        computed = hashlib.pbkdf2_hmac(
            PBKDF2_ALGORITHM,
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


def validate_auth_payload(email: str, password: str):
    if not email or "@" not in email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid email is required",
        )
    if not password or len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters",
        )


@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest):
    email = normalize_email(payload.email)
    validate_auth_payload(email, payload.password)

    with get_relational_session() as db:
        existing = db.query(DBUser).filter(DBUser.email == email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Account already exists for this email",
            )
        user = DBUser(
            id=str(uuid.uuid4()),
            username=_unique_username(db, _username_from_email(email)),
            email=email,
            password_hash=hash_password(payload.password),
            name=(payload.name or "").strip() or None,
            plan_type="free",
            is_active=True,
        )
        db.add(user)
        token = create_access_token({"sub": user.id, "email": email})
    return AuthResponse(access_token=token)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    email = normalize_email(payload.email)
    validate_auth_payload(email, payload.password)

    with get_relational_session() as db:
        user = db.query(DBUser).filter(DBUser.email == email).first()
        if not user or not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        token = create_access_token({"sub": str(user.id), "email": email})
    return AuthResponse(access_token=token)
