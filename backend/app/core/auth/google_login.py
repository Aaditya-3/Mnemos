"""
Google authentication endpoint backed by the relational database.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.core.auth.jwt_auth import create_access_token
from backend.app.core.db.relational import DBUser, get_relational_session


router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


class GoogleLoginRequest(BaseModel):
    id_token: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _username_from_email(email: str) -> str:
    base = (email or "").split("@", 1)[0].strip().lower() or "user"
    base = re.sub(r"[^a-z0-9._-]+", "_", base).strip("._-") or "user"
    return base[:120]


async def verify_google_id_token(id_token: str) -> dict[str, Any]:
    url = "https://oauth2.googleapis.com/tokeninfo"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params={"id_token": id_token})
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token",
        )
    data = resp.json()
    aud = data.get("aud")
    if GOOGLE_CLIENT_ID and aud != GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google client_id",
        )
    return data


@router.post("/google", response_model=AuthResponse)
async def login_with_google(payload: GoogleLoginRequest):
    token_info = await verify_google_id_token(payload.id_token)

    email = str(token_info.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token missing email",
        )

    with get_relational_session() as db:
        user = db.query(DBUser).filter(DBUser.email == email).first()
        if user is None:
            username_base = _username_from_email(email)
            username = username_base
            idx = 1
            while db.query(DBUser.id).filter(DBUser.username == username).first():
                suffix = f"_{idx}"
                username = f"{username_base[: max(1, 120 - len(suffix))]}{suffix}"
                idx += 1
            user = DBUser(
                id=str(uuid.uuid4()),
                username=username,
                email=email,
                password_hash="google_oauth",
                name=str(token_info.get("name") or username),
                plan_type="free",
                is_active=True,
            )
            db.add(user)

        access_token = create_access_token({"sub": str(user.id), "email": email})
    return AuthResponse(access_token=access_token)
