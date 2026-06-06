"""api/routes_auth.py — Login / token endpoints."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas import LoginRequest, TokenResponse
from api.dependencies import USERS_DB, create_access_token
from utils.logger import get_logger

logger   = get_logger(__name__)
router   = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse, summary="Obtain JWT access token")
def login(req: LoginRequest):
    """
    Authenticate with username/password and receive a JWT token.
    Use the token as `Bearer <token>` in the Authorization header.
    """
    stored_pw = USERS_DB.get(req.username)
    if not stored_pw or stored_pw != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": req.username})
    logger.info("User '%s' authenticated successfully", req.username)
    return TokenResponse(access_token=token)


@router.get("/me", summary="Get current user info")
def me(username: str = ""):
    """Return current authenticated user info."""
    return {"username": username, "role": "analyst"}
