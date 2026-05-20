"""FastAPI dependencies for Auth0 JWT authentication and authorisation."""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer

from domain.models import UserContext
from domain.ports import AuthPort
from interactors.api.deps import get_auth

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}

_auth0_domain = os.getenv("AUTH0_DOMAIN", "")
_oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"https://{_auth0_domain}/authorize",
    tokenUrl=f"https://{_auth0_domain}/oauth/token",
    auto_error=False,
)

# When AUTH_DISABLED=true, all auth checks are bypassed and every request is
# treated as a local admin user.  Never set this in production.
_AUTH_DISABLED = os.getenv("AUTH_DISABLED", "").lower() == "true"

_LOCAL_DEV_USER = UserContext(
    user_id="local-dev",
    email="local@dev",
    roles=["user", "admin"],
)


def require_auth(
    token: str | None = Depends(_oauth2_scheme),
    auth_port: AuthPort = Depends(get_auth),
) -> None:
    if _AUTH_DISABLED:
        return
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    if auth_port.authenticate(token) is None:
        raise HTTPException(status_code=401, detail="Invalid token", headers=_WWW_AUTH)


def get_current_user(
    token: str | None = Depends(_oauth2_scheme),
    auth_port: AuthPort = Depends(get_auth),
) -> UserContext:
    if _AUTH_DISABLED:
        return _LOCAL_DEV_USER
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    user = auth_port.authenticate(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token", headers=_WWW_AUTH)
    return user


def require_approved(user: UserContext = Depends(get_current_user)) -> UserContext:
    if _AUTH_DISABLED:
        return _LOCAL_DEV_USER
    if "user" not in user.roles and "admin" not in user.roles:
        raise HTTPException(
            status_code=403,
            detail="Access not approved. Contact an administrator.",
        )
    return user


def require_admin(user: UserContext = Depends(get_current_user)) -> UserContext:
    if _AUTH_DISABLED:
        return _LOCAL_DEV_USER
    if "admin" not in user.roles:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user