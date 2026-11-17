"""
Pydantic response schemas for the Face Auth & eKYC API.
Used by FastAPI for automatic validation, serialization, and OpenAPI docs.
"""
from pydantic import BaseModel, Field
from typing import Literal


# ─── Auth ──────────────────────────────────────────────────────────────────────

class AuthResponse(BaseModel):
    status: Literal["GRANTED", "DENIED", "NO_FACE"]
    user_name: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    liveness_score: float = Field(ge=0.0, le=1.0)
    is_kyc_verified: bool
    id_number: str | None
    message: str


# ─── KYC ───────────────────────────────────────────────────────────────────────

class KYCResponse(BaseModel):
    status: Literal[
        "APPROVED",
        "REJECTED",
        "NO_ID_FACE",
        "LIVENESS_FAILED",
        "IDENTITY_CONFLICT",
        "INVALID_DOCUMENT",
    ]
    is_verified: bool
    user_name: str
    id_number: str | None
    similarity_score: float = Field(ge=0.0, le=1.0)
    liveness_score: float = Field(ge=0.0, le=1.0)
    message: str
    metadata: dict = Field(default_factory=dict)


# ─── Users ─────────────────────────────────────────────────────────────────────

class UserRecord(BaseModel):
    id: str
    name: str
    created_at: str
    is_kyc_verified: bool
    id_number: str | None
    document_type: str | None


class UsersListResponse(BaseModel):
    total: int
    users: list[UserRecord]


# ─── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db_backend: str
    registered_faces: int
    version: str = "1.0.0"
