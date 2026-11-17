"""
Users routes:
  GET /api/users        - List all registered users with KYC status
  DELETE /api/users/{name} - Remove a user by name
"""
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import PipelineState, get_pipeline
from api.schemas import UserRecord, UsersListResponse

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get(
    "",
    response_model=UsersListResponse,
    summary="List all registered users",
    description="Returns all registered users with their KYC verification status and ID number.",
)
def list_users(pipeline: PipelineState = Depends(get_pipeline)) -> UsersListResponse:
    records = pipeline.repo.get_all_faces()
    users = []
    for r in records:
        meta = r.metadata or {}
        users.append(
            UserRecord(
                id=str(r.id),
                name=r.name,
                created_at=r.created_at,
                is_kyc_verified=bool(meta.get("kyc_verified", False)),
                id_number=meta.get("id_number"),
                document_type=meta.get("document_type"),
            )
        )
    return UsersListResponse(total=len(users), users=users)


@router.delete(
    "/{name}",
    summary="Delete a user by name",
    description="Permanently removes a registered user and their biometric data from the database.",
)
def delete_user(name: str, pipeline: PipelineState = Depends(get_pipeline)) -> dict:
    deleted = pipeline.repo.delete_face(name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No user found with name '{name}'.")
    # Refresh the in-memory authenticator cache
    pipeline.authenticator.reload_faces()
    return {"message": f"Deleted {deleted} record(s) for user '{name}'."}
