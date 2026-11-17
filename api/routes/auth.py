"""
Auth routes:
  POST /api/auth/verify  - Verify a live selfie against registered KYC users
"""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from api.dependencies import PipelineState, get_pipeline
from api.schemas import AuthResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Unable to decode image. Send a valid JPEG or PNG.")
    return img


@router.post(
    "/verify",
    response_model=AuthResponse,
    summary="Verify a live selfie against registered faces",
    description=(
        "Upload a selfie (JPEG/PNG). The system runs:\n"
        "1. YuNet face detection\n"
        "2. MiniFASNet anti-spoofing liveness check\n"
        "3. SFace 128D biometric matching against all registered users\n\n"
        "Returns `GRANTED` if a match is found above threshold, `DENIED` otherwise."
    ),
)
async def verify_face(
    selfie: UploadFile = File(..., description="Live selfie image (JPEG or PNG)"),
    pipeline: PipelineState = Depends(get_pipeline),
) -> AuthResponse:
    raw = await selfie.read()
    frame = _decode_image(raw)

    # Detect face
    faces = pipeline.detector.detect(frame)
    if not faces:
        return AuthResponse(
            status="NO_FACE",
            user_name="Unknown",
            similarity_score=0.0,
            liveness_score=0.0,
            is_kyc_verified=False,
            id_number=None,
            message="No face detected in the submitted image.",
        )

    best_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

    # Liveness check
    liveness = pipeline.liveness_detector.check_liveness(frame, best_face)
    liveness_score = float(liveness.liveness_score)

    if not liveness.is_real:
        return AuthResponse(
            status="DENIED",
            user_name="Unknown",
            similarity_score=0.0,
            liveness_score=liveness_score,
            is_kyc_verified=False,
            id_number=None,
            message=f"Anti-spoofing check failed: {liveness.attack_type}. Presentation attack detected.",
        )

    # Recognition
    auth = pipeline.authenticator.authenticate_face(frame, best_face)
    is_kyc = bool(auth.metadata and auth.metadata.get("kyc_verified"))
    id_num = auth.metadata.get("id_number") if auth.metadata else None

    return AuthResponse(
        status=auth.status,
        user_name=auth.name,
        similarity_score=round(auth.similarity_score, 4),
        liveness_score=round(liveness_score, 4),
        is_kyc_verified=is_kyc,
        id_number=id_num,
        message=(
            f"Access granted: {auth.name} (similarity: {auth.similarity_score:.2%})"
            if auth.is_authenticated
            else f"Access denied: No matching identity found (best score: {auth.similarity_score:.2%})."
        ),
    )
