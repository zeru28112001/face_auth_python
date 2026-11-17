"""
KYC routes:
  POST /api/kyc/verify  - Verify ID card photo against live selfie and register user
"""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from api.dependencies import PipelineState, get_pipeline
from api.schemas import KYCResponse

router = APIRouter(prefix="/api/kyc", tags=["eKYC Onboarding"])


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Unable to decode image. Send a valid JPEG or PNG.")
    return img


@router.post(
    "/verify",
    response_model=KYCResponse,
    summary="eKYC: Verify ID card vs live selfie and register user",
    description=(
        "Full eKYC onboarding workflow:\n"
        "1. ID card is deskewed, and the portrait photo is extracted from it\n"
        "2. Liveness check is run on the selfie (MiniFASNet anti-spoofing)\n"
        "3. SFace 128D embeddings are compared: ID portrait vs live face\n"
        "4. If similarity >= threshold AND liveness passes → user is registered into the database\n\n"
        "Returns `APPROVED` on success with complete KYC metadata stored."
    ),
)
async def kyc_verify(
    id_card: UploadFile = File(..., description="Government ID card or passport image (JPEG/PNG)"),
    selfie: UploadFile = File(..., description="Live selfie image (JPEG/PNG)"),
    name: str = Form(..., description="User's full legal name"),
    id_number: str | None = Form(None, description="ID card or passport number"),
    document_type: str = Form("NATIONAL_ID", description="Document type: NATIONAL_ID, PASSPORT, DRIVER_LICENSE"),
    allow_duplicate: bool = Form(False, description="Allow re-registration of an already registered face"),
    pipeline: PipelineState = Depends(get_pipeline),
) -> KYCResponse:
    id_raw = await id_card.read()
    selfie_raw = await selfie.read()

    id_img = _decode_image(id_raw)
    selfie_img = _decode_image(selfie_raw)

    result = pipeline.kyc_verifier.verify_and_onboard(
        id_image=id_img,
        live_frame=selfie_img,
        user_name=name,
        id_number=id_number,
        document_type=document_type,
        allow_duplicate=allow_duplicate,
        auto_register=True,
    )

    return KYCResponse(
        status=result.status,
        is_verified=result.is_verified,
        user_name=result.user_name,
        id_number=result.id_number,
        similarity_score=round(result.similarity_score, 4),
        liveness_score=round(result.liveness_score, 4),
        message=result.message,
        metadata=result.metadata,
    )
