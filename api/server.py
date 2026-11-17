"""
Face Authentication & eKYC FastAPI Application.

Endpoints:
  GET  /                      → Welcome message
  GET  /health                → System health & DB status
  POST /api/auth/verify       → Biometric face authentication
  POST /api/kyc/verify        → eKYC onboarding (ID card + selfie)
  GET  /api/users             → List registered users
  DELETE /api/users/{name}    → Delete a user
  GET  /docs                  → Swagger UI (interactive API docs)
  GET  /redoc                 → ReDoc documentation

Environment Variables:
  DB_BACKEND          : "sqlite" (default) or "mongodb"
  MONGO_URI           : MongoDB connection string (default: mongodb://localhost:27017)
  MONGO_DB            : MongoDB database name (default: face_auth_db)
  AUTH_THRESHOLD      : Cosine similarity threshold (default: 0.363)
  LIVENESS_THRESHOLD  : MiniFASNet liveness threshold (default: 0.50)
  API_HOST            : Host to bind (default: 0.0.0.0)
  API_PORT            : Port to listen on (default: 8000)
"""
import os
import sys
from contextlib import asynccontextmanager

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import init_pipeline, get_pipeline
from api.routes import auth, kyc, users
from api.schemas import HealthResponse

# ─── Configuration from environment ───────────────────────────────────────────
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "face_auth_db")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


# ─── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("  Face Auth & eKYC API  –  Starting up")
    print("=" * 55)
    print(f"  DB Backend   : {DB_BACKEND}")
    if DB_BACKEND == "mongodb":
        print(f"  MongoDB URI  : {MONGO_URI}")
    print(f"  Auth Thresh  : {os.getenv('AUTH_THRESHOLD', '0.363')}")
    print(f"  Liveness     : {os.getenv('LIVENESS_THRESHOLD', '0.50')}")
    print("-" * 55)
    print("  Loading YuNet / SFace / MiniFASNet models...")

    pipeline = init_pipeline(db_backend=DB_BACKEND, mongo_uri=MONGO_URI)
    registered = pipeline.repo.count()
    print(f"  Pipeline ready. Registered faces: {registered}")
    print(f"  Docs available at: http://{API_HOST}:{API_PORT}/docs")
    print("=" * 55 + "\n")

    yield  # App runs here

    print("\n[Shutdown] Face Auth API shutting down cleanly.")


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Face Authentication & eKYC API",
    description=(
        "Real-time biometric face authentication and KYC identity verification backend.\n\n"
        "**Features:**\n"
        "- YuNet face detection + 5-landmark localization\n"
        "- SFace 128D cosine-similarity face recognition\n"
        "- MiniFASNet v2 deep-learning anti-spoofing (detects printed photos & screens)\n"
        "- eKYC: ID card deskewing, portrait extraction, 1-to-1 biometric cross-verification\n"
        "- Swappable database: **SQLite** (default) or **MongoDB** (set `DB_BACKEND=mongodb`)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(kyc.router)
app.include_router(users.router)


# ─── Root & Health ────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "name": "Face Authentication & eKYC API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        db_backend=DB_BACKEND,
        registered_faces=pipeline.repo.count(),
    )


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
