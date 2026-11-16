from typing import Literal
from .base import BaseFaceRepository, FaceRecord
from .sqlite_repo import SQLiteFaceRepository


def get_repository(
    backend: Literal["sqlite", "mongodb"] = "sqlite",
    sqlite_path: str = "faces.db",
    mongo_uri: str = "mongodb://localhost:27017",
    mongo_db: str = "face_auth_db",
) -> BaseFaceRepository:
    """Factory to get the face storage repository."""
    if backend == "sqlite":
        return SQLiteFaceRepository(db_path=sqlite_path)
    elif backend == "mongodb":
        from .mongo_repo import MongoFaceRepository

        return MongoFaceRepository(connection_uri=mongo_uri, database_name=mongo_db)
    else:
        raise ValueError(f"Unknown database backend: {backend}. Choose 'sqlite' or 'mongodb'.")


__all__ = ["BaseFaceRepository", "FaceRecord", "SQLiteFaceRepository", "get_repository"]
