from datetime import datetime
from typing import Any
import numpy as np

from .base import BaseFaceRepository, FaceRecord


class MongoFaceRepository(BaseFaceRepository):
    """MongoDB implementation of the face storage repository.
    
    Easily pluggable when switching from SQLite to MongoDB.
    """

    def __init__(
        self,
        connection_uri: str = "mongodb://localhost:27017",
        database_name: str = "face_auth_db",
        collection_name: str = "faces",
    ):
        try:
            import pymongo
        except ImportError as e:
            raise ImportError(
                "pymongo is not installed. To use MongoDB, run:\n"
                "pip install pymongo"
            ) from e

        self.client = pymongo.MongoClient(connection_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        # Index on name for quick lookups
        self.collection.create_index("name")

    def add_face(self, name: str, embedding: np.ndarray, metadata: dict | None = None) -> Any:
        emb_list = embedding.flatten().tolist()
        doc = {
            "name": name.strip(),
            "embedding": emb_list,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def get_all_faces(self) -> list[FaceRecord]:
        docs = self.collection.find()
        records = []
        for doc in docs:
            emb = np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
            records.append(
                FaceRecord(
                    id=str(doc["_id"]),
                    name=doc["name"],
                    embedding=emb,
                    created_at=doc.get("created_at", ""),
                    metadata=doc.get("metadata"),
                )
            )
        return records

    def get_faces_by_name(self, name: str) -> list[FaceRecord]:
        docs = self.collection.find({"name": name.strip()})
        records = []
        for doc in docs:
            emb = np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
            records.append(
                FaceRecord(
                    id=str(doc["_id"]),
                    name=doc["name"],
                    embedding=emb,
                    created_at=doc.get("created_at", ""),
                    metadata=doc.get("metadata"),
                )
            )
        return records

    def delete_face(self, name: str) -> int:
        result = self.collection.delete_many({"name": name.strip()})
        return result.deleted_count

    def delete_by_id(self, record_id: Any) -> bool:
        from bson import ObjectId
        result = self.collection.delete_one({"_id": ObjectId(str(record_id))})
        return result.deleted_count > 0

    def count(self) -> int:
        return self.collection.count_documents({})
