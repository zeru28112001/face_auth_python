from contextlib import closing
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

from .base import BaseFaceRepository, FaceRecord


class SQLiteFaceRepository(BaseFaceRepository):
    """SQLite implementation of the face storage repository."""

    def __init__(self, db_path: str = "faces.db"):
        self.db_path = Path(db_path).resolve()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._get_connection()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata TEXT
                    )
                    """
                )

    def add_face(self, name: str, embedding: np.ndarray, metadata: dict | None = None) -> int:
        clean_name = name.strip()
        emb_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
        now = datetime.now().isoformat()
        meta_str = json.dumps(metadata) if metadata else None

        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO faces (name, embedding, created_at, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (clean_name, emb_bytes, now, meta_str),
                )
                return cursor.lastrowid

    def get_all_faces(self) -> list[FaceRecord]:
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, embedding, created_at, metadata FROM faces ORDER BY id ASC")
            rows = cursor.fetchall()

        records = []
        for row in rows:
            emb_array = np.frombuffer(row["embedding"], dtype=np.float32).reshape(1, -1)
            meta = json.loads(row["metadata"]) if row["metadata"] else None
            records.append(
                FaceRecord(
                    id=row["id"],
                    name=row["name"],
                    embedding=emb_array,
                    created_at=row["created_at"],
                    metadata=meta,
                )
            )
        return records

    def get_faces_by_name(self, name: str) -> list[FaceRecord]:
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, embedding, created_at, metadata FROM faces WHERE name = ? ORDER BY id ASC",
                (name.strip(),),
            )
            rows = cursor.fetchall()

        records = []
        for row in rows:
            emb_array = np.frombuffer(row["embedding"], dtype=np.float32).reshape(1, -1)
            meta = json.loads(row["metadata"]) if row["metadata"] else None
            records.append(
                FaceRecord(
                    id=row["id"],
                    name=row["name"],
                    embedding=emb_array,
                    created_at=row["created_at"],
                    metadata=meta,
                )
            )
        return records

    def delete_face(self, name: str) -> int:
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM faces WHERE name = ?", (name.strip(),))
                return cursor.rowcount

    def delete_by_id(self, record_id: Any) -> bool:
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM faces WHERE id = ?", (int(record_id),))
                return cursor.rowcount > 0

    def count(self) -> int:
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM faces")
            return cursor.fetchone()[0]
