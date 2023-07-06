from typing import List, Union
from dataclasses import asdict
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScoredPoint, Record, VectorParams, Distance
)

from ..dd_model import Array, Doc, B, H


class Indexer:

    def __init__(
        self,
        host: str,
        port: int,
    ):
        self.client = QdrantClient(
            host=host, grpc_port=port, prefer_grpc=True
        )

    def search(
        self,
        collection: str,
        qv: Union[List[float], Array["H", np.float32]],
        topn: int,
        score_threshold: float,
    ) -> List[ScoredPoint]:
        hits = self.client.search(
            collection_name=collection,
            query_vector=qv,
            limit=topn,
            score_threshold=score_threshold,
        )
        return hits

    def create_index(
        self,
        collection: str,
        dim: int
    ) -> None:
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=dim, distance=Distance.COSINE
            )
        )

    def delete_index(
        self,
        collection: str
    ) -> None:
        self.client.delete_collection(collection)

    def count(
        self,
        collection: str
    ) -> int:
        return self.client.count(collection).count

    @property
    def collections(self) -> List[str]:
        res = [v.name for v in self.client.get_collections().collections]
        return res

    def add_docs(
        self,
        collection: str,
        data: List[Doc],
        vectors: Array["B,H", np.float32]
    ):
        self.client.upload_records(
            collection_name=collection,
            records=[
                Record(
                    id=doc.uuid,
                    vector=vectors[_idx].tolist(),
                    payload=asdict(doc),
                ) for _idx, doc in enumerate(data)
            ]
        )

    def delete_docs(
        self,
        collection: str,
        doc_ids: List[str]
    ):
        self.client.delete(
            collection, points_selector=doc_ids
        )
