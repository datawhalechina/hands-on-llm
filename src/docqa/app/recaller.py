from typing import List, Dict

import pnlp

from .embedding_triton import Embedding
from .indexer import Indexer
from ..dd_model import Doc
from ..utils import Fuse
from ..config import emd_config, FUSE_COUNT


class Recaller:

    def __init__(
        self,
        host: str,
        port: int
    ):
        self.vec_index_ins = Indexer(host, port)
        self.emd_ins = Embedding(
            emd_config.triton_host,
            emd_config.triton_model,
            emd_config.model_id
        )

    @Fuse
    def _get_embedding(self, texts: List[str]):
        q_emd = self.emd_ins.get_embedding(texts)
        return q_emd

    def recall(
        self,
        collection: str,
        q: str,
        topn: int,
        score_threshold: float
    ) -> List[Dict]:
        if Recaller._get_embedding.nfails >= FUSE_COUNT:
            return
        q_emd = self._get_embedding([q])
        if q_emd is None:
            return
        qv = q_emd[0]
        hits = self.vec_index_ins.search(
            collection, qv, topn, score_threshold
        )
        res = []
        for hit in hits:
            hit.payload["score"] = hit.score
            res.append(hit.payload)
        return res

    def add_docs(
        self,
        collection: str,
        docs: List[Doc],
        batch_size: int,
    ):
        total = len(docs)
        batch_num = total // batch_size
        if total % batch_size != 0:
            batch_num += 1
        batches = pnlp.generate_batches_by_size(docs, batch_size)
        import time
        for i, batch in enumerate(batches, start=1):
            texts = [v.content for v in batch]
            vectors = self.emd_ins.get_embedding(texts)
            self.vec_index_ins.add_docs(collection, batch, vectors)
            print(f"uploading batch: {i} / {batch_num}")
            time.sleep(1)

    def delete_docs(
        self,
        collection: str,
        doc_ids: List[str]
    ):
        self.vec_index_ins.delete_docs(collection, doc_ids)
