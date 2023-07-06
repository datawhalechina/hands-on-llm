import pytest
import pnlp
from qdrant_client.models import Distance, VectorParams

from docqa.app.recaller import Recaller
from docqa.dd_model import Doc
from docqa.config import recaller_config


coll_name = "docqa_for_recaller_test"
recaller_ins = Recaller(
    recaller_config.host,
    recaller_config.port
)
recaller_ins.vec_index_ins.create_index(
    coll_name, recaller_config.dim
)

@pytest.fixture(scope="class")
def delete_index():
    print("setup")
    yield
    recaller_ins.vec_index_ins.delete_index(coll_name)


class TestRecaller:

    def test_add_delete_docs(self):
        texts = [
            "doc: 你好",
            "doc: 爱情",
            "doc: 友情",
            "doc: 世界",
            "doc: 爱因斯坦",
            "doc: 人工智能"
        ]
        uids = [pnlp.generate_uuid(s) for s in texts]
        docs = []
        for i, s in enumerate(texts):
            d = Doc(uids[i], s)
            docs.append(d)
        recaller_ins.add_docs(coll_name, docs, 2)
        assert recaller_ins.vec_index_ins.count(coll_name) == len(texts)
        recaller_ins.delete_docs(coll_name, [uids[0]])
        assert recaller_ins.vec_index_ins.count(coll_name) == len(texts) - 1
    
    
    @pytest.mark.parametrize("q, threshold, res_num", [
        ("你好", 0.1, 3),
        ("你好", 0.8, 0),
        ("爱情", 0.1, 3),
        ("爱情", 0.8, 1),
        ("爱情", 0.9, 0),
    ])
    def test_recall(self, q, threshold, res_num, delete_index):
        res = recaller_ins.recall(
            coll_name, q, 3, threshold)
        assert len(res) == res_num
        if len(res) > 0:
            assert type(res[0]) == dict