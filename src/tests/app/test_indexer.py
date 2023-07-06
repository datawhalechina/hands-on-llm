import pytest
import pnlp


from docqa.app.indexer import Indexer
from docqa.config import recaller_config


indexer_ins = Indexer(
    recaller_config.host,
    recaller_config.port,
)
coll_name = "docqa_for_indexer_test"


def test_create():
    indexer_ins.create_index(coll_name, 100)
    assert coll_name in indexer_ins.collections


def test_delete():
    indexer_ins.delete_index(coll_name)
    assert coll_name not in indexer_ins.collections