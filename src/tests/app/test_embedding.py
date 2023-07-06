import pytest
import numpy as np

from docqa.app.embedding_triton import Embedding
from docqa.config import emd_config


emd_ins = Embedding(
    emd_config.triton_host,
    emd_config.triton_model,
    emd_config.model_id,
)


@pytest.mark.parametrize("inp, batch_size", [
    ([""], 1),
    (["你好"], 1),
    (["你好" * 1024], 1),
    (["你好" * 2048], 1),
    (["你好", "hi"], 2),
    (["你好" * 1024, "hi" * 2048], 2),
    (["你好"] * 32, 32),
])
def test_embedding(inp, batch_size):
    emds = emd_ins.get_embedding(inp)
    assert type(emds) == np.ndarray
    shape = emds.shape
    assert shape[0] == len(emds) == batch_size
    assert shape[1] == 768
