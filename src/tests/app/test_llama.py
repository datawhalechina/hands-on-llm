import pytest
from dataclasses import asdict
import numpy as np

from docqa.app.llama_triton_ft import LlamaLlm
from docqa.dd_model import GenerateConfig
from docqa.config import llm_config


llm_ins = LlamaLlm(
    llm_config.triton_host,
    llm_config.triton_model,
    llm_config.model_id,
)


@pytest.mark.parametrize("inp", [
    (["写一首诗，赞美大自然"]),
    (["写一首中文歌曲，赞美大自然", "写一首诗，赞美大自然"]),
])
def test_llama(inp):
    gen_config = GenerateConfig(max_new_tokens=5)
    for out in llm_ins.generate(
        inp,
        **asdict(gen_config)
    ):
        assert len(out) == len(inp)
