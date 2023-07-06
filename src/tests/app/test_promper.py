import pytest

from docqa.app.prompter import Prompter


@pytest.mark.parametrize("btype, docs", [
    ("docqa", ["doc1", "doc2"]),
    ("docqa", ["doc1", ]),
])
def test_prompter(btype, docs):
    prompt_ins = Prompter(btype)
    prompt = prompt_ins.build_prompt("q", docs)
    assert type(prompt) == str
    assert docs[0] in prompt