import pytest

from docqa.app.prompter import Prompter


@pytest.mark.parametrize("btype, docs", [
    ("docqa", [{"content": "cont1"}, {"content": "cont2"}]),
    ("docqa", [{"content": "cont1"}]),
])
def test_prompter(btype, docs):
    prompt_ins = Prompter(btype)
    prompt = prompt_ins.build_prompt("q", docs)
    assert type(prompt) == str
    assert docs[0]["content"] in prompt