from typing import List


class Prompter:

    def __init__(
        self,
        business_type: str
    ):
        self.supported_types = ["docqa", "qa"]
        if business_type not in self.supported_types:
            msg = f"{self} does not support type: {business_type}"
            raise ValueError(msg)
        self.business_type = business_type

    def build_prompt(
        self,
        *args,
        **kwargs,
    ):
        func_name = f"build_{self.business_type}_prompt"
        return getattr(self, func_name)(*args, **kwargs)

    def build_docqa_prompt(
        self,
        q: str,
        docs: List[str],
    ) -> str:
        if not docs:
            prompt = "你现在是一台服务器，你的服务不可用了。用户正在访问，你这样回复用户："
            return prompt
        doc_str = "\n".join(docs)
        prompt = f"""根据给定文本回答问题，如果没有答案或答案不确定，回复“未涉及”。

给定文本：
```
{doc_str}
```
问题：
{q}
"""
        return prompt
