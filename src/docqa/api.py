from typing import Literal, List, Generator, Dict
from dataclasses import asdict
import asyncio
import json
import uuid
from fastapi import Request

from .app.base import Register
from .app import Recaller, Prompter
from .dd_model import GenerateConfig, LlmMessage
from .config import (
    llm_config, recaller_config,
    STREAM_DELAY
)


class LlmServer:

    def __init__(
        self,
        model_type: Literal["llama", "gpt"] = "llama",
        business_type: Literal["docqa", "qa"] = "docqa",
    ):
        self.gen_config = GenerateConfig()
        llm_cls = Register.get(model_type)
        if not llm_cls:
            msg = f"{self} do not support model {model_type}"
            raise NotImplementedError(msg)
        self.llm_ins = llm_cls(
            model_id=llm_config.model_id,
            triton_host=llm_config.triton_host,
            triton_model=llm_config.triton_model,
        )
        self.recall_ins = Recaller(recaller_config.host, recaller_config.port)
        self.prompt_ins = Prompter(business_type)

    def get_prompt(
        self,
        q: str
    ) -> str:
        docs = self.recall_ins.recall(
            recaller_config.collection,
            q,
            recaller_config.topn,
            recaller_config.threshold)
        prompt = self.prompt_ins.build_prompt(q, docs)
        return prompt

    def stream_generate(
        self,
        prompts: List[str],
    ) -> Generator[List[str], None, None]:
        for resp_texts in self.llm_ins.generate(
            prompts,
            self.gen_config.max_new_tokens,
            self.gen_config.temperature,
            self.gen_config.top_p,
            self.gen_config.top_k,
            self.gen_config.repetition_penalty,
        ):
            yield resp_texts

    def build_event_msg(
        self,
        request_id: str,
        message_id: str,
        resp_str: str,
        status: str,
    ) -> Dict:
        msg = LlmMessage(request_id, message_id, resp_str, status)
        dct = asdict(msg)
        out_msg = json.dumps(dct)
        return out_msg

    async def stream_run(
        self,
        request: Request,
        q: str,
    ) -> Generator[Dict, None, None]:
        rid = str(uuid.uuid4())
        if await request.is_disconnected():
            yield

        prompt = self.get_prompt(q)

        texts = [prompt]
        i = 0
        sents = []
        for txt in texts:
            sents.append(f"Human: {txt} \n\nAssistant: ")

        for resp_text in self.stream_generate(sents):
            resp_str = resp_text[0].replace(sents[0], "")
            yield self.build_event_msg(rid, str(i), resp_str, "in_progress")
            await asyncio.sleep(STREAM_DELAY)
            i += 1

        yield self.build_event_msg(rid, str(i), "", "stop")


llm_server = LlmServer()
