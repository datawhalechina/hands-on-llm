from typing import List, Generator, Dict
import json
from functools import partial
import queue  # block_queue
import asyncio
import threading

import google.protobuf.json_format
import numpy as np
import tritonclient.grpc as client
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import np_to_triton_dtype
from transformers import LlamaTokenizer

from .base import Register

Queue = queue.Queue


@Register.regist
class LlamaLlm:

    def __init__(
        self,
        triton_host: str,
        triton_model: str,
        model_id: str,
    ):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.padding_side = "left"

        self.triton_host = triton_host
        self.triton_model = triton_model

        self.triton_stream_timeout = None

    def build_parameters(
        self,
        sents: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> List[Dict[str, np.array]]:
        inp = self.tokenizer(
            sents,
            padding=True,
            add_special_tokens=True,
            return_tensors="np",
            return_length=True)
        batch_range = range(len(sents))
        input_ids = inp.input_ids.astype(np.uint32)
        input_len = np.array(
            [[v] for v in inp.length], dtype=np.uint32)
        output_len = np.array(
            [[max_new_tokens] for v in batch_range], dtype=np.uint32)
        temps = np.array(
            [[temperature] for v in batch_range], dtype=np.float32)
        top_ps = np.array([[top_p] for v in batch_range], dtype=np.float32)
        top_ks = np.array([[top_k] for v in batch_range], dtype=np.uint32)
        repetition_penalties = np.array(
            [[repetition_penalty] for v in batch_range], dtype=np.float32)

        res = []
        res.append({
            "name": "input_ids",
            "data": input_ids,
        })
        res.append({
            "name": "input_lengths",
            "data": input_len,
        })
        res.append({
            "name": "request_output_len",
            "data": output_len,
        })

        res.append({
            "name": "temperature",
            "data": temps,
        })
        res.append({
            "name": "runtime_top_p",
            "data": top_ps,
        })
        res.append({
            "name": "runtime_top_k",
            "data": top_ks,
        })
        res.append({
            "name": "repetition_penalty",
            "data": repetition_penalties,
        })
        return res

    def prepare_tensor(
        self,
        param_name: str,
        param_value: np.array,
    ) -> client.InferInput:
        t = client.InferInput(
            param_name,
            param_value.shape,
            np_to_triton_dtype(
                param_value.dtype))
        t.set_data_from_numpy(param_value)
        return t

    def stream_consume(
        self,
        queue: Queue,
        thread,
    ) -> Generator[List[str], None, None]:
        while True:
            try:
                item = queue.get()
                if item is None:
                    break

                message = ModelInferResponse()
                google.protobuf.json_format.Parse(json.dumps(item), message)

                result = client.InferResult(message)
                seq_len = result.as_numpy("sequence_length")
                out_ids = result.as_numpy("output_ids")

                idx = seq_len[0, 0]
                gen_texts = self.tokenizer.batch_decode(
                    out_ids[:, 0, :idx], skip_special_tokens=True
                )
                yield gen_texts
            except Exception:
                thread.join()

    def stream_callback(
        self,
        queue: Queue,
        result,
        error
    ):
        if error:
            queue.put(error)
        else:
            queue.put(result.get_response(as_json=True))

    async def _do_generate(
        self,
        req_params,
        result_queue,
    ):
        with client.InferenceServerClient(self.triton_host) as cl:
            payload = [
                self.prepare_tensor(v["name"], v["data"])
                for v in req_params
            ]
            cl.start_stream(
                callback=partial(self.stream_callback, result_queue),
                stream_timeout=self.triton_stream_timeout,
            )
            cl.async_stream_infer(self.triton_model, payload)
        result_queue.put(None)

    def generate(
        self,
        sents: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
        req_params = self.build_parameters(
            sents,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty)
        result_queue = Queue()
        t = threading.Thread(
            target=asyncio.run, args=(
                self._do_generate(req_params, result_queue),
            )
        )
        t.start()
        return self.stream_consume(result_queue, t)
