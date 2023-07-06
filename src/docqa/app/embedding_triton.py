from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer
import tritonclient.grpc as client

from ..utils import retry_else_stop
from ..dd_model import Array, B, L, H
from ..config import TRITON_CLIENT_TIMEOUT


class Embedding:

    def __init__(
        self,
        triton_host: str,
        triton_model: str,
        model_id: str
    ):
        self.triton_model = triton_model
        self.client = client.InferenceServerClient(url=triton_host)
        self.triton_outputs = [
            client.InferRequestedOutput("last_hidden_state")
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _build_inputs(
        self,
        inp: Dict[str, Array["B,L", np.int64]]
    ) -> List[client.InferInput]:
        shape = inp.input_ids.shape
        input_tensors = [
            client.InferInput("input_ids", shape, datatype="INT64"),
            client.InferInput("attention_mask", shape, datatype="INT64"),
            client.InferInput("token_type_ids", shape, datatype="INT64"),
        ]
        input_tensors[0].set_data_from_numpy(inp.input_ids)
        input_tensors[1].set_data_from_numpy(inp.attention_mask)
        input_tensors[2].set_data_from_numpy(inp.token_type_ids)
        return input_tensors

    @retry_else_stop
    def get_embedding(
        self,
        texts: List[str]
    ) -> Array["B,H", np.float32]:
        inp = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            max_length=512,
            truncation=True)
        input_tensors = self._build_inputs(inp)
        query_response = self.client.infer(
            model_name=self.triton_model,
            inputs=input_tensors,
            outputs=self.triton_outputs,
            client_timeout=TRITON_CLIENT_TIMEOUT
        )
        last_hidden_state = query_response.as_numpy("last_hidden_state")
        return self._pooling(inp.attention_mask, last_hidden_state)

    def _pooling(
        self,
        attention_mask: Array["B,L", np.int64],
        token_embeddings: Array["B,L,H", np.float32],
    ) -> Array["B, H", np.float32]:
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        np.clip(sum_mask, a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
