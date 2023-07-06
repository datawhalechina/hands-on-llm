from dataclasses import dataclass
from typing import TypeVar, Generic, NewType

import numpy as np


Shape = TypeVar("Shape")
DType = TypeVar("DType", np.int32, np.int64, np.float32)
B = NewType("B", int)
L = NewType("L", int)
H = NewType("H", int)


class Array(np.ndarray, Generic[Shape, DType]):
    ...


@dataclass
class GenerateConfig:

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.7
    top_k: int = 30
    repetition_penalty: float = 1.0


@dataclass
class LlmMessage:

    request_id: str
    message_id: str
    content: str
    status: str


@dataclass
class Doc:

    uuid: str
    content: str
