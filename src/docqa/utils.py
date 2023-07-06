from functools import wraps
import time
import types
from pydantic import BaseModel

from fastapi import HTTPException
import tritonclient.grpc as grpcclient
from tenacity import (
    retry_if_exception_type, wait_exponential, stop_after_attempt,
    Retrying, RetryError
)

from .config import logger, TRITON_RETRY_TIMES, FUSE_COUNT


def log_everything(func):

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            for key, val in kwargs.items():
                if isinstance(val, BaseModel):
                    reqbody = val.dict()
            else:
                reqbody = kwargs

            func_name = func.__name__
            inp = f"{func_name} Input: {reqbody}"
            logger.info(inp)

            start = time.perf_counter()
            res = await func(*args, **kwargs)
            end = time.perf_counter()
            cost = round(end - start, 3) * 1000

            out = f"{func_name} Output: {res}"
            logger.info(out)

            cost_info = f"{func_name} Cost: {cost:.{3}f} milliseconds"
            logger.info(cost_info)
            return res
        except Exception as e:
            msg = f"Error with {e}"
            logger.exception(msg)
            raise HTTPException(
                status_code=500,
            )
    return wrapper


def notify(msg: str):
    print(msg)
    # send_msg_to_admin(msg)


def retry_else_stop(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            for attempt in Retrying(
                retry=retry_if_exception_type(
                    grpcclient.InferenceServerException), wait=wait_exponential(
                    min=0, max=3), stop=stop_after_attempt(TRITON_RETRY_TIMES)):
                with attempt:
                    return func(*args, **kwargs)
        except RetryError:
            func_name = func.__name__
            msg = f"{func_name} error after retry!!!"
            logger.error(msg)
            notify(msg)
            return
    return wrapper


class Fuse:

    def __init__(self, func):
        wraps(func)(self)
        self.nfails = 0
        self._func = func

    def __call__(self, *args, **kwargs):
        res = self.__wrapped__(*args, **kwargs)
        if res is None:
            self.nfails += 1
        else:
            self.nfails = 0
        if self.nfails >= FUSE_COUNT:
            msg = f"{self._func} is fused!!!"
            logger.error(msg)
            notify(msg)
        return res

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)
