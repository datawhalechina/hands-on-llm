import pytest
from typing import Optional

import tritonclient.grpc as grpcclient

from docqa.utils import retry_else_stop, Fuse


class OtherError(BaseException):
    ...


@retry_else_stop
def func(error):
    raise error("msg")


@pytest.mark.parametrize("error", [
    (grpcclient.InferenceServerException),
    (OtherError),
])
def test_retry_else_stop(error):
    if error is OtherError:
        with pytest.raises(error) as e_info:
            func(error)
        assert "msg" in str(e_info)
    else:
        assert func(error) is None


class Spam:

    @Fuse
    def foo(self, x: Optional[int]):
        # NOTE: None will trigger count
        return x
    
    def run(self, x: Optional[int]):
        if Spam.foo.nfails > 0:
            return 0
        else:
            return self.foo(x)


@pytest.mark.parametrize("x, expected", [
    (1, 1),
    (None, None),  # NOTE: after run will trigger fuse
    (1, 0),
])
def test_fuse(x, expected):
    sp = Spam()
    assert sp.run(x) == expected