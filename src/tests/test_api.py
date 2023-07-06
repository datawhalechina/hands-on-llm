# https://github.com/encode/starlette/issues/1102
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
from sse_starlette.sse import EventSourceResponse
from httpx_sse import aconnect_sse


from docqa.api import llm_server


def test_root(client):
    resp = client.get("/")
    assert resp.json() == {"msg": "hello"}


@pytest.fixture
def stream_app() -> ASGIApp:
    async def llm_events(request: Request) -> Response:
        event_generator = llm_server.stream_run(request, q="写一首诗，赞美大自然")
        return EventSourceResponse(event_generator)
    return Starlette(routes=[Route("/stream_ask/", endpoint=llm_events)])


@pytest_asyncio.fixture
async def stream_client(stream_app: ASGIApp) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(app=stream_app) as client:
        yield client


@pytest.mark.asyncio
async def test_asgi_test(stream_client: httpx.AsyncClient) -> None:
    async with aconnect_sse(
        stream_client, "GET", "http://testserver/stream_ask/",
    ) as event_source:
        msg = [sse async for sse in event_source.aiter_sse()]
        sse = msg[0]
        assert sse.event == "message"
        dct = sse.json()
        assert "message_id" in dct
        assert "request_id" in dct
        assert "content" in dct
        assert "status" in dct
