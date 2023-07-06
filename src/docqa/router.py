from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse


from .api import llm_server
from .utils import log_everything

api_router = APIRouter(prefix="/api/v1")


@api_router.get("/steam_ask")
@log_everything
async def ask_sse_svc(request: Request, q: str):
    event_generator = llm_server.stream_run(request, q)
    return EventSourceResponse(event_generator)
