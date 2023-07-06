from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware

from docqa.router import api_router
from docqa.config import PROFILE


app = FastAPIOffline(
    title="LLM based DocQA",
    description="Anaswer Your Question based on LLM and Document.",
    version="v0.0.1",
    openapi_url="/api/openapi.json"
)

if PROFILE == "dev":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, tags=["LLM"])


@app.get("/")
async def hello():
    msg = "hello"
    return {
        "msg": msg,
    }
