import os
import contextlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from mcp_server import mcp
from agent.langchain_agent import generate_gemini, generate_openai

load_dotenv()


class ChatRequest(BaseModel):
    message: str
    model: str = "gemini"


class ChatResponse(BaseModel):
    reply: str


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if hasattr(mcp, "session_manager") and hasattr(mcp.session_manager, "run"):
        async with mcp.session_manager.run():
            yield
    else:
        yield


app = FastAPI(title="Agentic App (LangChain)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if hasattr(mcp, "streamable_http_app"):
    app.mount("/mcp", mcp.streamable_http_app())


@app.get("/")
def root():
    return {
        "message": "Agentic App (LangChain + MCP + RAG). POST /api/chat with { message, model }.",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    model = (req.model or "gemini").strip().lower()
    if model not in ("gemini", "openai"):
        raise HTTPException(status_code=400, detail="model must be 'gemini' or 'openai'")

    try:
        if model == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            if not api_key:
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
            reply = generate_gemini(api_key, gemini_model, req.message.strip())
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
            if not api_key:
                raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
            reply = generate_openai(api_key, openai_model, req.message.strip())
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
