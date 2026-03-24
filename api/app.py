"""
DuctAI Copilot — FastAPI app.

Endpoints:
    POST /chat    — send a message; history is managed server-side by session_id
    GET  /health  — check DB and model artifacts are reachable

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from agent.agent import build_agent
from api.schemas import ChatRequest, ChatResponse

app = FastAPI(title="DuctAI Copilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build agent once at startup; MemorySaver lives in-process.
_agent = build_agent()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = _agent.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"configurable": {"thread_id": req.session_id}},
        )
        reply = result["messages"][-1].content
        return ChatResponse(response=reply, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    status = {"status": "ok", "db": False, "models": False}

    try:
        from etl.utils import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        status["db"] = True
    except Exception:
        status["status"] = "degraded"

    upsell_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "upsell", "metadata.json")
    pricing_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "pricing", "metadata.json")
    if os.path.exists(upsell_dir) and os.path.exists(pricing_dir):
        status["models"] = True
    else:
        status["status"] = "degraded"

    return status
