"""
FastAPI backend for Kent Hologram Dashboard.

Provides SSE streaming for chat + activity events.
Simplified: No WebSocket, no event emitter singleton.
"""

import json
import re
import secrets
import time
from datetime import datetime
from html import escape
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import os
from pathlib import Path

# Hologram imports
from hologram.container import HologramContainer
from hologram.chat.interface import ChatInterface

app = FastAPI(title="Kent Hologram Dashboard", version="1.0.0")

# CORS - restricted to dashboard origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kent-dashboard.fly.dev",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Session storage (in production, use Redis or similar)
sessions: dict = {}
SESSION_TTL = 28800  # 8 hours


# --- Input Validation ---

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 5000:
            raise ValueError("Message too long (max 5000 chars)")
        # Block obvious XSS attempts
        if re.search(r"<script|javascript:|on\w+=", v, re.IGNORECASE):
            raise ValueError("Invalid content")
        return v


# --- Authentication ---

async def get_current_user(
    authorization: Optional[str] = Header(None),
    request: Request = None,
) -> str:
    """Validate session token and return user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")

    token = authorization[7:]
    session = sessions.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    # Check expiry
    if time.time() - session["created"] > SESSION_TTL:
        del sessions[token]
        raise HTTPException(status_code=401, detail="Session expired")

    # IP validation (optional but recommended)
    if request and session["ip"] != request.client.host:
        raise HTTPException(status_code=401, detail="Session IP mismatch")

    return session["user_id"]


# --- Chatbot Instance ---

# Single chatbot instance per process
_chatbot = None


def get_chatbot():
    """Get or create chatbot instance."""
    global _chatbot
    if _chatbot is None:
        interface = ChatInterface(
            persist_dir="./data/cadence_test_facts",
            persistent=True,
            enable_ventriloquist=False,
            force_neural=True,   # Enable neural memory for advanced capabilities
            dimensions=10000,     # Full 10K dimensions for maximum capacity
            conversational=True
        )
        _chatbot = interface.chatbot
    return _chatbot


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/login")
async def login(request: Request):
    """
    Create session for dashboard user.

    In production, add proper authentication (OAuth, etc.)
    """
    token = secrets.token_urlsafe(32)
    user_id = f"user_{secrets.token_hex(4)}"

    sessions[token] = {
        "user_id": user_id,
        "ip": request.client.host,
        "created": time.time(),
    }

    # Clean expired sessions
    now = time.time()
    expired = [k for k, v in sessions.items() if now - v["created"] > SESSION_TTL]
    for k in expired:
        del sessions[k]

    return {"token": token, "user_id": user_id, "expires_in": SESSION_TTL}


@app.post("/api/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,
    current_user: str = Depends(get_current_user),
):
    """
    SSE streaming endpoint for chat + activity events.

    Streams events:
    - {"event": "activity", "data": {...}} - HDC activity
    - {"event": "thinking", "data": {...}} - Thinking status
    - {"event": "response", "data": {...}} - Final response
    - {"event": "error", "data": {...}} - Errors
    """
    message = chat_request.message

    async def event_generator():
        activity_events = []

        def activity_callback(event: dict):
            """Capture activity events from chatbot."""
            activity_events.append(event)

        try:
            chatbot = get_chatbot()

            # Set up activity callback
            chatbot.set_activity_callback(activity_callback)

            # Emit thinking start
            yield _format_sse("thinking", {"status": "processing", "message": message[:50]})

            # Get response (synchronous for now)
            response = chatbot.respond(message)

            # Clear callback
            chatbot.set_activity_callback(None)

            # Emit collected activity events
            for event in activity_events:
                # Escape content for XSS prevention
                safe_event = {k: escape(str(v)) if isinstance(v, str) else v for k, v in event.items()}
                yield _format_sse("activity", safe_event)

            # Emit final response
            yield _format_sse("response", {
                "content": escape(response),
                "timestamp": datetime.utcnow().isoformat(),
            })

        except Exception as e:
            # Don't expose internal errors
            yield _format_sse("error", {"message": "Service error"})
            # Log internally
            print(f"Chat error for {current_user}: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stats")
async def get_stats(current_user: str = Depends(get_current_user)):
    """Get chatbot statistics."""
    try:
        chatbot = get_chatbot()
        stats = chatbot.get_session_stats()
        return {"user_id": current_user, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Service error")


def _format_sse(event: str, data: dict) -> str:
    """Format Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- Static File Serving for Next.js Frontend ---

# Mount Next.js static assets (_next directory)
next_static_dir = Path("/app/frontend/_next")
if next_static_dir.exists():
    app.mount("/_next", StaticFiles(directory=str(next_static_dir)), name="next_static")

# Serve the Next.js frontend index page
@app.get("/")
async def serve_frontend():
    """Serve the Next.js frontend index page."""
    index_path = Path("/app/frontend/index.html")
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "Frontend not found. API endpoints available at /api/"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
