# Kent Hologram Dashboard - Getting Started

A terminal-style web dashboard for visualizing real-time HDC (Hyperdimensional Computing) activity, neural consolidation, and conversation flow.

## Architecture

```
Frontend (Next.js 15.2.8)          Backend (FastAPI)              Core (Hologram)
┌─────────────────────────┐        ┌──────────────────┐          ┌────────────────────┐
│ Terminal UI             │  SSE   │ main.py          │ callback │ chatbot.py         │
│ (chat + activity)       │◄───────│ (SSE endpoint)   │◄─────────│ (activity hooks)   │
│                         │        │                  │          │                    │
│ - Default: ops+metrics  │        │ - /api/chat/stream│          │ - 3 event types    │
│ - Debug: raw events     │        │ - /api/login     │          │                    │
└─────────────────────────┘        └──────────────────┘          └────────────────────┘
```

## Files Overview

### Modified Files
| File | Changes |
|------|---------|
| `src/hologram/conversation/chatbot.py` | Added `set_activity_callback()` and `_emit_activity()` |
| `pyproject.toml` | Added `[web]` optional dependencies |

### New Files
| File | Purpose |
|------|---------|
| `web/backend/main.py` | FastAPI server with SSE streaming, auth, validation |
| `web/backend/__init__.py` | Package marker |
| `web/frontend/package.json` | Pinned React 19.2.1, Next.js 15.2.8 |
| `web/frontend/next.config.mjs` | Next.js configuration with API proxy |
| `web/frontend/tailwind.config.js` | Tailwind CSS config |
| `web/frontend/app/layout.js` | Root layout |
| `web/frontend/app/page.js` | Terminal UI with activity panel |
| `web/frontend/app/globals.css` | Dark terminal styling |
| `fly.toml` | Fly.io deployment config |
| `Dockerfile` | Multi-stage build (Node + Python) |
| `.dockerignore` | Docker build exclusions |

## Running Locally

### 1. Install Backend Dependencies

```bash
cd /path/to/kent_hologram
uv pip install -e ".[web]"
```

Or install manually:
```bash
pip install fastapi uvicorn[standard] python-multipart
```

### 2. Start the Backend

```bash
uvicorn web.backend.main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

Test it:
```bash
curl http://localhost:8000/health
# {"status":"healthy","timestamp":"2025-12-15T..."}
```

### 3. Install Frontend Dependencies

```bash
cd web/frontend
npm install
```

### 4. Start the Frontend

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/api/login` | POST | No | Get session token |
| `/api/chat/stream` | POST | Bearer | SSE stream for chat + activity |
| `/api/stats` | GET | Bearer | Get chatbot statistics |

## Activity Events

The dashboard streams 3 types of HDC activity events:

### 1. Intent Classification
```json
{"type": "intent", "intent": "question", "confidence": 0.92}
```
Emitted when the chatbot classifies user intent.

### 2. Fact Storage
```json
{"type": "fact", "subject": "Paris", "predicate": "capital", "object": "France"}
```
Emitted when a new fact is learned and stored.

### 3. Consolidation (via existing callback)
```json
{"type": "consolidation", "facts_count": 20, "loss": 0.023}
```
Emitted when neural memory consolidation completes.

## UI Layers

### Default View
Shows operations and metrics combined:
```
┌─────────────────────────────────────────────────────┐
│ [Memory ●] [Thinking ○]                   [Debug ▼] │
├─────────────────────────────────────────────────────┤
│ ► Intent: Question (92% confidence)                 │
│ ► Stored: Paris --capital--> France                 │
│ ► Neural: Consolidated 20 facts (loss=0.023)        │
└─────────────────────────────────────────────────────┘
```

### Debug View (Toggle)
Shows raw JSON event payloads:
```
│ [10:30:00] {"type":"intent","confidence":0.92,...}  │
│ [10:30:00] {"type":"fact","subject":"Paris",...}    │
```

## Security Features

- **Version Pinning**: React 19.2.1+, Next.js 15.2.8+ (mitigates CVE-2025-55182)
- **Token Auth**: Session tokens instead of URL-based IDs
- **IP Validation**: Sessions bound to client IP
- **Input Validation**: Length limits, XSS pattern blocking
- **XSS Prevention**: Uses `textContent` not `innerHTML`
- **CORS Restricted**: Only allows dashboard origins

## Deploying to Fly.io

### 1. Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Launch App
```bash
fly launch --name kent-hologram-dashboard
```

### 3. Deploy
```bash
fly deploy
```

### 4. Set Secrets (if needed)
```bash
fly secrets set SESSION_SECRET=$(openssl rand -hex 32)
```

## Development Notes

### Adding New Activity Events

1. Emit from chatbot:
```python
self._emit_activity("my_event", key1=value1, key2=value2)
```

2. Handle in frontend `formatActivity()`:
```javascript
case 'my_event':
  return `My Event: ${item.key1}`;
```

### Customizing Styles

Edit `web/frontend/app/globals.css` - uses CSS variables:
```css
:root {
  --bg-primary: #0d1117;
  --accent-green: #3fb950;
  /* ... */
}
```

## Troubleshooting

### Backend won't start
```bash
# Check if port is in use
lsof -i :8000

# Try different port
uvicorn web.backend.main:app --port 8001
```

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check `next.config.mjs` rewrites configuration
- Check browser console for CORS errors

### Activity events not showing
- Verify callback is set in chatbot
- Check browser Network tab for SSE events
- Enable Debug mode to see raw events

## Project Structure

```
kent_hologram/
├── src/hologram/
│   └── conversation/
│       └── chatbot.py          # Activity hooks added here
├── web/
│   ├── backend/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI server
│   └── frontend/
│       ├── package.json
│       ├── next.config.mjs
│       ├── tailwind.config.js
│       └── app/
│           ├── layout.js
│           ├── page.js         # Terminal UI
│           └── globals.css
├── fly.toml
├── Dockerfile
└── .dockerignore
```
