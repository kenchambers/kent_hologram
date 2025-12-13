# Install EmbeddixDB Vector Search

You are helping the user install and configure EmbeddixDB vector database with Novita AI embeddings for semantic code search.

## Your Task

Guide the user through the complete EmbeddixDB installation process by performing the following steps in order.

**IMPORTANT LESSONS LEARNED**:
- Use the REST API server (`embeddix-api`), NOT the MCP server
- Config file MUST have full database file path, not just directory
- Must explicitly disable embedding engine in config
- Build BOTH servers but use `embeddix-api` for running

---

### Step 1: Setup .gitignore (CRITICAL - Do this FIRST!)

**IMPORTANT**: Add embeddix directories to .gitignore BEFORE creating them to ensure repo-agnostic installation.

Check if `.gitignore` exists and contains embeddix entries:

```bash
# Check current gitignore
grep -E "embeddixdb/|embeddix/" .gitignore
```

If not present, add these entries to `.gitignore`:

```
# EmbeddixDB - Complete installation (repo-agnostic setup)
embeddixdb/
embeddix/
.embeddixdb/
*.embeddixdb.yml
```

**Why this matters**: EmbeddixDB should be installed per-repository, not tracked in version control. Each developer/environment should run `/install-embeddix` to set up their own local instance.

---

### Step 2: Check Prerequisites

First, verify the system is ready:

1. **Check for Novita API key**:
   - Look for `NOVITA_KEY` or `NOVITA_API_KEY` in the `.env` file
   - If not found, ask the user to obtain a key from https://novita.ai
   - Offer to add it to `.env` once they provide it

2. **Check for Go installation**:
   - Run `go version` to verify Go is installed (version 1.21+)
   - If not installed, guide the user to install it from https://go.dev/dl/

3. **Check directory structure**:
   - Verify we're in a git repository
   - Check if `embeddixdb/` directory already exists (skip clone if it does)

---

### Step 3: Clone and Build EmbeddixDB

If embeddixdb doesn't exist:

```bash
# Clone the repository
git clone https://github.com/dshills/EmbeddixDB.git embeddixdb
```

**Build BOTH servers** (we'll use embeddix-api):

```bash
cd embeddixdb

# Build the API server (this is what we'll use)
go build -o build/embeddix-api ./cmd/embeddix-api

# Also build MCP server (for Claude Desktop integration if needed)
make build-mcp

cd ..
```

Verify both binaries were created:
```bash
ls -la embeddixdb/build/embeddix-api
ls -la embeddixdb/build/embeddix-mcp
```

**Why embeddix-api?**
- The MCP server (`embeddix-mcp`) tries to initialize local embedding models by default
- We handle embeddings externally via Novita's API
- The REST API server (`embeddix-api`) is simpler and doesn't require embedding configuration

---

### Step 4: Create Embeddix Directory Structure

Create the organized directory structure:

```bash
mkdir -p embeddix/{config,scripts,src,tests,docs}
```

---

### Step 5: Create Configuration Files

Create `embeddix/config/.embeddixdb.yml`:

**CRITICAL**: Use full file path, not just directory!

```yaml
server:
  port: 8080
  host: "localhost"

persistence:
  type: "bolt"
  path: "./embeddixdb/data/embeddix.db"  # MUST be full file path!

# Explicitly disable AI embeddings - handled externally via Novita API
# The Python client (novita_integration.py) generates embeddings
# and passes them directly to the vector store
ai:
  embedding:
    engine: ""  # Empty string = no engine, no embedding initialization
```

**Common mistakes to avoid**:
- ❌ `path: "./embeddixdb/data"` (directory only) - causes "is a directory" error
- ✅ `path: "./embeddixdb/data/embeddix.db"` (full file path) - correct!
- ❌ Omitting `ai.embedding.engine: ""` - server tries to load models
- ✅ Explicitly setting `engine: ""` - disables embedding initialization

---

### Step 6: Create Python Integration Files

1. **Create `embeddix/requirements.txt`**:
```
requests>=2.31.0
python-dotenv>=1.0.0
```

2. **Set up Python package structure**:

**CRITICAL**: Create `__init__.py` files to make embeddix a proper Python package:

```bash
# Create package __init__.py files
cat > embeddix/__init__.py << 'EOF'
"""
EmbeddixDB Integration Package for kent_hologram

This package provides vector database integration using EmbeddixDB
with Novita AI embeddings for semantic code search and document retrieval.
"""

__version__ = "0.1.0"
EOF

cat > embeddix/src/__init__.py << 'EOF'
"""
EmbeddixDB Source Module

Core integration classes for vector database operations with Novita embeddings.
"""

from .novita_integration import (
    TradingDocumentManager,
    NovitaEmbedding,
    EmbeddixDBClient
)

__all__ = [
    'TradingDocumentManager',
    'NovitaEmbedding',
    'EmbeddixDBClient'
]
EOF
```

3. **Create `embeddix/src/novita_integration.py`**:
   - Copy from llm_prophet repository or fetch from known source
   - Should include: `NovitaEmbedding`, `EmbeddixDBClient`, `TradingDocumentManager` classes
   - **Note**: The `TradingDocumentManager` class was written for MCP protocol
   - For REST API, you may need to update endpoint calls or use direct REST calls

4. **Create `embeddix/src/index_codebase.py`**:
   - Script to index the codebase for semantic search
   - Should support command-line arguments for collection name and root directory

4. **Create `search_code.py` in project root**:
   - User-facing script for semantic code search
   - Returns **absolute paths** for compatibility with Claude Code Read tool
   - Example implementation:

```python
#!/usr/bin/env python3
"""
Semantic code search using EmbeddixDB
Usage: python search_code.py "your search query"
"""

import sys
import os
from pathlib import Path
from embeddix.src.novita_integration import TradingDocumentManager

def search_code(query: str, limit: int = 5):
    """Search the indexed codebase"""

    # Get project root for absolute paths
    project_root = Path(__file__).parent.absolute()

    # Initialize manager
    manager = TradingDocumentManager(
        novita_api_key=os.getenv("NOVITA_API_KEY"),
        novita_model="qwen/qwen3-embedding-0.6b",
        collection_name="kent_hologram_code"
    )

    # Search
    print(f"\n🔍 Searching for: '{query}'\n")
    print("=" * 80)

    results = manager.search(query, limit=limit)

    if not results:
        print("No results found.")
        return

    # Handle both list and dict responses
    if isinstance(results, dict):
        results = results.get('results', [])

    for i, result in enumerate(results, 1):
        score = result.get('score', 0)
        metadata = result.get('metadata', {})

        file_path = metadata.get('file_path', 'Unknown')
        file_type = metadata.get('file_type', 'Unknown')
        block_type = metadata.get('block_type', '')
        name = metadata.get('name', '')

        # Convert to absolute path (IMPORTANT: Claude Code needs absolute paths)
        abs_path = project_root / file_path

        print(f"\n[Result {i}] Score: {score:.4f}")
        print(f"File: {abs_path}")
        if block_type and name:
            print(f"Type: {block_type} '{name}'")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"Found {len(results)} results")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_code.py 'your search query'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_code(query)
```

Make it executable:
```bash
chmod +x search_code.py
```

**Important**: The script returns **absolute paths** to ensure compatibility with Claude Code's Read tool. This prevents "EISDIR: illegal operation on a directory" errors.

---

### Step 7: Create Management Scripts

Create these scripts in `embeddix/scripts/`:

**IMPORTANT**: Scripts must use `embeddix-api`, not `embeddix-mcp`!

#### 1. **start_bg.sh** - Start server in background

```bash
#!/bin/bash

# Get project root directory (two levels up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Set PYTHONPATH for embeddix Python modules
export PYTHONPATH="$PROJECT_ROOT/embeddix/src:$PYTHONPATH"

# Check if server is already running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ EmbeddixDB server already running"
    return 0 2>/dev/null || exit 0
fi

echo ""
echo "🔮 ════════════════════════════════════════════════════════════"
echo "   🚀 Starting EmbeddixDB Vector Search Server"
echo "   ════════════════════════════════════════════════════════════"
echo "   📁 Data: $PROJECT_ROOT/embeddixdb/data/embeddix.db"
echo "   💾 Storage: BoltDB (persistent)"
echo "   🌐 Port: 8080"
echo "   🤖 Model: Qwen 3 Embedding (1024-dim)"
echo "   ════════════════════════════════════════════════════════════"
echo ""

# Create data directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/embeddixdb/data"

# Start the API server in background (NOT embeddix-mcp!)
nohup "$PROJECT_ROOT/embeddixdb/build/embeddix-api" \
    -config "$SCRIPT_DIR/../config/.embeddixdb.yml" \
    -db bolt \
    -path "$PROJECT_ROOT/embeddixdb/data/embeddix.db" \
    -host localhost \
    -port 8080 \
    > "$PROJECT_ROOT/embeddixdb/server.log" 2>&1 &

SERVER_PID=$!

# Wait a moment and check if server started successfully
sleep 1

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "📝 Logs: $PROJECT_ROOT/embeddixdb/server.log"
    echo ""
else
    echo "❌ Server failed to start. Check $PROJECT_ROOT/embeddixdb/server.log for details"
    return 1 2>/dev/null || exit 1
fi
```

#### 2. **start.sh** - Start server in foreground

```bash
#!/bin/bash

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if server is already running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ EmbeddixDB server already running on port 8080"
    echo "   Use $SCRIPT_DIR/stop.sh to stop it first"
    exit 0
fi

echo ""
echo "🔮 ════════════════════════════════════════════════════════════"
echo "   🚀 Starting EmbeddixDB Vector Search Server"
echo "   ════════════════════════════════════════════════════════════"
echo "   📁 Data: $PROJECT_ROOT/embeddixdb/data/embeddix.db"
echo "   💾 Storage: BoltDB (persistent)"
echo "   🌐 Port: 8080"
echo "   ════════════════════════════════════════════════════════════"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Create data directory
mkdir -p "$PROJECT_ROOT/embeddixdb/data"

# Start the API server in foreground
"$PROJECT_ROOT/embeddixdb/build/embeddix-api" \
    -config "$SCRIPT_DIR/../config/.embeddixdb.yml" \
    -db bolt \
    -path "$PROJECT_ROOT/embeddixdb/data/embeddix.db" \
    -host localhost \
    -port 8080

echo ""
echo "🛑 Server stopped."
```

#### 3. **stop.sh** - Stop the server

```bash
#!/bin/bash

echo ""
echo "🔍 Checking for EmbeddixDB server..."

# Find the process
PID=$(lsof -ti:8080)

if [ -z "$PID" ]; then
    echo "ℹ️  No EmbeddixDB server running on port 8080"
    exit 0
fi

echo "🛑 Stopping EmbeddixDB server (PID: $PID)..."
kill $PID

# Wait a moment and verify
sleep 1

if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Server didn't stop gracefully, force killing..."
    kill -9 $PID
    sleep 1
fi

if ! lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ Server stopped successfully"
else
    echo "❌ Failed to stop server"
    exit 1
fi
```

#### 4. **status.sh** - Check server status

```bash
#!/bin/bash

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo ""
echo "🔍 EmbeddixDB Server Status"
echo "════════════════════════════════════════════════════════════"

# Check if server is running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    PID=$(lsof -ti:8080)
    echo "✅ Status: RUNNING"
    echo "🆔 PID: $PID"
    echo "🌐 Port: 8080"
    echo "📝 Logs: $PROJECT_ROOT/embeddixdb/server.log"
    echo ""
    echo "Recent log entries:"
    echo "────────────────────────────────────────────────────────────"
    if [ -f "$PROJECT_ROOT/embeddixdb/server.log" ]; then
        tail -5 "$PROJECT_ROOT/embeddixdb/server.log" | sed 's/^/   /'
    else
        echo "   (No log file found)"
    fi
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    echo "To start the server:"
    echo "   $SCRIPT_DIR/start.sh         # Foreground"
    echo "   $SCRIPT_DIR/start_bg.sh      # Background"
    echo "   cd <away and back>           # Auto-start"
fi

echo "════════════════════════════════════════════════════════════"
echo ""
```

**Make all scripts executable**:
```bash
chmod +x embeddix/scripts/*.sh
```

---

### Step 8: Create Test Files

Create these test files in `embeddix/tests/`:

#### 1. **test_novita_embedding.py** - Test Novita API connection

Make sure imports work correctly:
```python
#!/usr/bin/env python3
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from novita_integration import NovitaEmbedding
# ... rest of test code
```

#### 2. **test_embeddix_integration.py** - Test full integration

**Note**: The full integration test may need updating to work with REST API endpoints instead of MCP protocol.

For now, create a simple REST API test that:
1. Creates a collection
2. Adds vectors with Novita embeddings
3. Performs a search query

---

### Step 9: Create Documentation

Create these docs in `embeddix/docs/`:

1. **SETUP.md** - Complete setup guide
2. **QUICKSTART.md** - Quick start guide
3. **AUTO_START.md** - Auto-start configuration

Include notes about:
- Using `embeddix-api` server
- Config file requirements (full path!)
- Novita API key setup

---

### Step 10: Create Claude Code Agents

Create these agent definitions in `.claude/agents/`:

1. **embeddixdb-expert.md** - Expert in EmbeddixDB operations
2. **code-search.md** - Semantic code search agent

Copy from llm_prophet repository or create fresh versions.

---

### Step 11: Setup MCP Server for Claude Code Integration

**NEW**: Create a Model Context Protocol (MCP) server to expose semantic search as a clean tool for all Claude Code subagents.

#### 11.1: Create MCP Server

Create `embeddix/src/mcp_server.py`:

**Important**: This MCP server provides clean tool access to semantic search for ALL subagents (cursor-code, gemini-code, etc.).

```python
#!/usr/bin/env python3
"""
EmbeddixDB MCP Server
Model Context Protocol server for semantic code search using EmbeddixDB + Novita embeddings.

This server exposes semantic search capabilities as MCP tools that can be used by
Claude Code and other MCP-compatible AI assistants.

Usage:
    python mcp_server.py

Environment Variables:
    NOVITA_API_KEY - API key for Novita AI embeddings
    PYTHONPATH - Should include embeddix/src directory
"""

import sys
import json
import logging
from typing import Any, Dict, List, Optional
from novita_integration import TradingDocumentManager, EmbeddixDBClient

# Configure logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MCP Server] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class EmbeddixMCPServer:
    """MCP server for EmbeddixDB semantic code search"""

    def __init__(self):
        self.manager = None
        self.client = None
        self.default_collection = "kent_hologram_code"
        logger.info("EmbeddixDB MCP Server initializing...")

    def _get_manager(self, collection_name: Optional[str] = None) -> TradingDocumentManager:
        """Lazy load document manager"""
        collection = collection_name or self.default_collection
        if self.manager is None or self.manager.collection_name != collection:
            logger.info(f"Initializing manager for collection: {collection}")
            self.manager = TradingDocumentManager(collection_name=collection)
        return self.manager

    def _get_client(self) -> EmbeddixDBClient:
        """Lazy load EmbeddixDB client"""
        if self.client is None:
            logger.info("Initializing EmbeddixDB client")
            self.client = EmbeddixDBClient()
        return self.client

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of available MCP tools"""
        return [
            {
                "name": "search_code",
                "description": "Semantic code search using EmbeddixDB vector database. Finds code by natural language description, understanding meaning rather than just keywords.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query describing what code you're looking for. Examples: 'agent creation', 'database connection logic', 'error handling patterns'"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5, max: 20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name to search (default: kent_hologram_code)",
                            "default": "kent_hologram_code"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters to narrow results. Common filters: file_type (code/documentation), extension (.py, .md), file_path (partial path), block_type (function/class)",
                            "properties": {
                                "file_type": {"type": "string", "enum": ["code", "documentation"]},
                                "extension": {"type": "string", "description": "File extension filter (e.g., '.py', '.md')"},
                                "file_path": {"type": "string", "description": "Partial path match (e.g., 'agents/' to search only in agents directory)"},
                                "block_type": {"type": "string", "enum": ["function", "class", "module"]}
                            }
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_collection_stats",
                "description": "Get statistics about an indexed code collection including total vectors, metadata info, and collection configuration.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name (default: kent_hologram_code)",
                            "default": "kent_hologram_code"
                        }
                    }
                }
            },
            {
                "name": "list_collections",
                "description": "List all available EmbeddixDB collections",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool execution"""
        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            if name == "search_code":
                return self._search_code(arguments)
            elif name == "get_collection_stats":
                return self._get_collection_stats(arguments)
            elif name == "list_collections":
                return self._list_collections(arguments)
            else:
                return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}
        except Exception as e:
            logger.error(f"Error executing {name}: {e}", exc_info=True)
            return {"isError": True, "content": [{"type": "text", "text": f"Error: {str(e)}"}]}

    def _search_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic code search"""
        query = args.get("query")
        limit = args.get("limit", 5)
        collection = args.get("collection", self.default_collection)
        filters = args.get("filters", {})

        if not query:
            return {"isError": True, "content": [{"type": "text", "text": "Query parameter is required"}]}

        manager = self._get_manager(collection)
        results = manager.search(query=query, limit=min(limit, 20), filters=filters)

        if not results:
            return {"content": [{"type": "text", "text": f"No results found for query: '{query}'\n\nTry:\n- Broadening your search terms\n- Using different keywords\n- Checking if the collection is indexed"}]}

        response_text = f"🔍 Found {len(results)} results for: '{query}'\n\n"
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            file_path = metadata.get("file_path", "Unknown")
            block_type = metadata.get("block_type", "")
            name = metadata.get("name", "")

            response_text += f"{i}. {file_path}"
            if name:
                response_text += f" ({block_type}: {name})"
            response_text += f"\n   Relevance: {score:.3f}\n"

            content = metadata.get("content", "")
            if content:
                preview = content[:200].replace("\n", " ")
                if len(content) > 200:
                    preview += "..."
                response_text += f"   Preview: {preview}\n"
            response_text += "\n"

        response_text += f"\n💡 Tip: Use the Read tool with these file paths to see full code context."
        return {"content": [{"type": "text", "text": response_text}]}

    def _get_collection_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get collection statistics"""
        collection = args.get("collection", self.default_collection)
        response_text = f"Collection: {collection}\n\nℹ️ Collection statistics:\n- Status: Active\n- Type: Code search with semantic embeddings\n- Embedding model: Novita qwen/qwen3-embedding-0.6b (1024 dims)\n"
        return {"content": [{"type": "text", "text": response_text}]}

    def _list_collections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all collections"""
        try:
            client = self._get_client()
            collections = client.list_collections()
            if not collections:
                return {"content": [{"type": "text", "text": "No collections found.\n\nRun the indexing script to create a collection:\n  python embeddix/src/index_codebase.py --collection kent_hologram_code"}]}
            response_text = f"📚 Available Collections ({len(collections)}):\n\n"
            for coll in collections:
                response_text += f"- {coll}\n"
            return {"content": [{"type": "text", "text": response_text}]}
        except Exception as e:
            return {"isError": True, "content": [{"type": "text", "text": f"Error listing collections: {str(e)}"}]}

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming MCP protocol message"""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": "0.1.0", "capabilities": {"tools": {}}, "serverInfo": {"name": "embeddixdb-mcp-server", "version": "1.0.0"}}}
        elif method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self.list_tools()}}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.handle_tool_call(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        elif method == "notifications/initialized":
            logger.info("Client initialized successfully")
            return None
        else:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def run(self):
        """Main server loop - reads from stdin, writes to stdout"""
        logger.info("EmbeddixDB MCP Server started - waiting for messages...")
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                    response = self.handle_message(message)
                    if response:
                        print(json.dumps(response), flush=True)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")


def main():
    server = EmbeddixMCPServer()
    server.run()


if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x embeddix/src/mcp_server.py
```

#### 11.2: Configure .mcp.json

Create or update `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "embeddixdb-search": {
      "command": "python3",
      "args": [
        "embeddix/src/mcp_server.py"
      ],
      "env": {
        "NOVITA_API_KEY": "${NOVITA_API_KEY}",
        "PYTHONPATH": "embeddix/src"
      }
    }
  }
}
```

**Note**: If `.mcp.json` already exists with other servers, add the `embeddixdb-search` entry to the existing `mcpServers` object.

#### 11.3: Update Agent Files to Use MCP Tools

Update `.claude/agents/cursor-code.md` and `.claude/agents/gemini-code.md` to use the MCP `search_code` tool instead of Bash commands.

**Key changes**:
- Replace: `NOVITA_API_KEY=... python search_code.py "query"`
- With: `search_code(query="query")`

Example sections to update:

**In cursor-code.md** (around line 20):
```markdown
## Using the MCP Tool

You now have access to the `search_code` MCP tool for semantic code search:

\```
Use the search_code tool with:
- query: "your semantic search query"
- limit: 5 (or adjust as needed)
- filters: {} (optional metadata filters)
\```
```

**Benefits of MCP approach**:
- ✅ Clean interface (no ugly Bash commands)
- ✅ ALL subagents get access automatically
- ✅ Better error handling
- ✅ Consistent across all agents

#### 11.4: Create MCP Integration Test

Create `embeddix/tests/test_mcp_integration.py`:

```python
#!/usr/bin/env python3
"""
Test script for EmbeddixDB MCP server integration
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from novita_integration import TradingDocumentManager, EmbeddixDBClient

def test_embeddixdb_connection():
    """Test connection to EmbeddixDB REST API"""
    print("Testing EmbeddixDB connection...")
    try:
        client = EmbeddixDBClient()
        collections = client.list_collections()
        print(f"✅ Connected to EmbeddixDB")
        print(f"   Collections: {collections}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

def test_mcp_tool_interface():
    """Test the MCP tool interface structure"""
    print("\nTesting MCP tool interface...")
    try:
        from mcp_server import EmbeddixMCPServer
        server = EmbeddixMCPServer()
        tools = server.list_tools()
        print(f"✅ MCP server initialized")
        print(f"   Available tools: {len(tools)}")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description'][:50]}...")
        return True
    except Exception as e:
        print(f"❌ MCP interface test failed: {e}")
        return False

if __name__ == "__main__":
    test_embeddixdb_connection()
    test_mcp_tool_interface()
```

Make it executable and test:
```bash
chmod +x embeddix/tests/test_mcp_integration.py
PYTHONPATH=embeddix/src python3 embeddix/tests/test_mcp_integration.py
```

#### 11.5: Restart Claude Code

After setup, restart Claude Code to load the new MCP server:
- The MCP server will start automatically when needed
- Test it by using cursor-code or gemini-code agents
- They can now use `search_code` tool cleanly!

---

### Step 12: Setup Auto-Start and PYTHONPATH (Optional but Recommended)

Ask the user if they want to configure auto-start for zsh:

If yes:
1. Detect the project directory name
2. Create zsh hook function that:
   - Detects when entering the project directory
   - **Exports PYTHONPATH for embeddix modules** (CRITICAL for code-search agent!)
   - Runs `embeddix/scripts/start_bg.sh` if not already running
   - Avoids re-running on subdirectory changes
3. Add to `~/.zshrc` with proper markers
4. Backup `~/.zshrc` first

**IMPORTANT**: The hook MUST export PYTHONPATH so that the code-search agent can import embeddix modules!

Example hook:
```bash
# 🔮 EmbeddixDB Auto-Start for <project_name>
autostart_embeddix_<project_name>() {
    if [[ "$PWD" == */<project_name>* ]]; then
        # Find project root
        local project_root="<path_to_project>"

        # CRITICAL: Export PYTHONPATH for embeddix modules
        export PYTHONPATH="$project_root/embeddix/src:$PYTHONPATH"

        # Start server if not running
        # (implementation details)
    fi
}
add-zsh-hook chpwd autostart_embeddix_<project_name>
```

You can use the `.zshrc_embeddix_update.sh` script in the project root to automate this:
```bash
# Run the update script
bash .zshrc_embeddix_update.sh

# Then manually update ~/.zshrc with the content from /tmp/embeddix_function.txt
# Or run the sed command shown by the script
```

---

### Step 13: Install Dependencies

```bash
# Using pip
pip install -r embeddix/requirements.txt

# Or using uv (if available)
uv pip install -r embeddix/requirements.txt
```

---

### Step 14: Test the Installation

Run the test scripts:

```bash
# Test Novita API
cd embeddix/tests
python test_novita_embedding.py

# Start the server
cd ../..
./embeddix/scripts/start_bg.sh

# Wait for server to start
sleep 2

# Check server is running
./embeddix/scripts/status.sh

# Test basic API endpoint
curl http://localhost:8080/health
curl http://localhost:8080/collections
```

**Expected output**:
```
✅ Status: RUNNING
🆔 PID: <some_pid>
🌐 Port: 8080
```

---

### Step 15: Verify API is Working

Test the REST API endpoints:

```bash
# Check health
curl http://localhost:8080/health

# List collections (should return empty array initially)
curl http://localhost:8080/collections

# Check API documentation
curl http://localhost:8080/docs
```

---

### Step 16: Setup Automatic Git Hooks for Vector Store Updates

**IMPORTANT**: This step automatically keeps your vector database fresh by indexing files whenever they change through git operations.

#### 16.1: Create Incremental Indexing Script

Create `embeddix/src/incremental_index.py`:

```python
#!/usr/bin/env python3
"""
Incremental indexing script for EmbeddixDB
Indexes only specific files that have changed
"""

import os
import sys
from pathlib import Path
from typing import List
from index_codebase import CodebaseIndexer


def index_files(files: List[str], collection_name: str = "kent_hologram_code"):
    """
    Index specific files into EmbeddixDB

    Args:
        files: List of file paths to index
        collection_name: Collection name in EmbeddixDB
    """
    if not files:
        print("No files to index")
        return

    # Get repository root
    repo_root = Path.cwd()

    # Get API key
    api_key = os.getenv("NOVITA_API_KEY") or os.getenv("NOVITA_KEY")
    if not api_key:
        print("Error: NOVITA_API_KEY not found in environment")
        return

    # Create indexer
    indexer = CodebaseIndexer(
        root_path=repo_root,
        collection_name=collection_name,
        novita_api_key=api_key
    )

    # Filter files to only those that should be indexed
    files_to_index = []
    for file_str in files:
        file_path = repo_root / file_str
        if file_path.exists() and file_path.is_file() and indexer.should_index_file(file_path):
            files_to_index.append(file_path)

    if not files_to_index:
        print("No indexable files found")
        return

    print(f"Incrementally indexing {len(files_to_index)} files...")

    # Index files
    total_chunks = 0
    for file_path in files_to_index:
        doc_ids = indexer.index_file(file_path)
        total_chunks += len(doc_ids)

    print(f"✅ Incremental indexing complete: {len(files_to_index)} files, {total_chunks} chunks")


def main():
    """Main entry point for incremental indexing"""
    import argparse

    parser = argparse.ArgumentParser(description="Incremental indexing for EmbeddixDB")
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to index"
    )
    parser.add_argument(
        "--collection",
        default="kent_hologram_code",
        help="Collection name"
    )

    args = parser.parse_args()

    index_files(args.files, args.collection)


if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x embeddix/src/incremental_index.py
```

#### 16.2: Create Git Hooks Directory

Ensure the git hooks directory exists:
```bash
mkdir -p .git/hooks
```

#### 16.3: Create Post-Commit Hook

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
#
# Post-commit hook for automatic EmbeddixDB vector store updates
# This hook runs after each commit and incrementally updates the vector database
#

# Configuration
REPO_ROOT="$(git rev-parse --show-toplevel)"
COLLECTION_NAME=$(basename "$REPO_ROOT")_code
LOG_FILE="$REPO_ROOT/.embeddixdb/hook-logs/post-commit.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Post-commit hook triggered ==="

# Check if EmbeddixDB server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log "⚠️  EmbeddixDB server not running - skipping indexing"
    exit 0
fi

# Get list of changed files in this commit
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

if [ -z "$CHANGED_FILES" ]; then
    log "No files changed in this commit"
    exit 0
fi

# Count changed files
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
log "Found $FILE_COUNT changed file(s) in commit $(git rev-parse --short HEAD)"

# Source .env file if it exists (for NOVITA_API_KEY)
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | grep 'NOVITA' | xargs)
fi

# Check if API key is available
if [ -z "$NOVITA_KEY" ] && [ -z "$NOVITA_API_KEY" ]; then
    log "⚠️  NOVITA_API_KEY not found - skipping indexing"
    exit 0
fi

# Run incremental indexing in background
log "Starting incremental indexing in background..."

# Navigate to repo root
cd "$REPO_ROOT" || exit 1

# Run indexing in background, redirecting output to log
(
    log "Indexing files: $(echo "$CHANGED_FILES" | tr '\n' ' ')"

    # Export environment variables
    export NOVITA_API_KEY="${NOVITA_API_KEY:-$NOVITA_KEY}"
    export PYTHONPATH="$REPO_ROOT/embeddix/src:$PYTHONPATH"

    # Run incremental indexing
    echo "$CHANGED_FILES" | xargs python "$REPO_ROOT/embeddix/src/incremental_index.py" \
        --collection "$COLLECTION_NAME" >> "$LOG_FILE" 2>&1

    log "Incremental indexing complete"
) &

log "Incremental indexing started in background (PID: $!)"
log "=== Post-commit hook finished ==="

exit 0
```

#### 16.4: Create Post-Merge Hook

Create `.git/hooks/post-merge`:

```bash
#!/bin/bash
#
# Post-merge hook for automatic EmbeddixDB vector store updates
# This hook runs after git pull/merge and incrementally updates the vector database
#

# Configuration
REPO_ROOT="$(git rev-parse --show-toplevel)"
COLLECTION_NAME=$(basename "$REPO_ROOT")_code
LOG_FILE="$REPO_ROOT/.embeddixdb/hook-logs/post-merge.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Post-merge hook triggered ==="

# Check if EmbeddixDB server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log "⚠️  EmbeddixDB server not running - skipping indexing"
    exit 0
fi

# Get list of files that changed in the merge
# Compare current HEAD with the previous HEAD (ORIG_HEAD)
CHANGED_FILES=$(git diff --name-only ORIG_HEAD HEAD 2>/dev/null)

if [ -z "$CHANGED_FILES" ]; then
    log "No files changed in this merge"
    exit 0
fi

# Count changed files
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
log "Found $FILE_COUNT changed file(s) from merge"

# Source .env file if it exists (for NOVITA_API_KEY)
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | grep 'NOVITA' | xargs)
fi

# Check if API key is available
if [ -z "$NOVITA_KEY" ] && [ -z "$NOVITA_API_KEY" ]; then
    log "⚠️  NOVITA_API_KEY not found - skipping indexing"
    exit 0
fi

# Run incremental indexing in background
log "Starting incremental indexing in background..."

# Navigate to repo root
cd "$REPO_ROOT" || exit 1

# Run indexing in background, redirecting output to log
(
    log "Indexing files: $(echo "$CHANGED_FILES" | tr '\n' ' ')"

    # Export environment variables
    export NOVITA_API_KEY="${NOVITA_API_KEY:-$NOVITA_KEY}"
    export PYTHONPATH="$REPO_ROOT/embeddix/src:$PYTHONPATH"

    # Run incremental indexing
    echo "$CHANGED_FILES" | xargs python "$REPO_ROOT/embeddix/src/incremental_index.py" \
        --collection "$COLLECTION_NAME" >> "$LOG_FILE" 2>&1

    log "Incremental indexing complete"
) &

log "Incremental indexing started in background (PID: $!)"
log "=== Post-merge hook finished ==="

exit 0
```

#### 16.5: Create Post-Checkout Hook

Create `.git/hooks/post-checkout`:

```bash
#!/bin/bash
#
# Post-checkout hook for automatic EmbeddixDB vector store updates
# This hook runs after checking out a branch and incrementally updates the vector database
#

# Arguments: previous HEAD, new HEAD, branch checkout flag (1=branch, 0=file)
PREV_HEAD=$1
NEW_HEAD=$2
IS_BRANCH_CHECKOUT=$3

# Only run for branch checkouts, not file checkouts
if [ "$IS_BRANCH_CHECKOUT" != "1" ]; then
    exit 0
fi

# Configuration
REPO_ROOT="$(git rev-parse --show-toplevel)"
COLLECTION_NAME=$(basename "$REPO_ROOT")_code
LOG_FILE="$REPO_ROOT/.embeddixdb/hook-logs/post-checkout.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Post-checkout hook triggered ==="
log "Switched from $PREV_HEAD to $NEW_HEAD"

# Check if EmbeddixDB server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log "⚠️  EmbeddixDB server not running - skipping indexing"
    exit 0
fi

# Get list of files that changed between branches
CHANGED_FILES=$(git diff --name-only $PREV_HEAD $NEW_HEAD 2>/dev/null)

if [ -z "$CHANGED_FILES" ]; then
    log "No files changed between branches"
    exit 0
fi

# Count changed files
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
log "Found $FILE_COUNT changed file(s) between branches"

# Source .env file if it exists (for NOVITA_API_KEY)
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | grep 'NOVITA' | xargs)
fi

# Check if API key is available
if [ -z "$NOVITA_KEY" ] && [ -z "$NOVITA_API_KEY" ]; then
    log "⚠️  NOVITA_API_KEY not found - skipping indexing"
    exit 0
fi

# Run incremental indexing in background
log "Starting incremental indexing in background..."

# Navigate to repo root
cd "$REPO_ROOT" || exit 1

# Run indexing in background, redirecting output to log
(
    log "Indexing files: $(echo "$CHANGED_FILES" | tr '\n' ' ')"

    # Export environment variables
    export NOVITA_API_KEY="${NOVITA_API_KEY:-$NOVITA_KEY}"
    export PYTHONPATH="$REPO_ROOT/embeddix/src:$PYTHONPATH"

    # Run incremental indexing
    echo "$CHANGED_FILES" | xargs python "$REPO_ROOT/embeddix/src/incremental_index.py" \
        --collection "$COLLECTION_NAME" >> "$LOG_FILE" 2>&1

    log "Incremental indexing complete"
) &

log "Incremental indexing started in background (PID: $!)"
log "=== Post-checkout hook finished ==="

exit 0
```

#### 16.6: Make Hooks Executable

```bash
chmod +x .git/hooks/post-commit
chmod +x .git/hooks/post-merge
chmod +x .git/hooks/post-checkout
```

#### 16.7: Test the Git Hooks

Test that the post-commit hook works:

```bash
# Create a test file
echo "Test file for git hook verification" > test_hook.txt

# Stage and commit it
git add test_hook.txt
git commit -m "Test git hook auto-indexing"

# Check the hook log
cat .embeddixdb/hook-logs/post-commit.log
```

**Expected output in log**:
```
[2025-XX-XX XX:XX:XX] === Post-commit hook triggered ===
[2025-XX-XX XX:XX:XX] Found 1 changed file(s) in commit XXXXXXX
[2025-XX-XX XX:XX:XX] Starting incremental indexing in background...
[2025-XX-XX XX:XX:XX] Incremental indexing started in background (PID: XXXXX)
[2025-XX-XX XX:XX:XX] === Post-commit hook finished ===
```

#### 16.8: Verify Hook Functionality

Verify hooks are working properly:

```bash
# Check all hooks exist and are executable
ls -la .git/hooks/post-* | grep -v sample

# View recent hook activity
tail -20 .embeddixdb/hook-logs/post-commit.log
```

**What the hooks do**:
- **post-commit**: Triggers after every `git commit`, indexes changed files
- **post-merge**: Triggers after `git pull` or `git merge`, indexes merged files
- **post-checkout**: Triggers when switching branches, indexes differing files

**Benefits**:
- ✅ Vector store automatically stays current with code changes
- ✅ No manual re-indexing needed
- ✅ Runs in background, doesn't slow down git operations
- ✅ Works repository-agnostic (adapts to any repo name)
- ✅ Logs all activity for troubleshooting

---

## Completion

After all steps:

1. Show the user how to use the system:
   - Starting/stopping the server
   - Using the REST API
   - Using the Claude Code agents
   - How auto-indexing works (git hooks automatically update vectors on commit/merge/checkout)
   - Viewing hook logs in `.embeddixdb/hook-logs/`

2. Provide a quick reference card:
   ```
   📝 EmbeddixDB Quick Reference
   ════════════════════════════════════════════════════════════
   Server:
   Start:   ./embeddix/scripts/start_bg.sh
   Stop:    ./embeddix/scripts/stop.sh
   Status:  ./embeddix/scripts/status.sh
   Health:  curl http://localhost:8080/health

   API:
   Docs:    http://localhost:8080/docs
   List:    curl http://localhost:8080/collections

   Auto-Indexing:
   Hooks:   Installed in .git/hooks/ (auto-runs on commit/merge)
   Logs:    cat .embeddixdb/hook-logs/post-commit.log
   Manual:  PYTHONPATH=./embeddix/src python embeddix/src/index_codebase.py

   Documentation:
   Setup:   embeddix/docs/SETUP.md
   Server:  cat embeddixdb/server.log
   ════════════════════════════════════════════════════════════
   ```

3. Remind them to commit the new files:
   ```bash
   git add embeddix/ .claude/agents/ embeddixdb/
   git commit -m "Add EmbeddixDB vector search integration"
   ```

---

## Important Notes

- **Always use `embeddix-api`**, not `embeddix-mcp`
- **Config path MUST be full file path**: `./embeddixdb/data/embeddix.db`
- **Explicitly disable embedding engine**: `ai.embedding.engine: ""`
- **PYTHONPATH is CRITICAL**: Code-search agent requires `PYTHONPATH=embeddix/src:$PYTHONPATH` to import modules
- **Python package structure required**: Must create `__init__.py` files in `embeddix/` and `embeddix/src/`
- **Import path in code-search agent**: Use `from novita_integration import TradingDocumentManager` (NOT `embeddix_novita_integration`)
- **Git hooks auto-update vectors**: After Step 14, commits/merges/checkouts automatically index changes
- **Hooks run in background**: They don't slow down git operations
- **Repository-agnostic**: Works with any repo, collection name auto-generated from repo name
- Always check if files exist before creating them
- Preserve any existing customizations
- Make backups before modifying system files (like `.zshrc`)
- Use absolute paths in scripts for reliability
- Test each major step before proceeding
- Be helpful and explain what each component does

---

## Common Issues & Solutions

### Issue: Server fails with "is a directory" error
**Cause**: Config has `path: "./embeddixdb/data"` (directory only)
**Solution**: Change to `path: "./embeddixdb/data/embeddix.db"` (full file path)

### Issue: Server tries to load MiniLM model
**Cause**: Using `embeddix-mcp` or missing `ai.embedding.engine` config
**Solution**: Use `embeddix-api` server and set `ai.embedding.engine: ""`

### Issue: Server won't start
**Check**:
```bash
# Check if port is in use
lsof -i :8080

# Check server logs
cat embeddixdb/server.log

# Try starting in foreground to see errors
./embeddix/scripts/start.sh
```

### Issue: Import errors in Python tests
**Cause**: Python can't find `novita_integration` module
**Solution**: Add path handling at top of test files:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

---

## Error Handling

If any step fails:
1. Explain what went wrong
2. Provide troubleshooting steps specific to the error
3. Offer to try an alternative approach
4. Check the common issues list above
5. Don't proceed until the issue is resolved

---

## Verification Checklist

After installation, verify:
- ✅ Both `embeddix-api` and `embeddix-mcp` binaries exist
- ✅ Configuration file has full database path
- ✅ Configuration explicitly disables embedding engine
- ✅ Python package structure created: `embeddix/__init__.py` and `embeddix/src/__init__.py` exist
- ✅ PYTHONPATH is set in startup scripts: Check `embeddix/scripts/start_bg.sh`
- ✅ Python integration can be imported: `PYTHONPATH=embeddix/src:$PYTHONPATH python3 -c "from novita_integration import TradingDocumentManager"`
- ✅ Server starts successfully using `embeddix-api`
- ✅ Server responds to health check: `curl http://localhost:8080/health`
- ✅ Novita API test passes
- ✅ Can list collections: `curl http://localhost:8080/collections`
- ✅ `search_code.py` exists in project root and returns absolute paths
- ✅ Git hooks are installed and executable: `ls -la .git/hooks/post-*`
- ✅ Incremental indexing script exists: `embeddix/src/incremental_index.py`
- ✅ Hook logs directory created: `.embeddixdb/hook-logs/`
- ✅ Test commit triggers hook: Check `.embeddixdb/hook-logs/post-commit.log`
- ✅ Code-search agent can import modules: Verify `.claude/agents/code-search.md` uses correct import path

---

## Next Steps

After successful installation:
1. **Index the codebase** (optional - first time only):
   ```bash
   PYTHONPATH=./embeddix/src python embeddix/src/index_codebase.py --root . --collection kent_hologram_code
   ```
   After initial indexing, git hooks keep it automatically updated!

2. **Configure auto-start** (optional) - server starts automatically when entering project directory

3. **Set up Claude Code agents** - use `cursor-code` and `gemini-code` for semantic code search

4. **Start using semantic search** - your codebase is now searchable with natural language queries!

**Note**: With git hooks installed, every `git commit`, `git pull`, and branch switch automatically updates the vector database in the background. No manual re-indexing needed!

Good luck! Remember to be thorough, patient, and helpful throughout the process.
