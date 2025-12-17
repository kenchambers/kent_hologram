#!/bin/bash
# Wrapper for MCP semantic search
export EMBEDDIX_PORT=${EMBEDDIX_PORT:-8080}
python scripts/mcp_client.py "$@"
