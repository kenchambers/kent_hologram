import sys
import json
import subprocess
import os

def run_mcp_search(query, collection="kent_hologram_code"):
    base_dir = os.getcwd()
    
    # Use Python MCP server
    python_executable = sys.executable
    mcp_script = os.path.join(base_dir, "embeddix/src/mcp_server.py")
    
    # Environment variables
    env = os.environ.copy()
    # Use provided env var or default to 8080 (which seems to be the active one)
    env["EMBEDDIX_PORT"] = os.environ.get("EMBEDDIX_PORT", "8080")
    env["PYTHONPATH"] = os.path.join(base_dir, "embeddix/src")
    
    cmd = [python_executable, mcp_script]
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Log logs to stderr
        env=env,
        text=True
    )
    
    # JSON-RPC helper
    def send_request(method, params=None, req_id=1):
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id
        }
        if params is not None:
            req["params"] = params
        
        json_req = json.dumps(req) + "\n"
        process.stdin.write(json_req)
        process.stdin.flush()
        
        # Read response
        while True:
            response_line = process.stdout.readline()
            
            if not response_line:
                return None
            
            line = response_line.strip()
            if not line:
                continue
                
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    print(f"DEBUG: Failed to decode JSON: {line}", file=sys.stderr)
                    continue
            else:
                # Log ignored lines to stderr
                print(f"IGNORED STDOUT: {line}", file=sys.stderr)

    try:
        # 1. Initialize
        init_params = {
            "protocolVersion": "2024-11-05", 
            "capabilities": {},
            "clientInfo": {"name": "gemini-cli", "version": "1.0"}
        }
        
        init_resp = send_request("initialize", init_params)
        
        if not init_resp or "error" in init_resp:
            print("Initialization failed:", init_resp, file=sys.stderr)
            return

        # 2. Initialized notification
        # The Python MCP SDK expects 'notifications/initialized' after initialize response
        # But for simple tool call, maybe not strictly required? 
        # mcp.server implementation might wait for it.
        # send_request("notifications/initialized", {}) 
        # Wait, 'notifications' don't expect response.
        
        req = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        process.stdin.write(json.dumps(req) + "\n")
        process.stdin.flush()

        # 3. Search
        search_params = {
            "name": "search_code",
            "arguments": {
                "collection": collection,
                "query": query,
                "limit": 5
            }
        }
        search_resp = send_request("tools/call", search_params, req_id=2)
        
        if search_resp and "result" in search_resp:
            content = search_resp["result"].get("content", [])
            for item in content:
                if item["type"] == "text":
                    print(item["text"])
        else:
            print("Error or no result:", search_resp)

    finally:
        process.terminate()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/mcp_client.py 'query'")
        sys.exit(1)
    
    run_mcp_search(sys.argv[1])