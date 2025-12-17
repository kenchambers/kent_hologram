#!/bin/bash
# PreToolUse hook: Remind Claude to prefer semantic search over Glob/Grep
# This hook does NOT block - it provides guidance via systemMessage

# Read JSON from stdin
INPUT=$(cat)

# Extract tool name
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# For Glob and Grep exploration, add a reminder
if [[ "$TOOL" == "Glob" || "$TOOL" == "Grep" ]]; then
    # Extract the pattern/query to check if this is exploration
    PATTERN=$(echo "$INPUT" | jq -r '.tool_input.pattern // empty')

    # If pattern looks like exploration (contains *, **, common search terms)
    if [[ "$PATTERN" == *"**"* || "$PATTERN" == *"*"* || -z "$PATTERN" ]]; then
        # Return JSON with systemMessage - allows tool but reminds Claude
        cat << 'EOF'
{
  "continue": true,
  "systemMessage": "REMINDER: For code exploration in this repository, semantic search (mcp__embeddixdb-kent-hologram__search_code) is faster and more accurate. Use it with natural language queries like 'ARC solver implementation' or 'neural consolidation'. Only use Glob/Grep for exact string matches or when semantic search returns no results."
}
EOF
        exit 0
    fi
fi

# For all other tools, allow without message
echo '{"continue": true}'
exit 0
