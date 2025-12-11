#!/bin/bash
#
# Overnight training script for Hologram chatbot
# Run with --background to detach, or watch live by default
#

# Configuration
ROUNDS=100
TURNS_PER_TOPIC=8
BACKGROUND=false

# Parse arguments
if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    BACKGROUND=true
fi

echo "╔════════════════════════════════════════════════════════╗"
echo "║  CrewAI Hologram Trainer - Overnight Mode             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  - Max rounds: $ROUNDS"
echo "  - Turns per topic: $TURNS_PER_TOPIC"
echo ""

# Change to project root
cd "$(dirname "$0")/.." || exit 1

if [ "$BACKGROUND" = true ]; then
    # Background mode - use nohup
    LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
    echo "Running in background mode..."
    echo "Output will be logged to: $LOG_FILE"
    echo ""
    
    PYTHONUNBUFFERED=1 nohup uv run python scripts/crew_trainer.py \
        --max-rounds "$ROUNDS" \
        --turns-per-topic "$TURNS_PER_TOPIC" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "✓ Training started! (PID: $PID)"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop training:"
    echo "  pkill -f crew_trainer.py"
    echo ""
else
    # Foreground mode - watch live (default)
    echo "Running in foreground mode (watch live)..."
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    echo "Tip: Run with --background to detach and run overnight"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo ""
    
    PYTHONUNBUFFERED=1 uv run python scripts/crew_trainer.py \
        --max-rounds "$ROUNDS" \
        --turns-per-topic "$TURNS_PER_TOPIC"
fi
