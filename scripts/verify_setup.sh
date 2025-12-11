#!/bin/bash
#
# Verify CrewAI trainer setup before running overnight training
#

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Hologram CrewAI Trainer - Setup Verification         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check if running from project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

echo "✓ Running from project root"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo ""
    echo "Create a .env file with your API keys:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env and add your keys"
    echo ""
    echo "Get API keys from:"
    echo "  Gemini:    https://aistudio.google.com/app/apikey"
    echo "  Anthropic: https://console.anthropic.com/account/keys"
    exit 1
fi

echo "✓ Found .env file"

# Check API keys
source .env

if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo "❌ Error: GEMINI_API_KEY not set in .env"
    echo "   Get your key at: https://aistudio.google.com/app/apikey"
    exit 1
fi

echo "✓ GEMINI_API_KEY is set"

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY not set in .env"
    echo "   Get your key at: https://console.anthropic.com/account/keys"
    exit 1
fi

echo "✓ ANTHROPIC_API_KEY is set"
echo ""

# Check Python environment
echo "Checking Python environment..."
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv not found"
    echo "   Install from: https://docs.astral.sh/uv/"
    exit 1
fi

echo "✓ UV package manager found"

# Check dependencies
echo "Checking dependencies..."
if ! uv run python -c "import langchain_google_genai, langchain_anthropic" &> /dev/null; then
    echo "⚠️  Warning: Dependencies not installed or outdated"
    echo "   Running: uv sync"
    uv sync
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ All dependencies installed"
echo ""

# Create necessary directories
mkdir -p conversation_logs
mkdir -p data/crew_training_facts
echo "✓ Created necessary directories"
echo ""

# Run a quick test
echo "Running quick initialization test..."
if ! timeout 10 uv run python -c "
from scripts.crew_trainer import CrewTrainer
from pathlib import Path
import tempfile
import sys
try:
    temp_dir = tempfile.mkdtemp()
    trainer = CrewTrainer(
        persist_dir=temp_dir,
        log_dir=Path(temp_dir),
        max_turns_per_topic=1,
        max_rounds=0
    )
    print('✓ Trainer initialized successfully', file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f'❌ Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    echo "❌ Error: Trainer initialization failed"
    echo "   Check that your API keys are valid"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ✓ All checks passed! Ready for training              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Start training with:"
echo "  ./scripts/train_overnight.sh"
echo ""
echo "Or run manually:"
echo "  uv run python scripts/crew_trainer.py --max-rounds 100"
echo ""
echo "See TRAINING_GUIDE.md for more options and troubleshooting."
echo ""
