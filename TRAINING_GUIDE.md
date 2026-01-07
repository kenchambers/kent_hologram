# Hologram Crew Trainer Guide

The CrewAI Hologram Trainer is a continuous training system that uses multiple LLMs (Gemini and Claude) to have conversations with your Hologram chatbot, teaching it facts and improving its knowledge base overnight.

## Quick Start

### Option 1: Use the Overnight Training Script (Recommended)

**Watch live (recommended for first time):**

```bash
./scripts/train_overnight.sh
```

**Run in background (for overnight):**

```bash
./scripts/train_overnight.sh --background
```

This runs the trainer with sensible defaults (100 rounds). You can edit the script to customize:

- `ROUNDS`: Number of conversation rounds (default: 100)
- `TURNS_PER_TOPIC`: Conversation turns per round (default: 8)

### Option 2: Run Manually

```bash
# Run unlimited rounds (until stopped with Ctrl+C)
uv run python scripts/crew_trainer.py

# Run specific number of rounds
uv run python scripts/crew_trainer.py --max-rounds 100

# Custom configuration
uv run python scripts/crew_trainer.py \
    --max-rounds 50 \
    --turns-per-topic 10 \
    --persist-dir ./my_facts \
    --log-dir ./my_logs
```

## Prerequisites

### 1. API Keys

Create a `.env` file in the project root with your API keys:

```bash
GEMINI_API_KEY=your_google_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Get API Keys:**

- **Gemini**: https://aistudio.google.com/app/apikey
- **Anthropic**: https://console.anthropic.com/account/keys

### 2. Install Dependencies

```bash
uv sync
```

## How It Works

1. **Gemini** starts a conversation or teaches a fact
2. **Hologram** responds and learns facts when detected
3. **Claude** continues the conversation naturally
4. **Hologram** responds again (may learn more facts)
5. Process repeats for the configured number of turns
6. All conversations are logged and facts persist to ChromaDB

### Example Conversation Flow

```
Gemini: "The capital of France is Paris"
Hologram: "Got it! I'll remember that France capital Paris."
Claude: "That's right! Paris is also known as the City of Light."
Hologram: "Interesting! Tell me more about Paris."
...
```

## Command-Line Options

| Option                | Description                    | Default                      |
| --------------------- | ------------------------------ | ---------------------------- |
| `--max-rounds N`      | Maximum conversation rounds    | Unlimited                    |
| `--turns-per-topic N` | Turns per conversation round   | 8                            |
| `--persist-dir DIR`   | ChromaDB persistence directory | `./data/crew_training_facts` |
| `--log-dir DIR`       | Conversation logs directory    | `./conversation_logs`        |

## Monitoring Training

### Check Progress

```bash
# Watch live output
tail -f conversation_logs/session_*.log

# Or if using overnight script
tail -f training_*.log
```

### Status Reports

The trainer automatically prints status reports every 10 rounds:

```
============================================================
  Status Report - Round 10
============================================================
  Rounds completed: 10
  Facts learned (this session): 5
  Total facts in store: 125
  Conversation turns: 20
  Recent facts:
    • France capital Paris
    • Python created_by Guido
    • sky color blue
============================================================
```

## Stopping Training

### If running in foreground

Press `Ctrl+C` for graceful shutdown

### If running in background

```bash
pkill -f crew_trainer.py
```

## Error Handling

The trainer includes robust error handling:

- **Automatic retry** with exponential backoff on API errors
- **Stops after 5 consecutive errors** to prevent infinite loops
- **All errors logged** to conversation log file
- **Graceful shutdown** preserves all learned facts

## Cost Optimization

### Model Configuration

You can change the LLM models via environment variables in your `.env`:

```bash
# Use cheaper/faster models
GEMINI_MODEL=gemini-2.0-flash-exp
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Or more capable models
GEMINI_MODEL=gemini-1.5-pro
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### Cost Estimates

Using default models (Gemini Flash + Claude Haiku):

- ~$0.001-0.002 per conversation round
- 100 rounds overnight ≈ $0.10-0.20
- Actual cost depends on conversation length and API pricing

## Outputs

### Conversation Logs

- Location: `./conversation_logs/session_TIMESTAMP.log`
- Format: Timestamped messages from each participant
- Includes system events and learned facts

### Learned Facts

- Location: `./data/crew_training_facts/` (ChromaDB)
- Persistent across sessions
- Queryable via Hologram chatbot

## Tips for Effective Training

1. **Start with moderate rounds** (50-100) to test
2. **Monitor first few rounds** to ensure quality conversations
3. **Check learned facts** to verify knowledge acquisition
4. **Adjust turns-per-topic** based on conversation quality:
   - Lower (4-6) for quick, focused fact learning
   - Higher (8-12) for deeper, more natural conversations

## Troubleshooting

### "API key not found" error

- Ensure `.env` file exists in project root
- Check that `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` are set

### "Model not found" error

- Google/Anthropic may have changed model names
- Update `GEMINI_MODEL` or `ANTHROPIC_MODEL` in `.env`
- See [model configuration](#model-configuration) above

### Training stops after a few rounds

- Check the log file for error messages
- May be rate limiting from API providers
- Try reducing `--turns-per-topic` or adding delays

### No facts being learned

- Check conversation logs to see if teaching patterns are used
- The chatbot looks for patterns like "X is Y" or "the X of Y is Z"
- Gemini and Claude should naturally produce these in conversation

## Advanced Usage

### Custom System Prompts

Edit `scripts/crew_trainer.py` to modify:

- `GEMINI_SYSTEM_PROMPT`: Gemini's conversation style
- `CLAUDE_SYSTEM_PROMPT`: Claude's conversation style

### Integration with CI/CD

Run nightly training automatically:

```bash
# Add to crontab for nightly 2 AM training
0 2 * * * cd /path/to/kent_hologram && ./scripts/train_overnight.sh
```

### Multiple Training Sessions

Run parallel trainers with different configurations:

```bash
# Geography facts
uv run python scripts/crew_trainer.py \
    --persist-dir ./data/geography_facts \
    --max-rounds 50 &

# Science facts (in separate terminal)
uv run python scripts/crew_trainer.py \
    --persist-dir ./data/science_facts \
    --max-rounds 50 &
```

## Training on Project Gutenberg

Scale up your Hologram's knowledge with the entire Project Gutenberg library - 75,570 books, automatically cleaned and chunked for optimal learning.

### Why Project Gutenberg?

- **Massive Scale**: 61,300+ English books (French, German, Dutch, Spanish, Portuguese, Italian, Chinese, Swedish, Polish, Russian also available)
- **Public Domain**: No copyright restrictions, perfect for training
- **Streaming**: Downloads only what you need - no multi-gigabyte dataset required
- **Automatic Cleaning**: Headers, footers, and metadata removed automatically

### Quick Start

```bash
# Start fresh ingestion (English books only)
uv run python scripts/ingest_gutenberg.py

# Resume from checkpoint if interrupted
uv run python scripts/ingest_gutenberg.py --resume
```

The script will:
1. Stream books from Hugging Face (`manu/project_gutenberg` dataset)
2. Clean Project Gutenberg metadata (headers/footers)
3. Split into chunks (default: 1000 characters)
4. Teach each chunk to your Hologram
5. Save progress to `./data/gutenberg_checkpoint.json`
6. Resume automatically if interrupted (Ctrl+C then re-run with `--resume`)

### Common Commands

**Limit books for testing** (recommended for first run):

```bash
# Try with just 10 books first
uv run python scripts/ingest_gutenberg.py --max-books 10

# Then ramp up when confident
uv run python scripts/ingest_gutenberg.py --max-books 100

# Process all 61,300 English books
uv run python scripts/ingest_gutenberg.py
```

**Choose a different language**:

```bash
# French (5,500 books)
uv run python scripts/ingest_gutenberg.py --language fr

# German (3,100 books)
uv run python scripts/ingest_gutenberg.py --language de

# Spanish (1,200 books)
uv run python scripts/ingest_gutenberg.py --language es
```

**Adjust chunk size** (for performance tuning):

```bash
# Larger chunks = fewer facts, but each captures more context
uv run python scripts/ingest_gutenberg.py --chunk-size 2000

# Smaller chunks = more facts, but less context per fact
uv run python scripts/ingest_gutenberg.py --chunk-size 500
```

### Command-Line Options

| Option              | Short | Description                              | Default                          |
| ------------------- | ----- | ---------------------------------------- | -------------------------------- |
| `--max-books`       | `-n`  | Maximum number of books to process       | All books                        |
| `--language`        | `-l`  | Language split to use                    | `en`                             |
| `--chunk-size`      | `-c`  | Characters per chunk                     | `1000`                           |
| `--checkpoint-file` |       | Path to checkpoint JSON                  | `./data/gutenberg_checkpoint.json` |
| `--persist-dir`     |       | Directory for fact persistence           | `./data/crew_training_facts`     |
| `--resume`          | `-r`  | Resume from existing checkpoint          | (default behavior)               |
| `--fresh`           | `-f`  | Start fresh, ignoring existing checkpoint | Off                              |

### Available Languages

| Language   | Code | Books |
| ---------- | ---- | ----- |
| English    | en   | 61.3k |
| French     | fr   | 5.5k  |
| German     | de   | 3.1k  |
| Dutch      | nl   | 1.4k  |
| Spanish    | es   | 1.2k  |
| Portuguese | pt   | 1.1k  |
| Italian    | it   | 1k    |
| Chinese    | zh   | 437   |
| Swedish    | sv   | 388   |
| Polish     | pl   | 34    |
| Russian    | ru   | 6     |

### Checkpoint System

The script is **resumable** - it saves progress automatically every 10 books:

```bash
# Start processing
uv run python scripts/ingest_gutenberg.py --max-books 1000

# Press Ctrl+C to pause gracefully (saves checkpoint)

# Resume later from where you left off
uv run python scripts/ingest_gutenberg.py --resume

# Or start completely fresh (ignoring checkpoint)
uv run python scripts/ingest_gutenberg.py --fresh
```

The checkpoint tracks:
- Which books have been processed
- Total facts learned
- Errors encountered
- Last update timestamp

### Performance Expectations

- **10 books**: ~5-10 minutes
- **100 books**: ~30-60 minutes
- **1000 books**: 4-8 hours (overnight)
- **Full dataset (61,300 books)**: Several days

Each book contributes **10-20 facts** depending on length. First run takes 30-60 seconds to initialize the trainer.

### Combining with Crew Training

For best results, combine Gutenberg ingestion with overnight crew training:

```bash
# Day: Ingest Gutenberg books
uv run python scripts/ingest_gutenberg.py --max-books 500

# Night: Let crew_trainer have conversations
uv run python scripts/crew_trainer.py --max-rounds 200

# Next day: Resume Gutenberg
uv run python scripts/ingest_gutenberg.py --resume
```

This creates a virtuous cycle:
1. **Gutenberg ingestion** provides broad, factual knowledge from literature
2. **Crew training** teaches dialogue patterns and reasoning
3. Together they create a Hologram with deep knowledge and natural conversation

### Troubleshooting

**Script hangs on startup**: First run initializes the trainer (30-60 seconds is normal). Just wait for progress output.

**Checkpoint file is corrupted**:
```bash
uv run python scripts/ingest_gutenberg.py --fresh
```

**Out of disk space**:
- Facts stored in `./data/crew_training_facts`
- Each fact takes ~1-2 KB
- 10,000 facts ≈ 10-20 MB
- Reduce `--max-books` to fit your disk space

**Want to clean up and start over**:
```bash
rm -rf ./data/gutenberg_checkpoint.json ./data/crew_training_facts
uv run python scripts/ingest_gutenberg.py
```

**Missing datasets library**:
```bash
uv pip install datasets
```

## Support

For issues or questions:

1. Check the conversation logs for error details
2. Review the [README.md](./README.md) for general setup
3. Ensure API keys are valid and have sufficient credits
