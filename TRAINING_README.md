# Hologram Training System

This document describes the three training methods available for the Hologram HDC memory system. All trainers share the same underlying memory, enabling cross-domain learning.

## Quick Start

```bash
# Conversational training with LLMs
python scripts/crew_trainer.py --max-rounds 50

# ARC pattern training (visual reasoning)
python scripts/arc_trainer.py --curriculum easy --max-rounds 10

# Code pattern training (SWE-bench style)
python scripts/code_trainer.py --max-rounds 5
```

## Shared Memory Architecture

All three trainers use the **same persistence directory** by default (`./data/crew_training_facts`). This means:

- Facts learned in crew_trainer are available to arc_trainer and code_trainer
- ARC transformation patterns reinforce code patterns and vice versa
- Neural consolidation happens across all domains

To use separate memories, specify different `--persist-dir` values.

---

## 1. Crew Trainer (Conversational Learning)

**Purpose:** Train Hologram through LLM-guided conversations. Gemini and Claude act as teachers, quizzing the system on facts and teaching natural language patterns.

**Script:** `scripts/crew_trainer.py`

### Input Data Types

| Data Type | How to Provide | Limits |
|-----------|----------------|--------|
| Web topics | `--web-teach "Topic1" "Topic2"` | Any number of topics |
| Code topics | `--web-teach-code "Python lists"` | Any number of topics |
| Documents | `--teach-document file.txt` | Any file size (chunked automatically) |
| Live conversation | Run without special flags | Unlimited rounds |

### Configuration Options

```bash
python scripts/crew_trainer.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--persist-dir` | `./data/crew_training_facts` | Where to save learned facts |
| `--log-dir` | `./conversation_logs` | Where to save conversation logs |
| `--max-rounds` | Unlimited | Stop after N conversation rounds |
| `--turns-per-topic` | 8 | Conversation turns before topic change |
| `--web-teach` | None | Topics to search and learn from web |
| `--web-teach-code` | None | Code topics to search from technical docs |
| `--web-results` | 3 | Web search results per topic |
| `--web-facts` | 5 | Max facts extracted per result |
| `--teach-document` | None | Path to document file for book-scale ingestion |
| `--chunk-size` | 500 | Characters per chunk for documents |

### Examples

```bash
# Basic conversational training (runs until Ctrl+C)
python scripts/crew_trainer.py

# Train for 100 rounds overnight
python scripts/crew_trainer.py --max-rounds 100

# Populate knowledge base from web, then train
python scripts/crew_trainer.py --web-teach "World Capitals" "Famous Scientists" --max-rounds 50

# Learn programming concepts
python scripts/crew_trainer.py --web-teach-code "Python decorators" "async await patterns"

# Ingest a book or long document
python scripts/crew_trainer.py --teach-document my_textbook.txt --chunk-size 1000

# Combined: web facts + document + conversation
python scripts/crew_trainer.py \
  --web-teach "Physics" \
  --teach-document physics_notes.txt \
  --max-rounds 20
```

### Document Format

The `--teach-document` option accepts any plain text file:
- `.txt` files (recommended)
- `.md` markdown files
- Any UTF-8 encoded text

**How it works:**
1. Document is split into overlapping chunks (default 500 chars, 100 char overlap)
2. Each chunk is processed by an LLM to extract (Subject, Predicate, Object) facts
3. Facts are stored in the shared memory

**Limits:**
- No hard size limit (chunking handles arbitrarily large files)
- LLM context window: 2000 chars per chunk (truncated if longer)
- Recommended chunk size: 300-1000 characters

---

## 2. ARC Trainer (Visual Reasoning Patterns)

**Purpose:** Train on ARC-AGI transformation patterns. Learns to recognize grid transformations like rotation, color mapping, and object manipulation.

**Script:** `scripts/arc_trainer.py`

### Input Data Types

| Data Type | How to Provide | Limits |
|-----------|----------------|--------|
| ARC-AGI-2 dataset | Automatic (downloads from GitHub) | ~400 training tasks |
| Curriculum tier | `--curriculum easy/medium/hard/all` | Filter by difficulty |

### Configuration Options

```bash
python scripts/arc_trainer.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--persist-dir` | `./data/crew_training_facts` | Shared memory directory |
| `--log-dir` | `./arc_training_logs` | Training logs |
| `--max-rounds` | 10 | Number of training passes over dataset |
| `--curriculum` | `all` | Difficulty: `easy`, `medium`, `hard`, or `all` |
| `--validate-every` | 5 | Run validation every N rounds |
| `--validation-limit` | 50 | Number of validation tasks |
| `--train-split` | `training` | Dataset split to use |
| `--consolidation-threshold` | 20 | Facts before neural consolidation |

### Examples

```bash
# Start with easy tasks (recommended for first run)
python scripts/arc_trainer.py --curriculum easy --max-rounds 5

# Full curriculum training
python scripts/arc_trainer.py --max-rounds 20 --validate-every 5

# Train only on hard tasks
python scripts/arc_trainer.py --curriculum hard --max-rounds 10

# Use same memory as crew_trainer
python scripts/arc_trainer.py --persist-dir ./data/crew_training_facts
```

### Dataset Setup

ARC-AGI-2 dataset is required. If not present, the trainer will prompt:

```bash
git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI-2
```

**Difficulty Tiers:**
- **Easy:** Grid ≤5x5, ≤3 objects, no grid size changes (~100 tasks)
- **Medium:** Grid ≤10x10, ≤6 objects (~150 tasks)
- **Hard:** Larger grids, complex transformations (~150 tasks)

---

## 3. Code Trainer (SWE-bench Patterns)

**Purpose:** Train on code transformation patterns. Learns to recognize and generate code fixes for common issues like null checks, input validation, and error handling.

**Script:** `scripts/code_trainer.py`

### Input Data Types

| Data Type | How to Provide | Limits |
|-----------|----------------|--------|
| Sample tasks | Default (built-in) | 3 sample tasks |
| JSON training file | `--data file.json` | Unlimited tasks |
| Web code patterns | `--web-teach "pattern"` | Any number of topics |
| Issue text | `--mode generate --issue "..."` | Single issue for generation |

### Configuration Options

```bash
python scripts/code_trainer.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--persist-dir` | `./data/crew_training_facts` | Shared memory directory |
| `--log-dir` | `./code_training_logs` | Training logs |
| `--max-rounds` | 5 | Number of training passes |
| `--validate-every` | 2 | Run validation every N rounds |
| `--mode` | `train` | Mode: `train` or `generate` |
| `--issue` | None | Issue text for generate mode |
| `--data` | None | JSON file with training tasks |
| `--web-teach` | None | Code topics to search |
| `--web-results` | 3 | Web results per topic |

### Examples

```bash
# Train on built-in sample tasks
python scripts/code_trainer.py --max-rounds 5

# Load custom training data
python scripts/code_trainer.py --data my_swe_data.json --max-rounds 10

# Search web for code patterns, then train
python scripts/code_trainer.py --web-teach "null check python" "input validation"

# Generate a patch for an issue
python scripts/code_trainer.py --mode generate --issue "Fix division by zero in calculate()"

# Combined: web patterns + file data
python scripts/code_trainer.py \
  --web-teach "error handling python" \
  --data extra_patterns.json \
  --max-rounds 10
```

### Training Data Format

JSON file format for `--data`:

```json
{
  "patterns": [
    {
      "pattern_id": "null_check_001",
      "example_issue": "Add null check to process function",
      "example_fix": "if x is None:\n    raise ValueError('x cannot be None')"
    }
  ],
  "examples": [
    {
      "task_id": "fix_001",
      "repo": "myproject/utils",
      "issue_text": "Function crashes on empty input",
      "file_path": "utils.py",
      "code_before": "def process(x):\n    return x * 2",
      "code_after": "def process(x):\n    if x is None:\n        return 0\n    return x * 2"
    }
  ]
}
```

---

## Environment Requirements

### API Keys

Set in `.env` or environment:

```bash
# Required for crew_trainer
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional (fallback for crew_trainer)
OPENAI_API_KEY=your_key_here
```

### Dependencies

```bash
# Core dependencies
uv sync

# Optional: web search for teaching
pip install ddgs

# For ARC training
git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI-2
```

---

## Memory Limits and Recommendations

| Aspect | Limit | Recommendation |
|--------|-------|----------------|
| Vector dimensions | 10,000 (default) | Don't change unless you know why |
| Fact store | ~100,000+ facts | Practically unlimited |
| Document chunk size | 300-2000 chars | 500 chars is balanced |
| Neural consolidation threshold | 10-100 facts | 20 is good default |
| HNSW for large datasets | Use when >10,000 vectors | Automatic O(log n) queries |

### Performance Tips

1. **Cold start:** Use `--web-teach` to bootstrap knowledge before conversation training
2. **Curriculum learning:** For ARC, start with `--curriculum easy` before harder tasks
3. **Document ingestion:** Use larger `--chunk-size` (800-1000) for technical docs
4. **Long training runs:** Use `--max-rounds` with `nohup` for overnight training:
   ```bash
   nohup python scripts/crew_trainer.py --max-rounds 500 > training.log 2>&1 &
   ```

---

## Logs and Monitoring

Each trainer creates timestamped logs:

```
conversation_logs/session_20241217_143022.log  # crew_trainer
arc_training_logs/arc_training_20241217_143022.log  # arc_trainer
code_training_logs/code_training_20241217_143022.log  # code_trainer
```

Logs include:
- All conversation turns (crew_trainer)
- Task IDs and success/failure (arc_trainer, code_trainer)
- Facts learned
- Error messages
- Periodic statistics

---

## Troubleshooting

### "GEMINI_API_KEY not found"
Set your API key: `export GEMINI_API_KEY=your_key`

### "ARC dataset not found"
Run: `git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI-2`

### "Web search failed"
Install search library: `pip install ddgs`

### Memory growing too large
- Reduce `--web-facts` and `--web-results`
- Use higher `--consolidation-threshold`
- Neural consolidation compresses facts automatically

### Training seems slow
- Use `--curriculum easy` for ARC to start
- Reduce `--max-rounds` for initial testing
- Check API rate limits (especially for crew_trainer)
