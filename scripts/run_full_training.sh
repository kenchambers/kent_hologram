#!/usr/bin/env bash
#
# Full Training Pipeline for Kent Hologram
#
# Executes the complete recommended training sequence:
#   Phase 0: Validate tests (optional, recommended)
#   Phase 1: Conversation training (50 rounds) - Foundation
#   Phase 2: Book ingestion (50 books) - Depth
#   Phase 3: Fine-tuning conversations (50 rounds) - Polish
#
# Usage:
#   ./scripts/run_full_training.sh              # Medium training (~1.5 hours)
#   ./scripts/run_full_training.sh --quick      # Quick test (~15 min)
#   ./scripts/run_full_training.sh --small      # Small training (~30 min)
#   ./scripts/run_full_training.sh --large      # Large training (~4 hours)
#   ./scripts/run_full_training.sh --overnight  # Overnight (~8 hours)
#   ./scripts/run_full_training.sh --skip-tests # Skip test validation
#   ./scripts/run_full_training.sh --background # Run in background
#
# Author: Claude Code

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default profile settings
PROFILE="medium"
CONV_ROUNDS_1=50
BOOKS=50
CONV_ROUNDS_2=50
TURNS_PER_TOPIC=8

# Flags
SKIP_TESTS=false
BACKGROUND=false
DRY_RUN=false

# Directories
PERSIST_DIR="./data/crew_training_facts"
LOG_DIR="./training_logs"
CHECKPOINT_FILE="./data/gutenberg_checkpoint.json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

print_phase() {
    echo ""
    echo "------------------------------------------------------------------------"
    echo "  PHASE $1: $2"
    echo "------------------------------------------------------------------------"
    echo ""
}

print_status() {
    echo "  [OK] $1"
}

print_error() {
    echo "  [ERROR] $1" >&2
}

print_info() {
    echo "  [INFO] $1"
}

get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_to_file() {
    local log_file="$1"
    shift
    echo "[$(get_timestamp)] $*" >> "$log_file"
}

format_duration() {
    local minutes=$1
    if [ "$minutes" -lt 60 ]; then
        echo "${minutes} minutes"
    else
        local hours=$(( minutes / 60 ))
        local mins=$(( minutes % 60 ))
        echo "${hours}h ${mins}m"
    fi
}

set_profile() {
    case $1 in
        quick)
            PROFILE="quick"
            CONV_ROUNDS_1=10
            BOOKS=10
            CONV_ROUNDS_2=0
            TURNS_PER_TOPIC=6
            ;;
        small)
            PROFILE="small"
            CONV_ROUNDS_1=25
            BOOKS=25
            CONV_ROUNDS_2=0
            TURNS_PER_TOPIC=8
            ;;
        medium)
            PROFILE="medium"
            CONV_ROUNDS_1=50
            BOOKS=50
            CONV_ROUNDS_2=50
            TURNS_PER_TOPIC=8
            ;;
        large)
            PROFILE="large"
            CONV_ROUNDS_1=100
            BOOKS=200
            CONV_ROUNDS_2=100
            TURNS_PER_TOPIC=10
            ;;
        overnight)
            PROFILE="overnight"
            CONV_ROUNDS_1=200
            BOOKS=500
            CONV_ROUNDS_2=200
            TURNS_PER_TOPIC=8
            ;;
    esac
}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            set_profile "quick"
            shift
            ;;
        --small)
            set_profile "small"
            shift
            ;;
        --medium)
            set_profile "medium"
            shift
            ;;
        --large)
            set_profile "large"
            shift
            ;;
        --overnight)
            set_profile "overnight"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Training Profiles:"
            echo "  --quick       Quick test: 10 convs + 10 books (~15 min)"
            echo "  --small       Small KB: 25 convs + 25 books (~30 min)"
            echo "  --medium      Medium KB: 50 convs + 50 books + 50 convs (~1.5 hours) [default]"
            echo "  --large       Large KB: 100 convs + 200 books + 100 convs (~4 hours)"
            echo "  --overnight   Overnight: 200 convs + 500 books + 200 convs (~8 hours)"
            echo ""
            echo "Options:"
            echo "  --skip-tests  Skip test validation phase"
            echo "  --background  Run training in background (logs to file)"
            echo "  --dry-run     Show what would be run without executing"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick --skip-tests    # Fast test run"
            echo "  $0 --medium                # Recommended for first training"
            echo "  $0 --large --background    # Large training overnight"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

# Change to project root
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

# Create log directory
mkdir -p "$LOG_DIR"

# Generate log file name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/training_${PROFILE}_${TIMESTAMP}.log"

# Calculate estimates
CONV_TIME=$(( (CONV_ROUNDS_1 + CONV_ROUNDS_2) ))
BOOK_TIME=$(( BOOKS * 3 / 10 ))
EST_MINUTES=$(( CONV_TIME + BOOK_TIME + 5 ))
EST_TIME=$(format_duration "$EST_MINUTES")
TOTAL_FACTS_EST=$(( CONV_ROUNDS_1 * 3 + BOOKS * 100 + CONV_ROUNDS_2 * 3 ))

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

print_header "Kent Hologram Full Training Pipeline"

echo "Configuration:"
echo "  Profile:              $PROFILE"
echo "  Phase 1 Convs:        $CONV_ROUNDS_1 rounds"
echo "  Phase 2 Books:        $BOOKS books"
echo "  Phase 3 Convs:        $CONV_ROUNDS_2 rounds"
echo "  Turns per topic:      $TURNS_PER_TOPIC"
echo "  Skip tests:           $SKIP_TESTS"
echo "  Background mode:      $BACKGROUND"
echo ""
echo "Estimates:"
echo "  Duration:             ~$EST_TIME"
echo "  Facts to learn:       ~$TOTAL_FACTS_EST"
echo ""
echo "Paths:"
echo "  Project root:         $PROJECT_ROOT"
echo "  Persist directory:    $PERSIST_DIR"
echo "  Main log:             $MAIN_LOG"
echo ""

# ============================================================================
# DRY RUN MODE
# ============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Commands that would be executed:"
    echo ""
    if [ "$SKIP_TESTS" = false ]; then
        echo "  pytest tests/reasoning/test_analogy_engine.py -v"
    fi
    echo "  uv run python scripts/crew_trainer.py --max-rounds $CONV_ROUNDS_1 --turns-per-topic $TURNS_PER_TOPIC"
    if [ "$BOOKS" -gt 0 ]; then
        echo "  uv run python scripts/ingest_gutenberg.py --max-books $BOOKS --chunk-size 1000"
    fi
    if [ "$CONV_ROUNDS_2" -gt 0 ]; then
        echo "  uv run python scripts/crew_trainer.py --max-rounds $CONV_ROUNDS_2 --turns-per-topic $TURNS_PER_TOPIC"
    fi
    exit 0
fi

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

run_training() {
    # Initialize log file
    echo "Kent Hologram Training Log" > "$MAIN_LOG"
    echo "Started: $(get_timestamp)" >> "$MAIN_LOG"
    echo "Profile: $PROFILE" >> "$MAIN_LOG"
    echo "========================================" >> "$MAIN_LOG"

    # Track start time
    START_TIME=$(date +%s)

    # ========================================================================
    # PHASE 0: Test Validation (Optional)
    # ========================================================================

    if [ "$SKIP_TESTS" = false ]; then
        print_phase "0" "Test Validation"

        print_info "Running HDC analogy tests..."
        log_to_file "$MAIN_LOG" "Phase 0: Starting test validation"

        if uv run pytest tests/reasoning/test_analogy_engine.py -v --tb=short 2>&1 | tee -a "$MAIN_LOG"; then
            print_status "All 13 analogy tests passed!"
            log_to_file "$MAIN_LOG" "Phase 0: Tests PASSED"
        else
            print_error "Some tests failed. Review output above."
            print_info "Continuing with training anyway..."
            log_to_file "$MAIN_LOG" "Phase 0: Some tests FAILED (continuing)"
        fi
    else
        print_info "Skipping test validation (--skip-tests)"
        log_to_file "$MAIN_LOG" "Phase 0: Skipped"
    fi

    # ========================================================================
    # PHASE 1: Foundation Conversations
    # ========================================================================

    print_phase "1" "Foundation Conversations ($CONV_ROUNDS_1 rounds)"

    print_info "Starting conversational training..."
    print_info "This teaches dialogue patterns and basic facts"
    log_to_file "$MAIN_LOG" "Phase 1: Starting $CONV_ROUNDS_1 conversation rounds"

    PHASE1_LOG="$LOG_DIR/phase1_conversations_${TIMESTAMP}.log"

    if PYTHONUNBUFFERED=1 uv run python scripts/crew_trainer.py \
        --max-rounds "$CONV_ROUNDS_1" \
        --turns-per-topic "$TURNS_PER_TOPIC" \
        --persist-dir "$PERSIST_DIR" \
        2>&1 | tee "$PHASE1_LOG"; then
        print_status "Phase 1 complete!"
        log_to_file "$MAIN_LOG" "Phase 1: COMPLETED"
    else
        print_error "Phase 1 encountered errors (check $PHASE1_LOG)"
        log_to_file "$MAIN_LOG" "Phase 1: ERRORS (see $PHASE1_LOG)"
    fi

    # Brief pause between phases
    echo ""
    print_info "Pausing 5 seconds before next phase..."
    sleep 5

    # ========================================================================
    # PHASE 2: Book Ingestion
    # ========================================================================

    if [ "$BOOKS" -gt 0 ]; then
        print_phase "2" "Book Ingestion ($BOOKS books from Project Gutenberg)"

        print_info "Starting book ingestion..."
        print_info "This provides deep factual knowledge from literature"
        log_to_file "$MAIN_LOG" "Phase 2: Starting $BOOKS book ingestion"

        PHASE2_LOG="$LOG_DIR/phase2_books_${TIMESTAMP}.log"

        if PYTHONUNBUFFERED=1 uv run python scripts/ingest_gutenberg.py \
            --max-books "$BOOKS" \
            --chunk-size 1000 \
            --persist-dir "$PERSIST_DIR" \
            --checkpoint-file "$CHECKPOINT_FILE" \
            2>&1 | tee "$PHASE2_LOG"; then
            print_status "Phase 2 complete!"
            log_to_file "$MAIN_LOG" "Phase 2: COMPLETED"
        else
            print_error "Phase 2 encountered errors (check $PHASE2_LOG)"
            log_to_file "$MAIN_LOG" "Phase 2: ERRORS (see $PHASE2_LOG)"
        fi

        # Brief pause between phases
        echo ""
        print_info "Pausing 5 seconds before next phase..."
        sleep 5
    else
        print_info "Skipping Phase 2 (no books configured)"
        log_to_file "$MAIN_LOG" "Phase 2: Skipped (0 books)"
    fi

    # ========================================================================
    # PHASE 3: Fine-tuning Conversations
    # ========================================================================

    if [ "$CONV_ROUNDS_2" -gt 0 ]; then
        print_phase "3" "Fine-tuning Conversations ($CONV_ROUNDS_2 rounds)"

        print_info "Starting fine-tuning conversations..."
        print_info "This reinforces and connects learned knowledge"
        log_to_file "$MAIN_LOG" "Phase 3: Starting $CONV_ROUNDS_2 fine-tuning rounds"

        PHASE3_LOG="$LOG_DIR/phase3_finetuning_${TIMESTAMP}.log"

        if PYTHONUNBUFFERED=1 uv run python scripts/crew_trainer.py \
            --max-rounds "$CONV_ROUNDS_2" \
            --turns-per-topic "$TURNS_PER_TOPIC" \
            --persist-dir "$PERSIST_DIR" \
            2>&1 | tee "$PHASE3_LOG"; then
            print_status "Phase 3 complete!"
            log_to_file "$MAIN_LOG" "Phase 3: COMPLETED"
        else
            print_error "Phase 3 encountered errors (check $PHASE3_LOG)"
            log_to_file "$MAIN_LOG" "Phase 3: ERRORS (see $PHASE3_LOG)"
        fi
    else
        print_info "Skipping Phase 3 (no fine-tuning configured)"
        log_to_file "$MAIN_LOG" "Phase 3: Skipped (0 rounds)"
    fi

    # ========================================================================
    # COMPLETION
    # ========================================================================

    END_TIME=$(date +%s)
    DURATION=$(( END_TIME - START_TIME ))
    DURATION_MIN=$(( DURATION / 60 ))
    DURATION_SEC=$(( DURATION % 60 ))

    print_header "Training Complete!"

    echo "Summary:"
    echo "  Profile:              $PROFILE"
    echo "  Duration:             ${DURATION_MIN}m ${DURATION_SEC}s"
    echo "  Conversation rounds:  $(( CONV_ROUNDS_1 + CONV_ROUNDS_2 ))"
    echo "  Books processed:      $BOOKS"
    echo ""
    echo "Log files:"
    echo "  Main log:             $MAIN_LOG"
    [ -f "$PHASE1_LOG" ] && echo "  Phase 1 log:          $PHASE1_LOG"
    [ -f "$PHASE2_LOG" ] && echo "  Phase 2 log:          $PHASE2_LOG"
    [ -f "$PHASE3_LOG" ] && echo "  Phase 3 log:          $PHASE3_LOG"
    echo ""
    echo "Next steps:"
    echo "  1. Test your Hologram: uv run python -m hologram.conversation.chatbot"
    echo "  2. Run validation:     pytest tests/reasoning/test_analogy_engine.py -v"
    echo "  3. Check fact store:   ls -la $PERSIST_DIR"
    echo ""

    log_to_file "$MAIN_LOG" "========================================"
    log_to_file "$MAIN_LOG" "Training completed: $(get_timestamp)"
    log_to_file "$MAIN_LOG" "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
}

# ============================================================================
# EXECUTE TRAINING
# ============================================================================

if [ "$BACKGROUND" = true ]; then
    echo "Starting training in background..."
    echo "Output logged to: $MAIN_LOG"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f $MAIN_LOG"
    echo ""
    echo "To stop training:"
    echo "  pkill -f crew_trainer.py"
    echo "  pkill -f ingest_gutenberg.py"
    echo ""

    # Export variables for background process
    export MAIN_LOG LOG_DIR TIMESTAMP CONV_ROUNDS_1 BOOKS CONV_ROUNDS_2
    export TURNS_PER_TOPIC SKIP_TESTS PERSIST_DIR CHECKPOINT_FILE PROFILE

    # Run in background with nohup
    nohup bash -c '
        cd "'"$PROJECT_ROOT"'"
        source "'"$0"'"
        run_training
    ' > "$MAIN_LOG" 2>&1 &

    echo "Training started! (PID: $!)"
else
    # Run in foreground
    echo "Running in foreground mode..."
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    run_training
fi
