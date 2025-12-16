#!/bin/bash
# Run ARC benchmark after training completes

echo "=========================================="
echo "ARC Honest Benchmark"
echo "=========================================="
echo ""

# Check which persist directory to use
if [ -d "./data/arc_training" ]; then
    PERSIST_DIR="./data/arc_training"
    echo "Using ARC training directory: $PERSIST_DIR"
elif [ -d "./data/crew_training_facts" ]; then
    PERSIST_DIR="./data/crew_training_facts"
    echo "Using default training directory: $PERSIST_DIR"
else
    echo "ERROR: No training directory found!"
    echo "Please run training first."
    exit 1
fi

# Check if neural memory exists
if [ ! -f "$PERSIST_DIR/neural_memory.pt" ]; then
    echo "WARNING: No neural memory found at $PERSIST_DIR/neural_memory.pt"
    echo "The benchmark will use a fresh, untrained solver."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Running benchmark on evaluation set..."
echo ""

# Run the honest benchmark
python scripts/arc_benchmark_honest.py \
  --split evaluation \
  --limit 100 \
  --verbose

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo ""
echo "Results saved to: arc_benchmark_results.json"
