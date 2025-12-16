#!/bin/bash
# Run all three trainers overnight in parallel
# Each gets its own persist directory to avoid race conditions

set -e  # Exit on error

echo "=========================================="
echo "Kent Hologram - Overnight Training Suite"
echo "=========================================="
echo ""
echo "Starting three trainers in parallel:"
echo "  1. ARC Trainer    -> ./data/arc_training"
echo "  2. Crew Trainer   -> ./data/crew_training"
echo "  3. Code Trainer   -> ./data/code_training"
echo ""
echo "Logs will be saved to ./overnight_logs/"
echo ""

# Create log directory
mkdir -p ./overnight_logs

# Create persist directories
mkdir -p ./data/arc_training
mkdir -p ./data/crew_training
mkdir -p ./data/code_training

# Start ARC Trainer in background
echo "[$(date '+%H:%M:%S')] Starting ARC Trainer..."
nohup python scripts/arc_trainer.py \
  --persist-dir ./data/arc_training \
  --max-rounds 1000 \
  --validate-every 20 \
  --log-dir ./arc_training_logs \
  > ./overnight_logs/arc_trainer.log 2>&1 &
ARC_PID=$!
echo "  PID: $ARC_PID"

# Start Crew Trainer in background
echo "[$(date '+%H:%M:%S')] Starting Crew Trainer..."
nohup python scripts/crew_trainer.py \
  --persist-dir ./data/crew_training \
  --max-rounds 1000 \
  --turns-per-topic 8 \
  --log-dir ./conversation_logs \
  > ./overnight_logs/crew_trainer.log 2>&1 &
CREW_PID=$!
echo "  PID: $CREW_PID"

# Start Code Trainer in background
echo "[$(date '+%H:%M:%S')] Starting Code Trainer..."
nohup python scripts/code_trainer.py \
  --persist-dir ./data/code_training \
  --max-rounds 1000 \
  --validate-every 20 \
  --log-dir ./code_training_logs \
  > ./overnight_logs/code_trainer.log 2>&1 &
CODE_PID=$!
echo "  PID: $CODE_PID"

echo ""
echo "=========================================="
echo "All trainers started!"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  ARC Trainer:  $ARC_PID"
echo "  Crew Trainer: $CREW_PID"
echo "  Code Trainer: $CODE_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f ./overnight_logs/arc_trainer.log"
echo "  tail -f ./overnight_logs/crew_trainer.log"
echo "  tail -f ./overnight_logs/code_trainer.log"
echo ""
echo "Stop all trainers with:"
echo "  kill $ARC_PID $CREW_PID $CODE_PID"
echo ""
echo "Or save PIDs to file for later:"
echo "$ARC_PID $CREW_PID $CODE_PID" > ./overnight_logs/trainer_pids.txt
echo "  Saved to ./overnight_logs/trainer_pids.txt"
echo ""
echo "To stop later: kill \$(cat ./overnight_logs/trainer_pids.txt)"
echo ""

# Wait for all trainers to complete (optional)
# Uncomment if you want the script to wait
# wait $ARC_PID $CREW_PID $CODE_PID
# echo "All trainers completed!"
