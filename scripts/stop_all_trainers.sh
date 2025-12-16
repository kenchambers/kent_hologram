#!/bin/bash
# Stop all running trainers gracefully

if [ -f ./overnight_logs/trainer_pids.txt ]; then
    echo "Stopping trainers..."
    PIDS=$(cat ./overnight_logs/trainer_pids.txt)

    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Stopping PID $PID..."
            kill -SIGINT $PID
        else
            echo "  PID $PID already stopped"
        fi
    done

    echo "Done! Trainers received stop signal."
    echo "They will save their state and exit gracefully."
else
    echo "No trainer PIDs file found at ./overnight_logs/trainer_pids.txt"
    echo "Use: ps aux | grep trainer"
    echo "Then: kill -SIGINT <PID>"
fi
