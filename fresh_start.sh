#!/bin/bash

# Fresh start script - Clean up old sessions and prepare for new run
# Created on 2025-08-19 at 18:16

echo "=== FRESH START - CLEANING UP OLD SESSIONS ==="
echo "Current time: $(date)"

# Kill any existing hard-related screen sessions
echo "Cleaning up old screen sessions..."
for session in hard hard1 hard2 hard3; do
    if screen -list | grep -q "\.$session\s"; then
        echo "Killing screen session: $session"
        screen -S $session -X quit
        sleep 1
    fi
done

# Archive old log files
echo "Archiving old log files..."
timestamp=$(date +"%Y%m%d_%H%M%S")
if [ -f "hard_execution.log" ]; then
    cp hard_execution.log "hard_execution_backup_$timestamp.log"
fi

for stage in 1 2 3; do
    if [ -f "hard${stage}_execution.log" ]; then
        cp "hard${stage}_execution.log" "hard${stage}_execution_backup_$timestamp.log"
    fi
done

# Clear current logs (but keep backups)
echo "=== FRESH START AT $(date) ===" > hard_execution.log
> hard1_execution.log
> hard2_execution.log  
> hard3_execution.log

echo ""
echo "✅ Clean up completed!"
echo "📅 New execution schedule:"
echo "   Stage 1 (hard/1.py): 19:13 today"
echo "   Stage 2 (hard/2.py): 20:13 today"
echo "   Stage 3 (hard/3.py): 21:13 today"
echo ""
echo "🔍 Monitor with: ./check_hard_status.sh"
echo "📋 Current screen sessions:"
screen -list
