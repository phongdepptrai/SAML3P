#!/bin/bash

# Helper script to monitor hard script executions
# Usage: ./check_hard_status.sh

echo "=== HARD SCRIPTS EXECUTION STATUS ==="
echo "Current time: $(date)"
echo ""

# Check execution log
if [ -f "hard_execution.log" ]; then
    echo "📋 Execution Timeline:"
    echo "----------------------------------------"
    cat hard_execution.log
    echo "----------------------------------------"
    echo ""
fi

# Check for active hard screen sessions
echo "🖥️  Active Hard Screen Sessions:"
echo "----------------------------------------"
active_sessions=$(screen -list | grep -E "hard[0-9]*" || echo "No hard screen sessions found")
echo "$active_sessions"
echo "----------------------------------------"
echo ""

# Check individual log files and their status
for stage in 1 2 3; do
    session_name="hard$stage"
    log_file="hard${stage}_execution.log"
    
    echo "📊 STAGE $stage STATUS (hard/$stage.py in screen '$session_name'):"
    
    if screen -list | grep -q "$session_name"; then
        echo "   ✅ Screen session '$session_name' is RUNNING"
        echo "   📂 To attach: screen -r $session_name"
    else
        echo "   ❌ Screen session '$session_name' is NOT running"
    fi
    
    if [ -f "$log_file" ]; then
        echo "   📋 Log file exists: $log_file"
        echo "   📝 Last 3 lines of log:"
        tail -3 "$log_file" | sed 's/^/      /'
    else
        echo "   📋 No log file yet: $log_file"
    fi
    echo ""
done

echo "🕐 SCHEDULED TIMES:"
echo "   Stage 1 (hard/1.py): 19:13 today"
echo "   Stage 2 (hard/2.py): 20:13 today" 
echo "   Stage 3 (hard/3.py): 21:13 today"
echo ""

echo "🔧 USEFUL COMMANDS:"
echo "   Monitor all logs: tail -f hard*_execution.log"
echo "   Check main log: tail -f hard_execution.log"
echo "   List screens: screen -list"
echo "   Kill a screen: screen -S <name> -X quit"
