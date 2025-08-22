#!/bin/bash

# Helper script to check on the SAMP3P script execution
# Usage: ./check_samp3p.sh

echo "=== SAMP3P Script Status Check ==="
echo "Current time: $(date)"
echo ""

# Check if screen session exists
if screen -list | grep -q "samp3p_run"; then
    echo "✅ Screen session 'samp3p_run' is ACTIVE"
    echo "To attach to the session, run: screen -r samp3p_run"
    echo ""
    
    # Show recent log entries if log file exists
    if [ -f "samp3p_execution.log" ]; then
        echo "📋 Recent log entries (last 10 lines):"
        echo "----------------------------------------"
        tail -10 samp3p_execution.log
        echo "----------------------------------------"
        echo "Full log file: samp3p_execution.log"
    fi
else
    echo "❌ Screen session 'samp3p_run' is NOT running"
    
    # Check if log file exists to see if it completed
    if [ -f "samp3p_execution.log" ]; then
        echo "📋 Script may have completed. Last 10 lines of log:"
        echo "----------------------------------------"
        tail -10 samp3p_execution.log
        echo "----------------------------------------"
    fi
fi

echo ""

# Check scheduled runs log
if [ -f "scheduled_runs.log" ]; then
    echo "📅 Scheduled run history:"
    echo "----------------------------------------"
    cat scheduled_runs.log
    echo "----------------------------------------"
fi

echo ""
echo "🔍 To monitor in real-time: tail -f samp3p_execution.log"
echo "🖥️  To attach to screen: screen -r samp3p_run"
echo "📋 To list all screens: screen -list"
