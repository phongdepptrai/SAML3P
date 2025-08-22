#!/bin/bash

# Stage 1: Kill 'hard' screen, run hard/1.py in 'hard1' screen
# Updated on 2025-08-19 at 18:13

echo "$(date): Stage 1 - Starting hard/1.py execution" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log

# Kill existing 'hard' screen session if it exists
if screen -list | grep -q "\.hard\s"; then
    echo "$(date): Killing existing 'hard' screen session" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
    screen -S hard -X quit
    sleep 2
fi

# Change to the script directory
cd /home/nguyenchiphong2909/powerpeak/SAML3P

# Create new screen session 'hard1' and run hard/1.py
screen -dmS hard1 bash -c "
    echo 'Starting hard/1.py at $(date)'
    echo 'Log file: /home/nguyenchiphong2909/powerpeak/SAML3P/hard1_execution.log'
    echo '==========================================='
    python3 hard/1.py 2>&1 | tee hard1_execution.log
    echo '==========================================='
    echo 'hard/1.py completed at $(date)'
    echo 'This screen will be terminated in stage 2...'
    sleep 3600  # Keep screen alive for 1 hour
"

echo "$(date): hard/1.py started in screen session 'hard1'" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
