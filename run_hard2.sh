#!/bin/bash

# Stage 2: Kill 'hard1' screen, run hard/2.py in 'hard2' screen
# Updated on 2025-08-19 at 18:13

echo "$(date): Stage 2 - Starting hard/2.py execution" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log

# Kill existing 'hard1' screen session if it exists
if screen -list | grep -q "\.hard1\s"; then
    echo "$(date): Killing 'hard1' screen session" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
    screen -S hard1 -X quit
    sleep 2
fi

# Change to the script directory
cd /home/nguyenchiphong2909/powerpeak/SAML3P

# Create new screen session 'hard2' and run hard/2.py
screen -dmS hard2 bash -c "
    echo 'Starting hard/2.py at $(date)'
    echo 'Log file: /home/nguyenchiphong2909/powerpeak/SAML3P/hard2_execution.log'
    echo '==========================================='
    python3 hard/2.py 2>&1 | tee hard2_execution.log
    echo '==========================================='
    echo 'hard/2.py completed at $(date)'
    echo 'This screen will be terminated in stage 3...'
    sleep 3600  # Keep screen alive for 1 hour
"

echo "$(date): hard/2.py started in screen session 'hard2'" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
