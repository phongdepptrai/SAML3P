#!/bin/bash

# Stage 3: Kill 'hard2' screen, run hard/3.py in 'hard3' screen
# Updated on 2025-08-19 at 18:13

echo "$(date): Stage 3 - Starting hard/3.py execution" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log

# Kill existing 'hard2' screen session if it exists
if screen -list | grep -q "\.hard2\s"; then
    echo "$(date): Killing 'hard2' screen session" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
    screen -S hard2 -X quit
    sleep 2
fi

# Change to the script directory
cd /home/nguyenchiphong2909/powerpeak/SAML3P

# Create new screen session 'hard3' and run hard/3.py
screen -dmS hard3 bash -c "
    echo 'Starting hard/3.py at $(date)'
    echo 'Log file: /home/nguyenchiphong2909/powerpeak/SAML3P/hard3_execution.log'
    echo '==========================================='
    python3 hard/3.py 2>&1 | tee hard3_execution.log
    echo '==========================================='
    echo 'hard/3.py completed at $(date)'
    echo 'All hard scripts completed! Press any key to exit...'
    read -n 1
"

echo "$(date): hard/3.py started in screen session 'hard3'" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
echo "$(date): All hard script executions completed!" >> /home/nguyenchiphong2909/powerpeak/SAML3P/hard_execution.log
