#!/bin/bash

# Script to run SAMP3P_cadical.py in a screen session
# Created on 2025-08-18

# Change to the script directory
cd /home/nguyenchiphong2909/powerpeak/SAML3P

# Create a new screen session named 'samp3p_run' and run the Python script
screen -dmS samp3p_run bash -c "
    echo 'Starting SAMP3P_cadical.py at $(date)'
    echo 'Log file: /home/nguyenchiphong2909/powerpeak/SAML3P/samp3p_execution.log'
    echo '===========================================' 
    python3 SAMP3P_cadical.py 2>&1 | tee samp3p_execution.log
    echo '==========================================='
    echo 'Script completed at $(date)'
    echo 'Press any key to exit screen session...'
    read -n 1
"

# Log that the script was started
echo "$(date): SAMP3P_cadical.py started in screen session 'samp3p_run'" >> scheduled_runs.log
