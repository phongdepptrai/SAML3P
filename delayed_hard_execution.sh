#!/bin/bash

# Script to wait 1 hour, then delete 'hard' screen and run incremental_cadical_hard.py in 'hard1' screen

echo "$(date): Starting delayed execution script..."
echo "$(date): Will wait for 1 hour before executing..."

# Wait for 1 hour (3600 seconds)
sleep 3600

echo "$(date): 1 hour has passed. Proceeding with screen management..."

# Check if screen named 'hard' exists and delete it
if screen -list | grep -q "\.hard\s"; then
    echo "$(date): Found screen 'hard'. Terminating it..."
    screen -S hard -X quit
    echo "$(date): Screen 'hard' terminated."
else
    echo "$(date): No screen named 'hard' found."
fi

# Wait a moment to ensure the screen is fully terminated
sleep 2

# Create new screen named 'hard1' and run the Python script
echo "$(date): Creating new screen 'hard1' and running incremental_cadical_hard.py..."

# Navigate to the correct directory and run the script in a new screen
cd /home/nguyenchiphong2909/powerpeak/SAML3P
screen -dmS hard1 python3 incremental_cadical_hard.py

# Verify the screen was created
if screen -list | grep -q "\.hard1\s"; then
    echo "$(date): Screen 'hard1' created successfully and script is running."
    echo "$(date): You can attach to it using: screen -r hard1"
else
    echo "$(date): ERROR: Failed to create screen 'hard1'."
fi

# Write "ok" to check.txt
echo "$(date): Writing 'ok' to check.txt..."
echo "ok" > check.txt

echo "$(date): Delayed execution script completed."
