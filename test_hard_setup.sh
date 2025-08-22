#!/bin/bash

# Test script to verify the hard execution setup
# This will run Stage 1 immediately for testing

echo "=== TESTING HARD EXECUTION SETUP ==="
echo "This will kill the current 'hard' screen and start 'hard1' immediately for testing..."
echo ""

read -p "Do you want to proceed with the test? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Test cancelled."
    exit 0
fi

echo "Running test..."
/home/nguyenchiphong2909/powerpeak/SAML3P/run_hard1.sh

echo ""
echo "Test completed! Check status with:"
echo "./check_hard_status.sh"
echo ""
echo "To attach to the test session:"
echo "screen -r hard1"
