#!/bin/bash
# entrypoint.sh
# Entry point script for the Hailo AI inference container.
# Initializes the environment and starts the main application in webcam or RTSP mode.

set -e  # Exit on any error

# Accept mode parameter (webcam or rtsp)
MODE="$1"

# Check mode and start the application accordingly
if [ -z "$MODE" ] || [ "$MODE" == "webcam" ]; then
    echo "Starting app with webcam mode (default)..."
    echo "Using WEBCAM_INDEX: $WEBCAM_INDEX"
    exec python3 /local/workspace/main.py --webcam
elif [ "$MODE" == "rtsp" ]; then
    echo "Starting app with RTSP mode..."
    echo "Using RTSP_URL: $RTSP_URL"
    exec python3 /local/workspace/main.py
else
    echo "Usage: $0 [webcam|rtsp]"
    exit 1
fi

# Store the application process ID
app_pid=$!

# Tail the log file and stream its content to stdout for monitoring
tail -n 0 -F /var/log/hailo.log >&1 &
tail_pid=$!

# Trap SIGTERM/SIGINT to ensure clean shutdown
trap 'echo "Received signal, terminating..."; kill $app_pid $tail_pid; wait $app_pid; exit 0' SIGTERM SIGINT

# Wait for the application process to exit, forward signals
wait $app_pid

# Ensure tail process is terminated
if ps -p $tail_pid > /dev/null; then
    kill $tail_pid
    wait $tail_pid 2>/dev/null || true
fi