# main.py
# Orchestrates the startup and coordination of inference, MQTT, and API processes for the Hailo AI inference application.
# Ensures proper initialization, resource sharing, and graceful termination of all processes.
# Key Changes for Steady Stream:
# - Increased shared_queue maxsize to 500 for buffering during multi-consumer loads (e.g., React retries).
# - Added queue monitoring in main loop with MQTT warnings for low queue (integrates with backend logger).
# - Enhanced process restart logic to handle crashes gracefully with backoff.
# - Rationale: Prevents starvation; scalable for prod (e.g., alert on low FPS via MQTT to Prometheus/Grafana).
# Maintainability: Signal handling for clean Docker stops; processes list for easy extension.
# Security: No privileged ops; MQTT auth if needed in config.py.
from multiprocessing import Process, Queue
from config import MQTT_TOPIC, MQTT_BROKER, MQTT_PORT, RTSP_URL, QUEUE_WARN_THRESHOLD, QUEUE_DROP_THRESHOLD, QUEUE_MONITOR_INTERVAL, WARN_TOPIC, LOG_LEVEL, RTSP_URLS
import signal
import time
import inference
import mqtt_app
import api_app
import argparse
import paho.mqtt.client as mqtt
import json
import os
import sys  # Added for sys.exit
import logging

logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('/var/log/hailo_main.log')])
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application. Initializes shared resources and starts child processes."""
    # Parse command-line arguments to determine input source (webcam or RTSP)
    parser = argparse.ArgumentParser(description="Hailo AI inference application with webcam or RTSP input")
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input source instead of RTSP')
    args = parser.parse_args()
    
    # Parse multi-camera URLs (comma-separated; fallback to single RTSP_URL)
    rtsp_urls = os.environ.get("RTSP_URLS", RTSP_URL).split(",")
    num_cameras = len(rtsp_urls)
    if args.webcam:
        rtsp_urls = [None]  # Single process for webcam to avoid device conflicts
        num_cameras = 1
        logger.info("Webcam mode: Starting single inference process")
    else:
        logger.info("RTSP mode: Starting %d inference processes for cameras: %s", num_cameras, rtsp_urls)
    logger.info(f"Starting with {num_cameras} camera(s): {rtsp_urls}")
    
    # List to track all child processes for management and cleanup
    processes = []
    camera_queues = []  # Per-camera queues for independence and steady streaming
    
    # Start inference processes per camera
    for idx, url in enumerate(rtsp_urls):
        camera_queue = Queue(maxsize=5000)  # Per-camera buffer
        camera_queues.append(camera_queue)
        p_inference = Process(target=inference.run_inference_main, args=(args.webcam, camera_queue, url.strip() if not args.webcam else None))
        processes.append(p_inference)
        p_inference.start()
    
    # Start MQTT process for handling inference result subscriptions
    p_mqtt = Process(target=mqtt_app.run_mqtt)
    processes.append(p_mqtt)
    p_mqtt.start()
    
    # Start API process with list of camera queues for multi-stream support
    p_api = Process(target=api_app.run_api, args=(camera_queues,))
    processes.append(p_api)
    p_api.start()
    
    # Track inference processes separately for restart (better than name checks)
    inference_processes_indices = list(range(len(camera_queues)))  # Indices 0 to num_cameras-1 are inference
    mqtt_process_index = len(processes) - 2  # After inferences, before API (adjust if adding more)
    api_process_index = len(processes) - 1
    
    try:
        # Keep the main process alive to monitor child processes and queues
        queue_history = [[] for _ in range(num_cameras)]  # Per-camera histories
        restart_delays = [1] * len(processes)  # Per-process backoff (starts at 1s)
        MAX_BACKOFF = 30  # Cap delay
        while True:
            time.sleep(QUEUE_MONITOR_INTERVAL)
            for idx, q in enumerate(camera_queues):
                current_size = q.qsize()
                queue_history[idx].append(current_size)
                if len(queue_history[idx]) > 5:  # Average over last 5 checks
                    queue_history[idx].pop(0)
                avg_size = sum(queue_history[idx]) / len(queue_history[idx])
                if avg_size < QUEUE_WARN_THRESHOLD:
                    logger.warning(f"Average queue low ({avg_size:.1f}) for camera {idx} - Potential capture issue")
                    client = mqtt.Client(client_id=f"wago-inference-{os.getpid()}", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
                    client.username_pw_set(os.environ.get("MQTT_USER", ""), os.environ.get("MQTT_PASS", ""))  # Use env vars
                    try:
                        client.connect(MQTT_BROKER, MQTT_PORT, 60)
                        client.publish(WARN_TOPIC, json.dumps({"timestamp": time.time() * 1000, "warning": f"Low frame queue for camera {idx} - Check source", "avg_queue": avg_size}))
                    except Exception as e:
                        logger.error(f"Failed to publish queue warning: {e}")
                    finally:
                        client.disconnect()
                # Drop stale frames if queue nearing full (steady stream: prevents lag on WAGO RAM)
                while q.qsize() > QUEUE_DROP_THRESHOLD:
                    q.get()  # Drop oldest
                    logger.info(f"Dropped stale frame for camera {idx} to maintain steady stream")
            
            # Monitor and restart crashed processes
            for i, p in enumerate(processes):
                if not p.is_alive() and p.exitcode != 0:  # Ignore clean exits
                    logger.warning(f"Process {p.name} (index {i}) crashed (code {p.exitcode}) - Restarting after {restart_delays[i]}s backoff...")
                    time.sleep(restart_delays[i])
                    if i in inference_processes_indices:
                        url = rtsp_urls[i] if not args.webcam else None
                        new_p = Process(target=inference.run_inference_main, args=(args.webcam, camera_queues[i], url.strip() if url else None))
                    elif i == mqtt_process_index:
                        new_p = Process(target=mqtt_app.run_mqtt)
                    elif i == api_process_index:
                        new_p = Process(target=api_app.run_api, args=(camera_queues,))
                    else:
                        continue  # Unknown process—skip
                    processes[i] = new_p
                    new_p.start()
                    restart_delays[i] = min(restart_delays[i] * 2, MAX_BACKOFF)  # Exponential backoff
    except KeyboardInterrupt:
        # Handle graceful shutdown on interrupt (Ctrl+C)
        logger.info("Received KeyboardInterrupt, terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
    except SystemExit:
        # Handle SIGTERM (Docker stop) gracefully
        logger.info("Received SIGTERM, terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

if __name__ == "__main__":
    """Entry point for script execution."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))  # Map SIGTERM to SystemExit
    main()