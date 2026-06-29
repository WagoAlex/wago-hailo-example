# mqtt_app.py
# MQTT client for subscribing to inference results published by the inference process.
# Logs received messages with timestamps, detections, and metadata for debugging and monitoring.

import paho.mqtt.client as mqtt
import json
import uuid
from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

# Generate a unique client ID for the MQTT client
short_uuid = str(uuid.uuid4())[:6]
MQTT_UUID = f"wago-hailo-inference-sender-{short_uuid}"

def on_connect(client, userdata, flags, reason_code, properties=None):
    """
    Callback for when the MQTT client connects to the broker.
    
    Args:
        client: MQTT client instance
        userdata: User data (not used)
        flags: Connection flags
        reason_code: Connection result code
        properties: MQTT v5 properties (not used)
    """
    if reason_code == 0:
        # Successfully connected, subscribe to the inference results topic
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {reason_code}")

def on_message(client, userdata, msg):
    """
    Callback for processing incoming MQTT messages with inference data.
    
    Args:
        client: MQTT client instance
        userdata: User data (not used)
        msg: Received MQTT message
    """
    try:
        print(json.dumps(json.loads(msg.payload.decode()), indent=2))  # Print full JSON (complete, unredundant)
    except json.JSONDecodeError:
        print("Error decoding MQTT message payload")

def run_mqtt():
    """Runs the MQTT client to listen for inference results."""
    # Initialize MQTT client with unique ID and modern callback API
    client = mqtt.Client(client_id=MQTT_UUID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        # Connect to the MQTT broker and start the listening loop
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Error in MQTT client: {e}")

if __name__ == "__main__":
    """Entry point for running the MQTT client directly."""
    run_mqtt()