# Dockerfile
FROM hailo_docker_hailort_ub2204:4.20.0
LABEL maintainer="alexander.fugmann@wago.com"

USER root
WORKDIR /local/workspace

# Install dependencies
RUN apt-get update && \
    apt-get -y install g++ libqt5gui5 qtbase5-dev xvfb \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x \
    gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio gstreamer1.0-rtsp curl v4l-utils libv4l-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /local/workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /local/workspace/requirements.txt

# Copy application files individually
COPY main.py /local/workspace/main.py
COPY api_app.py /local/workspace/api_app.py
COPY config.py /local/workspace/config.py
COPY mqtt_app.py /local/workspace/mqtt_app.py
COPY entrypoint.sh /local/workspace/entrypoint.sh
COPY hef/yolov5m-helmet-wago.hef /local/workspace/yolov5m-helmet-wago.hef
COPY hef/yolov5m-helmet.hef /local/workspace/yolov5m-helmet.hef
COPY hef/yolov5m-helmet-wago_20251014_183320.hef /local/workspace/yolov5m-helmet-wago_20251014_183320.hef
COPY onnx/yolov5-helmet.onnx /local/workspace/yolov5-helmet.onnx
COPY inference.py /local/workspace/inference.py
RUN mkdir -p /local/workspace/share && \
    convert -size 640x480 xc:gray /local/workspace/share/wago.jpeg 2>/dev/null || \
    python3 -c "import numpy as np, cv2; cv2.imwrite('/local/workspace/share/wago.jpeg', np.full((480,640,3),128,dtype=np.uint8))"

# Ensure entrypoint.sh is executable
RUN chmod +x /local/workspace/entrypoint.sh

# Set environment variables
ENV DISPLAY=":0" \
    XDG_RUNTIME_DIR="/run/user/0" \
    hailort_enable_service="yes" \
    HAILO_MONITOR="1" \
    HAILORT_LOGGER_PATH="/var/log/hailo.log"

# Set entrypoint
ENTRYPOINT ["/local/workspace/entrypoint.sh"]
