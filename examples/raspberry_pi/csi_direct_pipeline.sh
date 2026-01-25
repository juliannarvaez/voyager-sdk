#!/bin/bash
# CSI Camera to V4L2 Virtual Device for Raspberry Pi
# Streams CSI camera to a virtual v4l2 device for docker container consumption
#
# Usage: ./csi_stream.sh <width> <height> <fps> [device_number]
# Example: ./csi_stream.sh 1280 720 30
# Example: ./csi_stream.sh 1280 720 30 10

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <width> <height> <fps> [device_number]"
    echo "Example: $0 1280 720 30"
    echo ""
    echo "Streams CSI camera to virtual v4l2 device /dev/video10 (default)"
    echo "Docker containers can read with: usb:10"
    echo ""
    echo "Requires v4l2loopback kernel module. Install and load with:"
    echo "  sudo apt-get install v4l2loopback-dkms"
    echo "  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"CSI Camera\" exclusive_caps=1"
    exit 1
fi

WIDTH=$1
HEIGHT=$2
FPS=$3
DEVICE_NR=${4:-10}

echo "Starting CSI camera stream..."
echo "Resolution: ${WIDTH}x${HEIGHT} @ ${FPS}fps"
echo "Virtual device: /dev/video${DEVICE_NR}"
echo ""
echo "Docker containers can consume this with: usb:${DEVICE_NR}"
echo ""

# Stream CSI camera to v4l2loopback device using rpicam-vid
# Using YUY2 format (optimized for low latency)
rpicam-vid \
  --width ${WIDTH} \
  --height ${HEIGHT} \
  --framerate ${FPS} \
  --nopreview \
  --codec yuv420 \
  --timeout 0 \
  --inline \
  --flush \
  --hflip \
  --awb auto \
  --ev 2.5 \
  --brightness 0.1 \
  --contrast 1.1 \
  -o - | \
gst-launch-1.0 -e \
  fdsrc do-timestamp=true ! \
  rawvideoparse width=${WIDTH} height=${HEIGHT} format=i420 framerate=${FPS}/1 ! \
  videoconvert ! \
  video/x-raw,format=YUY2 ! \
  v4l2sink device=/dev/video${DEVICE_NR} sync=false max-lateness=-1 qos=false
