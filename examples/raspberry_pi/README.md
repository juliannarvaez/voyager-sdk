# Raspberry Pi CSI Camera Integration

Stream CSI camera to Voyager SDK running in Docker using v4l2loopback.

## Prerequisites

- Raspberry Pi with CSI camera connected
- Docker container running Voyager SDK
- v4l2loopback kernel module

## One-Time Setup

Install v4l2loopback (if not already installed):
```bash
sudo apt-get install v4l2loopback-dkms v4l2loopback-utils
```

Load the kernel module to create virtual video device:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="CSI Camera"
```

To load automatically on boot, add to `/etc/modules-load.d/v4l2loopback.conf`:
```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
```

And configure in `/etc/modprobe.d/v4l2loopback.conf`:
```bash
echo "options v4l2loopback devices=1 video_nr=10 card_label=\"CSI Camera\"" | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

## Usage

### Terminal 1: Start CSI Camera Stream (on host)

```bash
cd ~/axeleras/voyager-sdk-1.4.2
./examples/raspberry_pi/csi_direct_pipeline.sh 1280 720 30
```

Parameters:
- Width: 1280 pixels
- Height: 720 pixels  
- FPS: 30 frames per second
- Device (optional): 10 (creates /dev/video10)

**Wait for:** `Setting pipeline to PLAYING ...`

Leave this terminal running.

### Terminal 2: Run Inference (in Docker)

```bash
docker exec -it voyager-sdk-1.4.2 /bin/bash
cd voyager-sdk
source venv/bin/activate

AXELERA_CONFIGURE_BOARD=,20 ./inference.py yolov8spose-coco usb:10/yuyv --show-host-fps --show-device-fps
```

Replace `yolov8spose-coco` with your deployed model.

**Important:** Use `usb:10/yuyv` (not just `usb:10`) to specify YUY2 format.

## Supported Resolutions

Your IMX296 camera supports:
- 1456x1088 @ 60fps (native)
- 1280x720 @ 60fps
- 640x480 @ 60fps

Adjust the script parameters based on your needs.

## Troubleshooting

### "Cannot access device at 10"
- Ensure streaming script is running in Terminal 1
- Check `/dev/video10` exists: `ls -l /dev/video10`
- Restart docker container: `docker restart voyager-sdk-1.4.2`

### "not-negotiated" error
- Make sure to use `usb:10/yuyv` (not `usb:10`)
- Verify streaming script shows "Setting pipeline to PLAYING"

### Camera not detected
```bash
rpicam-hello --list-cameras
```

### Stop streaming
Press Ctrl+C in Terminal 1

## How It Works

1. `rpicam-vid` captures from CSI camera in YUV420 format
2. GStreamer converts to YUY2 and streams to v4l2loopback device
3. v4l2loopback creates `/dev/video10` virtual camera
4. Docker container reads from `/dev/video10` like a USB camera
5. Voyager SDK processes video with minimal latency
