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

**Configure auto-load on boot** (recommended - survives reboots):
```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo "options v4l2loopback devices=1 video_nr=10 card_label=\"CSI Camera\"" | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="CSI Camera"
```

**OR** load manually (temporary - lost after reboot):
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="CSI Camera"
```

## Usage

### Terminal 1: Start CSI Camera Stream (on host)

```bash
cd ~/axeleras/voyager-sdk-1.4.2
./examples/raspberry_pi/csi_direct_pipeline.sh 640 640 120
```

Parameters:
- Width: 640 pixels
- Height: 640 pixels  
- FPS: 120 frames per second
- Device (optional): 10 (creates /dev/video10)

**Note:** Use 640x640 for pose estimation models (yolov8pose) to avoid keypoint misalignment. Use 640x480 for detection models.

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

## Recommended Resolutions

For best results, use standard aspect ratios:
- **640x480 @ 120fps** - **Recommended** (4:3, high frame rate for smooth tracking)
- **1280x720 @ 60fps** - Higher resolution (16:9, ~39fps end-to-end, 552ms latency)
- **640x640 @ 60fps** - Square format (optimal for some YOLO models)

**Note:** Avoid using the camera's native 1456x1088 resolution as the non-standard aspect ratio may cause issues with model processing.

Adjust the script parameters based on your needs and model requirements.

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

## Performance Optimization: NEON-Accelerated Normalization

To improve preprocessing performance (~2x faster normalization), rebuild the operators with SIMDE/NEON optimizations:

```bash
# Enter Docker container
docker exec -it voyager-sdk-1.4.2 /bin/bash

# Navigate to operators and build with NEON
cd /voyager-sdk/operators
rm -rf Build-NEON
PKG_CONFIG_PATH=$AXELERA_RUNTIME_DIR/lib/pkgconfig cmake -B Build-NEON -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_CXX_FLAGS="-O3 -DSIMDE_ENABLE_NATIVE_ALIASES -D__AVX2__" \
  -DCMAKE_C_FLAGS="-O3 -DSIMDE_ENABLE_NATIVE_ALIASES -D__AVX2__"

# Build the normalize library
cd Build-NEON
ninja -j2 libinplace_normalize.so

# Install to lib directory
cp libinplace_normalize.so ../lib/libinplace_normalize.so

# Verify
ls -lh ../lib/libinplace_normalize.so

# Exit container
exit

# Restart Docker to reload libraries
docker restart voyager-sdk-1.4.2
```

This enables ARM NEON SIMD instructions via SIMDE library, significantly improving normalization performance (from ~75fps to 150-200fps potential).

## How It Works

1. `rpicam-vid` captures from CSI camera in YUV420 format
2. GStreamer converts to YUY2 and streams to v4l2loopback device
3. v4l2loopback creates `/dev/video10` virtual camera
4. Docker container reads from `/dev/video10` like a USB camera
5. Voyager SDK processes video with minimal latency
