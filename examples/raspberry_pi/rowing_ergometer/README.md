# Rowing Ergometer Keypoint Recording

This module integrates your cameraerg keypoint collection, logging, and analysis patterns into the Axelera Voyager SDK's `inference.py` workflow.

## Key Features

Based on your [cameraerg](https://github.com/juliannarvaez/cameraerg) implementation:

### 1. Circular Buffering
- Maintains last 30 frames in memory before events
- Uses `deque(maxlen=30)` for O(1) operations
- Automatically captures pre-event context

### 2. Phase-Based Event Detection
- Stroke phases: IDLE (0), WAIT_ACCEL (1), **DRIVE (2)**, DWELLING (3), RECOVERY (4)
- Triggers on DRIVE phase (phase 2)
- Collects: 30 pre-frames + drive frames + 30 post-frames

### 3. Async File Saving
- Background thread handles gzip compression
- Non-blocking saves prevent frame drops
- Gzipped JSON format: `stroke_<timestamp>.json.gz`

### 4. Data Format
```json
{
  "timestamp": 1706140123,
  "frame_count": 120,
  "keypoints": [
    [
      {"name": "right_shoulder", "x": 450, "y": 280, "confidence": 0.95, "phase": 2},
      {"name": "right_hip", "x": 460, "y": 420, "confidence": 0.92, "phase": 2},
      {"name": "right_knee", "x": 380, "y": 560, "confidence": 0.89, "phase": 2},
      {"name": "right_ankle", "x": 320, "y": 680, "confidence": 0.87, "phase": 2},
      {"name": "right_wrist", "x": 350, "y": 300, "confidence": 0.93, "phase": 2}
    ],
    ...
  ],
  "velocities": [...],  // Future: Kalman filter velocities from C++
  "phases": [2, 2, 2, 3, 3, 4, 4, ...]
}
```

### 5. Rowing-Specific Keypoints
Tracks 5 COCO keypoints essential for rowing biomechanics:
- `right_shoulder` (idx 6): Shoulder angle, body lean
- `right_hip` (idx 12): Hip angle, core engagement
- `right_knee` (idx 14): Leg drive extension
- `right_ankle` (idx 16): Foot placement, ankle dorsiflexion
- `right_wrist` (idx 10): Handle trajectory, catch/finish positions

## Usage

### Basic Recording
```bash
# Enable keypoint recording with defaults
python inference.py --network yolov8n-pose-coco --record-keypoints

# Custom buffer size and save location
python inference.py \
  --network yolov8n-pose-coco \
  --record-keypoints \
  --keypoint-buffer-size 60 \
  --keypoint-save-dir /home/julian/rowing_data
```

### Manual Phase Control
```python
from rowing_ergometer.keypoint_recorder import KeypointRecorder
from rowing_ergometer.phase_controller import PhaseController

# Initialize recorder
recorder = KeypointRecorder(buffer_size=30, save_dir="/tmp/stroke_data")

# Manual control (keyboard)
controller = PhaseController(recorder, auto_detect=False)
controller.start_keyboard_control()

# Keyboard: Press 0-4 to set phase, 's' to save, 'q' to quit
```

### External Sensor Integration (Concept2 PM5)
```python
from rowing_ergometer.phase_controller import PhaseController
import pyrow

# Initialize ergometer connection
erg = pyrow.PyErg(pyrow.find()[0])

# Set phase callback
controller = PhaseController(recorder)
controller.set_external_phase_callback(
    lambda: erg.get_forceplot().get('strokestate', 0)
)

# Update phase in main loop
def main_loop():
    for frame in stream:
        controller.update_phase_from_external()
        # ... process frame
```

## Integration with Your Cameraerg Workflow

### Data Analysis
Use your existing `plotter.py` to analyze recorded data:

```bash
# Your plotter expects stroke_data/ folder with stroke_*.json.gz files
cp /tmp/stroke_data/*.json.gz ~/path/to/cameraerg/hailo_apps/apps/pose_estimation/stroke_data/
cd ~/path/to/cameraerg/hailo_apps/apps/pose_estimation/
python plotter.py
```

### Velocity Data
Your cameraerg uses Kalman-filtered velocities from C++ postprocessing. To add this:

1. **Option A**: Extract from Axelera C++ pipeline (requires SDK modification)
2. **Option B**: Post-process Python with velocity calculation:
```python
# Add to keypoint_recorder.py
def calculate_velocity(self, prev_kp, curr_kp, dt):
    dx = curr_kp['x'] - prev_kp['x']
    dy = curr_kp['y'] - prev_kp['y']
    vx, vy = dx / dt, dy / dt
    speed = (vx**2 + vy**2)**0.5
    return {"vx": vx, "vy": vy, "speed": speed}
```

## File Structure
```
voyager-sdk-1.5.3/
├── inference.py                     # Modified with --record-keypoints flag
├── rowing_ergometer/
│   ├── __init__.py
│   ├── keypoint_recorder.py         # Circular buffer + async saving
│   ├── phase_controller.py          # Phase detection & control
│   └── README.md                    # This file
└── /tmp/stroke_data/                # Default save location
    ├── stroke_1706140123.json.gz
    ├── stroke_1706140156.json.gz
    └── ...
```

## Performance Notes

Based on your experience at ~94 FPS (Raspberry Pi CSI camera):

- **Buffer size 30**: ~330ms pre-event context @ 90 FPS
- **Async saves**: Gzip compression level 3 for speed (saves in background)
- **Deque operations**: O(1) append/popleft vs list O(n)
- **Phase caching**: Reduced lock contention (update every 10 frames)

## Comparison with Cameraerg

| Feature | Cameraerg (Hailo) | Axelera Integration |
|---------|-------------------|---------------------|
| Keypoint extraction | C++ postprocess → Python callback | Python meta extraction |
| Kalman filtering | C++ (kalman_filter.hpp) | TODO: Add C++ integration |
| Circular buffer | `deque(maxlen=30)` | ✅ Same |
| Phase detection | PM5 USB → `get_forceplot()` | Manual + optional external |
| Async saving | ✅ Background thread | ✅ Same pattern |
| Data format | Gzipped JSON | ✅ Same |
| Force data | PM5 `forceplot` | External integration needed |

## Next Steps

1. **Add velocity calculation**: Either integrate with Axelera C++ Kalman filter or calculate in Python
2. **Concept2 integration**: Connect PM5 for automatic phase detection and force data
3. **Auto phase detection**: Implement ML model or heuristics based on keypoint velocities
4. **Real-time feedback**: Add stroke quality metrics (catch angle, finish position, etc.)

## Example Workflow

```bash
# Terminal 1: Start camera stream (if using CSI camera)
cd examples/raspberry_pi
./csi_direct_pipeline.sh 640 480 90 10

# Terminal 2: Run inference with keypoint recording
python inference.py \
  --network yolov8n-pose-coco \
  --source /dev/video10 \
  --record-keypoints \
  --keypoint-buffer-size 30 \
  --keypoint-save-dir /tmp/stroke_data

# Manual phase control via keyboard:
# - Press '2' when drive phase starts (pulling)
# - Press '3'/'4' during dwelling/recovery
# - Press 's' to manually save current buffer
# - Press 'q' to quit keyboard control

# Terminal 3: Monitor recorded data
ls -lh /tmp/stroke_data/
# stroke_1706140123.json.gz
# stroke_1706140156.json.gz

# Extract and analyze
gunzip -c /tmp/stroke_data/stroke_1706140123.json.gz | python -m json.tool | head -50
```

## Troubleshooting

### No keypoints saved
- Check that pose detection model is loaded (`--network yolov8n-pose-coco`)
- Verify person is in frame and detectable
- Check phase transitions (must go through phase 2 = DRIVE)
- Look for errors in logs

### Empty detections
- Ensure proper meta extraction in `extract_keypoints_from_meta()`
- Verify AxMeta structure with debug logging
- Check keypoint confidence thresholds

### Performance issues
- Reduce buffer size if memory constrained
- Check async save queue depth (`get_stats()`)
- Verify background save thread is running

## License
Copyright Axelera AI, 2025
