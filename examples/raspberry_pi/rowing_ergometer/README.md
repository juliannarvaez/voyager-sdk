# Rowing Ergometer Module

Python module for rowing ergometer keypoint recording and analysis. Automatically uses C++ backend for performance-critical Kalman filtering when available.

## Features

- **Circular Buffering**: Maintains keypoint history for event capture
- **Phase Detection**: Stroke phase tracking (IDLE, DRIVE, RECOVERY, etc.)
- **Concept2 PM5 Integration**: USB communication with PM5 ergometers via pyrow
- **Async File Saving**: Background gzip compression prevents frame drops
- **C++ Backend**: Zero-allocation Kalman filtering for <5ms jitter

## Module Structure

```
rowing_ergometer/
├── __init__.py           # Backend selector (C++ or Python)
├── keypoint_recorder.py  # Circular buffer + event capture (Python)
├── phase_controller.py   # Phase detection + PM5 integration
├── pyrow.py              # Concept2 PM5 USB communication
└── csafe/                # CSAFE protocol for PM5
```

## Usage

```python
# Imports automatically select C++ backend if available
from rowing_ergometer import KeypointRecorder, PhaseController, pyrow, USE_CPP

# Check which backend is active
print(f"Using C++ backend: {USE_CPP}")

# Initialize recorder
recorder = KeypointRecorder(fps=30, save_dir="/tmp/stroke_data")

# Connect to PM5 (if available)
ergs = list(pyrow.find())
if ergs:
    erg = pyrow.PyErg(ergs[0])
    controller = PhaseController(recorder)
    controller.set_external_phase_callback(
        lambda: erg.get_forceplot().get('strokestate', 0)
    )
```

## Building C++ Backend

See [../cpp/](../cpp/) for C++ source and build instructions.

## Data Format

Saved strokes use gzipped JSON:
```json
{
  "timestamp": 1706140123,
  "frame_count": 120,
  "keypoints": [
    [{"x": 450, "y": 280, "confidence": 0.95}, ...],
    ...
  ],
  "phases": [2, 2, 2, 3, 3, 4, 4, ...]
}
```

## License

Copyright Axelera AI, 2025
