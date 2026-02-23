#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Example: Rowing Ergometer Keypoint Recording with Phase Control

"""
Rowing Ergometer Keypoint Recording Script

Records pose keypoints during rowing strokes with phase-based event capture.
Uses embedded pyrow module (hidraw-based) for Concept2 PM5 integration.
    Report ID #2: 120-byte CSAFE payload via kernel HID driver.

Usage:
    # Basic recording (requires Concept2 PM5 ergometer)
    python examples/raspberry_pi/rowing_ergometer_recording.py --network yolov8n-pose-coco --source /dev/video10
    
    # With custom buffer size
    python examples/rowing_ergometer_recording.py \
        --network yolov8n-pose-coco \
        --source /dev/video10 \
        --keypoint-buffer-size 60
    
    # Data always saved to /tmp/stroke_data/stroke_*.json.gz
"""

import os
import sys
import time
import argparse
import threading
import glob
import numpy as np
from collections import deque
import ctypes

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

from axelera.app import (
    config,
    create_inference_stream,
    display,
    inf_tracers,
    logging_utils,
    statistics,
    yaml_parser,
)

# Import Python modules (no C++ dependency)
from rowing_ergometer import KeypointRecorder, PhaseController, pyrow
from kalman_filter import KeypointSmoother  # Kalman filter smoother
from one_euro_filter import OneEuroSmoother  # One Euro filter smoother (adaptive)
from extended_kalman_filter import EKFSmoother  # Extended Kalman filter (nonlinear)

LOG = logging_utils.getLogger(__name__)


def setup_capture_thread_priority(cpu_core=3):
    """
    Pin capture thread to dedicated CPU core and increase priority.
    
    Args:
        cpu_core: CPU core to pin to (default: 3 for RPI 5's last core)
                  Use cores 2-3 for capture to leave 0-1 for USB/system tasks
    """
    try:
        # Get current thread/process ID
        tid = threading.get_native_id()
        
        # Set CPU affinity - pin to specific core
        os.sched_setaffinity(0, {cpu_core})
        LOG.info(f"Pinned capture thread to CPU core {cpu_core}")
        
        # Increase thread priority using nice value (lower = higher priority)
        # Range: -20 (highest) to 19 (lowest), default is 0
        try:
            current_nice = os.nice(0)
            os.nice(-10)  # Increase priority (requires root or CAP_SYS_NICE)
            LOG.info(f"Increased thread priority (nice: {current_nice} -> {current_nice - 10})")
        except PermissionError:
            LOG.warning("Cannot set priority (need sudo for nice < 0). Running with default priority.")
        
        # Alternative: Try to set real-time scheduling policy (requires root)
        try:
            # SCHED_FIFO = 1, priority range 1-99 (higher = higher priority)
            param = os.sched_param(50)  # Medium-high RT priority
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            LOG.info("Set real-time FIFO scheduling policy (priority 50)")
        except (PermissionError, AttributeError, OSError) as e:
            # Not available or no permission - continue with nice value
            pass
            
    except Exception as e:
        LOG.warning(f"Could not optimize thread settings: {e}. Continuing with defaults.")


class FPSBenchmark:
    """Ultra-efficient FPS monitoring with drop detection"""
    __slots__ = ['intervals', 'frame_count', 'start_time', 'drop_threshold', 'last_timestamp']
    
    def __init__(self, window_size=300, drop_threshold_ms=25.0):
        self.intervals = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = None
        self.last_timestamp = None
        self.drop_threshold = drop_threshold_ms / 1000.0
    
    def record_frame(self, timestamp):
        """Record frame arrival"""
        if self.start_time is None:
            self.start_time = timestamp
            self.last_timestamp = timestamp
            return
        
        interval = timestamp - self.last_timestamp
        self.intervals.append(interval)
        self.last_timestamp = timestamp
        self.frame_count += 1
    
    def get_stats(self):
        """Get performance statistics (optimized for ARM NEON)"""
        if not self.intervals:
            return None
        
        # Convert to numpy array (optimized path - no iteration overhead)
        # asarray() is fastest when deque contains numeric values
        intervals_arr = np.asarray(self.intervals, dtype=np.float32)
        elapsed = self.last_timestamp - self.start_time
        
        # Vectorized statistics (single pass for NEON efficiency)
        mean_interval = intervals_arr.mean()
        
        # Pre-compute to avoid redundant calculations
        if mean_interval > 0:
            avg_fps = 1.0 / mean_interval
        else:
            avg_fps = 0.0
        
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        
        # Single-pass variance calculation (more efficient than .std())
        variance = ((intervals_arr - mean_interval) ** 2).mean()
        std_interval = np.sqrt(variance)
        
        # Min/max in single pass
        min_interval = intervals_arr.min()
        max_interval = intervals_arr.max()
        
        # Vectorized threshold check (NEON-optimized)
        drops = (intervals_arr > self.drop_threshold).sum()
        drop_rate = (drops / len(intervals_arr)) * 100.0
        
        return {
            'avg_fps': avg_fps,
            'actual_fps': actual_fps,
            'jitter_ms': std_interval * 1000.0,
            'min_ms': min_interval * 1000.0,
            'min_ms': min_interval * 1000,
            'max_ms': max_interval * 1000,
            'drops': int(drops),
            'drop_rate': drop_rate,
        }


class DisplayWorker:
    """Non-blocking display worker running in separate thread"""
    
    def __init__(self, wnd):
        self.wnd = wnd
        self.latest_frame = None  # Lock-free: only latest frame matters
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="DisplayWorker")
        self.thread.start()
        LOG.info("Display worker thread started (lock-free)")
    
    def _run(self):
        """Display worker loop - runs in separate thread"""
        while self.running:
            frame_data = self.latest_frame
            
            if frame_data:
                try:
                    self.wnd.show(frame_data['image'], frame_data['meta'], frame_data['stream_id'])
                    self.latest_frame = None  # Clear after display
                except Exception as e:
                    LOG.error(f"Display error: {e}")
            else:
                time.sleep(0.001)  # 1ms sleep when idle
    
    def push_frame(self, image, meta, stream_id):
        """Push frame to display (lock-free atomic swap)"""
        self.latest_frame = {'image': image, 'meta': meta, 'stream_id': stream_id}
    
    def stop(self):
        """Stop display worker"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def create_erg_phase_callback():
    """
    Create phase callback for Concept2 PM5 ergometer with auto-reconnect.
    
    Returns:
        Callable that returns current stroke phase (0-4)
    """
    erg = [None]  # Mutable container for ergometer instance
    last_error_log = [0.0]  # Rate-limit error logging
    consecutive_errors = [0]  # Track error count
    
    def connect_ergometer():
        """Attempt to connect/reconnect to ergometer with retry"""
        # Clean up old connection
        if erg[0] is not None:
            try:
                erg[0].close()
            except Exception:
                pass
            erg[0] = None
            time.sleep(0.5)  # Give USB stack time to cleanup
        
        # Retry connection up to 3 times for flaky cables
        for attempt in range(3):
            try:
                ergs = pyrow.find()
                if not ergs:
                    if attempt == 2:
                        return False
                    time.sleep(0.5)
                    continue
                
                erg[0] = pyrow.PyErg(ergs[0])
                LOG.info(f"Connected to Concept2 ergometer: {ergs[0]}  "
                         f"(report ID #{pyrow.REPORT_ID}, {pyrow.REPORT_DATA_SIZE}-byte CSAFE payload)")
                
                # Give PM5 extra time to stabilize after USB connection
                time.sleep(2.0)
                
                # Wake PM5 if it's in sleep mode (send commands until PM5 responds)
                LOG.info("Waking PM5...")
                for wake_attempt in range(5):
                    try:
                        erg[0].send(['CSAFE_PM_GET_STROKESTATE'])
                        # Success - PM5 is awake and responding
                        consecutive_errors[0] = 0
                        LOG.info("PM5 connection verified")
                        return True
                    except (ConnectionError, Exception) as e:
                        if wake_attempt < 4:
                            time.sleep(0.3)  # Brief delay between wake attempts
                        # Keep trying - PM5 wakes up progressively
                
                # Final wake attempt
                try:
                    erg[0].send(['CSAFE_GETSTATUS_CMD'])
                    consecutive_errors[0] = 0
                    LOG.info("PM5 connection verified")
                    return True
                except ConnectionError:
                    if attempt < 2:
                        LOG.warning(f"Connection test failed, retry {attempt + 2}/3...")
                        erg[0].close()
                        erg[0] = None
                        time.sleep(1.0)
                        continue
                    return False
            except Exception as e:
                if attempt == 2:
                    if time.time() - last_error_log[0] > 10.0:
                        LOG.warning(f"Failed to connect to ergometer: {e}")
                        last_error_log[0] = time.time()
                    return False
                time.sleep(0.5)
        return False
    
    # Initial connection
    if not connect_ergometer():
        LOG.warning("No Concept2 ergometer found - will retry on poll")
        # Continue with callback that retries
    
    # Set up workout for force data collection using direct CSAFE commands
    if erg[0] is not None:
        LOG.info("Setting up PM5 workout for force data collection...")
        try:
            # Reset PM5 first (clears Finished/Pause states from previous workouts)
            # PM5 state machine requires: GOFINISHED -> GOREADY
            LOG.info("Resetting PM5 to Ready state...")
            try:
                erg[0].send(['CSAFE_GOFINISHED_CMD'])
                time.sleep(0.5)
                erg[0].send(['CSAFE_GOREADY_CMD'])
                time.sleep(0.5)
                LOG.info("PM5 reset to Ready state")
            except Exception as e:
                LOG.warning(f"Reset failed ({e}), continuing anyway...")
            
            # Wait for PM5 to reach Ready state (states 1/2/3 can accept workout setup)
            LOG.info("Waiting for PM5 Ready state...")
            start_time = time.time()
            pm5_ready = False
            last_state = None
            while time.time() - start_time < 10:  # 10 second timeout
                try:
                    status = erg[0].send(['CSAFE_GETSTATUS_CMD'])
                    state_raw = status.get('CSAFE_GETSTATUS_CMD', [0])[0]
                    state = state_raw & 0x7F  # Mask high bit (transitional flag)
                    
                    if state != last_state:
                        state_names = ['Error','Ready','Idle','Have ID','N/A','In Use','Pause','Finished','Manual','Offline']
                        state_name = state_names[state] if state < len(state_names) else f'Unknown({state})'
                        LOG.info(f"PM5 state: {state_raw} -> {state} ({state_name})")
                        last_state = state
                    
                    # States 1 (Ready), 2 (Idle), or 3 (Have ID) can accept workout setup
                    if state in [1, 2, 3]:
                        pm5_ready = True
                        break
                    
                    time.sleep(0.3)
                except Exception as e:
                    if time.time() - start_time < 10:
                        time.sleep(0.5)
                        continue
                    break
            
            if not pm5_ready:
                raise ConnectionError("PM5 not ready - select 'New Workout' on PM5 display")
            
            # Send workout commands individually (PM5 firmware limitation)
            erg[0].send(['CSAFE_SETHORIZONTAL_CMD', 2000, 36])  # 2000m distance
            time.sleep(0.3)
            erg[0].send(['CSAFE_PM_SET_SPLITDURATION', 128, 100])  # 100m splits
            time.sleep(0.3)
            powerpace = int(round(2.8 / ((120 / 500.) ** 3)))
            erg[0].send(['CSAFE_SETPOWER_CMD', powerpace, 88])  # 120W pace
            time.sleep(0.3)
            erg[0].send(['CSAFE_SETPROGRAM_CMD', 0, 0])  # Program 0 enables force data
            time.sleep(0.2)
            erg[0].send(['CSAFE_GOINUSE_CMD'])  # Activate workout
            LOG.info("Workout configured: 2000m, 100m splits, 120W")
            time.sleep(0.5)
            
            # Verify PM5 entered "In Use" state
            try:
                state_r = erg[0].send(['CSAFE_GETSTATUS_CMD'])
                pm5_state = (state_r.get('CSAFE_GETSTATUS_CMD', [0])[0] & 0x7F) if state_r.get('CSAFE_GETSTATUS_CMD') else 0
                if pm5_state == 5:
                    LOG.info("✓ PM5 state: InUse - force data enabled")
                else:
                    LOG.warning(f"PM5 state: {pm5_state} (expected 5=InUse) - select workout on display if needed")
            except Exception as e:
                LOG.warning(f"Could not verify PM5 state: {e}")
        except Exception as e:
            # Fall back to simple activation without workout parameters
            LOG.warning(f"Full workout setup failed ({e}), trying simple activation...")
            try:
                # Just activate PM5 for "In Use" state to enable force data
                erg[0].send(['CSAFE_GOINUSE_CMD'])
                time.sleep(0.5)
                LOG.info("✓ PM5 activated for data collection")
                LOG.warning("="*60)
                LOG.warning("MANUAL WORKOUT RECOMMENDED:")
                LOG.warning("  Select 'Just Row' or set distance/time on PM5 display")
                LOG.warning("  Force data will be collected during rowing")
                LOG.warning("="*60)
            except Exception as e2:
                LOG.error(f"Could not activate PM5: {e2}")
                LOG.warning("="*60)
                LOG.warning("MANUAL WORKOUT REQUIRED:")
                LOG.warning("  1. Select 'New Workout' on PM5 display")
                LOG.warning("  2. Choose 'Just Row' or set distance/time")
                LOG.warning("  3. Begin rowing - app will collect force data")
                LOG.warning("="*60)
    
    last_logged_phase = [0]
    force_collected = [False]  # Track if force was collected for this stroke
    draining = [False]         # True while incrementally draining force buffer
    pending_force = [[]]       # Accumulate force samples across multiple calls
    
    def get_phase():
        """Get current stroke phase (and drain force data incrementally) from PM5.
        
        Uses COMBINED CSAFE frames: STROKESTATE + FORCEPLOTDATA in a single
        USB round-trip (~50ms).  Phase updates never stall during force drain.
        During non-drain states, sends STROKESTATE only.
        """
        # Auto-reconnect if disconnected
        if erg[0] is None:
            if not connect_ergometer():
                return {'phase': 0, 'force': []}
        
        try:
            # Build command: always STROKESTATE, add FORCEPLOTDATA if draining
            if draining[0]:
                # Combined frame: both commands in ONE USB round-trip
                combined = erg[0].send(['CSAFE_PM_GET_STROKESTATE',
                                        'CSAFE_PM_GET_FORCEPLOTDATA', 32])
                phase = combined.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
                fp = combined.get('CSAFE_PM_GET_FORCEPLOTDATA', [0])
                byte_count = fp[0] if fp else 0
                datapoints = byte_count // 2
                samples = fp[1:datapoints + 1] if len(fp) > datapoints else []
            else:
                resp = erg[0].send(['CSAFE_PM_GET_STROKESTATE'])
                phase = resp.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
                samples = []
            
            consecutive_errors[0] = 0
            
            if phase != last_logged_phase[0]:
                phase_names = {0: 'IDLE', 1: 'PREP', 2: 'DRIVE', 3: 'DWELLING', 4: 'RECOVERY'}
                LOG.info(f"Ergometer phase: {last_logged_phase[0]} -> {phase} ({phase_names.get(phase, 'UNKNOWN')})")
                
                if phase == 2:  # DRIVE start — reset drain state
                    force_collected[0] = False
                    draining[0] = False
                    pending_force[0] = []
                
                last_logged_phase[0] = phase
            
            result = {'phase': phase, 'force': []}
            
            # Start drain on DWELLING/RECOVERY after drive.
            # Skip drain-result processing on THIS call because we sent
            # STROKESTATE-only (draining was False at the top of this call).
            # The next poll iteration will send the combined command.
            just_started_drain = False
            if phase in [3, 4] and not force_collected[0] and not draining[0]:
                draining[0] = True
                pending_force[0] = []
                just_started_drain = True
            
            # Process force samples from combined frame
            if draining[0] and not just_started_drain:
                if samples:
                    pending_force[0].extend(samples)
                else:
                    # Empty response — buffer exhausted, drain complete
                    draining[0] = False
                    force_collected[0] = True
                    
                    if pending_force[0]:
                        result['force'] = pending_force[0]
                        LOG.info(f"Force curve collected: {len(pending_force[0])} samples "
                                 f"(drive duration ~{len(pending_force[0])/500:.2f}s @ 500Hz)")
                        
                        # Fetch drag factor + stroke stats (combined frame, 1 USB call)
                        try:
                            meta = erg[0].send(['CSAFE_PM_GET_DRAGFACTOR',
                                                'CSAFE_PM_GET_STROKESTATS', 0])
                            drag = meta.get('CSAFE_PM_GET_DRAGFACTOR', [None])
                            if drag and drag[0] is not None:
                                result['drag_factor'] = drag[0]
                                LOG.info(f"Drag factor: {drag[0]}")
                            ss = meta.get('CSAFE_PM_GET_STROKESTATS', [])
                            if len(ss) >= 9:
                                result['stroke_stats'] = {
                                    'stroke_distance': ss[0],
                                    'drive_time': ss[1],
                                    'recovery_time': ss[2],
                                    'stroke_length': ss[3],
                                    'stroke_count': ss[4],
                                    'peak_force': ss[5],
                                    'impulse_force': ss[6],
                                    'avg_force': ss[7],
                                    'work_per_stroke': ss[8],
                                }
                                LOG.info(f"Stroke stats: peak={ss[5]}, avg={ss[7]}, "
                                         f"drive_time={ss[1]}, count={ss[4]}")
                        except Exception as e:
                            LOG.debug(f"Drag/stats query failed: {e}")
                    else:
                        LOG.warning("No force data available for this stroke")
                    
                    pending_force[0] = []
            
            return result
        except ConnectionError as e:
            # Connection lost - trigger reconnect
            consecutive_errors[0] += 1
            if consecutive_errors[0] == 1 or (time.time() - last_error_log[0] > 30.0):  # Log first error and every 30s
                LOG.warning(f"PM5 disconnected (will auto-reconnect): {e}")
                last_error_log[0] = time.time()
            erg[0] = None  # Force reconnect on next poll
            return {'phase': 0, 'force': []}
        except Exception as e:
            consecutive_errors[0] += 1
            if time.time() - last_error_log[0] > 10.0:  # Rate-limit error logging
                LOG.error(f"Error reading ergometer phase: {e}")
                last_error_log[0] = time.time()
            return {'phase': 0, 'force': []}
    
    return get_phase


def inference_loop_with_recording(args, log_file_path, stream, app, wnd, recorder, controller, tracers=None):
    """
    Main inference loop with integrated keypoint recording.
    Optimized for ARM Cortex-A76 with NEON vectorization.
    
    Args:
        args: Command-line arguments
        log_file_path: Path for statistics logging (from --show-stats)
        stream: Axelera inference stream
        app: Display application
        wnd: Display window
        recorder: KeypointRecorder instance
        controller: PhaseController instance
        tracers: Inference tracers for performance monitoring
    """
    # ARM optimization: Pin to dedicated core and elevate priority
    cpu_core = getattr(args, 'cpu_core', 3)
    setup_capture_thread_priority(cpu_core=cpu_core)
    
    # ARM optimization: Ensure NumPy uses NEON (if available)
    try:
        np_config = np.__config__
        if hasattr(np_config, 'show'):
            pass  # NumPy build info available if needed
    except:
        pass

    from tqdm import tqdm
    
    PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    
    # Setup window
    if len(stream.sources) > 1:
        for sid, source in stream.sources.items():
            wnd.options(sid, title=f"#{sid} - {source}")
    
    wnd.options(-1, speedometer_smoothing=args.speedometer_smoothing)
    
    display_worker = DisplayWorker(wnd) if not args.headless else None
    fps_benchmark = FPSBenchmark()
    
    # Detailed timing diagnostics
    frame_arrival_times = deque(maxlen=300)
    processing_times = deque(maxlen=300)
    pipeline_timestamps = deque(maxlen=300)
    
    # Kalman timing tracking
    kalman_times = []
    kalman_max_us = 0.0
    
    frame_number = 0
    width, height = 0, 0
    frame_data = None
    
    phase_update_frames = 3  # Poll every 3 frames (~50ms @ 60fps) for accurate phase detection
    stats_interval_frames = args.stats_interval
    
    LOG.info("Starting inference loop with keypoint recording...")
    LOG.info(f"Recorder: buffer_size={recorder.buffer_size}, post_buffer_size={recorder.post_buffer_size}, save_dir={recorder.save_dir}")
    LOG.info(f"Phase control: {controller.get_phase_name()}")
    if args.headless:
        LOG.info("Headless mode: display disabled for minimal latency")
    if args.no_progress:
        LOG.info("Progress bar disabled for performance")
    if stats_interval_frames > 0:
        LOG.info(f"Stats logging every {stats_interval_frames} frames")
    
    # Smoothing filter selection
    smoother = None
    if args.filter == 'none':
        LOG.info("No filtering - using raw keypoints")
    elif args.filter == 'kalman':
        smoother = KeypointSmoother(
            process_noise=0.005,     # Lower: trust filter state more
            measurement_noise=1.2,   # Higher: more skeptical of noisy detections
            velocity_alpha=0.4       # Lower: smoother velocity estimation
        )
        LOG.info("Using Kalman filter: Q=0.005, R=1.2, alpha=0.4 (smoother)")
    elif args.filter == 'ekf':
        # Extended Kalman Filter - nonlinear motion model with acceleration
        # Tuned to reduce overshoot: trust measurements more, predict less
        smoother = EKFSmoother(
            process_noise_pos=0.1,    # Position process noise
            process_noise_vel=2.0,    # Velocity process noise (↓ from 5 - less prediction momentum)
            process_noise_acc=15.0,   # Acceleration process noise (↓ from 30 - less aggressive predictions)
            measurement_noise=4.0,    # Measurement noise (keep high - trust measurements)
            damping=0.4,              # Velocity damping (↑ from 0.2 - more aggressive slowdown)
            acc_decay=1.5             # Acceleration decay (↑ from 0.8 - faster decay to prevent overshoot)
        )
        LOG.info("Using Extended Kalman filter: balanced for overshoot reduction (acc=15, vel=2, damping=0.4)")
    else:
        # One Euro Filter - AGGRESSIVE for high jitter reduction
        # Decreased min_cutoff: 7.0→3.0 Hz (stronger low-pass filtering)
        # Adjusted for 896x504 @ 80fps (40% higher resolution, 20% lower framerate)
        # Beta scaled: 0.008 * (896/640) = 0.0112 for proportional pixel velocity
        # Min_cutoff increased for stronger jitter reduction
        smoother = OneEuroSmoother(
            min_cutoff=1.5,   # Hz - increased for aggressive jitter reduction
            beta=0.011,       # Speed coefficient - scaled for 896px width (0.008 * 896/640)
            d_cutoff=1.0,     # Derivative cutoff
        )
        LOG.info("Using One Euro filter: ADJUSTED (5.0Hz/beta=0.011/1.0Hz) for 896x504@80fps")
    
    for event in tqdm(
        stream.with_events(),
        desc=f"Recording... {' ':>30}",
        unit='frames',
        leave=False,
        bar_format=PBAR,
        disable=args.no_progress,
    ):
        if not event.result:
            continue
        
        # Capture frame arrival timestamp immediately
        arrival_time = time.perf_counter()
        
        # Use perf_counter for higher precision (nanosecond resolution)
        frame_timestamp = time.perf_counter()
        fps_benchmark.record_frame(frame_timestamp)
        
        # Track arrival timing for diagnostics
        frame_arrival_times.append(arrival_time)
        
        frame_result = event.result
        frame_number += 1
        
        image, meta = frame_result.image, frame_result.meta
        if image is None and meta is None:
            if not args.headless and wnd.is_closed:
                break
            continue
        
        # Update phase periodically
        if frame_number % phase_update_frames == 0:
            controller.update_phase_from_external()
            recorder.set_phase(controller.current_phase)
        
        # Get image dimensions first if needed
        if meta and image is not None and width == 0:
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
            else:
                width, height = image.size
        
        # Apply Kalman filter BEFORE extraction and display
        kalman_start = time.perf_counter()
        if meta:
            for item in meta:
                value = meta[item]
                if hasattr(value, 'keypoints') and value.keypoints is not None:
                    kpts = value.keypoints
                    if len(kpts) > 0 and len(kpts[0]) > 0:
                        # Smooth first person's keypoints in-place (if filter enabled)
                        if smoother is not None:
                            smoother.smooth_inplace(kpts[0], frame_timestamp)
                        
                        # Log velocity every 300 frames (reduce I/O overhead)
                        if frame_number % 300 == 0:
                            if smoother is not None:
                                vx, vy = smoother.get_velocities(17)
                                LOG.info(f"Frame {frame_number}: kpts[0][6] after filter = ({kpts[0][6][0]:.1f}, {kpts[0][6][1]:.1f}) vx={vx[6]:.1f}")
                            else:
                                LOG.info(f"Frame {frame_number}: kpts[0][6] (raw) = ({kpts[0][6][0]:.1f}, {kpts[0][6][1]:.1f})")
                        
                        # Log filter delta every 180 frames
                        if frame_number % 180 == 0 and smoother is not None:
                            before = kpts[0][0].copy() if len(kpts[0][0]) >= 2 else None
                            if before is not None:
                                after = kpts[0][0]
                                delta = np.sqrt((after[0]-before[0])**2 + (after[1]-before[1])**2)
                                LOG.info(f"filtered: nose before=({before[0]:.1f},{before[1]:.1f}) after=({after[0]:.1f},{after[1]:.1f}) delta={delta:.2f}px")
                    break
        kalman_time_us = (time.perf_counter() - kalman_start) * 1_000_000
        kalman_times.append(kalman_time_us)
        kalman_max_us = max(kalman_max_us, kalman_time_us)
        
        # Keep only last 180 frames for stats
        if len(kalman_times) > 180:
            kalman_times.pop(0)
        
        # Extract keypoints WITH Kalman velocities (much more accurate than differentiating positions)
        if meta and width > 0:
            frame_data = recorder.extract_keypoints_with_velocity(
                meta, width, height, frame_number, frame_timestamp, smoother
            )
            if frame_data:
                # ADD TO BUFFER - this is what triggers event-based recording!
                recorder.add_frame(frame_data)
        
        # Display the smoothed frame
        if image and display_worker:
            display_worker.push_frame(image, meta, frame_result.stream_id)
        
        # Track processing time
        processing_end = time.perf_counter()
        processing_times.append((processing_end - arrival_time) * 1000)  # ms
        
        if stats_interval_frames > 0 and frame_number % stats_interval_frames == 0:
            stats = recorder.get_stats()
            has_keypoints = "✓" if frame_data else "✗"
            fps_stats = fps_benchmark.get_stats()
            kalman_avg_us = sum(kalman_times) / len(kalman_times) if kalman_times else 0
            
            # Compute pipeline jitter vs processing jitter
            if len(frame_arrival_times) >= 2:
                arrival_intervals = np.diff(list(frame_arrival_times)) * 1000  # ms
                pipeline_jitter = np.std(arrival_intervals)
                pipeline_avg = np.mean(arrival_intervals)
                processing_avg = np.mean(list(processing_times))
                processing_max = np.max(list(processing_times))
            else:
                pipeline_jitter = 0
                pipeline_avg = 0
                processing_avg = 0
                processing_max = 0
            
            if fps_stats and 'fps' in fps_stats:
                LOG.info(
                    f"Stats: frames={stats['frames_processed']}, "
                    f"events_saved={stats['events_saved']}, "
                    f"phase={controller.get_phase_name()}, "
                    f"keypoints={has_keypoints} | "
                    f"FPS={fps_stats['fps']:.1f}, "
                    f"jitter={fps_stats['jitter_ms']:.2f}ms, "
                    f"drops={fps_stats['drops']}({fps_stats['drop_rate']:.1f}%) | "
                    f"pipeline_jitter={pipeline_jitter:.2f}ms, "
                    f"proc_avg={processing_avg:.2f}ms, "
                    f"proc_max={processing_max:.2f}ms, "
                    f"kalman_avg={kalman_avg_us:.0f}µs, "
                    f"kalman_max={kalman_max_us:.0f}µs"
                )
                kalman_max_us = 0.0  # Reset max after logging
            else:
                LOG.info(
                    f"Stats: frames={stats['frames_processed']}, "
                    f"events_saved={stats['events_saved']}, "
                    f"phase={controller.get_phase_name()}, "
                    f"queue_depth={stats['queue_depth']}, "
                    f"keypoints={has_keypoints} | "
                    f"Kalman: avg={kalman_avg_us:.0f}µs max={kalman_max_us:.0f}µs"
                )
                kalman_max_us = 0.0  # Reset max after logging
        
        if not args.headless and wnd.is_closed:
            break
    
    if display_worker:
        display_worker.stop()
    
    if log_file_path:
        print(statistics.format_table(log_file_path, tracers))
    
    stats = recorder.get_stats()
    LOG.info("Recording complete!")
    LOG.info(f"Final stats: {stats}")
    LOG.info(f"Saved data location: {recorder.save_dir}")


def main():
    """Main entry point"""
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, 
        description='Rowing ergometer keypoint recording with Axelera Voyager SDK'
    )
    
    # Keypoint recording arguments
    parser.add_argument(
        '--keypoint-buffer-size',
        type=int,
        default=30,
        help="Number of frames to buffer before drive (default: 30 frames @ 90 FPS = ~330ms)",
    )
    parser.add_argument(
        '--post-drive-frames',
        type=int,
        default=60,
        help="Number of frames to collect after drive ends (default: 60 frames @ 90 FPS = ~670ms)",
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help="Disable progress bar (reduces terminal I/O overhead at high FPS)",
    )
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=180,
        help="Log stats every N frames (default: 180 frames = 3s @ 60 FPS). 0 disables stats.",
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help="Disable display output for minimal latency (reduces jitter by ~30-50%%)",
    )
    parser.add_argument(
        '--filter',
        type=str,
        choices=['none', 'kalman', 'oneeuro', 'ekf'],
        default='oneeuro',
        help="Smoothing filter: 'none' (no filtering), 'kalman' (predictive), 'oneeuro' (adaptive, default), or 'ekf' (nonlinear with acceleration). "
             "One Euro is smoother when stationary, more responsive when moving. EKF models acceleration explicitly.",
    )
    parser.add_argument(
        '--cpu-core',
        type=int,
        default=3,
        help='CPU core to pin capture thread to (0-3 for RPI 5, default: 3)'
    )
    
    args = parser.parse_args()
    
    # Clean up old stroke data files at startup
    save_dir = "/tmp/stroke_data"
    old_files = glob.glob(os.path.join(save_dir, "stroke_*.json"))
    if old_files:
        LOG.info(f"Cleaning up {len(old_files)} old stroke file(s)...")
        for filepath in old_files:
            try:
                os.remove(filepath)
            except Exception as e:
                LOG.warning(f"Could not remove {filepath}: {e}")
        LOG.info("Old stroke data cleared")
    
    # Validate network is a pose model
    if 'pose' not in args.network.lower():
        LOG.warning(f"Network '{args.network}' may not be a pose detection model")
        LOG.warning("Recommended: --network yolov8n-pose-coco or similar")
    
    # Initialize keypoint recorder with fixed save directory
    recorder = KeypointRecorder(
        buffer_size=args.keypoint_buffer_size,
        post_buffer_size=args.post_drive_frames,
        save_dir="/tmp/stroke_data"
    )
    
    # Initialize phase controller
    controller = PhaseController(recorder)
    
    # Link recorder to controller for force data collection
    recorder.phase_controller = controller
    
    # Setup ergometer integration with auto-reconnect
    erg_callback = create_erg_phase_callback()
    controller.set_external_phase_callback(erg_callback)
    LOG.info("Ergometer phase detection enabled (will auto-connect/reconnect)")
    
    # Performance tips
    if not args.no_progress:
        LOG.info("TIP: Use --no-progress to reduce terminal I/O overhead at high FPS")
    if '--low-latency' not in sys.argv:
        LOG.info("TIP: Use --low-latency to disable buffering and reduce display latency")
    
    # Create inference stream
    try:
        tracers = inf_tracers.create_tracers_from_args(args)
        
        # Initialize pipeline statistics logging if --show-stats enabled
        log_file, log_file_path = None, None
        if hasattr(args, 'show_stats') and args.show_stats:
            try:
                log_file, log_file_path = statistics.initialise_logging()
                LOG.info("Pipeline statistics logging enabled")
            except Exception as e:
                LOG.warning(f"Failed to initialize statistics logging: {e}")
        
        stream = create_inference_stream(
            config.SystemConfig.from_parsed_args(args),
            config.InferenceStreamConfig.from_parsed_args(args),
            config.PipelineConfig.from_parsed_args(args),
            config.LoggingConfig.from_parsed_args(args),
            config.DeployConfig.from_parsed_args(args),
            tracers=tracers,
        )
        
        with display.App(
            renderer=args.display,
            opengl=stream.hardware_caps.opengl,
            buffering=not stream.is_single_image(),
        ) as app:
            wnd = app.create_window('Rowing Ergometer Recording', size=args.window_size)
            app.start_thread(
                inference_loop_with_recording,
                (args, log_file_path, stream, app, wnd, recorder, controller, tracers),
                name='InferenceThread',
            )
            app.run()  # Default 60 FPS for smooth display
            
    except KeyboardInterrupt:
        LOG.info("Recording stopped by user")
    except Exception as e:
        import traceback
        LOG.error(f"Error during recording: {e}")
        LOG.error(traceback.format_exc())
        LOG.exit_with_error_log(e)
    finally:
        # Stop background USB polling thread
        controller.stop()
        
        if 'stream' in locals():
            stream.stop()
        
        # Final summary
        stats = recorder.get_stats()
        print("\n" + "="*60)
        print("RECORDING SUMMARY")
        print("="*60)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Events saved: {stats['events_saved']}")
        print(f"Save location: {recorder.save_dir}")
        print(f"Files: ls -lh {recorder.save_dir}/")
        print("="*60)


if __name__ == "__main__":
    main()
