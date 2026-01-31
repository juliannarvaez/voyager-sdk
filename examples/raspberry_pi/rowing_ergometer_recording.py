#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Example: Rowing Ergometer Keypoint Recording with Phase Control

"""
Rowing Ergometer Keypoint Recording Script

Records pose keypoints during rowing strokes with phase-based event capture.
Uses embedded pyrow module from cameraerg repository for Concept2 PM5 integration.

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
from kalman_filter import KeypointSmoother  # Use original NumPy version

LOG = logging_utils.getLogger(__name__)


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
        self.frame_queue = deque(maxlen=1)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="DisplayWorker")
        self.thread.start()
        LOG.info("Display worker thread started (non-blocking)")
    
    def _run(self):
        """Display worker loop - runs in separate thread"""
        while self.running:
            with self.lock:
                frame_data = self.frame_queue.popleft() if self.frame_queue else None
            
            if frame_data:
                try:
                    self.wnd.show(frame_data['image'], frame_data['meta'], frame_data['stream_id'])
                except Exception as e:
                    LOG.error(f"Display error: {e}")
            else:
                time.sleep(0.001)  # 1ms sleep when idle
    
    def push_frame(self, image, meta, stream_id):
        """Push frame to display queue (non-blocking)"""
        with self.lock:
            self.frame_queue.append({'image': image, 'meta': meta, 'stream_id': stream_id})
    
    def stop(self):
        """Stop display worker"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def create_erg_phase_callback():
    """
    Create phase callback for Concept2 PM5 ergometer.
    
    Returns:
        Callable that returns current stroke phase (0-4)
    """
    try:
        # Find and connect to ergometer
        ergs = list(pyrow.find())
        if not ergs:
            LOG.warning("No Concept2 ergometer found")
            return None
        
        erg = pyrow.PyErg(ergs[0])
        LOG.info(f"Connected to Concept2 ergometer: {ergs[0]}")
        
        # Give PM5 time to stabilize after USB connection
        time.sleep(3.0)
        
        # Initialize workout - CRITICAL for force data collection!
        # PM5 only provides force plot data during an active workout
        try:
            erg.set_workout(distance=2000, split=100, pace=120)
            LOG.info("Workout initialized: 2000m, split=100m, pace=120s - force data now enabled")
        except Exception as e:
            LOG.warning(f"Could not set workout programmatically: {e}")
            LOG.warning("Please start a workout manually on the PM5 for force data collection")
        
        last_logged_phase = [0]
        force_collected = [False]  # Track if force was collected for this stroke
        
        def get_phase():
            """Get current stroke phase and force data from PM5 (complete curve via multiple polls)"""
            try:
                stroke_result = erg.send(['CSAFE_PM_GET_STROKESTATE'])
                phase = stroke_result.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
                
                if phase != last_logged_phase[0]:
                    phase_names = {0: 'IDLE', 1: 'PREP', 2: 'DRIVE', 3: 'DWELLING', 4: 'RECOVERY'}
                    LOG.info(f"Ergometer phase: {last_logged_phase[0]} -> {phase} ({phase_names.get(phase, 'UNKNOWN')})")
                    
                    # Reset force collection flag on DRIVE start
                    if phase == 2:
                        force_collected[0] = False
                    
                    last_logged_phase[0] = phase
                
                force_curve = []
                
                # Collect complete force curve at START of RECOVERY phase (per PM5 spec)
                # PM5 accumulates force during DRIVE, available in RECOVERY
                # Must poll IMMEDIATELY when RECOVERY starts - buffer clears quickly
                if phase == 4 and not force_collected[0]:
                    LOG.debug("RECOVERY phase - collecting complete force curve IMMEDIATELY...")
                    force_collected[0] = True
                    
                    # Aggressive polling: drain PM5 buffer as fast as possible
                    # PM5 buffer clearing timing is undocumented - poll continuously until empty
                    for attempt in range(20):
                        forceplot = erg.get_forceplot()
                        samples = forceplot.get('forceplot', []) if forceplot else []
                        
                        if not samples:
                            # Buffer empty - done collecting
                            if attempt == 0:
                                LOG.warning("No force data available at RECOVERY start")
                            else:
                                LOG.debug(f"Force buffer exhausted after {attempt} polls")
                            break
                        
                        force_curve.extend(samples)
                        LOG.debug(f"Force poll {attempt + 1}: {len(samples)} samples (total: {len(force_curve)})")
                        
                        # No delay - continuous polling to beat PM5's buffer clearing
                    
                    if force_curve:
                        LOG.info(f"Complete force curve collected: {len(force_curve)} samples")
                
                return {'phase': phase, 'force': force_curve}
            except Exception as e:
                LOG.error(f"Error reading ergometer phase: {e}")
                return {'phase': 0, 'force': []}
        
        return get_phase
        
    except Exception as e:
        LOG.error(f"Failed to connect to ergometer: {e}")
        return None


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
    # ARM optimization: Set CPU affinity to performance cores (cores 0-3 on RPI 5)
    try:
        import os
        # Pin to cores 2-3 (leave 0-1 for USB polling process)
        os.sched_setaffinity(0, {2, 3})
        LOG.info("Set CPU affinity to cores 2-3 (ARM Cortex-A76 performance cores)")
    except Exception as e:
        LOG.debug(f"Could not set CPU affinity: {e}")
    
    # ARM optimization: Ensure NumPy uses NEON (if available)
    try:
        np_config = np.__config__
        if hasattr(np_config, 'show'):
            LOG.debug("NumPy build info: optimizations may include NEON")
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
    
    # Kalman timing tracking
    kalman_times = []
    kalman_max_us = 0.0
    
    frame_number = 0
    width, height = 0, 0
    frame_data = None
    
    phase_update_frames = 15
    stats_interval_frames = args.stats_interval
    
    LOG.info("Starting inference loop with keypoint recording...")
    LOG.info(f"Recorder: buffer_size={recorder.buffer_size}, save_dir={recorder.save_dir}")
    LOG.info(f"Phase control: {controller.get_phase_name()}")
    if args.headless:
        LOG.info("Headless mode: display disabled for minimal latency")
    if args.no_progress:
        LOG.info("Progress bar disabled for performance")
    if stats_interval_frames > 0:
        LOG.info(f"Stats logging every {stats_interval_frames} frames")
    
    # Pure Python Kalman smoother (no C++ needed)
    # Parameters tuned for FAST RESPONSE while removing YOLO jitter:
    #   process_noise (Q=0.01): Allow model to predict motion changes quickly
    #   measurement_noise (R=0.5): Trust measurements more = faster tracking, still filters jitter
    #   velocity_alpha (α=0.7): Fast velocity adaptation for quick direction changes
    smoother = KeypointSmoother(
        process_noise=0.01,      # Allow model to predict motion changes (increased from 0.003)
        measurement_noise=0.5,   # Trust measurements more for quick tracking (reduced from 1.0)
        velocity_alpha=0.7       # Fast velocity adaptation for quick response (increased from 0.3)
    )
    LOG.info("Using pure Python Kalman filter for keypoint smoothing")
    
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
        
        frame_timestamp = time.time()
        fps_benchmark.record_frame(frame_timestamp)
        
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
        
        # Apply Kalman filter BEFORE extraction and display (smooths skeleton for display)
        kalman_start = time.perf_counter()
        if meta:
            for item in meta:
                value = meta[item]
                if hasattr(value, 'keypoints') and value.keypoints is not None:
                    kpts = value.keypoints
                    if len(kpts) > 0 and len(kpts[0]) > 0:
                        # Debug: capture before/after to verify filtering
                        if frame_number % 180 == 0 and len(kpts[0]) > 0:
                            before = kpts[0][0].copy() if len(kpts[0][0]) >= 2 else None
                        
                        # Smooth first person's keypoints IN-PLACE
                        smoother.smooth_inplace(kpts[0], frame_timestamp)
                        
                        # Debug: print delta to verify filtering is happening
                        if frame_number % 180 == 0 and before is not None:
                            after = kpts[0][0]
                            delta = np.sqrt((after[0]-before[0])**2 + (after[1]-before[1])**2)
                            LOG.info(f"Kalman: nose before=({before[0]:.1f},{before[1]:.1f}) after=({after[0]:.1f},{after[1]:.1f}) delta={delta:.2f}px")
                    break
        kalman_time_us = (time.perf_counter() - kalman_start) * 1_000_000
        kalman_times.append(kalman_time_us)
        kalman_max_us = max(kalman_max_us, kalman_time_us)
        
        # Keep only last 180 frames for stats
        if len(kalman_times) > 180:
            kalman_times.pop(0)
        
        # Extract keypoints AFTER smoothing
        if meta and width > 0:
            frame_data = recorder.extract_keypoints_from_meta(meta, width, height, frame_number, frame_timestamp)
            if not frame_data and frame_number % 100 == 0:
                LOG.warning(f"Frame {frame_number}: No keypoints detected in meta")
        
        # Display the smoothed frame
        if image and display_worker:
            display_worker.push_frame(image, meta, frame_result.stream_id)
        
        if stats_interval_frames > 0 and frame_number % stats_interval_frames == 0:
            stats = recorder.get_stats()
            has_keypoints = "✓" if frame_data else "✗"
            fps_stats = fps_benchmark.get_stats()
            kalman_avg_us = sum(kalman_times) / len(kalman_times) if kalman_times else 0
            
            if fps_stats and 'fps' in fps_stats:
                LOG.info(
                    f"Stats: frames={stats['frames_processed']}, "
                    f"events_saved={stats['events_saved']}, "
                    f"phase={controller.get_phase_name()}, "
                    f"keypoints={has_keypoints} | "
                    f"FPS={fps_stats['fps']:.1f}, "
                    f"jitter={fps_stats['jitter_ms']:.2f}ms, "
                    f"drops={fps_stats['drops']}({fps_stats['drop_rate']:.1f}%) | "
                    f"Kalman: avg={kalman_avg_us:.0f}µs max={kalman_max_us:.0f}µs"
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
        help="Number of frames to buffer before events (default: 30 frames @ 90 FPS = ~330ms)",
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
        save_dir="/tmp/stroke_data"
    )
    
    # Initialize phase controller
    controller = PhaseController(recorder)
    
    # Link recorder to controller for force data collection
    recorder.phase_controller = controller
    
    # Setup ergometer integration (required)
    erg_callback = create_erg_phase_callback()
    if erg_callback:
        controller.set_external_phase_callback(erg_callback)
        LOG.info("Ergometer connected - automatic phase detection enabled")
    else:
        LOG.error("Concept2 PM5 ergometer required but not found")
        LOG.error("Please connect ergometer or install pyrow: pip install pyrow")
        sys.exit(1)
    
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
