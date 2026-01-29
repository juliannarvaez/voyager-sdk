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

from rowing_ergometer import KeypointRecorder, PhaseController, pyrow

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
        """Get performance statistics (optimized with cached arrays)"""
        if not self.intervals:
            return None
        
        # Use numpy array directly from deque (avoid extra allocation)
        intervals_arr = np.fromiter(self.intervals, dtype=np.float32, count=len(self.intervals))
        elapsed = self.last_timestamp - self.start_time
        
        mean_interval = intervals_arr.mean()  # Faster than np.mean()
        std_interval = intervals_arr.std()    # Faster than np.std()
        min_interval = intervals_arr.min()
        max_interval = intervals_arr.max()
        
        avg_fps = 1.0 / mean_interval if mean_interval > 0 else 0
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        drops = (intervals_arr > self.drop_threshold).sum()  # Faster vectorized operation
        drop_rate = drops / len(intervals_arr) * 100.0
        
        return {
            'avg_fps': avg_fps,
            'actual_fps': actual_fps,
            'jitter_ms': std_interval * 1000,
            'min_ms': min_interval * 1000,
            'max_ms': max_interval * 1000,
            'drops': int(drops),
            'drop_rate': drop_rate,
        }


class DisplayWorker:
    """Non-blocking display worker running in separate thread"""
    
    def __init__(self, wnd):
        self.wnd = wnd
        self.frame_queue = deque(maxlen=1)  # Keep only latest frame (minimize latency)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="DisplayWorker")
        self.thread.start()
        LOG.info("Display worker thread started (non-blocking)")
    
    def _run(self):
        """Display worker loop - runs in separate thread"""
        while self.running:
            with self.lock:
                if self.frame_queue:
                    frame_data = self.frame_queue.popleft()
                else:
                    frame_data = None
            
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
            self.frame_queue.append({
                'image': image,
                'meta': meta,
                'stream_id': stream_id
            })
    
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
        
        # Track last phase for transition logging
        last_logged_phase = [0]  # Use list for closure
        
        # Track last phase for transition logging
        last_logged_phase = [0]  # Use list for closure
        
        def get_phase():
            """Get current stroke phase and force data from PM5"""
            try:
                # First, get stroke state to know current phase
                stroke_result = erg.send(['CSAFE_PM_GET_STROKESTATE'])
                phase = stroke_result.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
                
                # Log phase transitions
                if phase != last_logged_phase[0]:
                    phase_names = {0: 'IDLE', 1: 'PREP', 2: 'DRIVE', 3: 'DWELLING', 4: 'RECOVERY'}
                    LOG.info(f"Ergometer phase: {last_logged_phase[0]} -> {phase} ({phase_names.get(phase, 'UNKNOWN')})")
                    last_logged_phase[0] = phase
                
                # Only request force data during Drive phase (matches cameraerg)
                force_curve = []
                if phase == 2:  # Drive phase
                    forceplot = erg.get_forceplot()
                    force_curve = forceplot.get('forceplot', []) if forceplot else []
                    
                    # Conditional logging - only if no force data (debugging)
                    if not force_curve:
                        LOG.debug("Drive phase but no force data returned")
                
                # Return dict with both phase and force data
                return {
                    'phase': phase,
                    'force': force_curve
                }
            except Exception as e:
                LOG.error(f"Error reading ergometer phase: {e}")
                import traceback
                LOG.error(traceback.format_exc())
                return {'phase': 0, 'force': []}
        
        return get_phase
        
    except Exception as e:
        LOG.error(f"Failed to connect to ergometer: {e}")
        return None


def inference_loop_with_recording(args, log_file_path, stream, app, wnd, recorder, controller, tracers=None):
    """
    Main inference loop with integrated keypoint recording.
    Optimized with non-blocking display and early timestamp capture.
    
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

    from tqdm import tqdm
    
    PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    
    # Setup window
    if len(stream.sources) > 1:
        for sid, source in stream.sources.items():
            wnd.options(sid, title=f"#{sid} - {source}")
    
    wnd.options(-1, speedometer_smoothing=args.speedometer_smoothing)
    
    # Start non-blocking display worker
    display_worker = DisplayWorker(wnd) if not args.headless else None
    
    # Initialize FPS benchmark
    fps_benchmark = FPSBenchmark()
    
    frame_number = 0
    width, height = 0, 0  # Cache dimensions
    frame_data = None  # Cache last frame_data for stats
    
    # Frame-based timing (avoid time.time() syscalls)
    phase_update_frames = 15  # ~170ms @ 90 FPS (reduced callback overhead)
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
    LOG.info("Pipeline optimizations: reduced allocations, batched operations, cached stats")
    
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
        
        # Capture timestamp FIRST (earliest possible point in pipeline)
        frame_timestamp = time.time()
        
        # Record for FPS benchmark
        fps_benchmark.record_frame(frame_timestamp)
        
        frame_result = event.result
        frame_number += 1
        
        image, meta = frame_result.image, frame_result.meta
        if image is None and meta is None:
            if not args.headless and wnd.is_closed:
                break
            continue
        
        # Update phase from cached ergometer value every N frames (fast - no USB I/O)
        if frame_number % phase_update_frames == 0:
            controller.update_phase_from_external()
            # Sync phase to recorder for event triggering
            recorder.set_phase(controller.current_phase)
        
        # Extract and record keypoint data (only if meta exists)
        if meta and width > 0:
            # Use early-captured timestamp (frame arrival time, not processing completion)
            frame_data = recorder.extract_keypoints_from_meta(meta, width, height, frame_number, frame_timestamp)
            if frame_data:
                recorder.add_frame(frame_data)
            elif frame_number % 100 == 0:  # Log every 100 frames if no keypoints
                LOG.warning(f"Frame {frame_number}: No keypoints detected in meta")
        elif meta and image is not None:
            # First frame - get dimensions
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
            else:
                width, height = image.size
        
        # Apply Kalman filtering to display (in-place modification, zero overhead)
        if meta and display_worker:
            recorder.apply_kalman_to_meta(meta, frame_timestamp)
        
        # Display frame (non-blocking push to display worker)
        if image and display_worker:
            display_worker.push_frame(image, meta, frame_result.stream_id)
        
        # Periodic stats logging (frame-based, deferred computation)
        if stats_interval_frames > 0 and frame_number % stats_interval_frames == 0:
            # Compute stats off critical path (only when logging)
            stats = recorder.get_stats()
            has_keypoints = "✓" if frame_data else "✗"
            fps_stats = fps_benchmark.get_stats()
            
            if fps_stats:
                # Format once to reduce string operations
                LOG.info(
                    f"Stats: frames={stats['frames_processed']}, "
                    f"events_saved={stats['events_saved']}, "
                    f"phase={controller.get_phase_name()}, "
                    f"keypoints={has_keypoints} | "
                    f"FPS={fps_stats['actual_fps']:.1f}, "
                    f"jitter={fps_stats['jitter_ms']:.2f}ms, "
                    f"drops={fps_stats['drops']}({fps_stats['drop_rate']:.1f}%)"
                )
            else:
                LOG.info(
                    f"Stats: frames={stats['frames_processed']}, "
                    f"events_saved={stats['events_saved']}, "
                    f"phase={controller.get_phase_name()}, "
                    f"queue_depth={stats['queue_depth']}, "
                    f"keypoints={has_keypoints}"
                )
        
        if not args.headless and wnd.is_closed:
            break
    
    # Stop display worker
    if display_worker:
        display_worker.stop()
    
    # Print pipeline statistics if enabled
    if log_file_path:
        print(statistics.format_table(log_file_path, tracers))
    
    # Final stats
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
