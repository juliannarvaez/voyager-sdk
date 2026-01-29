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
from collections import deque

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

from axelera.app import (
    config,
    create_inference_stream,
    display,
    inf_tracers,
    logging_utils,
    yaml_parser,
)

from rowing_ergometer import KeypointRecorder, PhaseController, pyrow

LOG = logging_utils.getLogger(__name__)


class DisplayWorker:
    """Non-blocking display worker running in separate thread"""
    
    def __init__(self, wnd):
        self.wnd = wnd
        self.frame_queue = deque(maxlen=2)  # Keep only 2 latest frames (drop old ones)
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
        
        def get_phase():
            """Get current stroke phase and force data from PM5"""
            try:
                # First, get stroke state to know current phase
                stroke_result = erg.send(['CSAFE_PM_GET_STROKESTATE'])
                phase = stroke_result.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
                
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


def inference_loop_with_recording(args, stream, app, wnd, recorder, controller):
    """
    Main inference loop with integrated keypoint recording.
    Optimized with non-blocking display and early timestamp capture.
    
    Args:
        args: Command-line arguments
        stream: Axelera inference stream
        app: Display application
        wnd: Display window
        recorder: KeypointRecorder instance
        controller: PhaseController instance
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
    
    frame_number = 0
    width, height = 0, 0  # Cache dimensions
    
    # Frame-based timing (avoid time.time() syscalls)
    phase_update_frames = 9  # ~100ms @ 90 FPS
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
        
        # Display frame (non-blocking push to display worker)
        if image and display_worker:
            display_worker.push_frame(image, meta, frame_result.stream_id)
        
        # Periodic stats logging (frame-based, no time.time() syscalls)
        if stats_interval_frames > 0 and frame_number % stats_interval_frames == 0:
            stats = recorder.get_stats()
            LOG.info(
                f"Stats: frames={stats['frames_processed']}, "
                f"events_saved={stats['events_saved']}, "
                f"phase={controller.get_phase_name()}, "
                f"queue_depth={stats['queue_depth']}"
            )
        
        if not args.headless and wnd.is_closed:
            break
    
    # Stop display worker
    if display_worker:
        display_worker.stop()
    
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
        default=450,
        help="Log stats every N frames (default: 450 frames = 5s @ 90 FPS). 0 disables stats.",
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
                (args, stream, app, wnd, recorder, controller),
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
