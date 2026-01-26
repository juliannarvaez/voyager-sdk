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
        
        def get_phase():
            """Get current stroke phase and force data from PM5"""
            try:
                forceplot = erg.get_forceplot()  # Uses default 32 samples
                phase = forceplot.get('strokestate', 0) if forceplot else 0
                force_curve = forceplot.get('forceplot', []) if forceplot else []
                
                # Detailed debug logging
                if forceplot:
                    status = forceplot.get('status', -1)
                    
                    # Log raw response for debugging
                    if len(force_curve) > 0:
                        LOG.info(f"PM5 FORCE DATA: {len(force_curve)} samples, phase={phase}, status={status}")
                        LOG.info(f"  Force values: {force_curve[:10]}...")  # First 10 values
                    else:
                        # Log why force data is empty
                        LOG.debug(f"PM5: phase={phase}, force=EMPTY, status={status}")
                else:
                    LOG.warning("PM5 returned empty forceplot response")
                
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
    
    frame_number = 0
    last_stats_time = time.time()
    stats_interval = 5.0  # Log stats every 5 seconds
    last_phase_update = 0
    phase_update_interval = 0.1  # Check cached phase every 100ms (no USB I/O)
    width, height = 0, 0  # Cache dimensions
    
    LOG.info("Starting inference loop with keypoint recording...")
    LOG.info(f"Recorder: buffer_size={recorder.buffer_size}, save_dir={recorder.save_dir}")
    LOG.info(f"Phase control: {controller.get_phase_name()}")
    
    for event in tqdm(
        stream.with_events(),
        desc=f"Recording... {' ':>30}",
        unit='frames',
        leave=False,
        bar_format=PBAR,
        disable=None,
    ):
        if not event.result:
            continue
        
        frame_result = event.result
        frame_number += 1
        now = time.time()
        
        image, meta = frame_result.image, frame_result.meta
        if image is None and meta is None:
            if wnd.is_closed:
                break
            continue
        
        # Update phase from cached ergometer value (fast - no USB I/O)
        if now - last_phase_update > phase_update_interval:
            controller.update_phase_from_external()
            last_phase_update = now
        
        # Extract and record keypoint data (only if meta exists)
        if meta and width > 0:
            frame_data = recorder.extract_keypoints_from_meta(meta, width, height, frame_number, now)
            if frame_data:
                recorder.add_frame(frame_data)
        elif meta and image is not None:
            # First frame - get dimensions
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
            else:
                width, height = image.size
        
        # Display frame
        if image:
            wnd.show(image, meta, frame_result.stream_id)
        
        # Periodic stats logging (reduced frequency)
        if now - last_stats_time > stats_interval:
            stats = recorder.get_stats()
            LOG.info(
                f"Stats: frames={stats['frames_processed']}, "
                f"events_saved={stats['events_saved']}, "
                f"phase={controller.get_phase_name()}, "
                f"queue_depth={stats['queue_depth']}"
            )
            last_stats_time = now
        
        if wnd.is_closed:
            break
    
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
    
    args = parser.parse_args()
    
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
            app.run(interval=1/10)
            
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
