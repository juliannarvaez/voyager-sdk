// Copyright Axelera AI, 2025
// High-Performance Keypoint Recorder Implementation

#include "keypoint_recorder.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

// Extract and add frame from raw keypoint data (Axelera kpt_xyv format: x, y, visibility)
void KeypointRecorder::add_frame(const int* keypoints_xyv, size_t num_keypoints,
                                  uint32_t frame_number, double timestamp, Phase phase) {
    // Create frame data (stack allocation)
    FrameData frame;
    frame.frame_number = frame_number;
    frame.timestamp = timestamp;
    frame.phase = phase;
    frame.keypoint_count = 0;
    
    // Extract only the keypoints we care about (fast indexed access)
    // Axelera kpt_xyv is struct {int x, int y, float visibility} = 12 bytes
    for (size_t i = 0; i < ROWING_KEYPOINTS && frame.keypoint_count < ROWING_KEYPOINTS; ++i) {
        uint8_t idx = KEYPOINT_INDICES[i];
        
        // Check bounds (COCO has 17 keypoints)
        if (idx < num_keypoints) {
            // Axelera format: array of kpt_xyv structs (12 bytes each)
            size_t offset = idx * 3;  // 3 ints per keypoint (x, y, visibility as int bits)
            
            frame.keypoints[frame.keypoint_count].x = keypoints_xyv[offset];
            frame.keypoints[frame.keypoint_count].y = keypoints_xyv[offset + 1];
            // Visibility is stored as float but in int array - reinterpret
            frame.keypoints[frame.keypoint_count].visibility = 
                *reinterpret_cast<const float*>(&keypoints_xyv[offset + 2]);
            frame.keypoints[frame.keypoint_count].index = idx;
            frame.keypoint_count++;
        }
    }
    
    // Add to circular buffer (lock-free)
    frame_buffer_.push(frame);
    frames_processed_.fetch_add(1, std::memory_order_relaxed);
    
    // Event capture state machine
    Phase prev_phase = current_phase_.load(std::memory_order_acquire);
    
    if (phase == Phase::DRIVE && !event_active_.load(std::memory_order_acquire)) {
        // Start of drive phase - capture event
        event_active_.store(true, std::memory_order_release);
        post_event_count_.store(0, std::memory_order_release);
        
        // Copy buffered frames to event buffer
        event_frame_count_.store(
            frame_buffer_.get_all(event_frames_, MAX_EVENT_FRAMES),
            std::memory_order_release
        );
    } else if (event_active_.load(std::memory_order_acquire)) {
        // During event capture - add frame
        size_t count = event_frame_count_.load(std::memory_order_acquire);
        if (count < MAX_EVENT_FRAMES) {
            event_frames_[count] = frame;
            event_frame_count_.fetch_add(1, std::memory_order_release);
        }
        
        // Check for event end
        if (phase != Phase::DRIVE) {
            uint32_t post_count = post_event_count_.fetch_add(1, std::memory_order_acq_rel);
            
            // Capture 30 frames after drive phase ends
            if (post_count >= 30) {
                event_active_.store(false, std::memory_order_release);
                // Event ready for async save
            }
        }
    }
}

// Legacy float array support
void KeypointRecorder::add_frame_float(const float* keypoints, size_t num_keypoints,
                                        uint32_t frame_number, double timestamp, Phase phase) {
    // Convert float array to int array format
    // This is less efficient but maintains compatibility
    FrameData frame;
    frame.frame_number = frame_number;
    frame.timestamp = timestamp;
    frame.phase = phase;
    frame.keypoint_count = 0;
    
    for (size_t i = 0; i < ROWING_KEYPOINTS && frame.keypoint_count < ROWING_KEYPOINTS; ++i) {
        uint8_t idx = KEYPOINT_INDICES[i];
        if (idx < num_keypoints) {
            size_t offset = idx * 3;
            frame.keypoints[frame.keypoint_count].x = static_cast<int>(keypoints[offset]);
            frame.keypoints[frame.keypoint_count].y = static_cast<int>(keypoints[offset + 1]);
            frame.keypoints[frame.keypoint_count].visibility = keypoints[offset + 2];
            frame.keypoints[frame.keypoint_count].index = idx;
            frame.keypoint_count++;
        }
    }
    
    frame_buffer_.push(frame);
    frames_processed_.fetch_add(1, std::memory_order_relaxed);
}

// Apply Kalman filtering in-place (Axelera int format)
void KeypointRecorder::apply_kalman_filtering(int* keypoints_xyv, size_t num_keypoints, double timestamp) {
    double current_time = timestamp;
    double last_time = last_timestamp_.load(std::memory_order_acquire);
    
    float dt = 0.0167f;  // Default ~60 FPS
    if (last_time > 0.0) {
        dt = static_cast<float>(std::max(0.001, current_time - last_time));
    }
    
    // Update each tracked keypoint
    for (size_t i = 0; i < ROWING_KEYPOINTS; ++i) {
        uint8_t idx = KEYPOINT_INDICES[i];
        
        if (idx < num_keypoints) {
            size_t offset = idx * 3;  // 3 values per keypoint (x, y, visibility)
            
            float raw_x = static_cast<float>(keypoints_xyv[offset]);
            float raw_y = static_cast<float>(keypoints_xyv[offset + 1]);
            float visibility = *reinterpret_cast<float*>(&keypoints_xyv[offset + 2]);
            
            // Only filter if visibility is good
            if (visibility > 0.3f) {
                float smooth_x, smooth_y;
                kalman_filters_[i].update(raw_x, raw_y, dt, smooth_x, smooth_y);
                
                // Update in-place (convert back to int)
                keypoints_xyv[offset] = static_cast<int>(smooth_x);
                keypoints_xyv[offset + 1] = static_cast<int>(smooth_y);
            }
        }
    }
    
    last_timestamp_.store(current_time, std::memory_order_release);
}

// Apply Kalman filtering (float format)
void KeypointRecorder::apply_kalman_filtering_float(float* keypoints, size_t num_keypoints, double timestamp) {
    double current_time = timestamp;
    double last_time = last_timestamp_.load(std::memory_order_acquire);
    
    float dt = 0.0167f;
    if (last_time > 0.0) {
        dt = static_cast<float>(std::max(0.001, current_time - last_time));
    }
    
    for (size_t i = 0; i < ROWING_KEYPOINTS; ++i) {
        uint8_t idx = KEYPOINT_INDICES[i];
        
        if (idx < num_keypoints) {
            size_t offset = idx * 3;
            
            float raw_x = keypoints[offset];
            float raw_y = keypoints[offset + 1];
            float confidence = keypoints[offset + 2];
            
            if (confidence > 0.3f) {
                float smooth_x, smooth_y;
                kalman_filters_[i].update(raw_x, raw_y, dt, smooth_x, smooth_y);
                
                keypoints[offset] = smooth_x;
                keypoints[offset + 1] = smooth_y;
            }
        }
    }
    
    last_timestamp_.store(current_time, std::memory_order_release);
}

// Set current phase
void KeypointRecorder::set_phase(Phase phase) {
    current_phase_.store(phase, std::memory_order_release);
}

// Save event to disk (JSON format)
bool KeypointRecorder::save_event_async(const char* save_dir) {
    if (!event_active_.load(std::memory_order_acquire)) {
        return false;
    }
    
    // Create directory if needed
    mkdir(save_dir, 0755);
    
    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::ostringstream filename;
    filename << save_dir << "/stroke_" << ms << ".json";
    
    // Open file
    std::ofstream file(filename.str());
    if (!file.is_open()) {
        return false;
    }
    
    // Write JSON (simple format, no dependencies)
    size_t count = event_frame_count_.load(std::memory_order_acquire);
    
    file << "{\n";
    file << "  \"event_type\": \"rowing_stroke\",\n";
    file << "  \"frame_count\": " << count << ",\n";
    file << "  \"frames\": [\n";
    
    for (size_t i = 0; i < count; ++i) {
        const FrameData& frame = event_frames_[i];
        
        file << "    {\n";
        file << "      \"frame_number\": " << frame.frame_number << ",\n";
        file << "      \"timestamp\": " << std::fixed << std::setprecision(6) << frame.timestamp << ",\n";
        file << "      \"phase\": " << static_cast<int>(frame.phase) << ",\n";
        file << "      \"keypoints\": [\n";
        
        for (size_t j = 0; j < frame.keypoint_count; ++j) {
            const Keypoint& kp = frame.keypoints[j];
            file << "        {\n";
            file << "          \"name\": \"" << KEYPOINT_NAMES[j] << "\",\n";
            file << "          \"x\": " << kp.x << ",\n";
            file << "          \"y\": " << kp.y << ",\n";
            file << "          \"confidence\": " << kp.visibility << "\n";
            file << "        }" << (j < frame.keypoint_count - 1 ? "," : "") << "\n";
        }
        
        file << "      ]\n";
        file << "    }" << (i < count - 1 ? "," : "") << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    
    events_saved_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Get FPS statistics
FPSBenchmark::Stats FPSBenchmark::get_stats() const {
    Stats stats = {0};
    
    uint64_t count = frame_count_.load(std::memory_order_acquire);
    if (count == 0) {
        return stats;
    }
    
    // Calculate stats from intervals array
    size_t valid_count = std::min(count, static_cast<uint64_t>(window_size_));
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float min_interval = 1e9f;
    float max_interval = 0.0f;
    uint32_t drops = 0;
    
    for (size_t i = 0; i < valid_count; ++i) {
        float interval = intervals_[i];
        if (interval > 0.0f) {
            sum += interval;
            sum_sq += interval * interval;
            min_interval = std::min(min_interval, interval);
            max_interval = std::max(max_interval, interval);
            
            if (interval > drop_threshold_) {
                drops++;
            }
        }
    }
    
    float mean = sum / valid_count;
    float variance = (sum_sq / valid_count) - (mean * mean);
    float std_dev = std::sqrt(std::max(0.0f, variance));
    
    double elapsed = last_timestamp_.load(std::memory_order_acquire) - 
                    start_time_.load(std::memory_order_acquire);
    
    stats.avg_fps = (mean > 0.0f) ? (1.0f / mean) : 0.0f;
    stats.actual_fps = (elapsed > 0.0) ? (static_cast<float>(count) / elapsed) : 0.0f;
    stats.jitter_ms = std_dev * 1000.0f;
    stats.min_ms = min_interval * 1000.0f;
    stats.max_ms = max_interval * 1000.0f;
    stats.drops = drops;
    stats.drop_rate = (valid_count > 0) ? ((100.0f * drops) / valid_count) : 0.0f;
    
    return stats;
}
