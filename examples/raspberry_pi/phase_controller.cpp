// Copyright Axelera AI, 2025
// Phase Controller Implementation

#include "phase_controller.hpp"
#include "keypoint_recorder.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructor
PhaseController::PhaseController(KeypointRecorder* recorder)
    : recorder_(recorder),
      current_phase_(StrokePhase::IDLE),
      cached_phase_(StrokePhase::IDLE),
      force_buffer_idx_(0),
      polling_active_(false) {
    
    // Initialize force data pointers
    force_data_ptr_.store(&force_data_buffers_[0], std::memory_order_release);
}

// Destructor
PhaseController::~PhaseController() {
    stop();
}

// Set external callback and start background polling
void PhaseController::set_external_callback(ErgometerCallback callback) {
    external_callback_ = callback;
    
    if (!polling_active_.load(std::memory_order_acquire)) {
        polling_active_.store(true, std::memory_order_release);
        polling_thread_ = std::thread(&PhaseController::polling_loop, this);
        
        std::cout << "External phase callback registered, background USB polling started "
                  << "(idle=" << POLL_INTERVAL_IDLE_MS << "ms, "
                  << "drive=" << POLL_INTERVAL_DRIVE_MS << "ms)" << std::endl;
    }
}

// Background polling loop (runs in separate thread - no blocking on main thread)
void PhaseController::polling_loop() {
    StrokePhase last_phase = StrokePhase::IDLE;
    
    while (polling_active_.load(std::memory_order_acquire)) {
        try {
            if (external_callback_) {
                // Call ergometer (USB I/O - blocks for ~20-30ms per PM5 spec)
                PhaseResult result = external_callback_();
                
                // Update cached phase (lock-free atomic write)
                cached_phase_.store(result.phase, std::memory_order_release);
                
                // Update force data if available (double-buffer swap)
                if (!result.force_curve.empty()) {
                    int write_idx = 1 - force_buffer_idx_.load(std::memory_order_acquire);
                    force_data_buffers_[write_idx] = std::move(result.force_curve);
                    force_buffer_idx_.store(write_idx, std::memory_order_release);
                } else if (result.phase != StrokePhase::DRIVE && last_phase == StrokePhase::DRIVE) {
                    // Clear force data when exiting drive
                    int write_idx = 1 - force_buffer_idx_.load(std::memory_order_acquire);
                    force_data_buffers_[write_idx].clear();
                    force_buffer_idx_.store(write_idx, std::memory_order_release);
                }
                
                // Adaptive polling interval
                uint32_t poll_interval = (result.phase == StrokePhase::DRIVE) 
                    ? POLL_INTERVAL_DRIVE_MS 
                    : POLL_INTERVAL_IDLE_MS;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval));
                last_phase = result.phase;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_IDLE_MS));
            }
        } catch (...) {
            // Suppress errors in background thread
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_IDLE_MS));
        }
    }
}

// Update phase from cached value (called from main thread - no USB I/O, lock-free)
void PhaseController::update_phase_from_external() {
    if (external_callback_) {
        StrokePhase new_phase = cached_phase_.load(std::memory_order_acquire);
        StrokePhase old_phase = current_phase_.load(std::memory_order_acquire);
        
        if (new_phase != old_phase) {
            set_phase_internal(new_phase);
        }
    }
}

// Set phase (manual or from external callback)
void PhaseController::set_phase(StrokePhase phase) {
    set_phase_internal(phase);
}

// Internal phase setter
void PhaseController::set_phase_internal(StrokePhase phase) {
    StrokePhase old_phase = current_phase_.exchange(phase, std::memory_order_acq_rel);
    
    if (phase != old_phase) {
        std::cout << "Phase transition: " << phase_name_lookup(old_phase) 
                  << " -> " << phase_name_lookup(phase) << std::endl;
        
        // Notify recorder if attached
        if (recorder_) {
            recorder_->set_phase(static_cast<Phase>(phase));
        }
    }
}

// Get phase name
const char* PhaseController::get_phase_name() const {
    return phase_name_lookup(current_phase_.load(std::memory_order_acquire));
}

// Phase name lookup
const char* PhaseController::phase_name_lookup(StrokePhase phase) {
    switch (phase) {
        case StrokePhase::IDLE:       return "IDLE";
        case StrokePhase::WAIT_ACCEL: return "WAIT_ACCEL";
        case StrokePhase::DRIVE:      return "DRIVE";
        case StrokePhase::DWELLING:   return "DWELLING";
        case StrokePhase::RECOVERY:   return "RECOVERY";
        default:                      return "UNKNOWN";
    }
}

// Get cached force data (lock-free read)
std::vector<uint16_t> PhaseController::get_force_data() const {
    int read_idx = force_buffer_idx_.load(std::memory_order_acquire);
    return force_data_buffers_[read_idx];
}

// Clear force data
void PhaseController::clear_force_data() {
    int write_idx = 1 - force_buffer_idx_.load(std::memory_order_acquire);
    force_data_buffers_[write_idx].clear();
    force_buffer_idx_.store(write_idx, std::memory_order_release);
}

// Stop background polling
void PhaseController::stop() {
    if (polling_active_.load(std::memory_order_acquire)) {
        polling_active_.store(false, std::memory_order_release);
        
        if (polling_thread_.joinable()) {
            polling_thread_.join();
            std::cout << "Ergometer USB polling thread stopped" << std::endl;
        }
    }
}

//=============================================================================
// Heuristic Phase Detector (alternative to ergometer)
//=============================================================================

HeuristicPhaseDetector::HeuristicPhaseDetector()
    : prev_hip_x_(0.0f),
      prev_hip_y_(0.0f),
      prev_knee_x_(0.0f),
      prev_knee_y_(0.0f),
      prev_timestamp_(0.0),
      hip_vx_smooth_(0.0f),
      hip_vy_smooth_(0.0f),
      current_phase_(StrokePhase::IDLE),
      phase_start_time_(0.0),
      frames_in_phase_(0) {}

// Update with keypoint positions
StrokePhase HeuristicPhaseDetector::update(const float* keypoints, size_t num_keypoints, double timestamp) {
    // Extract hip position (COCO index 12 = right_hip)
    constexpr size_t HIP_IDX = 12;
    
    if (HIP_IDX >= num_keypoints) {
        return current_phase_;  // Not enough keypoints
    }
    
    float hip_x = keypoints[HIP_IDX * 3];
    float hip_y = keypoints[HIP_IDX * 3 + 1];
    float hip_conf = keypoints[HIP_IDX * 3 + 2];
    
    if (hip_conf < 0.3f) {
        return current_phase_;  // Low confidence
    }
    
    // Calculate velocity
    float dt = static_cast<float>(timestamp - prev_timestamp_);
    if (prev_timestamp_ > 0.0 && dt > 0.001f && dt < 1.0f) {
        float hip_vx = (hip_x - prev_hip_x_) / dt;
        
        // Exponential smoothing (alpha=0.3)
        constexpr float alpha = 0.3f;
        hip_vx_smooth_ = alpha * hip_vx + (1.0f - alpha) * hip_vx_smooth_;
        
        // Phase detection based on hip velocity
        double time_in_phase = timestamp - phase_start_time_;
        frames_in_phase_++;
        
        // State machine
        switch (current_phase_) {
            case StrokePhase::IDLE:
            case StrokePhase::RECOVERY:
                // Detect drive start: strong leftward motion (negative vx)
                if (hip_vx_smooth_ < VX_DRIVE_THRESHOLD && time_in_phase > MIN_PHASE_DURATION) {
                    current_phase_ = StrokePhase::DRIVE;
                    phase_start_time_ = timestamp;
                    frames_in_phase_ = 0;
                }
                break;
                
            case StrokePhase::DRIVE:
                // Detect drive end: velocity slows or reverses
                if (hip_vx_smooth_ > -20.0f && time_in_phase > MIN_PHASE_DURATION) {
                    current_phase_ = StrokePhase::DWELLING;
                    phase_start_time_ = timestamp;
                    frames_in_phase_ = 0;
                }
                break;
                
            case StrokePhase::DWELLING:
                // Quick transition to recovery
                if (hip_vx_smooth_ > 0.0f && time_in_phase > 0.1) {
                    current_phase_ = StrokePhase::RECOVERY;
                    phase_start_time_ = timestamp;
                    frames_in_phase_ = 0;
                }
                break;
                
            case StrokePhase::WAIT_ACCEL:
                // Not used in heuristic detector
                break;
        }
    }
    
    // Update history
    prev_hip_x_ = hip_x;
    prev_hip_y_ = hip_y;
    prev_timestamp_ = timestamp;
    
    return current_phase_;
}

// Reset detector
void HeuristicPhaseDetector::reset() {
    prev_hip_x_ = 0.0f;
    prev_hip_y_ = 0.0f;
    prev_timestamp_ = 0.0;
    hip_vx_smooth_ = 0.0f;
    current_phase_ = StrokePhase::IDLE;
    phase_start_time_ = 0.0;
    frames_in_phase_ = 0;
}
