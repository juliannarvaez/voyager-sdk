// Copyright Axelera AI, 2025
// Phase Controller with PM5 Ergometer Integration

#ifndef PHASE_CONTROLLER_HPP
#define PHASE_CONTROLLER_HPP

#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <vector>
#include <cstdint>

// Forward declaration
class KeypointRecorder;

// Phase enumeration (matches Python StrokePhase)
enum class StrokePhase : uint8_t {
    IDLE = 0,
    WAIT_ACCEL = 1,
    DRIVE = 2,
    DWELLING = 3,
    RECOVERY = 4
};

// Phase result from ergometer (phase + optional force data)
struct PhaseResult {
    StrokePhase phase;
    std::vector<uint16_t> force_curve;  // Force data from PM5 (if available)
};

// Ergometer callback type
using ErgometerCallback = std::function<PhaseResult()>;

// High-performance phase controller with background USB polling
class PhaseController {
public:
    explicit PhaseController(KeypointRecorder* recorder = nullptr);
    ~PhaseController();
    
    // Set external phase callback (e.g., PM5 ergometer)
    void set_external_callback(ErgometerCallback callback);
    
    // Update phase from cached value (called from inference thread - no USB I/O)
    void update_phase_from_external();
    
    // Manual phase control
    void set_phase(StrokePhase phase);
    
    // Get current phase
    StrokePhase get_phase() const {
        return current_phase_.load(std::memory_order_acquire);
    }
    
    // Get phase name for logging
    const char* get_phase_name() const;
    
    // Get cached force data
    std::vector<uint16_t> get_force_data() const;
    
    // Clear force data
    void clear_force_data();
    
    // Stop background polling
    void stop();
    
private:
    KeypointRecorder* recorder_;
    
    // Current phase (atomic for lock-free access)
    std::atomic<StrokePhase> current_phase_;
    
    // Cached phase from background thread (avoids USB I/O in main thread)
    std::atomic<StrokePhase> cached_phase_;
    
    // Force curve data (protected by atomic swap pattern)
    std::atomic<std::vector<uint16_t>*> force_data_ptr_;
    std::vector<uint16_t> force_data_buffers_[2];  // Double buffering
    std::atomic<int> force_buffer_idx_;
    
    // External callback
    ErgometerCallback external_callback_;
    
    // Background polling thread
    std::thread polling_thread_;
    std::atomic<bool> polling_active_;
    
    // Polling intervals (adaptive based on phase)
    static constexpr uint32_t POLL_INTERVAL_IDLE_MS = 300;   // 300ms during idle/recovery
    static constexpr uint32_t POLL_INTERVAL_DRIVE_MS = 50;   // 50ms during drive (catch force data)
    
    // Background polling loop
    void polling_loop();
    
    // Internal phase setter
    void set_phase_internal(StrokePhase phase);
    
    // Phase name lookup
    static const char* phase_name_lookup(StrokePhase phase);
};

// Heuristic phase detector (alternative to ergometer - uses keypoint motion)
class HeuristicPhaseDetector {
public:
    HeuristicPhaseDetector();
    
    // Update with current keypoint positions
    // Returns detected phase based on motion analysis
    StrokePhase update(const float* keypoints, size_t num_keypoints, double timestamp);
    
    // Reset detector state
    void reset();
    
private:
    // Previous keypoint positions for velocity calculation
    float prev_hip_x_;
    float prev_hip_y_;
    float prev_knee_x_;
    float prev_knee_y_;
    double prev_timestamp_;
    
    // Velocity smoothing
    float hip_vx_smooth_;
    float hip_vy_smooth_;
    
    // Current phase
    StrokePhase current_phase_;
    
    // Phase timing
    double phase_start_time_;
    uint32_t frames_in_phase_;
    
    // Thresholds
    static constexpr float VX_DRIVE_THRESHOLD = -50.0f;    // Negative = moving left (drive)
    static constexpr float VX_RECOVERY_THRESHOLD = 30.0f;  // Positive = moving right (recovery)
    static constexpr float MIN_PHASE_DURATION = 0.2f;      // Min 200ms per phase
};

#endif // PHASE_CONTROLLER_HPP
