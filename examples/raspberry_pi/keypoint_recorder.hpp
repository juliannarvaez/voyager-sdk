// Copyright Axelera AI, 2025
// High-Performance Keypoint Recorder - Zero-allocation design for minimal jitter

#ifndef KEYPOINT_RECORDER_HPP
#define KEYPOINT_RECORDER_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>

// Configuration constants - compile-time fixed for optimal performance
constexpr size_t MAX_BUFFER_SIZE = 128;         // Maximum frames in circular buffer
constexpr size_t MAX_EVENT_FRAMES = 512;        // Maximum frames per event capture
constexpr size_t MAX_KEYPOINTS = 17;            // COCO pose has 17 keypoints
constexpr size_t ROWING_KEYPOINTS = 5;          // We track 5 keypoints for rowing

// Keypoint indices for rowing (COCO format)
enum RowingKeypoint : uint8_t {
    RIGHT_SHOULDER = 6,
    RIGHT_WRIST = 10,
    RIGHT_HIP = 12,
    RIGHT_KNEE = 14,
    RIGHT_ANKLE = 16
};

// Static keypoint names lookup (no dynamic strings)
constexpr const char* KEYPOINT_NAMES[ROWING_KEYPOINTS] = {
    "right_shoulder",
    "right_wrist", 
    "right_hip",
    "right_knee",
    "right_ankle"
};

constexpr uint8_t KEYPOINT_INDICES[ROWING_KEYPOINTS] = {
    RIGHT_SHOULDER, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
};

// Stroke phases
enum Phase : uint8_t {
    IDLE = 0,
    WAIT_ACCEL = 1,
    DRIVE = 2,
    DWELLING = 3,
    RECOVERY = 4
};

// Single keypoint data (fixed size, cache-aligned)
// Compatible with Axelera's kpt_xyv structure
struct alignas(16) Keypoint {
    int x;              // Match Axelera kpt_xyv
    int y;              // Match Axelera kpt_xyv
    float visibility;   // Match Axelera kpt_xyv (confidence)
    uint8_t index;
    uint8_t padding[3];  // Align to 16 bytes
};

// Frame keypoint data (pre-allocated, no heap)
struct alignas(64) FrameData {
    Keypoint keypoints[ROWING_KEYPOINTS];
    double timestamp;
    uint32_t frame_number;
    Phase phase;
    uint8_t keypoint_count;
    uint8_t padding[2];
};

// Kalman filter state (2D position + velocity)
struct alignas(64) KalmanState {
    float x;      // Position X
    float y;      // Position Y
    float vx;     // Velocity X
    float vy;     // Velocity Y
    float P[4][4]; // State covariance (4x4)
    bool initialized;
    uint8_t padding[3];
};

// Kalman filter for keypoint smoothing (optimized, in-place operations)
class KalmanFilter2D {
public:
    KalmanFilter2D(float process_noise = 0.1f, float measurement_noise = 4.0f)
        : Q(process_noise), R(measurement_noise) {
        reset();
    }
    
    void reset() {
        state.x = 0.0f;
        state.y = 0.0f;
        state.vx = 0.0f;
        state.vy = 0.0f;
        
        // Initialize covariance to high uncertainty
        std::memset(state.P, 0, sizeof(state.P));
        state.P[0][0] = state.P[1][1] = state.P[2][2] = state.P[3][3] = 100.0f;
        
        state.initialized = false;
    }
    
    // Update with new measurement (in-place, minimal allocations)
    inline void update(float mx, float my, float dt, float& out_x, float& out_y) {
        if (!state.initialized) {
            state.x = mx;
            state.y = my;
            state.initialized = true;
            out_x = mx;
            out_y = my;
            return;
        }
        
        // Predict step (constant velocity model)
        state.x += state.vx * dt;
        state.y += state.vy * dt;
        
        // Simplified covariance prediction (diagonal Q optimization)
        state.P[0][0] += Q + state.P[2][2] * dt * dt;
        state.P[1][1] += Q + state.P[3][3] * dt * dt;
        state.P[2][2] += Q;
        state.P[3][3] += Q;
        
        // Innovation
        float y0 = mx - state.x;
        float y1 = my - state.y;
        
        // Innovation covariance
        float S00 = state.P[0][0] + R;
        float S11 = state.P[1][1] + R;
        
        // Kalman gain (simplified for diagonal R)
        float K00 = state.P[0][0] / S00;
        float K11 = state.P[1][1] / S11;
        
        // State update
        state.x += K00 * y0;
        state.y += K11 * y1;
        
        // Velocity update (estimate from residual)
        if (dt > 0.001f) {
            float alpha = 0.3f;  // Smoothing factor
            state.vx = (1.0f - alpha) * state.vx + alpha * (y0 / dt);
            state.vy = (1.0f - alpha) * state.vy + alpha * (y1 / dt);
        }
        
        // Covariance update (simplified Joseph form)
        state.P[0][0] *= (1.0f - K00);
        state.P[1][1] *= (1.0f - K11);
        
        out_x = state.x;
        out_y = state.y;
    }
    
private:
    KalmanState state;
    float Q;  // Process noise
    float R;  // Measurement noise
};

// Circular buffer for frame data (lock-free single producer, single consumer)
class FrameBuffer {
public:
    explicit FrameBuffer(size_t capacity = MAX_BUFFER_SIZE) 
        : capacity_(capacity), head_(0), tail_(0), size_(0) {}
    
    // Push frame (returns false if full)
    inline bool push(const FrameData& frame) {
        size_t current_size = size_.load(std::memory_order_acquire);
        if (current_size >= capacity_) {
            // Overwrite oldest (circular buffer behavior)
            tail_.fetch_add(1, std::memory_order_release);
            size_.fetch_sub(1, std::memory_order_release);
        }
        
        size_t head = head_.load(std::memory_order_acquire);
        buffer_[head % MAX_BUFFER_SIZE] = frame;
        head_.fetch_add(1, std::memory_order_release);
        size_.fetch_add(1, std::memory_order_release);
        return true;
    }
    
    // Pop frame (returns false if empty)
    inline bool pop(FrameData& frame) {
        size_t current_size = size_.load(std::memory_order_acquire);
        if (current_size == 0) {
            return false;
        }
        
        size_t tail = tail_.load(std::memory_order_acquire);
        frame = buffer_[tail % MAX_BUFFER_SIZE];
        tail_.fetch_add(1, std::memory_order_release);
        size_.fetch_sub(1, std::memory_order_release);
        return true;
    }
    
    inline size_t size() const {
        return size_.load(std::memory_order_acquire);
    }
    
    inline bool empty() const {
        return size() == 0;
    }
    
    inline void clear() {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
        size_.store(0, std::memory_order_release);
    }
    
    // Get frames without removing (for event capture)
    inline size_t get_all(FrameData* out_frames, size_t max_frames) const {
        size_t current_size = size_.load(std::memory_order_acquire);
        size_t n = std::min(current_size, max_frames);
        
        size_t tail = tail_.load(std::memory_order_acquire);
        for (size_t i = 0; i < n; ++i) {
            out_frames[i] = buffer_[(tail + i) % MAX_BUFFER_SIZE];
        }
        return n;
    }
    
private:
    std::array<FrameData, MAX_BUFFER_SIZE> buffer_;
    size_t capacity_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    std::atomic<size_t> size_;
};

// High-performance keypoint recorder (static allocation only)
class KeypointRecorder {
public:
    explicit KeypointRecorder(size_t buffer_size = 30)
        : buffer_size_(std::min(buffer_size, MAX_BUFFER_SIZE)),
          current_phase_(Phase::IDLE),
          event_active_(false),
          post_event_count_(0),
          frames_processed_(0),
          events_saved_(0),
          last_timestamp_(0.0) {
        
        // Pre-allocate Kalman filters
        for (size_t i = 0; i < ROWING_KEYPOINTS; ++i) {
            kalman_filters_[i].reset();
        }
        
        // Initialize event buffer
        std::memset(event_frames_, 0, sizeof(event_frames_));
        event_frame_count_ = 0;
    }
    
    // Extract and add frame from raw keypoint data (zero-copy when possible)
    // Accepts Axelera kpt_xyv array or raw float array
    void add_frame(const int* keypoints_xyv, size_t num_keypoints, 
                   uint32_t frame_number, double timestamp, Phase phase);
    
    // Overload for float array (legacy format)
    void add_frame_float(const float* keypoints, size_t num_keypoints,
                        uint32_t frame_number, double timestamp, Phase phase);
    
    // Apply Kalman filtering in-place (for int keypoints from Axelera)
    void apply_kalman_filtering(int* keypoints_xyv, size_t num_keypoints, double timestamp);
    
    // Overload for float array
    void apply_kalman_filtering_float(float* keypoints, size_t num_keypoints, double timestamp);
    
    // Set current phase (triggers event capture state machine)
    void set_phase(Phase phase);
    
    // Get statistics
    struct Stats {
        uint64_t frames_processed;
        uint32_t events_saved;
        uint32_t buffer_depth;
        Phase current_phase;
    };
    
    Stats get_stats() const {
        return Stats{
            frames_processed_.load(),
            events_saved_.load(),
            static_cast<uint32_t>(frame_buffer_.size()),
            current_phase_.load()
        };
    }
    
    // Save event to disk (called from background thread)
    bool save_event_async(const char* save_dir);
    
private:
    // Configuration
    size_t buffer_size_;
    
    // Circular frame buffer
    FrameBuffer frame_buffer_;
    
    // Phase state machine
    std::atomic<Phase> current_phase_;
    std::atomic<bool> event_active_;
    std::atomic<uint32_t> post_event_count_;
    
    // Event capture buffer (pre-allocated, no heap)
    FrameData event_frames_[MAX_EVENT_FRAMES];
    std::atomic<size_t> event_frame_count_;
    
    // Kalman filters (one per tracked keypoint)
    KalmanFilter2D kalman_filters_[ROWING_KEYPOINTS];
    
    // Statistics
    std::atomic<uint64_t> frames_processed_;
    std::atomic<uint32_t> events_saved_;
    std::atomic<double> last_timestamp_;
};

// FPS benchmark (lock-free, minimal overhead)
class FPSBenchmark {
public:
    explicit FPSBenchmark(size_t window_size = 300, float drop_threshold_ms = 25.0f)
        : window_size_(window_size),
          drop_threshold_(drop_threshold_ms / 1000.0f),
          write_idx_(0),
          frame_count_(0),
          start_time_(0.0),
          last_timestamp_(0.0) {
        std::memset(intervals_, 0, sizeof(intervals_));
    }
    
    inline void record_frame(double timestamp) {
        if (start_time_ == 0.0) {
            start_time_ = timestamp;
            last_timestamp_ = timestamp;
            return;
        }
        
        float interval = static_cast<float>(timestamp - last_timestamp_);
        size_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % window_size_;
        intervals_[idx] = interval;
        
        last_timestamp_ = timestamp;
        frame_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    struct Stats {
        float avg_fps;
        float actual_fps;
        float jitter_ms;
        float min_ms;
        float max_ms;
        uint32_t drops;
        float drop_rate;
    };
    
    Stats get_stats() const;
    
private:
    static constexpr size_t MAX_WINDOW = 1024;
    size_t window_size_;
    float drop_threshold_;
    
    std::atomic<size_t> write_idx_;
    std::atomic<uint64_t> frame_count_;
    std::atomic<double> start_time_;
    std::atomic<double> last_timestamp_;
    
    // Fixed-size ring buffer (no allocation)
    float intervals_[MAX_WINDOW];
};

#endif // KEYPOINT_RECORDER_HPP
