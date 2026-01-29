// Copyright Axelera AI, 2025
// Python binding for C++ KeypointRecorder and PhaseController

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "keypoint_recorder.hpp"
#include "phase_controller.hpp"

namespace py = pybind11;

// Python-accessible callback wrapper
class PythonCallback {
public:
    KeypointRecorder& recorder;
    PhaseController& controller;
    FPSBenchmark& fps_bench;
    
    uint32_t frame_count = 0;
    uint32_t phase_update_counter = 0;
    const uint32_t PHASE_UPDATE_INTERVAL = 15;  // Update phase every 15 frames (~167ms @ 90 FPS)
    
    PythonCallback(KeypointRecorder& rec, PhaseController& ctrl, FPSBenchmark& fps)
        : recorder(rec), controller(ctrl), fps_bench(fps) {}
    
    // Process frame from NumPy array
    void process_frame_numpy(py::array_t<int> kpts_array, uint32_t frame_num, double timestamp) {
        py::buffer_info buf = kpts_array.request();
        
        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("Expected shape (N, 3) for keypoints");
        }
        
        int* kpts_ptr = static_cast<int*>(buf.ptr);
        size_t num_kpts = buf.shape[0];
        
        // Update phase periodically (not every frame to reduce overhead)
        if (++phase_update_counter >= PHASE_UPDATE_INTERVAL) {
            controller.update_phase_from_external();
            phase_update_counter = 0;
        }
        
        // Add frame to recorder
        StrokePhase current_phase = controller.get_phase();
        recorder.add_frame(kpts_ptr, num_kpts, frame_num, timestamp, 
                          static_cast<Phase>(current_phase));
        
        // Update FPS benchmark
        fps_bench.record_frame(timestamp);
        
        ++frame_count;
    }
    
    // Get current stats
    py::dict get_stats() {
        auto rec_stats = recorder.get_stats();
        auto fps_stats = fps_bench.get_stats();
        
        py::dict stats;
        stats["frames_processed"] = rec_stats.frames_processed;
        stats["events_saved"] = rec_stats.events_saved;
        stats["buffer_depth"] = rec_stats.buffer_depth;
        stats["phase"] = static_cast<int>(controller.get_phase());
        stats["phase_name"] = controller.get_phase_name();
        stats["actual_fps"] = fps_stats.actual_fps;
        stats["jitter_ms"] = fps_stats.jitter_ms;
        stats["drops"] = fps_stats.drops;
        stats["drop_rate"] = fps_stats.drop_rate;
        
        return stats;
    }
};

PYBIND11_MODULE(rowing_cpp, m) {
    m.doc() = "C++ KeypointRecorder and PhaseController bindings for rowing ergometer";
    
    // Enum bindings
    py::enum_<Phase>(m, "Phase")
        .value("IDLE", Phase::IDLE)
        .value("WAIT_ACCEL", Phase::WAIT_ACCEL)
        .value("DRIVE", Phase::DRIVE)
        .value("DWELLING", Phase::DWELLING)
        .value("RECOVERY", Phase::RECOVERY)
        .export_values();
    
    py::enum_<StrokePhase>(m, "StrokePhase")
        .value("IDLE", StrokePhase::IDLE)
        .value("WAIT_ACCEL", StrokePhase::WAIT_ACCEL)
        .value("DRIVE", StrokePhase::DRIVE)
        .value("DWELLING", StrokePhase::DWELLING)
        .value("RECOVERY", StrokePhase::RECOVERY)
        .export_values();
    
    // Stats struct binding
    py::class_<KeypointRecorder::Stats>(m, "RecorderStats")
        .def_readonly("frames_processed", &KeypointRecorder::Stats::frames_processed)
        .def_readonly("events_saved", &KeypointRecorder::Stats::events_saved)
        .def_readonly("buffer_depth", &KeypointRecorder::Stats::buffer_depth)
        .def_readonly("current_phase", &KeypointRecorder::Stats::current_phase);
    
    // KeypointRecorder
    py::class_<KeypointRecorder>(m, "KeypointRecorder")
        .def(py::init([](size_t buffer_size, const char* save_dir) {
            // C++ version uses hardcoded /tmp/stroke_data, ignore save_dir for compatibility
            return new KeypointRecorder(buffer_size);
        }),
             py::arg("buffer_size") = 30,
             py::arg("save_dir") = "/tmp/stroke_data")
        .def("get_stats", [](const KeypointRecorder& self) -> py::dict {
            auto stats = self.get_stats();
            py::dict result;
            result["frames_processed"] = stats.frames_processed;
            result["events_saved"] = stats.events_saved;
            result["buffer_depth"] = stats.buffer_depth;
            result["queue_depth"] = stats.buffer_depth;  // Alias for compatibility
            return result;
        })
        .def("set_phase", [](KeypointRecorder& self, py::object phase_obj) {
            // Accept both int and Phase enum
            if (py::isinstance<py::int_>(phase_obj)) {
                int phase_int = phase_obj.cast<int>();
                self.set_phase(static_cast<Phase>(phase_int));
            } else {
                Phase phase = phase_obj.cast<Phase>();
                self.set_phase(phase);
            }
        })
        .def("add_frame", [](KeypointRecorder& self, const std::vector<int>& keypoints_xyv, 
                            size_t num_keypoints, uint32_t frame_num, double timestamp, int phase) {
            // Expose C++ add_frame to Python for direct calling
            self.add_frame(keypoints_xyv.data(), num_keypoints, frame_num, timestamp, static_cast<Phase>(phase));
        }, py::arg("keypoints_xyv"), py::arg("num_keypoints"), 
           py::arg("frame_num"), py::arg("timestamp"), py::arg("phase"))
        .def("apply_kalman_to_meta", [](KeypointRecorder& self, py::object meta, double timestamp) {
            // Python compatibility - does nothing in C++ (Kalman applied in add_frame)
            // The Python version modifies meta in-place; C++ does it during add_frame
        })
        .def("extract_keypoints_from_meta", [](KeypointRecorder& self, py::object meta, int width, int height, 
                                               uint32_t frame_num, double timestamp) -> py::object {
            // Extract keypoints from Axelera meta object and add to C++ recorder
            try {
                static int debug_frame_count = 0;
                static int success_count = 0;
                static int empty_count = 0;
                bool debug_this_frame = (debug_frame_count < 2);  // First 2 frames with structure
                
                // AxMeta is dict-like, iterate through values to find keypoint metadata
                py::object keypoints_meta;
                bool found = false;
                
                if (debug_this_frame) {
                    py::print("\n[C++ DEBUG] ===== Frame", frame_num, "- Exploring AxMeta =====");
                }
                
                for (auto item : meta) {
                    auto value = meta[item];
                    
                    if (debug_this_frame) {
                        py::print("  Task:", item, ", type:", py::type::of(value).attr("__name__"));
                        // Print attributes
                        if (py::hasattr(value, "__dict__")) {
                            auto attrs = value.attr("__dict__");
                            py::print("    __dict__:", attrs);
                        }
                        if (py::hasattr(value, "keypoints")) {
                            auto kpts = value.attr("keypoints");
                            py::print("    ✓ keypoints attr, type:", py::type::of(kpts).attr("__name__"));
                            if (py::hasattr(kpts, "shape")) {
                                py::print("      shape:", kpts.attr("shape"));
                            }
                        }
                        if (py::hasattr(value, "boxes")) {
                            auto boxes = value.attr("boxes");
                            py::print("    ✓ boxes attr, type:", py::type::of(boxes).attr("__name__"));
                            if (py::hasattr(boxes, "shape")) {
                                py::print("      shape:", boxes.attr("shape"));
                            }
                        }
                    }
                    
                    // Check if this value has keypoints attribute
                    if (py::hasattr(value, "keypoints")) {
                        keypoints_meta = value;
                        found = true;
                        if (!debug_this_frame) break;  // Only keep exploring if debugging
                    }
                }
                
                if (debug_this_frame) {
                    py::print("[C++ DEBUG] ===== End structure =====\n");
                }
                debug_frame_count++;
                
                if (!found) {
                    return py::none();
                }
                
                auto keypoints_list = keypoints_meta.attr("keypoints");
                
                // NumPy array - check shape instead of len
                if (keypoints_list.is_none()) {
                    return py::none();
                }
                
                // Get shape attribute for NumPy array
                auto shape = keypoints_list.attr("shape");
                auto shape_tuple = shape.cast<py::tuple>();
                int num_people = shape_tuple[0].cast<int>();
                
                // Check if array is empty or has no detections
                if (py::len(shape_tuple) < 2 || num_people == 0) {
                    empty_count++;
                    if (empty_count % 100 == 1 || empty_count <= 10) {
                        py::print("[C++ INFO] Frame", frame_num, "- No people detected (empty_count=", 
                                 empty_count, ", success_count=", success_count, ")");
                    }
                    return py::none();  // No people detected
                }
                
                // Get first person's keypoints
                // Expected shape: (num_people, num_keypoints, 3) where 3 = [x, y, confidence]
                auto first_person = keypoints_list[py::int_(0)];
                
                // Convert NumPy array to list for easier iteration
                auto kpts_array = first_person.cast<py::array_t<float>>();
                auto buf = kpts_array.request();
                float* ptr = static_cast<float*>(buf.ptr);
                size_t num_kpts = buf.shape[0];
                
                // Build keypoints array in Axelera format: [x, y, vis, x, y, vis, ...]
                std::vector<int> kpts_xyv;
                
                for (size_t i = 0; i < num_kpts; ++i) {
                    int x = static_cast<int>(ptr[i * 3 + 0]);
                    int y = static_cast<int>(ptr[i * 3 + 1]);
                    float vis = ptr[i * 3 + 2];
                    
                    kpts_xyv.push_back(x);
                    kpts_xyv.push_back(y);
                    // Store visibility as float bits in int
                    int vis_bits;
                    std::memcpy(&vis_bits, &vis, sizeof(float));
                    kpts_xyv.push_back(vis_bits);
                }
                
                if (!kpts_xyv.empty()) {
                    success_count++;
                    if (success_count <= 5 || success_count % 50 == 0) {
                        py::print("[C++ SUCCESS] Frame", frame_num, "- Extracted", kpts_xyv.size() / 3, 
                                 "keypoints (success_count=", success_count, ", empty_count=", empty_count, ")");
                    }
                    
                    // Call C++ add_frame directly
                    Phase phase = static_cast<Phase>(0);  // Will be set by controller
                    self.add_frame(kpts_xyv.data(), kpts_xyv.size() / 3, frame_num, timestamp, phase);
                }
            } catch (const std::exception& e) {
                static int error_count = 0;
                if (error_count++ < 5) {
                    py::print("[C++ ERROR] Frame", frame_num, "-", e.what());
                }
            }
            
            // Always return None - we already called add_frame internally
            return py::none();
        })
        .def_property_readonly("buffer_size", [](const KeypointRecorder& self) {
            return self.get_stats().buffer_depth;  // Use current buffer depth
        })
        .def_property_readonly("save_dir", [](const KeypointRecorder&) {
            return "/tmp/stroke_data";  // Hardcoded in C++ implementation
        });
    
    // PhaseController
    py::class_<PhaseController>(m, "PhaseController")
        .def(py::init([](py::object recorder_obj) {
            // Accept optional KeypointRecorder parameter
            if (recorder_obj.is_none()) {
                return new PhaseController(nullptr);
            } else {
                auto* recorder = recorder_obj.cast<KeypointRecorder*>();
                return new PhaseController(recorder);
            }
        }),
             py::arg("recorder") = py::none())
        .def("get_phase", &PhaseController::get_phase)
        .def("get_phase_name", &PhaseController::get_phase_name)
        .def("update_phase_from_external", &PhaseController::update_phase_from_external)
        .def("set_external_callback", [](PhaseController& self, py::function callback) {
            // Wrap Python callback
            auto cpp_callback = [callback]() -> PhaseResult {
                py::gil_scoped_acquire acquire;
                py::dict result = callback();
                PhaseResult data;
                data.phase = static_cast<StrokePhase>(result["phase"].cast<int>());
                if (result.contains("force")) {
                    auto force_list = result["force"].cast<std::vector<uint16_t>>();
                    data.force_curve = force_list;
                }
                return data;
            };
            self.set_external_callback(cpp_callback);
        })
        .def("set_external_phase_callback", [](PhaseController& self, py::function callback) {
            // Alias for compatibility with Python API
            auto cpp_callback = [callback]() -> PhaseResult {
                py::gil_scoped_acquire acquire;
                py::dict result = callback();
                PhaseResult data;
                data.phase = static_cast<StrokePhase>(result["phase"].cast<int>());
                if (result.contains("force")) {
                    auto force_list = result["force"].cast<std::vector<uint16_t>>();
                    data.force_curve = force_list;
                }
                return data;
            };
            self.set_external_callback(cpp_callback);
        })
        .def("stop", &PhaseController::stop)
        .def_property_readonly("current_phase", [](PhaseController& self) {
            return static_cast<int>(self.get_phase());
        });
    
    // FPSBenchmark
    py::class_<FPSBenchmark>(m, "FPSBenchmark")
        .def(py::init<>())
        .def("record_frame", &FPSBenchmark::record_frame)
        .def("get_stats", &FPSBenchmark::get_stats);
    
    // High-level callback wrapper
    py::class_<PythonCallback>(m, "InferenceCallback")
        .def(py::init<KeypointRecorder&, PhaseController&, FPSBenchmark&>())
        .def("process_frame", &PythonCallback::process_frame_numpy,
             py::arg("keypoints"),
             py::arg("frame_num"),
             py::arg("timestamp"))
        .def("get_stats", &PythonCallback::get_stats)
        .def_readonly("frame_count", &PythonCallback::frame_count);
}
