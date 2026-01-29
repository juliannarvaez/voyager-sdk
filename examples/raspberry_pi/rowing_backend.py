#!/usr/bin/env python3
# Copyright Axelera AI, 2025
# Backend selector for rowing ergometer - C++ or Python fallback

"""
Automatically selects C++ backend if available, otherwise falls back to Python.
This provides transparent performance improvements without code changes.
"""

import os
import sys

# Try to import C++ implementation first
try:
    import rowing_cpp
    USE_CPP = True
    print("[PERFORMANCE] Using C++ backend for keypoint recording (0.1ms jitter)")
    
    KeypointRecorder = rowing_cpp.KeypointRecorder
    PhaseController = rowing_cpp.PhaseController
    FPSBenchmark = rowing_cpp.FPSBenchmark
    Phase = rowing_cpp.Phase
    
except ImportError as e:
    print(f"[INFO] C++ backend not available ({e}), using Python fallback")
    USE_CPP = False
    
    # Import Python implementation
    try:
        from rowing_ergometer import KeypointRecorder, PhaseController, FPSBenchmark, Phase
    except ImportError:
        print("[ERROR] Neither C++ nor Python backend available!")
        sys.exit(1)

# Always import pyrow from rowing_ergometer module
try:
    from rowing_ergometer import pyrow
except ImportError:
    print("[WARNING] pyrow not available - ergometer integration disabled")
    pyrow = None

__all__ = ['KeypointRecorder', 'PhaseController', 'FPSBenchmark', 'Phase', 'pyrow', 'USE_CPP']
