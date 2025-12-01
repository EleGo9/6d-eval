#!/usr/bin/env python3
"""
Test script to compare CPU vs GPU metrics computation performance.
"""
import time
import torch
import numpy as np
from metrics import Metrics
from metrics_gpu import GPUMetrics
import sys
import argparse

def test_device_availability():
    """Test CUDA availability and device info."""
    print("=== Device Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")

        # Test basic GPU operations
        print("\n=== GPU Functionality Test ===")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            start = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()  # Wait for GPU computation to complete
            end = time.time()
            print(f"GPU matrix multiplication test: SUCCESS ({end-start:.4f}s)")

            # Test memory
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            print(f"GPU memory allocated: {memory_allocated:.1f} MB")

        except Exception as e:
            print(f"GPU test failed: {e}")
    else:
        print("CUDA not available. Will use CPU.")

    print("="*50)

def benchmark_metrics(conf_path: str, iterations: int = 1):
    """
    Benchmark CPU vs GPU metrics computation.
    """
    print(f"\n=== Benchmarking Metrics Computation ===")

    try:
        # Test GPU version
        print("Testing GPU implementation...")
        gpu_times = []
        gpu_success = True

        try:
            for i in range(iterations):
                print(f"GPU iteration {i+1}/{iterations}")
                start_time = time.time()
                gpu_metrics = GPUMetrics(conf_path=conf_path, device='cuda' if torch.cuda.is_available() else 'cpu')

                # Run a limited test (don't run full dataset for benchmark)
                # Just test initialization and basic operations
                test_prediction = [{
                    'obj_id': 0,
                    'cam_R_m2c': np.eye(3).tolist(),
                    'cam_t_m2c': [0.0, 0.0, 1.0],
                    'obj_bb': [100, 100, 50, 50]
                }]
                test_annotation = [{
                    'obj_id': 0,
                    'cam_R_m2c': np.eye(3).tolist(),
                    'cam_t_m2c': [0.1, 0.1, 1.1],
                    'obj_bb': [105, 105, 45, 45]
                }]

                # Test GPU operations
                if len(gpu_metrics.models_3d_gpu) > 0:
                    pts = gpu_metrics.models_3d_gpu[0][:100]  # Use first 100 points
                    R = torch.eye(3, device=gpu_metrics.device)
                    t = torch.zeros(3, 1, device=gpu_metrics.device)

                    # Test transformations
                    pts_transformed = gpu_metrics.transform_pts_rt_gpu(pts, R, t)

                    # Test error computations
                    re_error = gpu_metrics.re_gpu(R, R)
                    te_error = gpu_metrics.te_gpu(t, t)
                    add_error = gpu_metrics.add_gpu(R, t, R, t, pts)
                    adi_error = gpu_metrics.adi_gpu(R, t, R, t, pts)

                    print(f"  RE error: {re_error.item():.6f}")
                    print(f"  TE error: {te_error.item():.6f}")
                    print(f"  ADD error: {add_error.item():.6f}")
                    print(f"  ADI error: {adi_error.item():.6f}")

                end_time = time.time()
                gpu_time = end_time - start_time
                gpu_times.append(gpu_time)
                print(f"GPU iteration {i+1} time: {gpu_time:.3f}s")

                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"GPU implementation failed: {e}")
            gpu_success = False

        # Test CPU version for comparison
        print("\nTesting original CPU implementation...")
        cpu_times = []
        cpu_success = True

        try:
            for i in range(iterations):
                print(f"CPU iteration {i+1}/{iterations}")
                start_time = time.time()
                cpu_metrics = Metrics(conf_path=conf_path)
                # Just test initialization time
                end_time = time.time()
                cpu_time = end_time - start_time
                cpu_times.append(cpu_time)
                print(f"CPU iteration {i+1} time: {cpu_time:.3f}s")

        except Exception as e:
            print(f"CPU implementation failed: {e}")
            cpu_success = False

        # Results
        print(f"\n=== Benchmark Results ===")
        if gpu_success and gpu_times:
            print(f"GPU average time: {np.mean(gpu_times):.3f}s (±{np.std(gpu_times):.3f}s)")
        if cpu_success and cpu_times:
            print(f"CPU average time: {np.mean(cpu_times):.3f}s (±{np.std(cpu_times):.3f}s)")

        if gpu_success and cpu_success and gpu_times and cpu_times:
            speedup = np.mean(cpu_times) / np.mean(gpu_times)
            print(f"Speedup: {speedup:.2f}x")

    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test GPU metrics implementation")
    parser.add_argument('--conf_path', type=str,
                       default='/home/elena/repos/6d-eval/config/metrics_cfg.yml',
                       help='Path to metrics configuration file')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of benchmark iterations')
    parser.add_argument('--skip_benchmark', action='store_true',
                       help='Skip benchmarking, just test device availability')

    args = parser.parse_args()

    # Test device availability
    test_device_availability()

    if not args.skip_benchmark:
        # Check if config file exists
        import os
        if not os.path.exists(args.conf_path):
            print(f"Configuration file not found: {args.conf_path}")
            print("Available config files:")
            for root, dirs, files in os.walk('/home/elena/repos/6d-eval/config'):
                for file in files:
                    if file.endswith('.yml'):
                        print(f"  {os.path.join(root, file)}")
            return

        # Run benchmark
        benchmark_metrics(args.conf_path, args.iterations)

    print("\n=== Installation Instructions ===")
    print("To use GPU acceleration, install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nOr use the provided requirements file:")
    print("pip install -r requirements_gpu.txt")

if __name__ == "__main__":
    main()