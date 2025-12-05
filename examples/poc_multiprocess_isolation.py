"""
Proof of Concept: Multi-process Memory Isolation Test

This POC validates that spawning separate Python processes truly releases
GPU/RAM memory after model inference, solving Python's memory leak problem.

Test scenarios:
1. Load model in subprocess -> run inference -> kill subprocess -> measure memory
2. Load model in same process -> del model -> gc.collect() -> measure memory
3. Compare memory reclamation between approaches

Expected outcome: Subprocess approach should fully reclaim memory, while
in-process approach may retain memory due to Python's memory allocator.
"""

import gc
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psutil

# Attempt to import torch for GPU monitoring
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - GPU memory tracking disabled")


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""

    ram_used_mb: float
    ram_percent: float
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()

    def __str__(self):
        result = f"RAM: {self.ram_used_mb:.1f}MB ({self.ram_percent:.1f}%)"
        if self.gpu_allocated_mb is not None:
            result += f", GPU Allocated: {self.gpu_allocated_mb:.1f}MB"
        if self.gpu_reserved_mb is not None:
            result += f", GPU Reserved: {self.gpu_reserved_mb:.1f}MB"
        return result


def get_memory_snapshot() -> MemorySnapshot:
    """Get current memory usage snapshot"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    ram_percent = process.memory_percent()

    gpu_allocated = None
    gpu_reserved = None

    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)

    return MemorySnapshot(
        ram_used_mb=ram_mb, ram_percent=ram_percent, gpu_allocated_mb=gpu_allocated, gpu_reserved_mb=gpu_reserved
    )


def load_and_infer_subprocess(model_name: str, use_gpu: bool = False):
    """
    Worker function that runs in a subprocess.
    Loads model, runs inference, then exits (memory should be freed).
    """
    print(f"  [PID {os.getpid()}] Subprocess started")

    if TORCH_AVAILABLE:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"  [PID {os.getpid()}] Loading model on {device}...")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        # Run inference
        print(f"  [PID {os.getpid()}] Running inference...")
        inputs = tokenizer("Hello, this is a memory isolation test!", return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"  [PID {os.getpid()}] Inference complete. Output shape: {outputs.last_hidden_state.shape}")

        # Memory snapshot before exit
        snapshot = get_memory_snapshot()
        print(f"  [PID {os.getpid()}] Memory before exit: {snapshot}")
    else:
        print(f"  [PID {os.getpid()}] PyTorch not available, simulating with dummy data")
        # Simulate memory allocation
        dummy_data = [0] * (100 * 1024 * 1024)  # ~100MB
        time.sleep(2)
        print(f"  [PID {os.getpid()}] Dummy work complete")

    print(f"  [PID {os.getpid()}] Subprocess exiting...")


def test_subprocess_isolation(
    model_name: str, use_gpu: bool = False
) -> tuple[MemorySnapshot, MemorySnapshot, MemorySnapshot]:
    """
    Test memory isolation using subprocess approach.
    Returns: (baseline, during_subprocess, after_subprocess)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Subprocess Memory Isolation")
    print("=" * 70)

    # Baseline
    print("\n📊 Measuring baseline memory...")
    gc.collect()
    time.sleep(1)
    baseline = get_memory_snapshot()
    print(f"Baseline: {baseline}")

    # Start subprocess
    print("\n🚀 Starting subprocess with model inference...")
    process = mp.Process(target=load_and_infer_subprocess, args=(model_name, use_gpu))
    process.start()

    # Wait a bit for model to load
    time.sleep(3)

    # Measure parent process memory (should be similar to baseline)
    during = get_memory_snapshot()
    print(f"\n📊 Parent process memory during subprocess: {during}")

    # Wait for subprocess to complete
    process.join(timeout=60)
    if process.is_alive():
        print("⚠️  Subprocess timeout, terminating...")
        process.terminate()
        process.join()

    # Measure after subprocess exit
    print("\n⏳ Waiting for memory cleanup...")
    time.sleep(2)
    gc.collect()
    after = get_memory_snapshot()
    print(f"📊 After subprocess exit: {after}")

    # Calculate delta
    ram_delta = after.ram_used_mb - baseline.ram_used_mb
    print(f"\n✅ RAM delta from baseline: {ram_delta:+.1f}MB")
    if after.gpu_allocated_mb is not None and baseline.gpu_allocated_mb is not None:
        gpu_delta = after.gpu_allocated_mb - baseline.gpu_allocated_mb
        print(f"✅ GPU delta from baseline: {gpu_delta:+.1f}MB")

    return baseline, during, after


def test_same_process_cleanup(
    model_name: str, use_gpu: bool = False
) -> tuple[MemorySnapshot, MemorySnapshot, MemorySnapshot]:
    """
    Test memory cleanup using del + gc in same process.
    Returns: (before_load, after_load, after_cleanup)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Same-Process Memory Cleanup (del + gc.collect)")
    print("=" * 70)

    # Baseline
    print("\n📊 Measuring baseline memory...")
    gc.collect()
    time.sleep(1)
    before = get_memory_snapshot()
    print(f"Before load: {before}")

    if TORCH_AVAILABLE:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"\n🚀 Loading model on {device}...")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        # Run inference
        print("🔄 Running inference...")
        inputs = tokenizer("Hello, this is a memory isolation test!", return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ Inference complete. Output shape: {outputs.last_hidden_state.shape}")

        # Measure after load
        after_load = get_memory_snapshot()
        print(f"\n📊 After model load: {after_load}")
        ram_increase = after_load.ram_used_mb - before.ram_used_mb
        print(f"   RAM increase: {ram_increase:+.1f}MB")

        # Cleanup
        print("\n🧹 Cleaning up (del model, del tokenizer, gc.collect)...")
        del model
        del tokenizer
        del inputs
        del outputs
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        time.sleep(2)

    else:
        print("\n🚀 PyTorch not available, simulating...")
        dummy_data = [0] * (100 * 1024 * 1024)
        after_load = get_memory_snapshot()
        print(f"\n📊 After allocation: {after_load}")

        del dummy_data
        gc.collect()
        time.sleep(2)

    # Measure after cleanup
    after_cleanup = get_memory_snapshot()
    print(f"📊 After cleanup: {after_cleanup}")

    # Calculate remaining delta
    ram_delta = after_cleanup.ram_used_mb - before.ram_used_mb
    print(f"\n📈 RAM delta from baseline: {ram_delta:+.1f}MB")
    if after_cleanup.gpu_allocated_mb is not None and before.gpu_allocated_mb is not None:
        gpu_delta = after_cleanup.gpu_allocated_mb - before.gpu_allocated_mb
        print(f"📈 GPU delta from baseline: {gpu_delta:+.1f}MB")

    return before, after_load, after_cleanup


def print_comparison(subprocess_results: tuple, same_process_results: tuple):
    """Print comparison of both approaches"""
    print("\n" + "=" * 70)
    print("COMPARISON: Subprocess vs Same-Process")
    print("=" * 70)

    sub_baseline, _, sub_after = subprocess_results
    same_before, same_after_load, same_after_cleanup = same_process_results

    # RAM comparison
    sub_ram_delta = sub_after.ram_used_mb - sub_baseline.ram_used_mb
    same_ram_delta = same_after_cleanup.ram_used_mb - same_before.ram_used_mb
    same_ram_leaked = same_after_cleanup.ram_used_mb - same_before.ram_used_mb

    print(f"\n📊 RAM Memory Analysis:")
    print(f"  Subprocess approach:")
    print(f"    - Delta from baseline: {sub_ram_delta:+.1f}MB")
    print(f"  Same-process approach:")
    print(f"    - Model size (peak): {same_after_load.ram_used_mb - same_before.ram_used_mb:+.1f}MB")
    print(f"    - Delta after cleanup: {same_ram_delta:+.1f}MB")
    print(f"    - Potential leak: {same_ram_leaked:.1f}MB")

    # GPU comparison
    if sub_after.gpu_allocated_mb is not None and same_after_cleanup.gpu_allocated_mb is not None:
        sub_gpu_delta = sub_after.gpu_allocated_mb - sub_baseline.gpu_allocated_mb
        same_gpu_delta = same_after_cleanup.gpu_allocated_mb - same_before.gpu_allocated_mb

        print(f"\n🎮 GPU Memory Analysis:")
        print(f"  Subprocess approach:")
        print(f"    - Delta from baseline: {sub_gpu_delta:+.1f}MB")
        print(f"  Same-process approach:")
        print(f"    - Delta after cleanup: {same_gpu_delta:+.1f}MB")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT:")
    print(f"{'='*70}")

    if abs(sub_ram_delta) < abs(same_ram_delta) * 0.5:
        print("✅ PASS: Subprocess approach significantly better at memory reclamation")
        print(f"   Subprocess leaked ~{abs(sub_ram_delta):.1f}MB vs ~{abs(same_ram_delta):.1f}MB in-process")
        print("\n💡 Recommendation: Use multi-process architecture for ModelMora")
    elif abs(sub_ram_delta) < abs(same_ram_delta):
        print("⚠️  PARTIAL: Subprocess approach better, but difference is small")
        print(f"   Subprocess leaked ~{abs(sub_ram_delta):.1f}MB vs ~{abs(same_ram_delta):.1f}MB in-process")
        print("\n💡 Recommendation: Multi-process may help, but monitor production memory")
    else:
        print("❌ FAIL: No significant benefit from subprocess isolation")
        print(f"   Both approaches leaked similar amounts (~{abs(sub_ram_delta):.1f}MB)")
        print("\n⚠️  Warning: May need alternative memory management strategies")


def main():
    """Run all memory isolation tests"""
    print("=" * 70)
    print("Multi-Process Memory Isolation POC")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"PID: {os.getpid()}")

    if TORCH_AVAILABLE:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch: Not installed")

    # Configuration
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small model (~90MB)
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()

    print(f"\nTest model: {model_name}")
    print(f"Using GPU: {use_gpu}")

    try:
        # Test 1: Subprocess isolation
        subprocess_results = test_subprocess_isolation(model_name, use_gpu)

        # Wait between tests
        print("\n⏳ Waiting 5 seconds before next test...")
        time.sleep(5)

        # Test 2: Same-process cleanup
        same_process_results = test_same_process_cleanup(model_name, use_gpu)

        # Compare results
        print_comparison(subprocess_results, same_process_results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("POC Complete")
    print("=" * 70)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method("spawn", force=True)
    main()
