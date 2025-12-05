# POC: Multi-Process Memory Isolation Test

## Overview

This proof-of-concept validates that spawning separate Python processes for model inference effectively releases GPU/RAM memory, solving Python's notorious memory leak problem with ML models.

## What It Tests

1. **Subprocess Approach**: Load model in subprocess → run inference → kill subprocess → measure memory
2. **Same-Process Approach**: Load model → `del model` → `gc.collect()` → measure memory
3. **Comparison**: Which approach better reclaims memory?

## Prerequisites

```powershell
# Install dependencies
pip install torch transformers sentence-transformers psutil
```

For GPU testing, ensure CUDA-compatible PyTorch is installed:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Running the POC

```powershell
# Basic run (will download ~90MB model on first run)
python .\examples\poc_multiprocess_isolation.py

# Note: First run will cache the model from HuggingFace Hub
# Subsequent runs will be faster
```

## Expected Output

```
======================================================================
Multi-Process Memory Isolation POC
======================================================================
Python: 3.10.x
PyTorch: 2.x.x
CUDA available: True/False
GPU: NVIDIA ...

Test model: sentence-transformers/all-MiniLM-L6-v2
Using GPU: True/False

======================================================================
TEST 1: Subprocess Memory Isolation
======================================================================

📊 Measuring baseline memory...
Baseline: RAM: 245.3MB (2.1%)

🚀 Starting subprocess with model inference...
  [PID 12345] Subprocess started
  [PID 12345] Loading model on cpu...
  [PID 12345] Running inference...
  [PID 12345] Inference complete. Output shape: torch.Size([1, 9, 384])
  [PID 12345] Memory before exit: RAM: 398.7MB (3.4%)
  [PID 12345] Subprocess exiting...

📊 Parent process memory during subprocess: RAM: 247.1MB (2.1%)

⏳ Waiting for memory cleanup...
📊 After subprocess exit: RAM: 246.8MB (2.1%)

✅ RAM delta from baseline: +1.5MB

======================================================================
TEST 2: Same-Process Memory Cleanup (del + gc.collect)
======================================================================

📊 Measuring baseline memory...
Before load: RAM: 246.8MB (2.1%)

🚀 Loading model on cpu...
🔄 Running inference...
✅ Inference complete. Output shape: torch.Size([1, 9, 384])

📊 After model load: RAM: 401.2MB (3.4%)
   RAM increase: +154.4MB

🧹 Cleaning up (del model, del tokenizer, gc.collect)...
📊 After cleanup: RAM: 268.5MB (2.3%)

📈 RAM delta from baseline: +21.7MB

======================================================================
COMPARISON: Subprocess vs Same-Process
======================================================================

📊 RAM Memory Analysis:
  Subprocess approach:
    - Delta from baseline: +1.5MB
  Same-process approach:
    - Model size (peak): +154.4MB
    - Delta after cleanup: +21.7MB
    - Potential leak: 21.7MB

======================================================================
VERDICT:
======================================================================
✅ PASS: Subprocess approach significantly better at memory reclamation
   Subprocess leaked ~1.5MB vs ~21.7MB in-process

💡 Recommendation: Use multi-process architecture for ModelMora
```

## Interpretation

### Success Criteria

- ✅ **PASS**: Subprocess delta < 50% of same-process delta
  - Multi-process architecture is effective
  - Proceed with worker subprocess design

- ⚠️ **PARTIAL**: Subprocess delta < same-process delta, but close
  - Multi-process helps, but not dramatically
  - Monitor production memory carefully

- ❌ **FAIL**: No significant difference
  - Need alternative strategies (e.g., model unloading libraries, C++ bindings)

### Why Python Leaks Memory

Python's garbage collector may not immediately release memory back to the OS, even after `del` and `gc.collect()`. This is because:

1. **Memory Fragmentation**: Python's memory allocator (`pymalloc`) pools memory
2. **Reference Cycles**: Deep learning models have complex object graphs
3. **C Extension Memory**: PyTorch tensors allocated in C++ may not be tracked by Python GC

### Why Subprocesses Work

When a subprocess terminates, the OS **forcibly reclaims all memory** (RAM and GPU VRAM) allocated to that process. This guarantees cleanup regardless of Python's GC behavior.

## Adapting the Test

### Test with Different Models

```python
# In main() function, change:
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB
# to:
model_name = "gpt2"  # ~500MB
# or:
model_name = "stabilityai/stable-diffusion-2-1"  # ~2GB (requires diffusers)
```

### Test with GPU

The script automatically uses GPU if available. To force CPU:

```python
use_gpu = False  # In main() function
```

### Increase Iterations

To test multiple load/unload cycles:

```python
for i in range(5):
    print(f"\n--- Iteration {i+1} ---")
    subprocess_results = test_subprocess_isolation(model_name, use_gpu)
    time.sleep(5)
```

## Known Issues

### Windows Multiprocessing

On Windows, multiprocessing requires `if __name__ == "__main__"` guard and `spawn` start method (already implemented).

### Model Download Time

First run downloads the model (~90MB for test model). Use cached model path if offline:

```python
model_name = "/path/to/local/model"
```

### PyTorch Not Installed

Script runs in "dummy mode" without PyTorch, allocating plain Python objects to simulate memory patterns.

## Next Steps After POC

1. **If PASS**: Implement Phase 1.3 worker process with confidence
2. **Document Results**: Add findings to architecture decision record
3. **Design Process Pool**: Plan worker lifecycle management
4. **Benchmark Startup Overhead**: Measure process spawn time impact

## References

- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Multiprocessing Best Practices](https://docs.python.org/3/library/multiprocessing.html#programming-guidelines)
