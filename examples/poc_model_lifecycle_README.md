# Model Loading/Unloading Benchmark POC

## Overview

This POC measures model lifecycle performance to inform ModelMora's architecture decisions around model loading strategies, memory management, and resource planning.

## Purpose

Determine:

1. **Load Time Characteristics**: Cold vs warm load times for different model types
2. **Memory Footprint**: Actual RAM/GPU usage per model type and size
3. **Unload Effectiveness**: How well Python GC reclaims memory vs subprocess isolation
4. **Memory Leak Detection**: Whether repeated load/unload cycles leak memory
5. **First Inference Latency**: Warmup cost for initial inference

## Test Scenarios

### 1. Individual Model Load/Unload

Tests single model lifecycle with memory tracking:

- Capture baseline memory
- Load model (measure time)
- Capture loaded memory
- Run first inference
- Unload with GC
- Capture final memory

### 2. Memory Leak Detection

Repeated load/unload cycles (3x) to detect leaks:

- Track memory baseline
- Load → Unload → Load → Unload → Load → Unload
- Compare final memory to baseline
- PASS if leak < 10MB

### 3. Model Type Comparison

Tests different model architectures:

- **Embedding**: sentence-transformers (90MB - 420MB)
- **Text Generation**: GPT-2 variants (320MB - 500MB)
- **Image Generation**: Stable Diffusion (optional, 3-5GB)
- **Vision**: CLIP-like models (optional)

## Architecture

```python
ModelLifecycleBenchmark
├── TEST_MODELS: List[ModelConfig]  # Test configurations
├── load_results: List[LoadMetrics]  # Load performance data
├── unload_results: List[UnloadMetrics]  # Unload performance data
│
├── benchmark_load(config) → LoadMetrics
│   ├── Check cache status
│   ├── Measure baseline memory
│   ├── Load model (timed)
│   ├── Measure loaded memory
│   └── Run first inference
│
├── benchmark_unload_gc(model) → UnloadMetrics
│   ├── Measure before memory
│   ├── del model + gc.collect()
│   ├── torch.cuda.empty_cache()
│   └── Measure after memory
│
└── benchmark_repeated_cycles(config, n)
    └── Detect memory leaks over n cycles
```

## Metrics Tracked

### LoadMetrics

- `load_time_ms`: Time to load model (cold/warm)
- `memory_increase_mb`: RAM consumed by model
- `gpu_memory_increase_mb`: GPU VRAM consumed (if available)
- `cache_hit`: Whether model was cached
- `first_inference_ms`: Latency for first inference
- `error`: Any loading errors

### UnloadMetrics

- `unload_time_ms`: Time to unload and cleanup
- `memory_reclaimed_mb`: RAM freed after unload
- `reclamation_percentage`: % of memory successfully freed
- `gpu_memory_reclaimed_mb`: GPU VRAM freed

## How to Run

### Prerequisites

```bash
# Install dependencies
pip install transformers sentence-transformers torch psutil

# Optional for image generation tests
pip install diffusers
```

### Basic Execution

```bash
cd examples
python poc_model_lifecycle.py
```

### Expected Output

```bash
======================================================================
ModelMora: Model Lifecycle Benchmark (Production)
======================================================================

Dependency Check:
  PyTorch: ✅
  Transformers: ✅
  Sentence-Transformers: ✅
  GPU: ✅ NVIDIA GeForce RTX 3090

======================================================================
TEST 1: Individual Model Load/Unload
======================================================================

======================================================================
Loading: sentence-transformers/all-MiniLM-L6-v2 (embedding)
======================================================================
Cache status: HIT ✅
Memory before: RAM: 450.2MB (2.8%)
Memory after: RAM: 542.1MB (3.4%)
Load time: 1834.2ms
First inference: 45.3ms

======================================================================
Unloading (GC): sentence-transformers/all-MiniLM-L6-v2
======================================================================
Memory before: RAM: 542.1MB (3.4%)
Memory after: RAM: 458.7MB (2.9%)
Unload time: 123.4ms

...

======================================================================
BENCHMARK SUMMARY
======================================================================

📊 Load Performance:
  EMBEDDING:
    sentence-transformers/all-MiniLM-L6-v2
      Load: 1834ms (cached)
      Memory: +91.9MB
      First inference: 45ms

🗑️  Unload Performance:
  Average memory reclamation: 87.3%

======================================================================
RECOMMENDATIONS FOR MODELMORA
======================================================================

1. Model Loading Strategy:
   ✅ Lazy loading viable (avg 2145ms)
   💡 Load models on first request

2. Memory Management:
   ✅ Good reclamation rate (87.3%)
   💡 GC-based unload acceptable for memory management

3. Resource Planning:
   Largest model footprint: 420.5MB
   💡 Reserve 631MB per worker process
   💡 For 10 concurrent models: ~4.1GB RAM minimum
```

## Interpretation Guide

### Load Time Analysis

| Load Time | Verdict | Action |
|-----------|---------|--------|
| < 2s (cached) | ✅ Excellent | Lazy loading viable |
| 2-5s (cached) | ✅ Good | Lazy loading acceptable |
| 5-10s (cached) | ⚠️ Moderate | Consider pre-warming popular models |
| > 10s (cached) | ❌ Slow | Pre-warm at startup |

### Memory Reclamation

| Reclamation % | Verdict | Action |
|---------------|---------|--------|
| > 90% | ✅ Excellent | GC-based unload sufficient |
| 70-90% | ✅ Good | GC acceptable, monitor long-running processes |
| 50-70% | ⚠️ Moderate | Consider subprocess isolation for critical apps |
| < 50% | ❌ Poor | **CRITICAL**: Use subprocess isolation |

### Memory Leak Detection

| Final Leak | Verdict | Action |
|------------|---------|--------|
| < 10MB | ✅ PASS | No significant leak |
| 10-50MB | ⚠️ CAUTION | Monitor in production |
| > 50MB | ❌ FAIL | Memory leak detected, use subprocess isolation |

## Key Findings (Expected)

### Load Times

- **Tiny models** (90MB): 1-2s cached, 10-20s cold
- **Small models** (400MB): 2-3s cached, 30-60s cold
- **Large models** (3GB+): 5-10s cached, 5-10min cold

### Memory Characteristics

- **Embedding models**: Low overhead, efficient
- **Text generation**: Moderate overhead, good GC
- **Image generation**: High overhead, GPU-dependent

### GC Effectiveness

- **Expected**: 70-90% reclamation for most models
- **Reality**: Python GC may retain 10-30% due to allocator behavior
- **GPU**: CUDA cache requires explicit `empty_cache()` call

## Architecture Decisions Informed

### ✅ Use Lazy Loading

If cached load times < 3s, load on first request instead of pre-warming.

### ✅ Subprocess Isolation Critical

If GC reclamation < 70%, subprocess isolation is mandatory for production.

### ✅ Memory Budget Planning

Use `memory_increase_mb` to calculate per-worker resource limits:

- Reserve 1.5x model footprint per worker
- Plan for N concurrent models: `N × largest_model × 1.5`

### ✅ Cache Strategy

Cold load times justify:

- Persistent HuggingFace cache volume
- Pre-download models at build time
- Never clear cache in production

## Production Implications

### For ModelMora Architecture

1. **Lifecycle Manager**:
   - Use lazy loading (load on first request)
   - Implement LRU eviction when memory pressure detected
   - Track load time metrics for warmup predictions

2. **Worker Process Design**:
   - One model per subprocess (isolation)
   - Kill subprocess to unload (don't rely on GC)
   - Monitor memory per process

3. **Resource Limits**:
   - Set memory limits per worker: `model_size × 1.5`
   - Reserve 10% buffer for temporary allocations
   - Fail fast if memory limit exceeded

4. **Health Checks**:
   - Monitor load time (alert if > 2x expected)
   - Monitor memory leaks (alert if growth > 50MB/hour)
   - Track first inference latency

## Next Steps

1. ✅ Run this POC on target hardware
2. ✅ Record actual load times and memory usage
3. ✅ Update resource planning in architecture doc
4. → Implement Lifecycle Manager based on findings
5. → Add load time metrics to observability

## Related POCs

- **POC 1**: Multi-process memory isolation test (validates subprocess cleanup)
- **POC 2**: gRPC streaming performance test (validates inference throughput)
- **POC 4**: Priority queue implementation (validates request scheduling)

## Success Criteria

This POC is successful if:

- ✅ Load time data collected for 3+ model types
- ✅ Memory reclamation percentage measured
- ✅ Memory leak detection completed (3+ cycles)
- ✅ Clear recommendation: lazy vs pre-warm loading
- ✅ Clear recommendation: GC vs subprocess isolation
- ✅ Resource planning data available (MB per model)

## Troubleshooting

### Model Download Fails

```bash
# Set HuggingFace token if accessing gated models
export HF_TOKEN=your_token_here
```

### GPU Out of Memory

```bash
# Run CPU-only version
CUDA_VISIBLE_DEVICES="" python poc_model_lifecycle.py
```

### Import Errors

```bash
# Install all optional dependencies
pip install transformers sentence-transformers torch diffusers psutil
```

## References

- HuggingFace Model Hub: <https://huggingface.co/models>
- PyTorch Memory Management: <https://pytorch.org/docs/stable/notes/cuda.html>
- Python GC Documentation: <https://docs.python.org/3/library/gc.html>
