"""
Production-Level Model Loading/Unloading Benchmark

This POC measures load times and memory cleanup effectiveness for different
model types and loading strategies to inform ModelMora's lifecycle management.

Test Scenarios:
1. Cold load (first-time download + initialization)
2. Warm load (cached model initialization)
3. Repeated load/unload cycles (memory leak detection)
4. Concurrent model loading (resource contention)
5. Different model types (embedding, text gen, image gen, vision)

Metrics Tracked:
- Load time (cold vs warm)
- Memory footprint (RAM + GPU)
- Unload effectiveness (memory reclamation %)
- Cache efficiency
- First inference latency vs subsequent

Architecture Decision Points:
- Should we pre-warm models or lazy load?
- How effective is Python GC vs subprocess isolation?
- What's the cost of keeping models in memory?
- Which models benefit from persistent loading?
"""

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

# Optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MemorySnapshot:
    """Point-in-time memory measurement"""

    ram_used_mb: float
    ram_percent: float
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        result = f"RAM: {self.ram_used_mb:.1f}MB ({self.ram_percent:.1f}%)"
        if self.gpu_allocated_mb is not None:
            result += f", GPU: {self.gpu_allocated_mb:.1f}MB allocated"
        if self.gpu_reserved_mb is not None:
            result += f" / {self.gpu_reserved_mb:.1f}MB reserved"
        return result


@dataclass
class LoadMetrics:
    """Metrics for a single model load operation"""

    model_id: str
    model_type: str
    load_time_ms: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    cache_hit: bool
    first_inference_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def memory_increase_mb(self) -> float:
        """RAM increase from loading model"""
        return self.memory_after.ram_used_mb - self.memory_before.ram_used_mb

    @property
    def gpu_memory_increase_mb(self) -> Optional[float]:
        """GPU memory increase if available"""
        if self.memory_before.gpu_allocated_mb is not None and self.memory_after.gpu_allocated_mb is not None:
            return self.memory_after.gpu_allocated_mb - self.memory_before.gpu_allocated_mb
        return None

    def __str__(self):
        result = f"{self.model_type} ({self.model_id})\n"
        result += f"  Load time: {self.load_time_ms:.1f}ms ({'cached' if self.cache_hit else 'cold'})\n"
        result += f"  Memory increase: {self.memory_increase_mb:.1f}MB RAM"
        if self.gpu_memory_increase_mb is not None:
            result += f", {self.gpu_memory_increase_mb:.1f}MB GPU"
        if self.first_inference_ms:
            result += f"\n  First inference: {self.first_inference_ms:.1f}ms"
        if self.error:
            result += f"\n  ❌ Error: {self.error}"
        return result


@dataclass
class UnloadMetrics:
    """Metrics for model unloading operation"""

    model_id: str
    unload_method: str  # "gc", "subprocess", "explicit_del"
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    unload_time_ms: float

    @property
    def memory_reclaimed_mb(self) -> float:
        """RAM reclaimed after unload"""
        return self.memory_before.ram_used_mb - self.memory_after.ram_used_mb

    @property
    def reclamation_percentage(self) -> float:
        """Percentage of memory successfully reclaimed"""
        if self.memory_before.ram_used_mb <= self.memory_after.ram_used_mb:
            return 0.0
        return (self.memory_reclaimed_mb / self.memory_before.ram_used_mb) * 100

    @property
    def gpu_memory_reclaimed_mb(self) -> Optional[float]:
        """GPU memory reclaimed if available"""
        if self.memory_before.gpu_allocated_mb is not None and self.memory_after.gpu_allocated_mb is not None:
            return self.memory_before.gpu_allocated_mb - self.memory_after.gpu_allocated_mb
        return None

    def __str__(self):
        result = f"{self.model_id} ({self.unload_method})\n"
        result += f"  Unload time: {self.unload_time_ms:.1f}ms\n"
        result += f"  Memory reclaimed: {self.memory_reclaimed_mb:.1f}MB ({self.reclamation_percentage:.1f}%)"
        if self.gpu_memory_reclaimed_mb is not None:
            result += f"\n  GPU reclaimed: {self.gpu_memory_reclaimed_mb:.1f}MB"
        return result


@dataclass
class ModelConfig:
    """Configuration for a test model"""

    model_id: str
    model_type: str  # "embedding", "text_gen", "image_gen", "vision"
    size_category: str  # "tiny", "small", "medium", "large"
    expected_load_time_ms: float
    expected_memory_mb: float
    supports_gpu: bool = True


# ============================================================================
# Memory Monitoring
# ============================================================================


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory state"""
    process = psutil.Process()
    mem_info = process.memory_info()

    gpu_allocated = None
    gpu_reserved = None

    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)

    return MemorySnapshot(
        ram_used_mb=mem_info.rss / (1024 * 1024),
        ram_percent=process.memory_percent(),
        gpu_allocated_mb=gpu_allocated,
        gpu_reserved_mb=gpu_reserved,
    )


def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.1)  # Allow OS to process cleanup


# ============================================================================
# Model Loading Implementations
# ============================================================================


def load_embedding_model(model_id: str, use_gpu: bool = False) -> Tuple[object, float]:
    """Load embedding model and measure time"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available")

    start = time.time()
    device = "cuda" if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    load_time_ms = (time.time() - start) * 1000

    return model, load_time_ms


def load_text_generation_model(model_id: str, use_gpu: bool = False) -> Tuple[object, float]:
    """Load text generation model and measure time"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not available")

    start = time.time()
    device = 0 if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else -1
    model = pipeline("text-generation", model=model_id, device=device)
    load_time_ms = (time.time() - start) * 1000

    return model, load_time_ms


def load_image_generation_model(model_id: str, use_gpu: bool = False) -> Tuple[object, float]:
    """Load image generation model and measure time"""
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers not available")

    start = time.time()
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    else:
        model = StableDiffusionPipeline.from_pretrained(model_id)
    load_time_ms = (time.time() - start) * 1000

    return model, load_time_ms


def load_vision_model(model_id: str, use_gpu: bool = False) -> Tuple[object, float]:
    """Load vision/multimodal model and measure time"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not available")

    start = time.time()
    device = 0 if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else -1
    model = pipeline("image-to-text", model=model_id, device=device)
    load_time_ms = (time.time() - start) * 1000

    return model, load_time_ms


# ============================================================================
# Benchmark Implementation
# ============================================================================


class ModelLifecycleBenchmark:
    """Production-level model lifecycle benchmark"""

    # Test model configurations
    TEST_MODELS = [
        ModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            size_category="tiny",
            expected_load_time_ms=2000,
            expected_memory_mb=90,
        ),
        ModelConfig(
            model_id="sentence-transformers/all-mpnet-base-v2",
            model_type="embedding",
            size_category="small",
            expected_load_time_ms=3000,
            expected_memory_mb=420,
        ),
        ModelConfig(
            model_id="gpt2",
            model_type="text_gen",
            size_category="small",
            expected_load_time_ms=4000,
            expected_memory_mb=500,
        ),
        ModelConfig(
            model_id="distilgpt2",
            model_type="text_gen",
            size_category="tiny",
            expected_load_time_ms=2000,
            expected_memory_mb=320,
        ),
    ]

    def __init__(self, use_gpu: bool = False, cache_dir: Optional[str] = None):
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface")
        self.load_results: List[LoadMetrics] = []
        self.unload_results: List[UnloadMetrics] = []

        # Set cache directory
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir

    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is in cache"""
        # Simple heuristic: check if model directory exists
        model_path = Path(self.cache_dir) / "hub" / f"models--{model_id.replace('/', '--')}"
        return model_path.exists()

    def benchmark_load(self, config: ModelConfig) -> LoadMetrics:
        """Benchmark single model load"""

        print(f"\n{'='*70}")
        print(f"Loading: {config.model_id} ({config.model_type})")
        print(f"{'='*70}")

        cache_hit = self.is_model_cached(config.model_id)
        print(f"Cache status: {'HIT ✅' if cache_hit else 'MISS (downloading...)'}")

        # Force cleanup before load
        force_cleanup()
        mem_before = get_memory_snapshot()
        print(f"Memory before: {mem_before}")

        # Load model
        model = None
        load_time_ms = 0
        error = None

        try:
            if config.model_type == "embedding":
                model, load_time_ms = load_embedding_model(config.model_id, self.use_gpu)
            elif config.model_type == "text_gen":
                model, load_time_ms = load_text_generation_model(config.model_id, self.use_gpu)
            elif config.model_type == "image_gen":
                model, load_time_ms = load_image_generation_model(config.model_id, self.use_gpu)
            elif config.model_type == "vision":
                model, load_time_ms = load_vision_model(config.model_id, self.use_gpu)
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")

        except Exception as e:
            error = str(e)
            print(f"❌ Load failed: {error}")

        mem_after = get_memory_snapshot()
        print(f"Memory after: {mem_after}")
        print(f"Load time: {load_time_ms:.1f}ms")

        # Measure first inference if model loaded
        first_inference_ms = None
        if model is not None and error is None:
            first_inference_ms = self._measure_first_inference(model, config.model_type)
            print(f"First inference: {first_inference_ms:.1f}ms")

        metrics = LoadMetrics(
            model_id=config.model_id,
            model_type=config.model_type,
            load_time_ms=load_time_ms,
            memory_before=mem_before,
            memory_after=mem_after,
            cache_hit=cache_hit,
            first_inference_ms=first_inference_ms,
            error=error,
        )

        self.load_results.append(metrics)

        # Store model for unload test
        return metrics, model

    def _measure_first_inference(self, model, model_type: str) -> float:
        """Measure first inference latency"""
        try:
            start = time.time()

            if model_type == "embedding":
                model.encode(["Test sentence"], show_progress_bar=False)
            elif model_type == "text_gen":
                model("Hello", max_length=10, num_return_sequences=1)
            elif model_type == "image_gen":
                model("test prompt", num_inference_steps=1, height=64, width=64)
            elif model_type == "vision":
                # Would need actual image, skip for now
                return 0.0

            return (time.time() - start) * 1000

        except Exception as e:
            print(f"⚠️  First inference failed: {e}")
            return 0.0

    def benchmark_unload_gc(self, model, model_id: str) -> UnloadMetrics:
        """Benchmark unload with gc.collect()"""

        print(f"\n{'='*70}")
        print(f"Unloading (GC): {model_id}")
        print(f"{'='*70}")

        mem_before = get_memory_snapshot()
        print(f"Memory before: {mem_before}")

        start = time.time()

        # Delete model and force garbage collection
        del model
        force_cleanup()

        unload_time_ms = (time.time() - start) * 1000

        mem_after = get_memory_snapshot()
        print(f"Memory after: {mem_after}")
        print(f"Unload time: {unload_time_ms:.1f}ms")

        metrics = UnloadMetrics(
            model_id=model_id,
            unload_method="gc",
            memory_before=mem_before,
            memory_after=mem_after,
            unload_time_ms=unload_time_ms,
        )

        self.unload_results.append(metrics)
        return metrics

    def benchmark_repeated_cycles(self, config: ModelConfig, num_cycles: int = 3):
        """Test repeated load/unload to detect memory leaks"""

        print(f"\n{'='*70}")
        print(f"Repeated Cycles Test: {config.model_id} ({num_cycles} cycles)")
        print(f"{'='*70}")

        baseline_memory = get_memory_snapshot()
        print(f"Baseline memory: {baseline_memory}")

        cycle_results = []

        for i in range(num_cycles):
            print(f"\n--- Cycle {i+1}/{num_cycles} ---")

            # Load
            metrics, model = self.benchmark_load(config)

            # Small delay
            time.sleep(0.5)

            # Unload
            unload_metrics = self.benchmark_unload_gc(model, config.model_id)

            cycle_results.append(
                {
                    "cycle": i + 1,
                    "load_time_ms": metrics.load_time_ms,
                    "memory_increase_mb": metrics.memory_increase_mb,
                    "memory_reclaimed_mb": unload_metrics.memory_reclaimed_mb,
                    "final_memory_mb": unload_metrics.memory_after.ram_used_mb,
                }
            )

        # Analysis
        print(f"\n{'='*70}")
        print(f"Repeated Cycles Analysis")
        print(f"{'='*70}")

        print(f"\nBaseline: {baseline_memory.ram_used_mb:.1f}MB")

        for result in cycle_results:
            print(
                f"Cycle {result['cycle']}: "
                f"Load {result['load_time_ms']:.0f}ms, "
                f"+{result['memory_increase_mb']:.1f}MB, "
                f"-{result['memory_reclaimed_mb']:.1f}MB, "
                f"Final {result['final_memory_mb']:.1f}MB"
            )

        # Check for memory leak
        final_memory = cycle_results[-1]["final_memory_mb"]
        memory_leak_mb = final_memory - baseline_memory.ram_used_mb

        print(f"\n🔍 Memory Leak Detection:")
        print(f"  Baseline: {baseline_memory.ram_used_mb:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Leak: {memory_leak_mb:+.1f}MB")

        if memory_leak_mb > 50:
            print(f"  ⚠️  WARNING: Significant memory leak detected (>{memory_leak_mb:.1f}MB)")
        elif memory_leak_mb > 10:
            print(f"  ⚠️  CAUTION: Minor memory leak detected ({memory_leak_mb:.1f}MB)")
        else:
            print(f"  ✅ PASS: No significant memory leak")

    def print_summary(self):
        """Print benchmark summary"""

        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")

        if self.load_results:
            print(f"\n📊 Load Performance:")
            print(f"{'='*70}")

            # Group by model type
            by_type: Dict[str, List[LoadMetrics]] = {}
            for metric in self.load_results:
                by_type.setdefault(metric.model_type, []).append(metric)

            for model_type, metrics in by_type.items():
                print(f"\n{model_type.upper()}:")

                for m in metrics:
                    cache_str = "cached" if m.cache_hit else "cold"
                    print(f"  {m.model_id}")
                    print(f"    Load: {m.load_time_ms:.0f}ms ({cache_str})")
                    print(f"    Memory: +{m.memory_increase_mb:.1f}MB")
                    if m.first_inference_ms:
                        print(f"    First inference: {m.first_inference_ms:.0f}ms")

        if self.unload_results:
            print(f"\n🗑️  Unload Performance:")
            print(f"{'='*70}")

            avg_reclamation = sum(m.reclamation_percentage for m in self.unload_results) / len(self.unload_results)

            print(f"\nAverage memory reclamation: {avg_reclamation:.1f}%")

            for m in self.unload_results:
                print(f"  {m.model_id}: {m.memory_reclaimed_mb:.1f}MB ({m.reclamation_percentage:.1f}%)")

        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS FOR MODELMORA")
        print(f"{'='*70}")

        if self.load_results:
            avg_load_time = sum(m.load_time_ms for m in self.load_results if not m.error) / len(
                [m for m in self.load_results if not m.error]
            )

            print(f"\n1. Model Loading Strategy:")
            if avg_load_time < 3000:
                print(f"   ✅ Lazy loading viable (avg {avg_load_time:.0f}ms)")
                print(f"   💡 Load models on first request")
            else:
                print(f"   ⚠️  Consider pre-warming (avg {avg_load_time:.0f}ms)")
                print(f"   💡 Pre-load frequently used models at startup")

        if self.unload_results:
            avg_reclaim = sum(m.reclamation_percentage for m in self.unload_results) / len(self.unload_results)

            print(f"\n2. Memory Management:")
            if avg_reclaim < 70:
                print(f"   ⚠️  Low reclamation rate ({avg_reclaim:.1f}%)")
                print(f"   💡 CRITICAL: Use subprocess isolation for model workers")
            else:
                print(f"   ✅ Good reclamation rate ({avg_reclaim:.1f}%)")
                print(f"   💡 GC-based unload acceptable for memory management")

        print(f"\n3. Resource Planning:")
        if self.load_results:
            max_memory = max(m.memory_increase_mb for m in self.load_results if not m.error)
            print(f"   Largest model footprint: {max_memory:.1f}MB")
            print(f"   💡 Reserve {max_memory * 1.5:.0f}MB per worker process")
            print(f"   💡 For 10 concurrent models: ~{max_memory * 10 / 1024:.1f}GB RAM minimum")


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run comprehensive model lifecycle benchmark"""

    print("=" * 70)
    print("ModelMora: Model Lifecycle Benchmark (Production)")
    print("=" * 70)

    # Check dependencies
    print("\nDependency Check:")
    print(f"  PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"  Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    print(f"  Sentence-Transformers: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
    print(f"  Diffusers: {'✅' if DIFFUSERS_AVAILABLE else '❌'}")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"  GPU: ✅ {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print(f"  GPU: ❌ (CPU only)")
        use_gpu = False

    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ Transformers required. Install: pip install transformers sentence-transformers")
        return

    # Initialize benchmark
    benchmark = ModelLifecycleBenchmark(use_gpu=use_gpu)

    # Test 1: Individual model load/unload
    print(f"\n{'='*70}")
    print("TEST 1: Individual Model Load/Unload")
    print(f"{'='*70}")

    for config in benchmark.TEST_MODELS[:2]:  # Test first 2 models
        metrics, model = benchmark.benchmark_load(config)

        if model is not None and metrics.error is None:
            time.sleep(0.5)  # Small delay
            benchmark.benchmark_unload_gc(model, config.model_id)

    # Test 2: Repeated cycles (memory leak detection)
    print(f"\n{'='*70}")
    print("TEST 2: Memory Leak Detection")
    print(f"{'='*70}")

    benchmark.benchmark_repeated_cycles(benchmark.TEST_MODELS[0], num_cycles=3)

    # Print summary
    benchmark.print_summary()

    print(f"\n{'='*70}")
    print("Benchmark Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
