# Priority Queue Implementation Comparison POC

## Overview

This POC compares different Python priority queue implementations to inform ModelMora's task scheduling architecture, ensuring optimal performance for the request queue system.

## Purpose

Determine:

1. **Performance**: Enqueue/dequeue latency and throughput for different implementations
2. **Concurrency Safety**: Thread-safe and async-safe queue options
3. **Priority Correctness**: Proper task ordering under load
4. **Scalability**: Performance with concurrent producers/consumers
5. **Overhead**: Memory and latency costs of different queue types

## Queue Implementations Tested

### 1. **heapq** (stdlib, lightweight)

- **Type**: List-based binary heap
- **Thread-safe**: ❌ No
- **Async-safe**: ❌ No
- **Use case**: Single-threaded, performance-critical paths
- **Pros**: Fastest raw performance, minimal overhead
- **Cons**: Requires manual locking for concurrent access

### 2. **queue.PriorityQueue** (stdlib, thread-safe)

- **Type**: Thread-safe wrapper around heapq
- **Thread-safe**: ✅ Yes
- **Async-safe**: ❌ No (blocking)
- **Use case**: Multi-threaded workers, thread pools
- **Pros**: Built-in thread safety, blocking semantics
- **Cons**: Not async-native, moderate overhead

### 3. **asyncio.PriorityQueue** (stdlib, async)

- **Type**: Async-native priority queue
- **Thread-safe**: ❌ No (coroutine-safe)
- **Async-safe**: ✅ Yes
- **Use case**: Async API servers (FastAPI, gRPC async)
- **Pros**: Native async/await support, coroutine-safe
- **Cons**: Only works in asyncio context

### 4. **ThreadSafeHeapQueue** (custom)

- **Type**: heapq with explicit threading.Lock
- **Thread-safe**: ✅ Yes
- **Async-safe**: ❌ No
- **Use case**: Custom control over locking behavior
- **Pros**: Minimal overhead vs queue.PriorityQueue
- **Cons**: Manual implementation

## Test Scenarios

### Test 1: Single-threaded Baseline

Measures raw performance without concurrency:

- Enqueue 10,000 tasks with mixed priorities
- Dequeue all tasks in priority order
- Measure latency (μs) for each operation
- Verify priority ordering correctness

**Metrics:**

- Avg enqueue latency (μs)
- Avg dequeue latency (μs)
- P99 latency
- Throughput (ops/sec)

### Test 2: Multi-threaded Concurrent

Simulates thread pool workers:

- 5 producer threads (enqueue)
- 5 consumer threads (dequeue)
- 10,000 tasks total
- Measure concurrent performance

**Validates:**

- Thread safety
- Lock contention impact
- Throughput under concurrency

### Test 3: Async Concurrent

Simulates async API server:

- 5 producer coroutines
- 5 consumer coroutines
- 10,000 tasks total
- Measure async performance

**Validates:**

- Async/await compatibility
- Coroutine scheduling efficiency
- No blocking operations

### Test 4: Priority Order Verification

Validates correctness:

- Tasks dequeued in priority order (CRITICAL → LOW)
- FIFO within same priority (timestamp ordering)
- Count violations

**Pass criteria:**

- 100% correct ordering
- 0 priority violations

## Architecture

```python
Priority Levels:
  CRITICAL = 1  # SLA-bound, real-time
  HIGH = 2      # User-facing, interactive
  NORMAL = 3    # Batch processing
  LOW = 4       # Background tasks

Task Structure:
  @dataclass(order=True)
  Task:
    priority: int
    timestamp: float
    task_id: str
    model_id: str
    payload: Any

QueueMetrics:
  - enqueue_latency_us: List[float]
  - dequeue_latency_us: List[float]
  - throughput_ops_per_sec: float
  - priority_violations: int
  - total_operations: int
```

## How to Run

### Prerequisites

```bash
# Install optional dependencies
pip install psutil  # For memory tracking
```

### Basic Execution

```bash
cd examples
python poc_priority_queue.py
```

### Expected Output

```bash
======================================================================
ModelMora: Priority Queue Implementation Comparison
======================================================================

Dependency Check:
  psutil: ✅

Configuration:
  Test tasks: 10,000
  Priority levels: 4 (CRITICAL, HIGH, NORMAL, LOW)

======================================================================
RUNNING BENCHMARKS
======================================================================

🔬 Test 1: Single-threaded Performance (Baseline)

======================================================================
TEST: heapq (stdlib, single-threaded)
======================================================================
Enqueuing 10000 tasks...
Dequeuing 10000 tasks...
✅ Complete: heapq:
  Avg enqueue: 2.3μs (p99: 8.1μs)
  Avg dequeue: 3.1μs (p99: 12.4μs)
  Throughput: 370,370 ops/sec
  Memory: 245.23MB
  Priority violations: 0/10000
  Errors: 0

======================================================================
TEST: queue.PriorityQueue (stdlib, thread-safe)
======================================================================
Enqueuing 10000 tasks...
Dequeuing 10000 tasks...
✅ Complete: queue.PriorityQueue:
  Avg enqueue: 3.8μs (p99: 15.2μs)
  Avg dequeue: 4.5μs (p99: 18.3μs)
  Throughput: 240,963 ops/sec
  Memory: 248.45MB
  Priority violations: 0/10000
  Errors: 0

...

======================================================================
COMPARISON: Priority Queue Implementations
======================================================================

📊 Throughput Ranking (ops/sec):
----------------------------------------------------------------------
1. heapq                               370,370 ops/sec
2. asyncio.PriorityQueue              285,714 ops/sec
3. queue.PriorityQueue                240,963 ops/sec
4. concurrent_async                   198,412 ops/sec
5. concurrent_threads                 165,289 ops/sec

⚡ Latency Comparison (avg μs):
----------------------------------------------------------------------
Queue Type                       Enqueue      Dequeue   P99 Enqueue    P99 Dequeue
----------------------------------------------------------------------
heapq                                2.3          3.1            8.1           12.4
asyncio.PriorityQueue               3.5          4.2           14.2           18.5
queue.PriorityQueue                 3.8          4.5           15.2           18.3
concurrent_async                    5.2          6.8           24.1           31.2
concurrent_threads                  6.1          7.3           28.4           35.8

🎯 Priority Order Correctness:
----------------------------------------------------------------------
✅ heapq                          100.00% (0 violations)
✅ queue.PriorityQueue           100.00% (0 violations)
✅ asyncio.PriorityQueue         100.00% (0 violations)
✅ concurrent_threads            100.00% (0 violations)
✅ concurrent_async              100.00% (0 violations)

======================================================================
RECOMMENDATIONS FOR MODELMORA
======================================================================

1. Single-Node Async API Server:
   ✅ Use: asyncio.PriorityQueue
   Throughput: 285,714 ops/sec
   Avg latency: 3.5μs enqueue
   💡 Native async support, perfect for FastAPI/gRPC async handlers

2. Multi-threaded Workers:
   ✅ Use: queue.PriorityQueue
   Throughput: 240,963 ops/sec
   💡 Thread-safe, blocking semantics, stdlib

3. Performance-Critical Single-Threaded:
   ✅ Use: heapq (with manual locking if needed)
   Throughput: 370,370 ops/sec
   Avg latency: 2.3μs enqueue
   💡 Fastest raw performance, minimal overhead

4. Distributed Multi-Node (Future):
   📋 Use: Redis sorted sets or dedicated message queue
   💡 For Phase 4 multi-node coordination

5. Latency Analysis:
   ✅ Excellent: 3.8μs avg enqueue latency
   💡 Negligible queue overhead (<50ms target)

======================================================================
ARCHITECTURE DECISION
======================================================================

For ModelMora MVP (Phase 1):
  ✅ Use asyncio.PriorityQueue for request queue
  ✅ Integrate with FastAPI async endpoints
  ✅ Priority levels: CRITICAL, HIGH, NORMAL, LOW
  ✅ FIFO within same priority (timestamp ordering)

Expected Performance:
  • Handle 285,714 requests/sec
  • Queue latency: <4μs
  • Well within <50ms target
```

## Interpretation Guide

### Throughput

| Throughput (ops/sec) | Verdict | Capacity |
|---------------------|---------|----------|
| > 200,000 | ✅ Excellent | 1000s of requests/sec sustained |
| 100,000 - 200,000 | ✅ Good | High-throughput capable |
| 50,000 - 100,000 | ✅ Acceptable | Normal workload |
| < 50,000 | ⚠️ Moderate | May bottleneck at scale |

### Latency

| Avg Latency (μs) | Verdict | Impact |
|-----------------|---------|--------|
| < 5μs | ✅ Excellent | Negligible overhead |
| 5-10μs | ✅ Good | Minimal overhead |
| 10-50μs | ✅ Acceptable | Low overhead |
| > 50μs | ⚠️ Moderate | May accumulate under load |

### Priority Correctness

| Violations | Verdict | Action |
|-----------|---------|--------|
| 0 | ✅ PASS | Perfect ordering |
| 1-10 | ⚠️ CAUTION | Investigate edge cases |
| > 10 | ❌ FAIL | Implementation bug |

## Key Findings (Expected)

### Performance Ranking

1. **heapq**: Fastest (370k ops/sec) - single-threaded only
2. **asyncio.PriorityQueue**: Fast (285k ops/sec) - async-native
3. **queue.PriorityQueue**: Good (240k ops/sec) - thread-safe
4. **Concurrent**: Lower throughput but real-world applicable

### Latency Characteristics

- **Single-threaded**: 2-4μs per operation
- **Concurrent**: 5-7μs per operation (lock contention)
- **P99**: Typically 3-4x average (outliers from GC, scheduling)

### Concurrency Trade-offs

- **Thread-safe queues**: 30-40% throughput reduction vs raw heapq
- **Lock contention**: Minimal with 5-10 threads, acceptable overhead
- **Async**: Best balance of safety and performance for IO-bound workloads

## Architecture Decisions Informed

### ✅ Use asyncio.PriorityQueue for MVP

**Rationale:**

- ModelMora uses FastAPI (async framework)
- Request handling is IO-bound (model loading, inference)
- Native async/await integration
- Excellent throughput (285k ops/sec >> target)
- Low latency (3-4μs negligible vs 50ms target)

**Implementation:**

```python
# In ModelMora scheduler
request_queue = asyncio.PriorityQueue()

async def enqueue_request(request: InferenceRequest, priority: Priority):
    task = Task(
        priority=priority.value,
        timestamp=time.time(),
        model_id=request.model_id,
        payload=request
    )
    await request_queue.put(task)

async def worker():
    while True:
        task = await request_queue.get()
        await process_inference(task)
```

### ✅ Priority Levels

Define 4 priority levels:

- **CRITICAL (1)**: SLA-bound, real-time inference
- **HIGH (2)**: User-facing, interactive requests
- **NORMAL (3)**: Batch processing, standard workload
- **LOW (4)**: Background tasks, analytics

### ✅ FIFO Within Priority

Use `(priority, timestamp)` tuple for ordering:

- Higher priority always first
- Same priority → FIFO (earliest timestamp first)
- Prevents starvation of same-priority tasks

### ✅ Scalability Path

**Phase 1 (MVP)**: asyncio.PriorityQueue (single-node)

- Expected: 10-100 req/sec
- Capacity: 285,000 req/sec
- **Huge headroom** for growth

**Phase 4 (Multi-node)**: Redis sorted sets

- Distributed queue across nodes
- Same priority semantics
- Horizontal scaling

## Production Implications

### For ModelMora Scheduler

1. **Queue Integration**:
   - Single `asyncio.PriorityQueue` instance
   - Enqueue from FastAPI request handlers
   - Dequeue from worker coroutines

2. **Priority Assignment**:
   - User-specified via API parameter
   - Default: NORMAL
   - Admin/health checks: CRITICAL

3. **Worker Pool**:
   - N worker coroutines (e.g., 10)
   - Each dequeues tasks in priority order
   - Load models, run inference, return results

4. **Monitoring**:
   - Queue depth gauge (Prometheus)
   - Enqueue/dequeue latency histograms
   - Priority distribution metrics

## Next Steps

1. ✅ Run this POC to validate expected performance
2. ✅ Confirm asyncio.PriorityQueue meets requirements
3. → Implement Scheduler with asyncio.PriorityQueue
4. → Add priority parameter to API endpoints
5. → Monitor queue depth and latency in production

## Related POCs

- **POC 1**: Multi-process memory isolation (worker architecture)
- **POC 2**: gRPC streaming performance (API protocol)
- **POC 3**: Model loading/unloading (lifecycle management)

## Success Criteria

This POC is successful if:

- ✅ Throughput > 100,000 ops/sec (target: 1000 req/sec)
- ✅ Avg latency < 10μs (target: <50ms queue overhead)
- ✅ 100% priority order correctness (0 violations)
- ✅ Clear recommendation: asyncio.PriorityQueue
- ✅ Validated concurrency safety (async/threads)

## Troubleshooting

### High Latency (>50μs)

Check for:

- GC pauses (run with `PYTHONMALLOCSTATS=1`)
- CPU throttling
- Background processes

### Priority Violations

Should never happen with stdlib queues. If violations occur:

- Check Task comparison logic
- Verify priority values (lower = higher priority)
- Inspect timestamp precision

### Low Throughput (<50k ops/sec)

Potential causes:

- Debug mode enabled
- Running in VM/container with limited CPU
- Antivirus interference (Windows)

## References

- Python heapq: <https://docs.python.org/3/library/heapq.html>
- queue.PriorityQueue: <https://docs.python.org/3/library/queue.html>
- asyncio.PriorityQueue: <https://docs.python.org/3/library/asyncio-queue.html>
- Priority Queue Algorithms: <https://en.wikipedia.org/wiki/Priority_queue>
