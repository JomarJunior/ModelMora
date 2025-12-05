"""
Production-Level gRPC Streaming Performance Test

This implementation uses proper Protocol Buffer compilation and real
network communication for accurate performance testing.

Architecture:
- gRPC server runs on port 50051 with proper protobuf servicer
- REST server runs on port 8000 with FastAPI
- Both servers handle real network traffic
- Client makes actual network calls to both servers
"""

import asyncio
import io

# Import generated protobuf code
import os
import sys
import time
from concurrent import futures
from typing import AsyncIterator

import grpc
from grpc import aio as grpc_aio

# Add examples directory to path so protobuf imports work
examples_dir = os.path.dirname(os.path.abspath(__file__))
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

import inference_pb2
import inference_pb2_grpc

# Optional dependencies
try:
    import numpy as np
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ============================================================================
# gRPC Server Implementation
# ============================================================================


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Production gRPC servicer with proper protobuf messages"""

    async def StreamInference(
        self, request: inference_pb2.InferenceRequest, context: grpc_aio.ServicerContext
    ) -> AsyncIterator[inference_pb2.InferenceResponse]:
        """Server-side streaming RPC for large payloads"""

        request_id = f"req_{int(time.time() * 1000)}"
        num_results = request.batch_size
        chunk_size = 1024 * 1024  # 1MB chunks

        for i in range(num_results):
            # Generate payload
            if PIL_AVAILABLE:
                img_data = self._generate_image(1024, 1024)
            else:
                img_data = bytes(10 * 1024 * 1024)  # 10MB

            # Stream in chunks
            num_chunks = (len(img_data) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(img_data))
                chunk_data = img_data[start:end]

                response = inference_pb2.InferenceResponse(
                    request_id=request_id, chunk_index=chunk_idx, total_chunks=num_chunks, data=chunk_data
                )
                response.metadata["result_index"] = str(i)
                response.metadata["format"] = "png"

                yield response

    def _generate_image(self, width: int, height: int) -> bytes:
        """Generate random PNG image data"""
        if PIL_AVAILABLE:
            arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return bytes(width * height * 3)


async def serve_grpc(port: int = 50051):
    """Start production gRPC server"""
    server = grpc_aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.so_reuseport", 1),
        ],
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"🚀 gRPC server started on port {port}")

    return server


# ============================================================================
# REST Server Implementation
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="ModelMora Inference API")

    @app.post("/infer/stream")
    async def rest_stream_inference(request: dict):
        """REST endpoint with chunked streaming"""

        async def generate():
            batch_size = request.get("batch_size", 1)
            chunk_size = 1024 * 1024  # 1MB

            servicer = InferenceServicer()

            for i in range(batch_size):
                # Generate payload
                if PIL_AVAILABLE:
                    img_data = servicer._generate_image(1024, 1024)
                else:
                    img_data = bytes(10 * 1024 * 1024)

                # Stream in chunks
                for start in range(0, len(img_data), chunk_size):
                    end = min(start + chunk_size, len(img_data))
                    yield img_data[start:end]

        return StreamingResponse(
            generate(), media_type="application/octet-stream", headers={"X-Content-Type-Options": "nosniff"}
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}


# ============================================================================
# Client-side Testing
# ============================================================================


class PerformanceMetrics:
    """Performance measurement results"""

    def __init__(self, protocol: str, total_time_ms: float, bytes_transferred: int, num_chunks: int):
        self.protocol = protocol
        self.total_time_ms = total_time_ms
        self.bytes_transferred = bytes_transferred
        self.num_chunks = num_chunks
        self.throughput_mbps = (bytes_transferred / (1024 * 1024)) / (total_time_ms / 1000)
        self.latency_per_chunk_ms = total_time_ms / num_chunks if num_chunks > 0 else 0

    def __str__(self):
        return (
            f"{self.protocol}:\n"
            f"  Total time: {self.total_time_ms:.1f}ms\n"
            f"  Data transferred: {self.bytes_transferred / (1024*1024):.1f}MB\n"
            f"  Throughput: {self.throughput_mbps:.2f} MB/s\n"
            f"  Avg latency/chunk: {self.latency_per_chunk_ms:.2f}ms\n"
            f"  Chunks: {self.num_chunks}"
        )


async def test_grpc_client(host: str = "localhost", port: int = 50051, batch_size: int = 10) -> PerformanceMetrics:
    """Test gRPC client with real network calls"""

    print("\n" + "=" * 70)
    print("TEST 1: gRPC Streaming (Real Network)")
    print("=" * 70)

    # Create channel with proper options
    channel = grpc_aio.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
        ],
    )

    try:
        # Create stub (client)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        # Create request
        request = inference_pb2.InferenceRequest(model_id="test-model", prompt="Generate images", batch_size=batch_size)

        print(f"\n📤 Sending gRPC request for {batch_size} results...")

        start_time = time.time()
        total_bytes = 0
        num_chunks = 0

        # Stream responses
        async for response in stub.StreamInference(request):
            total_bytes += len(response.data)
            num_chunks += 1

            if num_chunks % 10 == 0:
                print(f"  Received chunk {num_chunks}/{response.total_chunks}...")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        metrics = PerformanceMetrics(
            protocol="gRPC", total_time_ms=elapsed_ms, bytes_transferred=total_bytes, num_chunks=num_chunks
        )

        print(f"\n✅ gRPC streaming complete")
        print(f"   {metrics}")

        return metrics

    finally:
        await channel.close()


async def test_rest_client(host: str = "localhost", port: int = 8000, batch_size: int = 10) -> PerformanceMetrics:
    """Test REST client with real HTTP calls"""

    print("\n" + "=" * 70)
    print("TEST 2: REST Streaming (Real HTTP)")
    print("=" * 70)

    try:
        import httpx
    except ImportError:
        print("⚠️  httpx not available")
        return None

    url = f"http://{host}:{port}/infer/stream"
    request_data = {"model_id": "test-model", "prompt": "Generate images", "batch_size": batch_size}

    print(f"\n📤 Sending HTTP POST request for {batch_size} results...")

    start_time = time.time()
    total_bytes = 0
    num_chunks = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=request_data) as response:
            async for chunk in response.aiter_bytes():
                total_bytes += len(chunk)
                num_chunks += 1

                if num_chunks % 10 == 0:
                    print(f"  Received chunk {num_chunks}...")

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    metrics = PerformanceMetrics(
        protocol="REST", total_time_ms=elapsed_ms, bytes_transferred=total_bytes, num_chunks=num_chunks
    )

    print(f"\n✅ REST streaming complete")
    print(f"   {metrics}")

    return metrics


def print_comparison(grpc_metrics: PerformanceMetrics, rest_metrics: PerformanceMetrics):
    """Print detailed comparison"""

    print("\n" + "=" * 70)
    print("COMPARISON: gRPC vs REST (Production Network)")
    print("=" * 70)

    if not rest_metrics:
        print("\n⚠️  REST test skipped")
        return

    speedup = grpc_metrics.throughput_mbps / rest_metrics.throughput_mbps

    print(f"\n📊 Throughput:")
    print(f"  gRPC: {grpc_metrics.throughput_mbps:.2f} MB/s")
    print(f"  REST: {rest_metrics.throughput_mbps:.2f} MB/s")
    print(f"  Speedup: {speedup:.2f}x {'✅' if speedup > 1.0 else '⚠️'}")

    latency_improvement = (
        (rest_metrics.latency_per_chunk_ms - grpc_metrics.latency_per_chunk_ms)
        / rest_metrics.latency_per_chunk_ms
        * 100
    )

    print(f"\n⏱️  Latency per chunk:")
    print(f"  gRPC: {grpc_metrics.latency_per_chunk_ms:.2f}ms")
    print(f"  REST: {rest_metrics.latency_per_chunk_ms:.2f}ms")
    print(f"  Improvement: {latency_improvement:+.1f}%")

    time_saved = rest_metrics.total_time_ms - grpc_metrics.total_time_ms

    print(f"\n🕐 Total time:")
    print(f"  gRPC: {grpc_metrics.total_time_ms:.1f}ms")
    print(f"  REST: {rest_metrics.total_time_ms:.1f}ms")
    print(f"  Time saved: {time_saved:+.1f}ms ({time_saved/rest_metrics.total_time_ms*100:+.1f}%)")

    print(f"\n{'='*70}")
    print("VERDICT:")
    print(f"{'='*70}")

    if speedup >= 2.0 and grpc_metrics.latency_per_chunk_ms < 100:
        print("✅ PASS: gRPC significantly faster")
        print(f"   - Throughput: {speedup:.2f}x improvement")
        print(f"   - Latency: {grpc_metrics.latency_per_chunk_ms:.1f}ms < 100ms target")
        print("\n💡 Recommendation: Use gRPC as primary API for ModelMora")
    elif speedup >= 1.5:
        print("⚠️  PARTIAL: gRPC shows moderate improvement")
        print(f"   - Throughput: {speedup:.2f}x improvement")
        print("\n💡 Recommendation: Use gRPC for large payloads, REST for simple queries")
    elif speedup >= 1.0:
        print("📊 OBSERVATION: gRPC slightly faster")
        print(f"   - Throughput: {speedup:.2f}x improvement")
        print("\n💡 Recommendation: Both protocols viable, choose based on ecosystem")
    else:
        print("⚠️  OBSERVATION: REST faster in this test")
        print(f"   - REST throughput: {rest_metrics.throughput_mbps:.2f} MB/s")
        print("\n💡 Note: May be due to HTTP/1.1 vs gRPC overhead for small payloads")


async def run_concurrent_test(num_clients: int = 10, batch_size: int = 5):
    """Test with concurrent clients"""

    print("\n" + "=" * 70)
    print(f"TEST 3: Concurrent Load ({num_clients} gRPC clients)")
    print("=" * 70)

    print(f"\n🚀 Starting {num_clients} concurrent clients...")

    start_time = time.time()
    tasks = [test_grpc_client(batch_size=batch_size) for _ in range(num_clients)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    metrics = [r for r in results if isinstance(r, PerformanceMetrics)]

    end_time = time.time()
    total_time = (end_time - start_time) * 1000

    if metrics:
        total_bytes = sum(m.bytes_transferred for m in metrics)
        avg_throughput = sum(m.throughput_mbps for m in metrics) / len(metrics)

        print(f"\n✅ {len(metrics)}/{num_clients} clients completed successfully")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Total data: {total_bytes / (1024*1024):.1f}MB")
        print(f"   Avg throughput/client: {avg_throughput:.2f} MB/s")
        print(f"   Aggregate throughput: {(total_bytes / (1024*1024)) / (total_time/1000):.2f} MB/s")


async def main():
    """Main test orchestrator"""

    print("=" * 70)
    print("Production gRPC Streaming Performance Test")
    print("=" * 70)
    print("\nUsing:")
    print(f"  - Protocol Buffers (compiled from .proto)")
    print(f"  - Real network communication (localhost)")
    print(f"  - PIL for image generation: {'✅' if PIL_AVAILABLE else '❌'}")

    batch_size = 10

    print(f"\nConfiguration:")
    print(f"  - Payload size: ~3MB per result (PNG image)")
    print(f"  - Batch size: {batch_size} results")
    print(f"  - Total data: ~{batch_size * 3}MB per request")

    try:
        # Start gRPC server
        grpc_server = await serve_grpc(port=50051)

        # Give server time to start
        await asyncio.sleep(0.5)

        # Test gRPC
        grpc_metrics = await test_grpc_client(batch_size=batch_size)

        # Test REST (if server running)
        rest_metrics = None
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    print("\n✅ REST server detected")
                    rest_metrics = await test_rest_client(batch_size=batch_size)
        except Exception:
            print("\n⚠️  REST server not running (expected if not started separately)")
            print("   To test REST: uvicorn examples.poc_grpc_streaming_prod:app --port 8000")

        # Compare
        if rest_metrics:
            print_comparison(grpc_metrics, rest_metrics)

        # Concurrent test
        await run_concurrent_test(num_clients=10, batch_size=5)

        # Cleanup
        await grpc_server.stop(grace=2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
