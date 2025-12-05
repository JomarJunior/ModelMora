# Production gRPC Streaming Performance Test

## Overview

This is a production-level implementation using:

- ✅ **Real Protocol Buffers** (compiled from `.proto` files)
- ✅ **Actual gRPC network communication** (not simulated)
- ✅ **Real HTTP REST API** with FastAPI
- ✅ **Concurrent load testing** with multiple clients

## Quick Start

### 1. Install Dependencies

```powershell
pip install grpcio grpcio-tools Pillow numpy fastapi uvicorn httpx
```

### 2. Generate Protocol Buffers (Already Done)

The `.proto` file has been compiled to `inference_pb2.py` and `inference_pb2_grpc.py`.

To regenerate if needed:

```powershell
cd examples
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
```

### 3. Run the Test

- **Option A: gRPC Only (No REST comparison)**

```powershell
python ./examples/poc_grpc_streaming_prod.py
```

- **Option B: Full Comparison (gRPC vs REST)**

Terminal 1 - Start REST server:

```powershell
uvicorn examples.poc_grpc_streaming_prod:app --port 8000
```

Terminal 2 - Run test:

```powershell
python ./examples/poc_grpc_streaming_prod.py
```

## Test Results (Sample)

### Single Request Performance

```text
gRPC:
  Total time: 977.1ms
  Data transferred: 30.1MB
  Throughput: 30.75 MB/s
  Avg latency/chunk: 24.43ms
  Chunks: 40
```

### Concurrent Load (10 clients)

```text
✅ 10/10 clients completed successfully
   Total time: 4698.2ms
   Total data: 150.2MB
   Avg throughput/client: 3.28 MB/s
   Aggregate throughput: 31.98 MB/s
```

## Architecture

### Protocol Buffer Definition (`inference.proto`)

```protobuf
syntax = "proto3";

package modelmora;

service InferenceService {
  rpc StreamInference(InferenceRequest) returns (stream InferenceResponse);
}

message InferenceRequest {
  string model_id = 1;
  string prompt = 2;
  int32 batch_size = 3;
}

message InferenceResponse {
  string request_id = 1;
  int32 chunk_index = 2;
  int32 total_chunks = 3;
  bytes data = 4;
  map<string, string> metadata = 5;
}
```

### gRPC Server

- Runs on port `50051`
- Uses proper `InferenceServiceServicer` from generated code
- Streams PNG images in 1MB chunks
- Supports concurrent clients with thread pool

### REST Server

- Runs on port `8000` (FastAPI)
- Same image generation logic
- HTTP chunked transfer encoding
- Endpoint: `POST /infer/stream`

### Client

- Creates proper gRPC stub from generated code
- Makes real network calls (not in-process)
- Measures throughput, latency, and data transfer
- Supports concurrent load testing

## Key Differences from Simplified POC

| Aspect | Simplified POC | Production POC |
|--------|----------------|----------------|
| Protocol Buffers | Simulated with dataclasses | Compiled from `.proto` |
| gRPC Communication | Direct servicer call | Real network via stub |
| Message Serialization | Python objects | Protobuf binary |
| Network Stack | Skipped | Full gRPC/HTTP stack |
| Server Registration | Manual `_servicer` attribute | Proper `add_*_to_server()` |

## Performance Expectations

### gRPC Advantages

- **HTTP/2 multiplexing**: Single connection for multiple streams
- **Protobuf efficiency**: 3-10x smaller than JSON
- **Binary framing**: Less parsing overhead
- **Built-in streaming**: Native support for large payloads

### When gRPC Wins

- Large payloads (>1MB): Images, embeddings, audio
- High throughput: Many concurrent requests
- Low latency: Binary protocol overhead <1ms
- Service-to-service: Internal microservices

### When REST Competitive

- Small payloads (<100KB): JSON responses
- Browser clients: Wide compatibility
- Debugging: Human-readable with curl
- Simple queries: Model listing, health checks

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'inference_pb2'`

Solution: Run from project root:

```powershell
python ./examples/poc_grpc_streaming_prod.py
```

### Port Already in Use

```bash
OSError: [WinError 10048] Only one usage of each socket address...
```

Solution: Kill existing process or change port:

```powershell
netstat -ano | findstr :50051
taskkill /F /PID <PID>
```

### REST Server Not Detected

Expected if not running REST server. Test will skip REST comparison and show:

```text
⚠️  REST server not running (expected if not started separately)
```

## Next Steps for ModelMora

Based on results:

1. **If gRPC >2x faster**: Use as primary API
   - Implement all inference endpoints with gRPC
   - Keep REST for admin/health endpoints

2. **If gRPC 1.5-2x faster**: Support both protocols
   - gRPC for large payloads (images, embeddings)
   - REST for simple queries (model list, status)

3. **If similar performance**: Choose by ecosystem
   - MiraVeja uses gRPC → proceed with gRPC
   - External integrations → provide both

## Production Checklist

- [ ] Add TLS/SSL for encrypted connections
- [ ] Implement authentication (API keys, JWT)
- [ ] Add request validation and error handling
- [ ] Implement proper logging and metrics
- [ ] Add rate limiting and quotas
- [ ] Create client SDKs (Python, TypeScript)
- [ ] Document API with examples
- [ ] Deploy with load balancer
- [ ] Monitor with Prometheus/Grafana

## References

- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
- [Protocol Buffers Guide](https://protobuf.dev/)
- [FastAPI Streaming](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
