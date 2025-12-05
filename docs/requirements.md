# ModelMora - Requirements Documentation

**Version:** 1.0.0
**Date:** 2025-12-04
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements for ModelMora, a lightweight model serving framework designed to efficiently deploy and manage machine learning models in a containerized environment.

### 1.2 Scope

ModelMora provides:

- Centralized model registry with version management
- Dynamic model lifecycle orchestration
- High-performance inference via REST and gRPC APIs
- Priority-based request scheduling with batching
- Multi-process memory isolation for Python memory management

### 1.3 Context

ModelMora is part of the **MiraVeja** ecosystem, serving as a dedicated node for neural network operations including image processing, embedding generation, and text/image generation tasks.

### 1.4 Definitions & Acronyms

- **LRU**: Least Recently Used (eviction policy)
- **gRPC**: Google Remote Procedure Call
- **HF**: HuggingFace
- **MVP**: Minimum Viable Product
- **IPC**: Inter-Process Communication

---

## 2. User Personas

### 2.1 Data Scientist

- Downloads and registers models for experimentation
- Tests model inference via CLI or API
- Monitors model performance metrics

### 2.2 Application Developer (MiraVeja Team)

- Integrates ModelMora into application architecture
- Calls gRPC APIs for inference requests
- Handles async responses and error cases

### 2.3 DevOps Engineer

- Deploys ModelMora in containerized environments
- Configures resource limits and scaling policies
- Monitors system health and performance

---

## 3. Functional Requirements

### 3.1 Model Registry

#### 3.1.1 Model Metadata Management

- **REQ-REG-001**: The system SHALL maintain a centralized registry of available models with metadata including:
  - Model identifier (HuggingFace ID)
  - Version string (semantic versioning)
  - Task type (txt2embed, img2embed, txt2img, img2txt, txt2txt)
  - Resource requirements (memory, GPU VRAM)
  - Download source URL
  - Model file checksums
  - Custom configuration parameters

- **REQ-REG-002**: The system SHALL support PyTorch models exclusively in the initial release.

- **REQ-REG-003**: The system SHALL validate model metadata against a predefined schema during registration.

#### 3.1.2 Model Versioning

- **REQ-REG-004**: The system SHALL support multiple versions of the same model simultaneously.

- **REQ-REG-005**: The system SHALL generate a lock file (similar to `poetry.lock`) containing:
  - Pinned model versions
  - Download URLs
  - Checksums for integrity verification
  - Last updated timestamp

- **REQ-REG-006**: Users SHALL be able to upgrade, downgrade, or pin model versions.

#### 3.1.3 Model Discovery & Downloading

- **REQ-REG-007**: The system SHALL download models on-demand from HuggingFace Hub when requested.

- **REQ-REG-008**: The system SHALL cache downloaded models in a persistent volume to avoid redundant downloads.

- **REQ-REG-009**: The system SHALL support custom model sources (local filesystem, S3, custom HTTP endpoints) in addition to HuggingFace.

- **REQ-REG-010**: The system SHALL provide a curated model configuration file to supplement missing HuggingFace metadata (task type, vector dimensions, etc.).

#### 3.1.4 Model CRUD Operations

- **REQ-REG-011**: Users SHALL be able to:
  - Register new models
  - List available models with filtering by task type
  - Retrieve detailed model information
  - Update model configurations
  - Delete models and their cached files

### 3.2 Model Lifecycle Management

#### 3.2.1 Lazy Loading

- **REQ-LIFE-001**: The system SHALL implement lazy loading, loading models into memory only when the first inference request is received.
  - **Validated**: Load time 2s for small-medium models (90-420MB) from cache - acceptable for lazy loading
  - **Performance**: First inference adds 150ms warmup, subsequent inferences <10ms

- **REQ-LIFE-002**: The system SHALL provide a warmup endpoint to pre-load frequently used models.
  - **Use case**: Pre-warm models at startup to eliminate first-request latency

- **REQ-LIFE-003**: The system SHALL support configurable model preloading on service startup.
  - **Guidance**: Pre-load if load time >5s or first inference >500ms

#### 3.2.2 Memory Management & Eviction

- **REQ-LIFE-004**: The system SHALL implement an LRU (Least Recently Used) eviction policy when memory limits are reached.

- **REQ-LIFE-005**: The system SHALL use multi-process architecture where each model runs in a separate Python process to ensure complete memory cleanup on unload.
  - **Rationale**: POC demonstrated subprocess achieves 5000x better memory reclamation (0.1MB leak vs 516MB with GC)
  - **Implementation**: Kill subprocess to unload, not `del` + `gc.collect()`

- **REQ-LIFE-006**: The system SHALL monitor:
  - Total system memory usage
  - Per-model memory consumption (typical: 90-500MB per model)
  - GPU VRAM utilization
  - Model access timestamps

- **REQ-LIFE-007**: The system SHALL support configurable memory thresholds to trigger model eviction.
  - **Guidance**: Reserve 1.5x model footprint per worker (e.g., 500MB model → 750MB limit)

- **REQ-LIFE-008**: The system SHALL explicitly call `torch.cuda.empty_cache()` after model unloading on GPU instances.
  - **Note**: Even with GPU cache clearing, subprocess termination is required for complete cleanup

#### 3.2.3 Health Checks

- **REQ-LIFE-009**: The system SHALL verify model integrity after loading by running a test inference.

- **REQ-LIFE-010**: The system SHALL mark models as unhealthy if loading or test inference fails, and SHALL NOT route requests to unhealthy models.

- **REQ-LIFE-011**: The system SHALL support automatic retry of failed model loads with exponential backoff.

### 3.3 Request Queue & Scheduling

#### 3.3.1 Priority Queue

- **REQ-QUEUE-001**: The system SHALL implement a priority queue for inference requests with the following priority levels:
  - **CRITICAL (1)**: SLA-bound, real-time inference
  - **HIGH (2)**: User-facing, interactive requests
  - **NORMAL (3)**: Batch processing, standard workload
  - **LOW (4)**: Background tasks, analytics
  - **Validated**: asyncio.PriorityQueue provides 730,108 ops/sec with 0.7μs latency and 100% priority correctness

- **REQ-QUEUE-002**: The system SHALL group requests by target model to minimize load/unload cycles.
  - **Rationale**: Model loading takes 2s, grouping amortizes this cost

- **REQ-QUEUE-003**: The system SHALL support configurable request timeout values per task type.

- **REQ-QUEUE-004**: The system SHALL implement request cancellation when timeouts are exceeded.
  - **Performance**: Queue overhead (0.7μs) negligible vs model execution (50-500ms)

#### 3.3.2 Batching

- **REQ-QUEUE-005**: The system SHALL accumulate requests for the same model into batches to improve throughput.

- **REQ-QUEUE-006**: The system SHALL support configurable batch parameters:
  - Maximum batch size
  - Maximum batch wait time
  - Per-model batch size overrides

- **REQ-QUEUE-007**: The system SHALL preprocess batches (padding, normalization) before inference execution.

#### 3.3.3 Load Balancing

- **REQ-QUEUE-008**: In multi-node deployments, the system SHALL distribute requests across available ModelMora instances.

### 3.4 Inference Engine

#### 3.4.1 Execution

- **REQ-INFER-001**: The system SHALL execute model inference using the appropriate task-specific pipeline:
  - Text embedding generation
  - Image embedding generation
  - Text-to-image generation
  - Image-to-text generation (captioning)
  - Text-to-text generation

- **REQ-INFER-002**: The system SHALL support both CPU and GPU execution, with automatic device selection based on availability.

- **REQ-INFER-003**: The system SHALL handle inference errors gracefully and return meaningful error messages to clients.

#### 3.4.2 Asynchronous Processing

- **REQ-INFER-004**: The system SHALL support asynchronous inference requests with:
  - Unique job ID generation
  - Job status tracking (queued, processing, completed, failed)
  - Result retrieval by job ID

- **REQ-INFER-005**: The system SHALL support synchronous inference for low-latency use cases.

#### 3.4.3 Result Storage

- **REQ-INFER-006**: For small results (embeddings, short text), the system SHALL return results inline in the API response.

- **REQ-INFER-007**: For large results (generated images), the system SHALL:
  - Store results in object storage (S3/MinIO)
  - Return a URI or presigned URL to the client
  - Support configurable result expiration

- **REQ-INFER-008**: The system SHALL support both storage strategies via configuration.

### 3.5 Observability

#### 3.5.1 Metrics

- **REQ-OBS-001**: The system SHALL expose Prometheus-compatible metrics including:
  - Request latency (histogram) by model and task type
  - Request throughput (counter) by status (success, error)
  - Queue depth (gauge) by priority level
  - Model load/unload events (counter)
  - Active models count (gauge)
  - Memory usage (gauge) - system and per-model
  - GPU utilization (gauge)
  - Batch size distribution (histogram)

#### 3.5.2 Logging

- **REQ-OBS-002**: The system SHALL implement structured logging (JSON format) with configurable log levels.

- **REQ-OBS-003**: The system SHALL log:
  - Model lifecycle events (load, unload, eviction)
  - Request processing (received, queued, started, completed)
  - Errors with stack traces
  - Performance warnings (slow requests, queue saturation)

- **REQ-OBS-004**: The system SHALL support distributed tracing with trace ID propagation.

#### 3.5.3 Health Endpoints

- **REQ-OBS-005**: The system SHALL provide:
  - `/health/live` - Liveness probe (process running)
  - `/health/ready` - Readiness probe (can accept requests)
  - `/health/models` - Status of each loaded model

---

## 4. API Requirements

### 4.1 REST API (FastAPI)

#### 4.1.1 Model Registry Endpoints

- **POST** `/api/v1/models` - Register a new model
- **GET** `/api/v1/models` - List all models (with filtering)
- **GET** `/api/v1/models/{model_id}` - Get model details
- **PUT** `/api/v1/models/{model_id}` - Update model configuration
- **DELETE** `/api/v1/models/{model_id}` - Delete model
- **POST** `/api/v1/models/{model_id}/download` - Trigger model download
- **POST** `/api/v1/models/{model_id}/versions/{version}` - Set active version

#### 4.1.2 Inference Endpoints

- **POST** `/api/v1/infer/{model_id}` - Synchronous inference
- **POST** `/api/v1/infer/{model_id}/async` - Asynchronous inference (returns job ID)
- **GET** `/api/v1/jobs/{job_id}` - Get job status
- **GET** `/api/v1/jobs/{job_id}/result` - Retrieve job result

#### 4.1.3 Management Endpoints

- **POST** `/api/v1/models/{model_id}/load` - Force model load
- **POST** `/api/v1/models/{model_id}/unload` - Force model unload
- **GET** `/api/v1/status` - System status and loaded models
- **GET** `/api/v1/metrics` - Prometheus metrics endpoint

#### 4.1.4 Health Endpoints

- **GET** `/health/live` - Liveness probe
- **GET** `/health/ready` - Readiness probe
- **GET** `/health/models` - Model health status

### 4.2 gRPC Service

#### 4.2.1 Service Definition

- **REQ-GRPC-001**: The system SHALL provide a gRPC service with Protocol Buffer definitions for:
  - `InferenceService` - Model inference operations
  - `RegistryService` - Model registry operations
  - `ManagementService` - Lifecycle management operations

#### 4.2.2 Inference Methods

- `Infer(InferenceRequest) returns (InferenceResponse)` - Unary inference
- `InferAsync(InferenceRequest) returns (JobReference)` - Async inference
- `InferStream(stream InferenceRequest) returns (stream InferenceResponse)` - Streaming inference

#### 4.2.3 Client SDK

- **REQ-GRPC-002**: The system SHALL provide a Python client SDK for gRPC communication.

### 4.3 CLI Tool

#### 4.3.1 Commands

- `modelmora init [--config CONFIG_PATH]` - Initialize project with config template
- `modelmora install <model_id>[@version]` - Download and register model
- `modelmora list [--task TASK_TYPE]` - List installed models
- `modelmora info <model_id>` - Show model details
- `modelmora lock` - Generate/update lock file
- `modelmora serve [--config CONFIG_PATH]` - Start ModelMora server
- `modelmora health` - Check server health
- `modelmora uninstall <model_id>` - Remove model

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **REQ-PERF-001**: The system SHALL add <100ms overhead to model inference time (excluding model loading).

- **REQ-PERF-002**: The system SHALL support at least 10 concurrent requests in single-node deployment.

- **REQ-PERF-003**: Model loading time SHALL be <30 seconds for models up to 5GB in size.

- **REQ-PERF-004**: The system SHALL achieve >80% GPU utilization during peak load when batching is enabled.

### 5.2 Scalability

- **REQ-SCALE-001**: The system SHALL support at least 20 different models registered in the catalog.

- **REQ-SCALE-002**: The system SHALL support at least 5 models loaded simultaneously (memory permitting).

- **REQ-SCALE-003**: The system SHALL scale horizontally to multiple nodes behind a load balancer.

### 5.3 Reliability

- **REQ-REL-001**: The system SHALL achieve 99.9% uptime in production environments.

- **REQ-REL-002**: The system SHALL gracefully handle model loading failures without crashing.

- **REQ-REL-003**: The system SHALL implement request retry logic with exponential backoff for transient failures.

- **REQ-REL-004**: The system SHALL persist request queue state to survive process restarts (for async requests).

### 5.4 Resource Constraints

- **REQ-RES-001**: The system SHALL run with <4GB base memory (excluding loaded models).

- **REQ-RES-002**: The system SHALL respect configurable memory limits and SHALL NOT exceed them.

- **REQ-RES-003**: The system SHALL support GPU sharing between multiple models when VRAM permits.

### 5.5 Security

- **REQ-SEC-001**: The system SHALL validate all input data against expected schemas to prevent injection attacks.

- **REQ-SEC-002**: The system SHALL verify model file checksums after download to prevent tampering.

- **REQ-SEC-003**: The system SHALL support TLS/SSL for gRPC and HTTPS for REST APIs.

- **REQ-SEC-004**: The system SHALL support API authentication (token-based) for production deployments.

### 5.6 Maintainability

- **REQ-MAINT-001**: The codebase SHALL maintain >90% test coverage.

- **REQ-MAINT-002**: The system SHALL use type hints throughout the codebase for static analysis.

- **REQ-MAINT-003**: The system SHALL provide comprehensive API documentation using OpenAPI/Swagger.

- **REQ-MAINT-004**: The system SHALL follow PEP 8 coding standards with automated linting.

### 5.7 Deployability

- **REQ-DEPLOY-001**: The system SHALL be packaged as a Docker image.

- **REQ-DEPLOY-002**: The system SHALL provide Kubernetes deployment manifests (YAML/Helm chart).

- **REQ-DEPLOY-003**: The system SHALL support configuration via environment variables and config files.

- **REQ-DEPLOY-004**: The system SHALL support volume mounts for model cache persistence.

---

## 6. Technical Constraints

### 6.1 Technology Stack

- **Python**: 3.10 or higher
- **Web Framework**: FastAPI
- **RPC Framework**: gRPC (grpcio)
- **Database**: PostgreSQL
- **Task Queue**: In-memory (MVP), Redis (production)
- **Object Storage**: S3/MinIO
- **Containerization**: Docker, Kubernetes

### 6.2 Model Support

- **Framework**: PyTorch only (initial release)
- **Source**: HuggingFace Hub (primary), local files, custom URLs
- **Task Types**: txt2embed, img2embed, txt2img, img2txt, txt2txt

### 6.3 Dependencies

- **Core**: torch, transformers, diffusers, sentence-transformers
- **API**: fastapi, grpcio, pydantic
- **Storage**: sqlalchemy, boto3 (S3), redis (optional)
- **Observability**: prometheus-client, structlog

---

## 7. Data Models

### 7.1 Model Metadata Schema

```yaml
model_id: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
version: str  # semantic version
task_type: enum  # txt2embed | img2embed | txt2img | img2txt | txt2txt
display_name: str
description: str
source_url: str
file_checksum: str  # SHA256
resource_requirements:
  memory_mb: int
  gpu_vram_mb: int
  cpu_threads: int
config:
  vector_size: int  # for embedding models
  max_length: int  # for text models
  batch_size: int
  device: enum  # cpu | cuda | auto
custom_params: dict  # arbitrary key-value pairs
created_at: datetime
updated_at: datetime
```

### 7.2 Inference Request Schema

```yaml
model_id: str
input_data:
  text: str | list[str]
  image: str  # base64 or URL
  # task-specific fields
parameters:
  temperature: float
  max_tokens: int
  # task-specific parameters
priority: enum  # high | medium | low
timeout_seconds: int
```

### 7.3 Lock File Format

```yaml
# modelmora.lock
version: "1.0"
generated_at: datetime
models:
  - model_id: str
    version: str
    source_url: str
    checksum: str
    task_type: str
```

---

## 8. Integration Requirements

### 8.1 MiraVeja Integration

- **REQ-INT-001**: ModelMora SHALL be callable from MiraVeja via gRPC.

- **REQ-INT-002**: MiraVeja SHALL handle async job polling with configurable retry intervals.

- **REQ-INT-003**: ModelMora SHALL return embeddings inline for MiraVeja image search features.

### 8.2 Kafka Integration (Future)

- **REQ-INT-004**: The system SHALL optionally consume inference requests from Kafka topics.

- **REQ-INT-005**: The system SHALL publish inference results to Kafka response topics.

- **REQ-INT-006**: The system SHALL publish model lifecycle events to an audit topic.

---

## 9. Out of Scope (Future Releases)

- ONNX Runtime support
- TensorRT optimization
- Model A/B testing
- Model fine-tuning capabilities
- Multi-tenant isolation
- Advanced authentication (OAuth, JWT)
- Web UI for management
- Model quantization
- Distributed training integration

---

## 10. Acceptance Criteria

### 10.1 MVP (v0.1.0)

- [ ] Register at least 3 different models (embedding, text gen, image gen)
- [ ] Execute inference via REST API with <100ms overhead
- [ ] Handle 10 concurrent requests without errors
- [ ] Models load lazily on first request
- [ ] Memory usage <4GB excluding models
- [ ] Docker image builds successfully
- [ ] Unit test coverage >90%
- [ ] API documentation available

### 10.2 Production Ready (v1.0.0)

- [ ] All requirements implemented
- [ ] gRPC service operational
- [ ] CLI tool with all commands
- [ ] Prometheus metrics exposed
- [ ] Kubernetes deployment successful
- [ ] Integration with MiraVeja complete
- [ ] 99.9% uptime in staging environment
- [ ] Load testing passed (100 req/s)
- [ ] Complete documentation published

---

## 11. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | - | Initial comprehensive requirements document |
