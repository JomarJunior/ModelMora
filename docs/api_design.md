# ModelMora - API Design Document

**Version:** 1.0.0
**Date:** 2025-12-04
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose

This document defines the API contracts for ModelMora, including REST endpoints (OpenAPI specification), gRPC service definitions (Protocol Buffers), and message schemas for event-driven communication. These specifications serve as the contract between ModelMora and its clients.

### 1.2 API Versioning Strategy

- **REST API**: Version prefix in URL path (`/api/v1/...`)
- **gRPC**: Version suffix in package name (`modelmora.v1`)
- **Breaking Changes**: Require major version increment
- **Deprecation**: Supported for 2 minor versions before removal

### 1.3 API Principles

- **Resource-Oriented**: REST endpoints follow RESTful conventions
- **Idempotent Operations**: PUT/DELETE operations are idempotent
- **Consistency**: Common error formats and status codes
- **Validation**: Input validation with detailed error messages
- **Documentation**: Self-documenting with OpenAPI/Swagger UI

### 1.4 POC-Validated Performance Characteristics

**REST API Performance:**

- Suitable for: Small requests/responses (embeddings, text, metadata)
- Typical latency: <50ms for simple inference
- Synchronous by default, async endpoints for long-running tasks

**gRPC Streaming Performance:** ✅ **(POC 2: Validated)**

- **Throughput** 31.92 MB/s (single client), 33.31 MB/s (10 concurrent clients)
- **Latency**: 23.53ms per chunk average
- **Use cases**:
  - Image generation (3MB+ per result)
  - Large batch embeddings
  - Video/audio processing
- **Protocol**: Server-side streaming with Protocol Buffers
- **Verdict**: gRPC performs well for large payloads, both protocols viable

**API Selection Guide:**

| Use Case | Protocol | Rationale |
|----------|----------|----------|
| Text embedding (<1KB) | REST | Simple, low overhead |
| Batch embeddings (>100KB) | gRPC | Streaming efficiency |
| Image generation (3MB+) | gRPC | Validated 31.92 MB/s throughput |
| Model management (CRUD) | REST | Resource-oriented operations |
| Health checks | REST | Simple request/response |
| Real-time streaming | gRPC | Bidirectional streaming support |

---

## 2. REST API Specification (OpenAPI 3.0)

### 2.1 OpenAPI Metadata

```yaml
openapi: 3.0.3
info:
  title: ModelMora API
  version: 1.0.0
  description: |
    ModelMora is a lightweight model serving framework for managing and executing
    machine learning model inference at scale.

    ## Features
    - Model registry with version management
    - Lazy model loading with LRU eviction
    - Synchronous and asynchronous inference
    - Priority-based request queuing
    - Batch processing for improved throughput

  contact:
    name: MiraVeja Team
    email: support@miraveja.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: http://localhost:8000/api/v1
    description: Local development server
  - url: https://modelmora.miraveja.com/api/v1
    description: Production server

tags:
  - name: Registry
    description: Model catalog management
  - name: Inference
    description: Model inference operations
  - name: Lifecycle
    description: Model lifecycle management
  - name: Jobs
    description: Asynchronous job management
  - name: Health
    description: Health check endpoints
  - name: Metrics
    description: Observability and metrics

security:
  - BearerAuth: []
```

### 2.2 Common Components

#### 2.2.1 Security Schemes

```yaml
components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT token for API authentication (optional in MVP)
```

#### 2.2.2 Common Schemas

```yaml
components:
  schemas:
    # Error Response
    Error:
      type: object
      required:
        - error_code
        - message
      properties:
        error_code:
          type: string
          example: "MODEL_NOT_FOUND"
        message:
          type: string
          example: "Model 'invalid-model-id' not found in registry"
        details:
          type: object
          additionalProperties: true
        trace_id:
          type: string
          format: uuid
          description: Distributed tracing ID

    # Validation Error
    ValidationError:
      type: object
      required:
        - error_code
        - message
        - validation_errors
      properties:
        error_code:
          type: string
          example: "VALIDATION_ERROR"
        message:
          type: string
          example: "Request validation failed"
        validation_errors:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
                example: "model_id"
              message:
                type: string
                example: "Field is required"

    # Pagination
    PaginationMeta:
      type: object
      properties:
        total:
          type: integer
          example: 42
        page:
          type: integer
          example: 1
        page_size:
          type: integer
          example: 20
        total_pages:
          type: integer
          example: 3
```

### 2.3 Registry API

#### 2.3.1 Register Model

```yaml
paths:
  /models:
    post:
      tags:
        - Registry
      summary: Register a new model
      description: |
        Registers a new model in the ModelMora catalog. The model files are not
        downloaded immediately; use the `/download` endpoint or lazy loading will
        occur on first inference request.
      operationId: registerModel
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RegisterModelRequest'
            examples:
              embedding_model:
                summary: Text embedding model
                value:
                  model_id: "sentence-transformers/all-MiniLM-L6-v2"
                  version: "v2.2.2"
                  task_type: "txt2embed"
                  display_name: "MiniLM-L6 Sentence Embeddings"
                  config:
                    vector_size: 384
                    max_length: 512
                    batch_size: 32
              image_gen_model:
                summary: Text-to-image model
                value:
                  model_id: "stabilityai/stable-diffusion-2-1"
                  version: "fp16"
                  task_type: "txt2img"
                  resource_requirements:
                    gpu_vram_mb: 8192
      responses:
        '201':
          description: Model registered successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelResponse'
        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationError'
        '409':
          description: Model already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    RegisterModelRequest:
      type: object
      required:
        - model_id
        - task_type
      properties:
        model_id:
          type: string
          description: HuggingFace model identifier
          example: "sentence-transformers/all-MiniLM-L6-v2"
        version:
          type: string
          description: Specific version or branch name
          example: "v2.2.2"
          default: "main"
        task_type:
          $ref: '#/components/schemas/TaskType'
        display_name:
          type: string
          example: "MiniLM-L6 Sentence Embeddings"
        description:
          type: string
          example: "Lightweight sentence embedding model"
        source_url:
          type: string
          format: uri
          description: Custom download URL (overrides HuggingFace)
        resource_requirements:
          $ref: '#/components/schemas/ResourceRequirements'
        config:
          type: object
          description: Model-specific configuration
          additionalProperties: true

    ModelResponse:
      type: object
      properties:
        model_id:
          type: string
        version:
          type: string
        task_type:
          $ref: '#/components/schemas/TaskType'
        display_name:
          type: string
        description:
          type: string
        status:
          type: string
          enum: [registered, downloading, ready, failed]
        resource_requirements:
          $ref: '#/components/schemas/ResourceRequirements'
        config:
          type: object
          additionalProperties: true
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    TaskType:
      type: string
      enum:
        - txt2embed
        - img2embed
        - txt2img
        - img2txt
        - txt2txt
      description: |
        - `txt2embed`: Text to embedding vector
        - `img2embed`: Image to embedding vector
        - `txt2img`: Text to image generation
        - `img2txt`: Image to text (captioning)
        - `txt2txt`: Text to text (generation, translation)

    ResourceRequirements:
      type: object
      properties:
        memory_mb:
          type: integer
          description: Estimated RAM requirement
          example: 2048
        gpu_vram_mb:
          type: integer
          description: Estimated VRAM requirement
          example: 4096
        cpu_threads:
          type: integer
          description: Recommended CPU threads
          example: 4
```

#### 2.3.2 List Models

```yaml
paths:
  /models:
    get:
      tags:
        - Registry
      summary: List registered models
      description: Retrieve all models with optional filtering
      operationId: listModels
      parameters:
        - name: task_type
          in: query
          schema:
            $ref: '#/components/schemas/TaskType'
          description: Filter by task type
        - name: status
          in: query
          schema:
            type: string
            enum: [registered, downloading, ready, failed]
        - name: page
          in: query
          schema:
            type: integer
            default: 1
            minimum: 1
        - name: page_size
          in: query
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 100
      responses:
        '200':
          description: List of models
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/ModelResponse'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'
```

#### 2.3.3 Get Model Details

```yaml
paths:
  /models/{model_id}:
    get:
      tags:
        - Registry
      summary: Get model details
      operationId: getModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
          description: URL-encoded model identifier
          example: "sentence-transformers%2Fall-MiniLM-L6-v2"
      responses:
        '200':
          description: Model details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelResponse'
        '404':
          description: Model not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
```

#### 2.3.4 Update Model

```yaml
paths:
  /models/{model_id}:
    put:
      tags:
        - Registry
      summary: Update model configuration
      operationId: updateModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateModelRequest'
      responses:
        '200':
          description: Model updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelResponse'
        '404':
          description: Model not found

components:
  schemas:
    UpdateModelRequest:
      type: object
      properties:
        display_name:
          type: string
        description:
          type: string
        config:
          type: object
          additionalProperties: true
        resource_requirements:
          $ref: '#/components/schemas/ResourceRequirements'
```

#### 2.3.5 Delete Model

```yaml
paths:
  /models/{model_id}:
    delete:
      tags:
        - Registry
      summary: Delete model
      description: Removes model from registry and deletes cached files
      operationId: deleteModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
        - name: force
          in: query
          schema:
            type: boolean
            default: false
          description: Force deletion even if model is loaded
      responses:
        '204':
          description: Model deleted successfully
        '404':
          description: Model not found
        '409':
          description: Model is currently loaded (use force=true)
```

#### 2.3.6 Download Model

```yaml
paths:
  /models/{model_id}/download:
    post:
      tags:
        - Registry
      summary: Download model files
      description: Explicitly downloads model files to cache
      operationId: downloadModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '202':
          description: Download started
          content:
            application/json:
              schema:
                type: object
                properties:
                  model_id:
                    type: string
                  status:
                    type: string
                    enum: [downloading]
                  message:
                    type: string
                    example: "Download started in background"
        '404':
          description: Model not found
```

#### 2.3.7 Generate Lock File

```yaml
paths:
  /models/lock:
    post:
      tags:
        - Registry
      summary: Generate lock file
      description: Creates a lock file with pinned model versions
      operationId: generateLock
      responses:
        '200':
          description: Lock file content
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LockFile'

components:
  schemas:
    LockFile:
      type: object
      properties:
        version:
          type: string
          example: "1.0"
        generated_at:
          type: string
          format: date-time
        models:
          type: array
          items:
            type: object
            properties:
              model_id:
                type: string
              version:
                type: string
              source_url:
                type: string
              checksum:
                type: string
              task_type:
                $ref: '#/components/schemas/TaskType'
```

### 2.4 Inference API

#### 2.4.1 Synchronous Inference

**POC-Validated Queue Performance:**

- Priority queue latency: 0.7μs enqueue (negligible)
- 4 priority levels: CRITICAL(1), HIGH(2), NORMAL(3), LOW(4)
- 100% priority ordering correctness under load
- Capacity: 730,108 ops/sec (730x above target)

```yaml
paths:
  /infer/{model_id}:
    post:
      tags:
        - Inference
      summary: Execute synchronous inference
      description: |
        Executes inference and waits for result. Suitable for low-latency
        requests like embeddings. For long-running tasks, use async endpoint.
      operationId: infer
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferenceRequest'
            examples:
              text_embedding:
                summary: Text embedding request
                value:
                  input_data:
                    text: "This is a sample sentence to embed"
                  parameters:
                    normalize: true
              text_generation:
                summary: Text generation request
                value:
                  input_data:
                    text: "Once upon a time"
                  parameters:
                    max_tokens: 100
                    temperature: 0.7
                  priority: "medium"
              image_generation:
                summary: Image generation (should use async)
                value:
                  input_data:
                    text: "A beautiful sunset over mountains"
                  parameters:
                    width: 512
                    height: 512
                    steps: 50
                  priority: "low"
                  timeout_seconds: 300
      responses:
        '200':
          description: Inference completed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InferenceResponse'
        '400':
          description: Invalid input
        '404':
          description: Model not found
        '408':
          description: Request timeout
        '503':
          description: Model loading or service unavailable

components:
  schemas:
    InferenceRequest:
      type: object
      required:
        - input_data
      properties:
        input_data:
          $ref: '#/components/schemas/InputData'
        parameters:
          type: object
          description: Task-specific inference parameters
          additionalProperties: true
        priority:
          type: string
          enum: [high, medium, low]
          default: medium
          description: |
            - `high`: Fast tasks (embeddings, <1s)
            - `medium`: Text generation (1-10s)
            - `low`: Image generation (>10s)
        timeout_seconds:
          type: integer
          minimum: 1
          maximum: 600
          default: 60

    InputData:
      type: object
      description: Task-specific input data structure
      properties:
        text:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          example: "Sample text input"
        image:
          type: string
          description: Base64-encoded image or URL
          example: "data:image/png;base64,iVBORw0KG..."
      additionalProperties: true

    InferenceResponse:
      type: object
      properties:
        model_id:
          type: string
        output_data:
          $ref: '#/components/schemas/OutputData'
        metadata:
          type: object
          properties:
            inference_time_ms:
              type: number
              description: Model inference duration
            total_time_ms:
              type: number
              description: Total request duration
            batch_size:
              type: integer
            device:
              type: string
              enum: [cpu, cuda]

    OutputData:
      type: object
      description: Task-specific output data structure
      properties:
        embedding:
          type: array
          items:
            type: number
          example: [0.123, -0.456, 0.789]
        embeddings:
          type: array
          description: For batch embedding requests
          items:
            type: array
            items:
              type: number
        text:
          type: string
          example: "Generated text output"
        image_url:
          type: string
          format: uri
          description: Presigned URL for generated image
        image_base64:
          type: string
          description: Base64-encoded image (if inline)
      additionalProperties: true
```

#### 2.4.2 Asynchronous Inference

```yaml
paths:
  /infer/{model_id}/async:
    post:
      tags:
        - Inference
      summary: Execute asynchronous inference
      description: |
        Submits inference request and returns immediately with job ID.
        Use `/jobs/{job_id}` endpoint to check status and retrieve result.
      operationId: inferAsync
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferenceRequest'
      responses:
        '202':
          description: Request accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobReference'
        '400':
          description: Invalid input
        '404':
          description: Model not found

components:
  schemas:
    JobReference:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
          example: "123e4567-e89b-12d3-a456-426614174000"
        status:
          $ref: '#/components/schemas/JobStatus'
        created_at:
          type: string
          format: date-time
        status_url:
          type: string
          format: uri
          example: "/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000"

    JobStatus:
      type: string
      enum:
        - queued
        - processing
        - completed
        - failed
        - cancelled
```

### 2.5 Job Management API

#### 2.5.1 Get Job Status

```yaml
paths:
  /jobs/{job_id}:
    get:
      tags:
        - Jobs
      summary: Get job status
      operationId: getJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatusResponse'
        '404':
          description: Job not found

components:
  schemas:
    JobStatusResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        model_id:
          type: string
        status:
          $ref: '#/components/schemas/JobStatus'
        progress:
          type: number
          minimum: 0
          maximum: 100
          description: Progress percentage (if available)
        created_at:
          type: string
          format: date-time
        started_at:
          type: string
          format: date-time
        completed_at:
          type: string
          format: date-time
        error:
          $ref: '#/components/schemas/Error'
```

#### 2.5.2 Get Job Result

```yaml
paths:
  /jobs/{job_id}/result:
    get:
      tags:
        - Jobs
      summary: Retrieve job result
      operationId: getJobResult
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Job result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InferenceResponse'
        '404':
          description: Job not found
        '409':
          description: Job not completed yet
          content:
            application/json:
              schema:
                type: object
                properties:
                  error_code:
                    type: string
                    example: "JOB_NOT_READY"
                  message:
                    type: string
                  status:
                    $ref: '#/components/schemas/JobStatus'
```

#### 2.5.3 Cancel Job

```yaml
paths:
  /jobs/{job_id}:
    delete:
      tags:
        - Jobs
      summary: Cancel job
      description: Attempts to cancel a queued or processing job
      operationId: cancelJob
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Job cancelled
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                  status:
                    type: string
                    example: "cancelled"
        '404':
          description: Job not found
        '409':
          description: Job already completed or failed
```

### 2.6 Lifecycle Management API

#### 2.6.1 Load Model

```yaml
paths:
  /models/{model_id}/load:
    post:
      tags:
        - Lifecycle
      summary: Force load model
      description: Explicitly loads model into memory (warmup)
      operationId: loadModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Model loaded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelStatusResponse'
        '404':
          description: Model not found
        '507':
          description: Insufficient memory

components:
  schemas:
    ModelStatusResponse:
      type: object
      properties:
        model_id:
          type: string
        state:
          type: string
          enum: [unloaded, loading, loaded, unhealthy]
        memory_usage_mb:
          type: number
        device:
          type: string
          enum: [cpu, cuda]
        loaded_at:
          type: string
          format: date-time
        last_used_at:
          type: string
          format: date-time
        health_status:
          type: object
          properties:
            healthy:
              type: boolean
            last_check_at:
              type: string
              format: date-time
            error_message:
              type: string
```

#### 2.6.2 Unload Model

```yaml
paths:
  /models/{model_id}/unload:
    post:
      tags:
        - Lifecycle
      summary: Unload model from memory
      operationId: unloadModel
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Model unloaded
        '404':
          description: Model not found or not loaded
```

#### 2.6.3 System Status

```yaml
paths:
  /status:
    get:
      tags:
        - Lifecycle
      summary: Get system status
      description: Returns overall system status and loaded models
      operationId: getStatus
      responses:
        '200':
          description: System status
          content:
            application/json:
              schema:
                type: object
                properties:
                  version:
                    type: string
                    example: "1.0.0"
                  uptime_seconds:
                    type: integer
                  memory:
                    type: object
                    properties:
                      total_mb:
                        type: number
                      used_mb:
                        type: number
                      available_mb:
                        type: number
                      threshold_mb:
                        type: number
                  gpu:
                    type: object
                    properties:
                      available:
                        type: boolean
                      devices:
                        type: array
                        items:
                          type: object
                          properties:
                            device_id:
                              type: integer
                            name:
                              type: string
                            memory_total_mb:
                              type: number
                            memory_used_mb:
                              type: number
                  loaded_models:
                    type: array
                    items:
                      $ref: '#/components/schemas/ModelStatusResponse'
                  queue_depth:
                    type: object
                    properties:
                      high_priority:
                        type: integer
                      medium_priority:
                        type: integer
                      low_priority:
                        type: integer
```

### 2.7 Health Check API

```yaml
paths:
  /health/live:
    get:
      tags:
        - Health
      summary: Liveness probe
      description: Kubernetes liveness probe endpoint
      operationId: liveness
      responses:
        '200':
          description: Service is alive
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "ok"

  /health/ready:
    get:
      tags:
        - Health
      summary: Readiness probe
      description: Kubernetes readiness probe endpoint
      operationId: readiness
      responses:
        '200':
          description: Service is ready
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "ready"
                  checks:
                    type: object
                    properties:
                      database:
                        type: boolean
                      queue:
                        type: boolean
        '503':
          description: Service not ready

  /health/models:
    get:
      tags:
        - Health
      summary: Model health status
      description: Health status of all loaded models
      operationId: modelHealth
      responses:
        '200':
          description: Model health status
          content:
            application/json:
              schema:
                type: object
                properties:
                  healthy_count:
                    type: integer
                  unhealthy_count:
                    type: integer
                  models:
                    type: array
                    items:
                      type: object
                      properties:
                        model_id:
                          type: string
                        healthy:
                          type: boolean
                        last_check_at:
                          type: string
                          format: date-time
                        error:
                          type: string
```

### 2.8 Metrics API

```yaml
paths:
  /metrics:
    get:
      tags:
        - Metrics
      summary: Prometheus metrics
      description: Exposes metrics in Prometheus format
      operationId: metrics
      responses:
        '200':
          description: Prometheus metrics
          content:
            text/plain:
              schema:
                type: string
              example: |
                # HELP modelmora_requests_total Total inference requests
                # TYPE modelmora_requests_total counter
                modelmora_requests_total{model_id="all-MiniLM-L6-v2",status="success"} 1234

                # HELP modelmora_inference_duration_seconds Inference duration
                # TYPE modelmora_inference_duration_seconds histogram
                modelmora_inference_duration_seconds_bucket{model_id="all-MiniLM-L6-v2",le="0.1"} 500
                modelmora_inference_duration_seconds_bucket{model_id="all-MiniLM-L6-v2",le="0.5"} 800
                modelmora_inference_duration_seconds_sum{model_id="all-MiniLM-L6-v2"} 245.3

                # HELP modelmora_queue_depth Current queue depth
                # TYPE modelmora_queue_depth gauge
                modelmora_queue_depth{priority="high"} 2
                modelmora_queue_depth{priority="medium"} 5
                modelmora_queue_depth{priority="low"} 12
```

---

## 3. gRPC Service Specification (Protocol Buffers)

### 3.1 Package Structure

```bash
protos/
├── modelmora/
│   └── v1/
│       ├── common.proto
│       ├── inference.proto
│       ├── registry.proto
│       └── management.proto
```

### 3.2 Common Types (`common.proto`)

```protobuf
syntax = "proto3";

package modelmora.v1;

option go_package = "github.com/miraveja/modelmora/gen/go/v1";

// Task types supported by ModelMora
enum TaskType {
  TASK_TYPE_UNSPECIFIED = 0;
  TASK_TYPE_TXT2EMBED = 1;
  TASK_TYPE_IMG2EMBED = 2;
  TASK_TYPE_TXT2IMG = 3;
  TASK_TYPE_IMG2TXT = 4;
  TASK_TYPE_TXT2TXT = 5;
}

// Priority levels for inference requests
enum Priority {
  PRIORITY_UNSPECIFIED = 0;
  PRIORITY_LOW = 1;
  PRIORITY_MEDIUM = 2;
  PRIORITY_HIGH = 3;
}

// Job status for async operations
enum JobStatus {
  JOB_STATUS_UNSPECIFIED = 0;
  JOB_STATUS_QUEUED = 1;
  JOB_STATUS_PROCESSING = 2;
  JOB_STATUS_COMPLETED = 3;
  JOB_STATUS_FAILED = 4;
  JOB_STATUS_CANCELLED = 5;
}

// Model state in lifecycle
enum ModelState {
  MODEL_STATE_UNSPECIFIED = 0;
  MODEL_STATE_UNLOADED = 1;
  MODEL_STATE_LOADING = 2;
  MODEL_STATE_LOADED = 3;
  MODEL_STATE_UNHEALTHY = 4;
}

// Resource requirements for a model
message ResourceRequirements {
  int32 memory_mb = 1;
  int32 gpu_vram_mb = 2;
  int32 cpu_threads = 3;
}

// Error details with code and message
message Error {
  string code = 1;
  string message = 2;
  map<string, string> details = 3;
  string trace_id = 4;
}

// Pagination metadata
message PaginationRequest {
  int32 page = 1;
  int32 page_size = 2;
}

message PaginationResponse {
  int32 total = 1;
  int32 page = 2;
  int32 page_size = 3;
  int32 total_pages = 4;
}
```

### 3.3 Inference Service (`inference.proto`)

```protobuf
syntax = "proto3";

package modelmora.v1;

import "modelmora/v1/common.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

option go_package = "github.com/miraveja/modelmora/gen/go/v1";

// InferenceService handles model inference operations
service InferenceService {
  // Execute synchronous inference
  rpc Infer(InferenceRequest) returns (InferenceResponse);

  // Execute asynchronous inference (returns job ID)
  rpc InferAsync(InferenceRequest) returns (JobReference);

  // Stream inference (for real-time use cases)
  rpc InferStream(stream InferenceRequest) returns (stream InferenceResponse);

  // Get job status
  rpc GetJobStatus(GetJobStatusRequest) returns (JobStatusResponse);

  // Get job result
  rpc GetJobResult(GetJobResultRequest) returns (InferenceResponse);

  // Cancel job
  rpc CancelJob(CancelJobRequest) returns (CancelJobResponse);
}

// Inference request message
message InferenceRequest {
  // Model identifier
  string model_id = 1;

  // Input data (task-specific)
  InputData input_data = 2;

  // Inference parameters (task-specific)
  google.protobuf.Struct parameters = 3;

  // Request priority
  Priority priority = 4;

  // Timeout in seconds
  int32 timeout_seconds = 5;

  // Client request ID (for tracing)
  string request_id = 6;
}

// Input data structure
message InputData {
  oneof data {
    string text = 1;
    TextBatch text_batch = 2;
    bytes image = 3;
    ImageBatch image_batch = 4;
  }

  // Additional custom fields
  google.protobuf.Struct custom_data = 10;
}

message TextBatch {
  repeated string texts = 1;
}

message ImageBatch {
  repeated bytes images = 1;
}

// Inference response message
message InferenceResponse {
  // Model identifier
  string model_id = 1;

  // Output data (task-specific)
  OutputData output_data = 2;

  // Inference metadata
  InferenceMetadata metadata = 3;

  // Error (if failed)
  Error error = 4;
}

// Output data structure
message OutputData {
  oneof data {
    Embedding embedding = 1;
    EmbeddingBatch embedding_batch = 2;
    string text = 3;
    string image_url = 4;
    bytes image_data = 5;
  }

  // Additional custom fields
  google.protobuf.Struct custom_data = 10;
}

message Embedding {
  repeated float values = 1;
}

message EmbeddingBatch {
  repeated Embedding embeddings = 1;
}

// Inference execution metadata
message InferenceMetadata {
  double inference_time_ms = 1;
  double total_time_ms = 2;
  int32 batch_size = 3;
  string device = 4;
  google.protobuf.Timestamp timestamp = 5;
}

// Job reference for async operations
message JobReference {
  string job_id = 1;
  JobStatus status = 2;
  google.protobuf.Timestamp created_at = 3;
}

// Get job status request
message GetJobStatusRequest {
  string job_id = 1;
}

// Job status response
message JobStatusResponse {
  string job_id = 1;
  string model_id = 2;
  JobStatus status = 3;
  float progress = 4;
  google.protobuf.Timestamp created_at = 5;
  google.protobuf.Timestamp started_at = 6;
  google.protobuf.Timestamp completed_at = 7;
  Error error = 8;
}

// Get job result request
message GetJobResultRequest {
  string job_id = 1;
}

// Cancel job request
message CancelJobRequest {
  string job_id = 1;
}

// Cancel job response
message CancelJobResponse {
  string job_id = 1;
  JobStatus status = 2;
}
```

### 3.4 Registry Service (`registry.proto`)

```protobuf
syntax = "proto3";

package modelmora.v1;

import "modelmora/v1/common.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

option go_package = "github.com/miraveja/modelmora/gen/go/v1";

// RegistryService manages model catalog
service RegistryService {
  // Register a new model
  rpc RegisterModel(RegisterModelRequest) returns (Model);

  // List all models
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);

  // Get model details
  rpc GetModel(GetModelRequest) returns (Model);

  // Update model configuration
  rpc UpdateModel(UpdateModelRequest) returns (Model);

  // Delete model
  rpc DeleteModel(DeleteModelRequest) returns (DeleteModelResponse);

  // Download model files
  rpc DownloadModel(DownloadModelRequest) returns (DownloadModelResponse);

  // Generate lock file
  rpc GenerateLock(GenerateLockRequest) returns (LockFile);
}

// Register model request
message RegisterModelRequest {
  string model_id = 1;
  string version = 2;
  TaskType task_type = 3;
  string display_name = 4;
  string description = 5;
  string source_url = 6;
  ResourceRequirements resource_requirements = 7;
  google.protobuf.Struct config = 8;
}

// Model message
message Model {
  string model_id = 1;
  string version = 2;
  TaskType task_type = 3;
  string display_name = 4;
  string description = 5;
  string status = 6;
  ResourceRequirements resource_requirements = 7;
  google.protobuf.Struct config = 8;
  google.protobuf.Timestamp created_at = 9;
  google.protobuf.Timestamp updated_at = 10;
}

// List models request
message ListModelsRequest {
  TaskType task_type = 1;
  string status = 2;
  PaginationRequest pagination = 3;
}

// List models response
message ListModelsResponse {
  repeated Model models = 1;
  PaginationResponse pagination = 2;
}

// Get model request
message GetModelRequest {
  string model_id = 1;
}

// Update model request
message UpdateModelRequest {
  string model_id = 1;
  string display_name = 2;
  string description = 3;
  google.protobuf.Struct config = 4;
  ResourceRequirements resource_requirements = 5;
}

// Delete model request
message DeleteModelRequest {
  string model_id = 1;
  bool force = 2;
}

// Delete model response
message DeleteModelResponse {
  string model_id = 1;
  bool deleted = 2;
}

// Download model request
message DownloadModelRequest {
  string model_id = 1;
}

// Download model response
message DownloadModelResponse {
  string model_id = 1;
  string status = 2;
  string message = 3;
}

// Generate lock request
message GenerateLockRequest {}

// Lock file message
message LockFile {
  string version = 1;
  google.protobuf.Timestamp generated_at = 2;
  repeated LockedModel models = 3;
}

message LockedModel {
  string model_id = 1;
  string version = 2;
  string source_url = 3;
  string checksum = 4;
  TaskType task_type = 5;
}
```

### 3.5 Management Service (`management.proto`)

```protobuf
syntax = "proto3";

package modelmora.v1;

import "modelmora/v1/common.proto";
import "google/protobuf/timestamp.proto";

option go_package = "github.com/miraveja/modelmora/gen/go/v1";

// ManagementService handles lifecycle and system operations
service ManagementService {
  // Load model into memory
  rpc LoadModel(LoadModelRequest) returns (ModelStatus);

  // Unload model from memory
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);

  // Get system status
  rpc GetSystemStatus(GetSystemStatusRequest) returns (SystemStatus);

  // Get model health status
  rpc GetModelHealth(GetModelHealthRequest) returns (ModelHealthResponse);

  // Health check endpoints
  rpc LivenessCheck(LivenessRequest) returns (LivenessResponse);
  rpc ReadinessCheck(ReadinessRequest) returns (ReadinessResponse);
}

// Load model request
message LoadModelRequest {
  string model_id = 1;
}

// Model status message
message ModelStatus {
  string model_id = 1;
  ModelState state = 2;
  double memory_usage_mb = 3;
  string device = 4;
  google.protobuf.Timestamp loaded_at = 5;
  google.protobuf.Timestamp last_used_at = 6;
  HealthStatus health_status = 7;
}

message HealthStatus {
  bool healthy = 1;
  google.protobuf.Timestamp last_check_at = 2;
  string error_message = 3;
}

// Unload model request
message UnloadModelRequest {
  string model_id = 1;
}

// Unload model response
message UnloadModelResponse {
  string model_id = 1;
  bool unloaded = 2;
}

// Get system status request
message GetSystemStatusRequest {}

// System status response
message SystemStatus {
  string version = 1;
  int64 uptime_seconds = 2;
  MemoryStatus memory = 3;
  GPUStatus gpu = 4;
  repeated ModelStatus loaded_models = 5;
  QueueStatus queue = 6;
}

message MemoryStatus {
  double total_mb = 1;
  double used_mb = 2;
  double available_mb = 3;
  double threshold_mb = 4;
}

message GPUStatus {
  bool available = 1;
  repeated GPUDevice devices = 2;
}

message GPUDevice {
  int32 device_id = 1;
  string name = 2;
  double memory_total_mb = 3;
  double memory_used_mb = 4;
}

message QueueStatus {
  int32 high_priority = 1;
  int32 medium_priority = 2;
  int32 low_priority = 3;
}

// Get model health request
message GetModelHealthRequest {}

// Model health response
message ModelHealthResponse {
  int32 healthy_count = 1;
  int32 unhealthy_count = 2;
  repeated ModelStatus models = 3;
}

// Liveness check
message LivenessRequest {}

message LivenessResponse {
  string status = 1;
}

// Readiness check
message ReadinessRequest {}

message ReadinessResponse {
  string status = 1;
  HealthChecks checks = 2;
}

message HealthChecks {
  bool database = 1;
  bool queue = 2;
}
```

---

## 4. Kafka Message Schemas (Future)

### 4.1 Topic Structure

```text
Topics:
- modelmora.inference.requests    # Inference request submissions
- modelmora.inference.responses   # Inference results
- modelmora.lifecycle.events      # Model load/unload events
- modelmora.audit.events          # Audit trail
```

### 4.2 Message Format (JSON Schema)

#### 4.2.1 Inference Request Message

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InferenceRequestMessage",
  "type": "object",
  "required": ["message_id", "model_id", "input_data", "timestamp"],
  "properties": {
    "message_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique message identifier"
    },
    "correlation_id": {
      "type": "string",
      "format": "uuid",
      "description": "Client correlation ID for response matching"
    },
    "model_id": {
      "type": "string"
    },
    "input_data": {
      "type": "object"
    },
    "parameters": {
      "type": "object"
    },
    "priority": {
      "type": "string",
      "enum": ["high", "medium", "low"]
    },
    "timeout_seconds": {
      "type": "integer"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

#### 4.2.2 Inference Response Message

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InferenceResponseMessage",
  "type": "object",
  "required": ["message_id", "correlation_id", "status", "timestamp"],
  "properties": {
    "message_id": {
      "type": "string",
      "format": "uuid"
    },
    "correlation_id": {
      "type": "string",
      "format": "uuid",
      "description": "Matches request correlation_id"
    },
    "model_id": {
      "type": "string"
    },
    "status": {
      "type": "string",
      "enum": ["success", "failure"]
    },
    "output_data": {
      "type": "object"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "inference_time_ms": {"type": "number"},
        "device": {"type": "string"}
      }
    },
    "error": {
      "type": "object",
      "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"}
      }
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

#### 4.2.3 Lifecycle Event Message

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LifecycleEventMessage",
  "type": "object",
  "required": ["event_id", "event_type", "model_id", "timestamp"],
  "properties": {
    "event_id": {
      "type": "string",
      "format": "uuid"
    },
    "event_type": {
      "type": "string",
      "enum": [
        "model_load_requested",
        "model_loaded",
        "model_load_failed",
        "model_unloaded",
        "model_evicted"
      ]
    },
    "model_id": {
      "type": "string"
    },
    "details": {
      "type": "object"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

---

## 5. Error Handling

### 5.1 HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful GET/PUT/DELETE |
| 201 | Created | Successful POST (resource created) |
| 202 | Accepted | Async operation started |
| 204 | No Content | Successful DELETE (no body) |
| 400 | Bad Request | Invalid input/validation error |
| 401 | Unauthorized | Missing/invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 408 | Request Timeout | Request exceeded timeout |
| 409 | Conflict | Resource conflict (e.g., already exists) |
| 422 | Unprocessable Entity | Semantic validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Service temporarily unavailable |
| 507 | Insufficient Storage | Not enough memory/disk |

### 5.2 gRPC Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| OK | Success | - |
| CANCELLED | Request cancelled | Client cancelled request |
| INVALID_ARGUMENT | Invalid input | Validation error |
| DEADLINE_EXCEEDED | Timeout | Request timeout |
| NOT_FOUND | Not found | Resource doesn't exist |
| ALREADY_EXISTS | Already exists | Duplicate resource |
| RESOURCE_EXHAUSTED | Resource exhausted | Memory/quota limits |
| FAILED_PRECONDITION | Precondition failed | Invalid state |
| UNAVAILABLE | Service unavailable | Temporary failure |
| INTERNAL | Internal error | Unexpected error |

### 5.3 Error Codes

```bash
# Registry Errors
MODEL_NOT_FOUND
MODEL_ALREADY_EXISTS
INVALID_MODEL_ID
INVALID_TASK_TYPE
DOWNLOAD_FAILED
CHECKSUM_MISMATCH

# Lifecycle Errors
MODEL_LOAD_FAILED
MODEL_NOT_LOADED
INSUFFICIENT_MEMORY
UNHEALTHY_MODEL
PROCESS_SPAWN_FAILED

# Inference Errors
INVALID_INPUT
INFERENCE_FAILED
INFERENCE_TIMEOUT
JOB_NOT_FOUND
JOB_NOT_READY
BATCH_SIZE_EXCEEDED

# System Errors
VALIDATION_ERROR
INTERNAL_ERROR
SERVICE_UNAVAILABLE
RATE_LIMIT_EXCEEDED
```

---

## 6. Client SDK Examples

### 6.1 Python REST Client

```python
from modelmora_client import ModelMoraClient

# Initialize client
client = ModelMoraClient(base_url="http://localhost:8000")

# Register model
model = client.registry.register_model(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    task_type="txt2embed",
    version="v2.2.2"
)

# Synchronous inference
result = client.inference.infer(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    input_data={"text": "Hello world"},
    parameters={"normalize": True}
)
print(result.output_data.embedding)

# Asynchronous inference
job = client.inference.infer_async(
    model_id="stabilityai/stable-diffusion-2-1",
    input_data={"text": "A beautiful sunset"},
    priority="low"
)
print(f"Job ID: {job.job_id}")

# Poll for result
status = client.jobs.get_status(job.job_id)
while status.status in ["queued", "processing"]:
    time.sleep(1)
    status = client.jobs.get_status(job.job_id)

result = client.jobs.get_result(job.job_id)
print(f"Image URL: {result.output_data.image_url}")
```

### 6.2 Python gRPC Client

```python
import grpc
from modelmora.v1 import inference_pb2, inference_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = inference_pb2_grpc.InferenceServiceStub(channel)

# Create request
request = inference_pb2.InferenceRequest(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    input_data=inference_pb2.InputData(text="Hello world"),
    priority=inference_pb2.PRIORITY_HIGH
)

# Execute inference
response = stub.Infer(request)
print(response.output_data.embedding.values)
```

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | - | Initial API design document |
