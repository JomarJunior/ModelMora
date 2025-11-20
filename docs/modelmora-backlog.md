# ModelMora Implementation Backlog

> **Project**: ModelMora Microservice Development \
> **Based on**: ModelMora Architecture Plan v1.0 \
> **Last Updated**: November 20, 2025

---

## Story Point Scale

- **1 point**: 1-2 hours (simple task, well-defined)
- **2 points**: 2-4 hours (straightforward implementation)
- **3 points**: 4-8 hours (moderate complexity)
- **5 points**: 1-2 days (complex, multiple components)
- **8 points**: 2-3 days (very complex, significant research)
- **13 points**: 3-5 days (epic-level, break down if possible)

## Priority Levels

- **P0**: Critical path, blocks other work
- **P1**: High priority, needed for phase completion
- **P2**: Medium priority, important but can be deferred
- **P3**: Low priority, nice-to-have

## Status Legend

✅ - Completed \
🔄 - Ongoing (optional)

---

## Phase 1: Foundation (Week 1-2)

**Goal**: Establish basic service structure, configuration, and single model provider

### 1.1 Project Scaffolding

#### ✅ Story 1.1.1: Initialize ModelMora Project Structure

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create the complete directory structure for ModelMora service including all folders (Configuration, Domain, Application, Infrastructure, API) following the architecture plan. Set up `__init__.py` files in all packages to ensure proper Python module structure.
- **Acceptance Criteria**:
  - All directories match architecture plan structure
  - All packages have `__init__.py` files
  - Directory structure validated with `tree` command
- **Dependencies**: None

#### ✅ Story 1.1.2: Configure Poetry Dependencies

- **Priority**: P0
- **Story Points**: 3
- **Description**: Update `pyproject.toml` with all required dependencies including FastAPI, gRPC, torch, open-clip-torch, and miraveja-core. Configure torch CPU source and set up development dependencies. Ensure Poetry can resolve all dependencies without conflicts.
- **Acceptance Criteria**:
  - All production dependencies listed with correct versions
  - Development dependencies configured (pytest, pytest-grpc, etc.)
  - `poetry install` completes successfully
  - No dependency conflicts
- **Dependencies**: Story 1.1.1

#### 🔄 Story 1.1.3: Create Pydantic Configuration Models

- **Priority**: P0
- **Story Points**: 3
- **Description**: Implement `ModelMoraConfig` class in `Configuration/ModelMoraConfig.py` using Pydantic BaseSettings. Include all configuration parameters: service identity, server ports, performance settings, model management, GPU configuration, Kafka integration, and monitoring settings with environment variable support.
- **Acceptance Criteria**:
  - All config fields defined with proper types and defaults
  - Environment variable mapping works correctly
  - Config loads from `.env` file
  - Validation works for invalid values
- **Dependencies**: Story 1.1.2

#### Story 1.1.4: Set Up Dependency Injection Container

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create `Dependencies.py` with DI container registration following miraveja-core patterns. Register configuration, logging, and placeholder services for repositories and infrastructure components to be implemented later.
- **Acceptance Criteria**:
  - DI container initialized properly
  - Configuration registered as singleton
  - Logger service registered
  - Container can resolve registered dependencies
- **Dependencies**: Story 1.1.3

### 1.2 gRPC Infrastructure

#### Story 1.2.1: Define Protocol Buffer Schema

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create `modelmora.proto` file in `Infrastructure/GRPC/protos/` with service definitions for `ModelInference` including `GenerateEmbedding`, `GenerateBatch`, and `GenerateEmbeddingStream` methods. Define message types for `EmbeddingRequest`, `EmbeddingResponse`, `BatchRequest`, and `InferenceStatus`.
- **Acceptance Criteria**:
  - Proto file compiles without errors
  - All message fields properly typed
  - Service methods defined with correct request/response types
  - Documentation comments added
- **Dependencies**: Story 1.1.1

#### Story 1.2.2: Generate Python gRPC Stubs

- **Priority**: P0
- **Story Points**: 1
- **Description**: Use `grpcio-tools` to generate Python code from proto file. Create script or Poetry task to automate stub generation. Ensure generated files are properly imported in the package.
- **Acceptance Criteria**:
  - `modelmora_pb2.py` and `modelmora_pb2_grpc.py` generated
  - Generation script works reliably
  - Generated code can be imported
  - `.gitignore` updated to exclude generated files (optional)
- **Dependencies**: Story 1.2.1

#### Story 1.2.3: Implement Basic gRPC Server

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create `Infrastructure/GRPC/Server.py` with async gRPC server setup. Implement server initialization, port binding, graceful shutdown, and error handling. Use configuration for port and worker settings.
- **Acceptance Criteria**:
  - Server starts and binds to configured port
  - Server accepts gRPC connections
  - Graceful shutdown on SIGTERM/SIGINT
  - Error logging for startup failures
- **Dependencies**: Story 1.2.2, Story 1.1.3

#### Story 1.2.4: Create gRPC Service Stub Implementation

- **Priority**: P0
- **Story Points**: 2
- **Description**: Implement `Infrastructure/GRPC/Services.py` with `ModelInferenceServicer` class. Create placeholder implementations for all RPC methods that return mock responses. This allows testing the gRPC layer before model logic is complete.
- **Acceptance Criteria**:
  - All RPC methods implemented with TODO markers
  - Methods return valid proto message types
  - Basic logging added for method calls
  - Can be invoked via gRPC client
- **Dependencies**: Story 1.2.3

### 1.3 REST API Infrastructure

#### Story 1.3.1: Create FastAPI Application

- **Priority**: P0
- **Story Points**: 2
- **Description**: Set up FastAPI application in `main.py` with CORS, exception handlers, and middleware. Configure OpenAPI documentation with service metadata. Implement application lifecycle events (startup/shutdown).
- **Acceptance Criteria**:
  - FastAPI app initializes correctly
  - CORS configured for development
  - OpenAPI docs accessible at `/docs`
  - Global exception handler catches errors
- **Dependencies**: Story 1.1.4

#### Story 1.3.2: Implement Health Check Endpoint

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create `API/Health.py` with comprehensive health check endpoint showing service status, uptime, loaded models (placeholder), memory usage, and dependency status. Include readiness and liveness probe variants.
- **Acceptance Criteria**:
  - `/health` endpoint returns 200 with status JSON
  - Response includes all required fields
  - Uptime calculation works correctly
  - Can be used as Docker HEALTHCHECK
- **Dependencies**: Story 1.3.1

#### Story 1.3.3: Implement Model Management Endpoints (Stub)

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create `API/Models.py` with REST endpoints for listing models, getting model info, loading/unloading models, and model warmup. Implement as stubs returning mock data until ModelManager is complete.
- **Acceptance Criteria**:
  - All endpoints defined with proper request/response models
  - OpenAPI schema generated correctly
  - Endpoints return mock data consistently
  - HTTP status codes correct
- **Dependencies**: Story 1.3.1

#### Story 1.3.4: Integrate gRPC and REST Servers in Main

- **Priority**: P0
- **Story Points**: 3
- **Description**: Update `main.py` to run both FastAPI (port 8080) and gRPC (port 50051) servers concurrently using asyncio. Implement proper startup sequence, concurrent execution with `asyncio.gather()`, and coordinated shutdown handling.
- **Acceptance Criteria**:
  - Both servers start successfully
  - Servers run concurrently without blocking
  - Graceful shutdown stops both servers
  - Startup errors handled properly
- **Dependencies**: Story 1.2.3, Story 1.3.1

### 1.4 Model Registry and Configuration

#### Story 1.4.1: Create Model Registry YAML Schema

- **Priority**: P0
- **Story Points**: 2
- **Description**: Design and document the ModelRegistry.yaml schema with fields for model name, type, provider, config, resources, warmup settings, and default flag. Create example configuration with CLIP model.
- **Acceptance Criteria**:
  - YAML schema well-documented
  - Example with CLIP model validates
  - All required fields defined
  - Schema supports extensibility
- **Dependencies**: None

#### Story 1.4.2: Implement Model Configuration Parser

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create Pydantic models in `Configuration/ModelRegistry.py` for parsing YAML configuration: `ModelConfig`, `ProviderConfig`, `ResourceConfig`, `WarmupConfig`. Implement YAML loader with validation.
- **Acceptance Criteria**:
  - YAML parses into Pydantic models
  - Validation catches invalid configurations
  - Type conversions work correctly
  - Helpful error messages on parse failure
- **Dependencies**: Story 1.4.1

#### Story 1.4.3: Implement Model Registry Domain Model

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create `Domain/Models/ModelRegistry.py` with registry logic for loading configuration, querying models by name, filtering by type, getting default models, and validating model existence.
- **Acceptance Criteria**:
  - Registry loads from YAML file
  - Query methods work correctly
  - Thread-safe access (if needed)
  - Errors handled gracefully
- **Dependencies**: Story 1.4.2

### 1.5 Domain Models

#### Story 1.5.1: Define IModelProvider Interface

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create `Domain/Models/IModelProvider.py` with abstract interface defining methods: `initialize()`, `generate_embedding()`, `generate_test_embedding()`, `cleanup()`, and properties for metadata, memory usage, and model state.
- **Acceptance Criteria**:
  - Interface uses Python ABC
  - All methods properly typed with async where needed
  - Docstrings explain each method's purpose
  - Example usage documented
- **Dependencies**: None

#### Story 1.5.2: Create Model Metadata Value Objects

- **Priority**: P1
- **Story Points**: 2
- **Description**: Implement `Domain/Models/ModelMetadata.py` with value objects for model metadata: name, version, dimension, memory usage, load time. Use frozen dataclasses or Pydantic for immutability.
- **Acceptance Criteria**:
  - Immutable value objects
  - Proper validation in constructors
  - Equality comparison works
  - JSON serialization supported
- **Dependencies**: None

### 1.6 First Model Provider (CLIP)

#### Story 1.6.1: Implement Base Model Provider

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create `Infrastructure/Providers/BaseProvider.py` with common functionality for all providers: configuration loading, error handling, logging, state management, and resource tracking.
- **Acceptance Criteria**:
  - Base class implements IModelProvider partially
  - Common error handling patterns
  - State transitions logged
  - Memory tracking methods
- **Dependencies**: Story 1.5.1

#### Story 1.6.2: Implement CLIP Model Provider

- **Priority**: P0
- **Story Points**: 5
- **Description**: Create `Infrastructure/Providers/ClipProvider.py` implementing full CLIP model loading with open-clip-torch. Handle model initialization, image preprocessing, embedding generation, error handling, and cleanup. Support CPU inference initially.
- **Acceptance Criteria**:
  - CLIP model loads successfully
  - Image preprocessing works (PIL/tensor conversion)
  - Embedding generation produces correct dimensions
  - Memory is freed on cleanup
  - Unit tests pass
- **Dependencies**: Story 1.6.1, Story 1.1.2

#### Story 1.6.3: Add CLIP Configuration to Registry

- **Priority**: P1
- **Story Points**: 1
- **Description**: Update `Configuration/ModelRegistry.yaml` with complete CLIP model configuration including pretrained weights, architecture, resource requirements, and warmup settings. Mark as default model.
- **Acceptance Criteria**:
  - Configuration validates correctly
  - All required fields populated
  - Resource limits appropriate for CLIP
  - Default flag set to true
- **Dependencies**: Story 1.4.1, Story 1.6.2

### 1.7 Docker Integration

#### Story 1.7.1: Optimize Dockerfile for Production

- **Priority**: P1
- **Story Points**: 3
- **Description**: Review and optimize existing Dockerfile: multi-stage build (if beneficial), layer caching optimization, security hardening (non-root user), health check refinement, and supervisor configuration for running both servers.
- **Acceptance Criteria**:
  - Image builds successfully
  - Build time optimized with layer caching
  - Image size reasonable (<2GB)
  - Security scan passes (no critical vulnerabilities)
  - Both servers start in container
- **Dependencies**: Story 1.3.4

#### Story 1.7.2: Create Docker Compose Configuration

- **Priority**: P1
- **Story Points**: 2
- **Description**: Add ModelMora service to docker-compose.yml with proper network configuration, environment variables, volume mounts for models and config, resource limits, and health checks. Configure dependencies on Kafka and MinIO.
- **Acceptance Criteria**:
  - Service starts with `docker-compose up`
  - Environment variables loaded correctly
  - Volumes mounted properly
  - Health check works
  - Can communicate with other services
- **Dependencies**: Story 1.7.1

#### Story 1.7.3: Create Supervisor Configuration

- **Priority**: P1
- **Story Points**: 2
- **Description**: Create `docker/supervisord.conf` for running FastAPI and gRPC servers as separate processes with automatic restart, log forwarding to stdout/stderr, and proper process management.
- **Acceptance Criteria**:
  - Both processes start under supervisor
  - Logs visible in docker logs
  - Auto-restart on crash works
  - Supervisor graceful shutdown works
- **Dependencies**: Story 1.7.1

### 1.8 Basic Testing

#### Story 1.8.1: Set Up Testing Infrastructure

- **Priority**: P1
- **Story Points**: 2
- **Description**: Configure pytest with async support, coverage reporting, and test fixtures. Create base test classes and utilities for mocking dependencies. Set up test database/config if needed.
- **Acceptance Criteria**:
  - `pytest` runs successfully
  - Coverage reporting configured
  - Async tests work with pytest-asyncio
  - Test fixtures for common dependencies
- **Dependencies**: Story 1.1.2

#### Story 1.8.2: Write Unit Tests for Configuration

- **Priority**: P1
- **Story Points**: 2
- **Description**: Create `tests/unit/test_configuration.py` with tests for ModelMoraConfig loading from environment variables, defaults, validation, and ModelRegistry YAML parsing.
- **Acceptance Criteria**:
  - All config fields tested
  - Environment variable override works
  - Invalid configs raise proper errors
  - Code coverage >80%
- **Dependencies**: Story 1.8.1, Story 1.1.3

#### Story 1.8.3: Write Unit Tests for CLIP Provider

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create `tests/unit/test_clip_provider.py` with tests for model initialization, embedding generation with mock images, error handling, cleanup, and resource tracking.
- **Acceptance Criteria**:
  - All provider methods tested
  - Mock images used (no real model loading in unit tests)
  - Error cases covered
  - Code coverage >80%
- **Dependencies**: Story 1.8.1, Story 1.6.2

#### Story 1.8.4: Write Integration Tests for gRPC Server

- **Priority**: P2
- **Story Points**: 3
- **Description**: Create `tests/integration/test_grpc_server.py` with tests using pytest-grpc to test actual gRPC calls, request/response serialization, and basic end-to-end flow with mock model.
- **Acceptance Criteria**:
  - gRPC server starts in test
  - Client can call RPC methods
  - Serialization works correctly
  - Error responses handled
- **Dependencies**: Story 1.8.1, Story 1.2.4

---

## Phase 2: Core Functionality (Week 3-4)

**Goal**: Implement model management, memory management, and monitoring

### 2.1 Model Manager Implementation

#### Story 2.1.1: Implement Basic ModelManager Class

- **Priority**: P0
- **Story Points**: 5
- **Description**: Create `Domain/Services/ModelManager.py` with core functionality: model loading, tracking loaded models, provider factory, and basic lifecycle management. Include async lock for thread safety.
- **Acceptance Criteria**:
  - ModelManager initializes with registry
  - Can load single model
  - Provider factory creates correct provider type
  - Thread-safe access with asyncio lock
  - Basic error handling
- **Dependencies**: Story 1.5.1, Story 1.4.3

#### Story 2.1.2: Implement LRU Cache for Models

- **Priority**: P0
- **Story Points**: 5
- **Description**: Add LRU cache logic to ModelManager using OrderedDict. Implement `get_model()` method that loads on demand and updates LRU ordering. Track model access patterns.
- **Acceptance Criteria**:
  - LRU ordering maintained correctly
  - Most recently used model at end
  - Access updates position
  - Cache queries don't trigger loads
- **Dependencies**: Story 2.1.1

#### Story 2.1.3: Implement Memory Management Logic

- **Priority**: P0
- **Story Points**: 8
- **Description**: Implement `_ensure_memory()` method that evicts least recently used models when memory threshold exceeded. Track current memory usage, implement eviction loop, add logging for evictions, and handle edge cases (cannot allocate even after full eviction).
- **Acceptance Criteria**:
  - Memory tracking accurate
  - LRU eviction works correctly
  - Evicts minimum number of models needed
  - OutOfMemoryError raised when appropriate
  - Detailed logging of evictions
- **Dependencies**: Story 2.1.2

#### Story 2.1.4: Implement Model Unloading

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create `unload_model()` method that calls provider cleanup, removes from cache, updates memory tracking, and publishes events. Support both manual unloading and automatic eviction.
- **Acceptance Criteria**:
  - Provider cleanup called properly
  - Memory updated correctly
  - Model removed from all tracking structures
  - Idempotent (safe to call multiple times)
  - Events published
- **Dependencies**: Story 2.1.3

#### Story 2.1.5: Implement Model Warmup Strategy

- **Priority**: P1
- **Story Points**: 5
- **Description**: Create `warm_up_models()` method that loads default models from registry, runs test inferences, measures warmup time, and publishes ModelLoadedEvents. Handle warmup failures gracefully.
- **Acceptance Criteria**:
  - Default models loaded on startup
  - Test inferences complete successfully
  - Timing metrics accurate
  - Failures don't crash service
  - Events published for successful warmups
- **Dependencies**: Story 2.1.1

### 2.2 Application Commands

#### Story 2.2.1: Implement GenerateEmbedding Command

- **Priority**: P0
- **Story Points**: 5
- **Description**: Create `Application/Commands/GenerateEmbedding.py` with command/handler pattern for embedding generation. Integrate ModelManager, handle image preprocessing, call provider, return results, and add comprehensive error handling.
- **Acceptance Criteria**:
  - Command accepts image data and model name
  - Handler coordinates ModelManager and provider
  - Returns embedding vector with metadata
  - All error cases handled
  - Logging and metrics instrumented
- **Dependencies**: Story 2.1.2

#### Story 2.2.2: Implement LoadModel Command

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create `Application/Commands/LoadModel.py` for explicit model loading via API/admin requests. Validate model exists in registry, trigger loading, return status, and handle already-loaded case.
- **Acceptance Criteria**:
  - Command loads model explicitly
  - Validation prevents invalid model names
  - Idempotent (safe to call if already loaded)
  - Returns load time metrics
  - Events published
- **Dependencies**: Story 2.1.1

#### Story 2.2.3: Implement UnloadModel Command

- **Priority**: P1
- **Story Points**: 2
- **Description**: Create `Application/Commands/UnloadModel.py` for explicit model unloading. Validate model is loaded, trigger unload, return status, and handle not-loaded case.
- **Acceptance Criteria**:
  - Command unloads model
  - Validation prevents errors
  - Idempotent
  - Returns freed memory amount
  - Events published
- **Dependencies**: Story 2.1.4

#### Story 2.2.4: Wire Commands to gRPC Service

- **Priority**: P0
- **Story Points**: 3
- **Description**: Update `Infrastructure/GRPC/Services.py` to inject and call command handlers. Replace stub implementations with real command execution. Add request ID tracking and error mapping to gRPC status codes.
- **Acceptance Criteria**:
  - gRPC methods invoke real commands
  - Request/response mapping correct
  - Errors mapped to proper gRPC status codes
  - Request IDs tracked for observability
- **Dependencies**: Story 2.2.1

#### Story 2.2.5: Wire Commands to REST Endpoints

- **Priority**: P1
- **Story Points**: 2
- **Description**: Update `API/Models.py` to inject and call command handlers. Replace stub implementations with real operations. Add proper HTTP status codes and error responses.
- **Acceptance Criteria**:
  - REST endpoints invoke real commands
  - HTTP status codes correct (200, 404, 500, etc.)
  - Error responses include helpful messages
  - OpenAPI schema updated
- **Dependencies**: Story 2.2.2, Story 2.2.3

### 2.3 Logging and Metrics

#### Story 2.3.1: Implement Structured Logging

- **Priority**: P1
- **Story Points**: 3
- **Description**: Integrate with miraveja-core logging infrastructure (structlog). Add structured logging throughout ModelManager, providers, and command handlers with consistent fields: request_id, model_name, timing, etc.
- **Acceptance Criteria**:
  - All components use structured logging
  - Log fields consistent across service
  - Request IDs tracked through call chain
  - Log levels appropriate (debug, info, warn, error)
  - JSON output for production
- **Dependencies**: Story 2.1.1

#### Story 2.3.2: Implement Prometheus Metrics

- **Priority**: P1
- **Story Points**: 5
- **Description**: Add Prometheus instrumentation: counters for requests/errors, histograms for inference time, gauges for loaded models/memory usage. Create `/metrics` endpoint. Instrument ModelManager and command handlers.
- **Acceptance Criteria**:
  - All key metrics instrumented
  - Metrics endpoint returns Prometheus format
  - Histogram buckets appropriate
  - Labels include model_name, status
  - No significant performance impact
- **Dependencies**: Story 2.1.1

#### Story 2.3.3: Enhanced Health Check with Metrics

- **Priority**: P2
- **Story Points**: 2
- **Description**: Update health check endpoint to include real data from ModelManager: loaded models list, actual memory usage, system resources (CPU, memory, disk), and dependency checks.
- **Acceptance Criteria**:
  - Health check shows real loaded models
  - Memory usage accurate
  - System metrics included
  - Dependency status checked
  - Response time <100ms
- **Dependencies**: Story 2.1.1, Story 1.3.2

### 2.4 Testing

#### Story 2.4.1: Unit Tests for ModelManager

- **Priority**: P1
- **Story Points**: 5
- **Description**: Comprehensive unit tests for ModelManager covering: model loading, LRU ordering, memory management, eviction logic, edge cases (OOM, invalid model), and concurrent access.
- **Acceptance Criteria**:
  - All public methods tested
  - LRU behavior verified
  - Eviction logic validated
  - Edge cases covered
  - Code coverage >85%
- **Dependencies**: Story 2.1.3

#### Story 2.4.2: Unit Tests for Commands

- **Priority**: P1
- **Story Points**: 3
- **Description**: Unit tests for all command handlers with mocked dependencies: GenerateEmbedding, LoadModel, UnloadModel. Test success paths and error handling.
- **Acceptance Criteria**:
  - All commands tested
  - Mocks used for dependencies
  - Error cases covered
  - Code coverage >80%
- **Dependencies**: Story 2.2.1, Story 2.2.2, Story 2.2.3

#### Story 2.4.3: Integration Tests for End-to-End Flow

- **Priority**: P2
- **Story Points**: 5
- **Description**: Integration tests using real CLIP model: load model, generate embedding from test image, verify vector dimensions, test unload, verify memory freed. May be slow tests marked with pytest markers.
- **Acceptance Criteria**:
  - Real model loading works
  - Embedding generation produces correct output
  - Memory management verified
  - Tests marked as slow/integration
  - Can be skipped in CI if needed
- **Dependencies**: Story 2.2.4

---

## Phase 3: Integration (Week 5-6)

**Goal**: Integrate ModelMora with existing services (API, Worker) via gRPC and Kafka

### 3.1 gRPC Client Library

#### Story 3.1.1: Create gRPC Client Module in miraveja-core

- **Priority**: P0
- **Story Points**: 3
- **Description**: Create reusable gRPC client in miraveja-core for other services to use. Include connection management, retry logic, timeout handling, and proper error mapping from gRPC status codes.
- **Acceptance Criteria**:
  - Client handles connection lifecycle
  - Configurable retry with exponential backoff
  - Timeouts configurable per-request
  - gRPC errors mapped to domain exceptions
  - Thread-safe
- **Dependencies**: Story 2.2.4

#### Story 3.1.2: Implement Synchronous Embedding Generation Client

- **Priority**: P0
- **Story Points**: 3
- **Description**: Add high-level `generate_embedding()` method to client that handles image serialization, request construction, gRPC call, response parsing, and returns embedding vector with metadata.
- **Acceptance Criteria**:
  - Simple API: `await client.generate_embedding(image_data, model_name)`
  - Image data accepts bytes or file path
  - Returns EmbeddingResponse object
  - Timeout support
  - Proper error handling
- **Dependencies**: Story 3.1.1

#### Story 3.1.3: Add Client to Dependency Injection

- **Priority**: P1
- **Story Points**: 2
- **Description**: Register ModelMoraGrpcClient in miraveja-core DI container with configuration for endpoint URL, default timeout, retry settings. Make available for injection in API and Worker services.
- **Acceptance Criteria**:
  - Client registered in DI container
  - Configuration loaded from environment
  - Singleton or scoped appropriately
  - Can be injected in services
- **Dependencies**: Story 3.1.2

### 3.2 API Service Integration

#### Story 3.2.1: Update UploadImage Handler with Synchronous Embedding

- **Priority**: P0
- **Story Points**: 5
- **Description**: Modify API service's UploadImageHandler to call ModelMora gRPC client synchronously after image upload. Implement try/catch for timeout with fallback to Kafka async processing. Store embedding immediately if successful.
- **Acceptance Criteria**:
  - gRPC client injected in handler
  - Synchronous call attempted with timeout
  - Success: embedding stored immediately
  - Timeout/failure: fallback to Kafka event
  - User experience improved (immediate search availability)
- **Dependencies**: Story 3.1.3

#### Story 3.2.2: Update Image Search to Use Embeddings

- **Priority**: P1
- **Story Points**: 3
- **Description**: Ensure API service's image search functionality uses embeddings generated by ModelMora (via Qdrant). Verify query embedding generation works through gRPC client.
- **Acceptance Criteria**:
  - Search generates query embedding via ModelMora
  - Qdrant search uses proper collection
  - Results ranked by similarity
  - Performance acceptable (<500ms)
- **Dependencies**: Story 3.2.1

#### Story 3.2.3: Add Retry Logic for Failed Embeddings

- **Priority**: P2
- **Story Points**: 3
- **Description**: Implement retry mechanism in API service for images that failed synchronous embedding generation. Could be background task, scheduled job, or subscriber to failed events.
- **Acceptance Criteria**:
  - Failed embeddings tracked
  - Retry attempts with backoff
  - Max retry limit enforced
  - Success updates metadata
  - Monitoring for stuck images
- **Dependencies**: Story 3.2.1

### 3.3 Kafka Event Integration

#### Story 3.3.1: Define Domain Events

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create domain event classes in ModelMora: `EmbeddingGeneratedEvent`, `ModelLoadedEvent`, `ModelUnloadedEvent`, `EmbeddingFailedEvent`. Register with event registry following miraveja-core patterns.
- **Acceptance Criteria**:
  - All events defined with proper schemas
  - Events registered with versions
  - JSON serialization works
  - Events inherit from DomainEvent base class
- **Dependencies**: None

#### Story 3.3.2: Implement Event Publisher

- **Priority**: P0
- **Story Points**: 3
- **Description**: Integrate Kafka producer in ModelMora. Publish events from ModelManager (model loaded/unloaded) and command handlers (embedding generated/failed). Configure Kafka connection from environment.
- **Acceptance Criteria**:
  - Kafka producer initialized on startup
  - Events published to correct topics
  - Serialization works correctly
  - Async publishing doesn't block operations
  - Errors logged
- **Dependencies**: Story 3.3.1

#### Story 3.3.3: Implement ImageMetadataRegistered Event Subscriber

- **Priority**: P0
- **Story Points**: 5
- **Description**: Create `Application/Subscribers/HandleImageRegistered.py` that listens for `ImageMetadataRegisteredEvent`, downloads image from MinIO, generates embedding, stores in Qdrant, updates database, and publishes completion event.
- **Acceptance Criteria**:
  - Subscriber registered with Kafka consumer
  - Processes events successfully
  - Downloads image from MinIO
  - Generates embedding via ModelManager
  - Stores in Qdrant
  - Updates database
  - Publishes completion event
  - Error handling and retries
- **Dependencies**: Story 3.3.2, Story 2.2.1

#### Story 3.3.4: Configure Kafka Consumer in Main

- **Priority**: P1
- **Story Points**: 3
- **Description**: Add Kafka consumer startup to `main.py` alongside gRPC and REST servers. Configure consumer group, topics to subscribe, and integrate with event subscriber registry. Handle consumer lifecycle (start/stop).
- **Acceptance Criteria**:
  - Consumer starts with service
  - Subscribes to configured topics
  - Routes events to subscribers
  - Graceful shutdown on service stop
  - Error handling and logging
- **Dependencies**: Story 3.3.3

### 3.4 Worker Service Migration

#### Story 3.4.1: Migrate Worker's Embedding Generation to ModelMora

- **Priority**: P1
- **Story Points**: 5
- **Description**: Update Worker service's `GenerateImageVector` subscriber to use ModelMora gRPC client instead of local model loading. Remove model loading code from worker. Simplify worker to orchestration only.
- **Acceptance Criteria**:
  - Worker calls ModelMora via gRPC
  - Local model loading code removed
  - Worker dependencies cleaned up (torch, open-clip removed if not used elsewhere)
  - Same functionality maintained
  - Performance acceptable
- **Dependencies**: Story 3.1.3

#### Story 3.4.2: Update Worker Configuration

- **Priority**: P1
- **Story Points**: 1
- **Description**: Update worker's configuration and environment variables to include ModelMora gRPC endpoint. Remove model-related configuration that's now in ModelMora.
- **Acceptance Criteria**:
  - Configuration updated
  - Environment variables set in docker-compose
  - Old config removed
  - Documentation updated
- **Dependencies**: Story 3.4.1

### 3.5 Integration Testing

#### Story 3.5.1: End-to-End Test: API Upload with Sync Embedding

- **Priority**: P1
- **Story Points**: 5
- **Description**: Integration test covering full flow: upload image via API → ModelMora generates embedding synchronously → embedding stored in Qdrant → search returns uploaded image. Use test containers or docker-compose for dependencies.
- **Acceptance Criteria**:
  - Full stack integration test
  - Uses real services (API, ModelMora, Qdrant)
  - Upload successful
  - Embedding generated and stored
  - Search finds image
  - Test isolated and repeatable
- **Dependencies**: Story 3.2.1

#### Story 3.5.2: End-to-End Test: Async Embedding via Kafka

- **Priority**: P1
- **Story Points**: 5
- **Description**: Integration test for async flow: publish `ImageMetadataRegisteredEvent` → ModelMora subscriber processes → embedding stored → database updated → `EmbeddingGeneratedEvent` published. Verify complete event chain.
- **Acceptance Criteria**:
  - Event published to Kafka
  - ModelMora processes event
  - Embedding stored correctly
  - Database updated
  - Completion event published
  - Test isolated
- **Dependencies**: Story 3.3.3

#### Story 3.5.3: Load Testing for ModelMora

- **Priority**: P2
- **Story Points**: 5
- **Description**: Performance testing with realistic load: concurrent embedding requests, measure throughput and latency, test LRU eviction under load, identify bottlenecks, validate resource limits.
- **Acceptance Criteria**:
  - Load test script created (locust or similar)
  - Tests various concurrency levels
  - Throughput measured (requests/sec)
  - Latency percentiles (p50, p95, p99)
  - Resource usage monitored
  - Results documented
- **Dependencies**: Story 3.2.1

---

## Phase 4: Enhancement (Week 7-8)

**Goal**: Add multiple model providers, batch processing, and streaming

### 4.1 Additional Model Providers

#### Story 4.1.1: Implement DINOv2 Provider

- **Priority**: P1
- **Story Points**: 5
- **Description**: Create `Infrastructure/Providers/DINOv2Provider.py` implementing DINOv2 model with transformers library. Follow same pattern as CLIP provider. Add configuration to ModelRegistry.yaml.
- **Acceptance Criteria**:
  - DINOv2 loads successfully
  - Embedding generation works
  - Dimensions correct (1024 for vitl)
  - Memory tracking accurate
  - Tests pass
  - Registry configuration added
- **Dependencies**: Story 1.6.1

#### Story 4.1.2: Implement SAM (Segment Anything) Provider

- **Priority**: P2
- **Story Points**: 8
- **Description**: Create `Infrastructure/Providers/SAMProvider.py` for segmentation model. More complex than embedding models - may need different interface methods for segmentation vs embedding. Consider if IModelProvider needs extension.
- **Acceptance Criteria**:
  - SAM loads successfully
  - Segmentation works with test images
  - Returns mask data properly
  - Handles large memory requirements (4GB+)
  - GPU preference supported
  - Tests pass
  - Registry configuration added
- **Dependencies**: Story 1.6.1

#### Story 4.1.3: Add Model Type Abstraction

- **Priority**: P2
- **Story Points**: 3
- **Description**: Refactor IModelProvider to support different model types (embedding, segmentation, classification). May need separate interfaces or capabilities pattern. Update ModelManager to handle different types.
- **Acceptance Criteria**:
  - Architecture supports multiple model types
  - Type checking enforced
  - Providers implement correct interface
  - ModelManager routes correctly
  - Backward compatible
- **Dependencies**: Story 4.1.2

### 4.2 Batch Processing

#### Story 4.2.1: Implement Batch Request Handling

- **Priority**: P1
- **Story Points**: 5
- **Description**: Implement `GenerateBatch` gRPC method in service. Accept multiple image requests, batch to model for efficient processing, return streaming responses. Handle partial failures.
- **Acceptance Criteria**:
  - Accepts multiple requests in single call
  - Batches efficiently (up to configured batch size)
  - Streams responses as completed
  - Partial failures handled gracefully
  - Performance improvement vs sequential
- **Dependencies**: Story 2.2.4

#### Story 4.2.2: Add Batch Support to Providers

- **Priority**: P1
- **Story Points**: 5
- **Description**: Update IModelProvider interface and implementations to support batch inference. Modify CLIP and DINOv2 providers to process multiple images in single forward pass for efficiency.
- **Acceptance Criteria**:
  - Interface includes batch methods
  - Providers implement batching
  - Batch size configurable
  - Actual performance improvement measured
  - Backward compatible with single inference
- **Dependencies**: Story 4.2.1

#### Story 4.2.3: Implement Request Queue with Batching

- **Priority**: P2
- **Story Points**: 8
- **Description**: Advanced feature: implement request queue that accumulates incoming requests and triggers batch processing when queue reaches threshold or timeout. Improves throughput for high-load scenarios.
- **Acceptance Criteria**:
  - Queue accumulates requests
  - Batches triggered by size or timeout
  - Requests mapped to correct responses
  - Lower latency under high load
  - Configuration options
  - Metrics for queue depth
- **Dependencies**: Story 4.2.2

### 4.3 Streaming Inference

#### Story 4.3.1: Implement Streaming gRPC Method

- **Priority**: P2
- **Story Points**: 5
- **Description**: Implement `GenerateEmbeddingStream` that returns stream of progress updates: model loading, preprocessing, inference progress, completion. Useful for long-running operations.
- **Acceptance Criteria**:
  - Streams progress updates
  - Updates include stage and percentage
  - Final update includes result
  - Client can cancel mid-stream
  - Error handling works
- **Dependencies**: Story 2.2.4

#### Story 4.3.2: Add Progress Tracking to Providers

- **Priority**: P2
- **Story Points**: 3
- **Description**: Update providers to emit progress events during inference. Hook into model internals if possible, or provide coarse-grained updates (loading, processing, complete).
- **Acceptance Criteria**:
  - Providers emit progress callbacks
  - Progress percentages accurate
  - Minimal performance impact
  - Optional (can be disabled)
- **Dependencies**: Story 4.3.1

### 4.4 Advanced Metrics and Monitoring

#### Story 4.4.1: Add Detailed Performance Metrics

- **Priority**: P2
- **Story Points**: 3
- **Description**: Expand Prometheus metrics: per-model latency histograms, request rate by model, error rate by error type, queue depth (if implemented), cache hit rate, warmup times.
- **Acceptance Criteria**:
  - Additional metrics implemented
  - Metrics useful for debugging
  - Grafana dashboard templates provided
  - Minimal overhead
- **Dependencies**: Story 2.3.2

#### Story 4.4.2: Implement Distributed Tracing

- **Priority**: P3
- **Story Points**: 5
- **Description**: Add OpenTelemetry instrumentation for distributed tracing. Trace requests from API → ModelMora → provider. Include span attributes for model name, inference time, memory usage.
- **Acceptance Criteria**:
  - Tracing configured (Jaeger/Zipkin)
  - Traces show full request path
  - Spans include useful attributes
  - Sampling configured appropriately
  - Documentation for viewing traces
- **Dependencies**: Story 2.3.1

#### Story 4.4.3: Add Alerting Rules

- **Priority**: P2
- **Story Points**: 2
- **Description**: Define Prometheus alerting rules for critical conditions: service down, high error rate, memory pressure, slow inference, model loading failures. Document runbook for each alert.
- **Acceptance Criteria**:
  - Alert rules defined
  - Thresholds tuned appropriately
  - Alerts fire correctly in test
  - Runbooks documented
  - Integration with alerting system (PagerDuty/Slack)
- **Dependencies**: Story 4.4.1

### 4.5 Testing

#### Story 4.5.1: Tests for Additional Providers

- **Priority**: P1
- **Story Points**: 5
- **Description**: Unit and integration tests for DINOv2 and SAM providers. Similar coverage as CLIP tests.
- **Acceptance Criteria**:
  - All providers tested
  - Similar test coverage as CLIP
  - Tests pass consistently
  - Performance tests included
- **Dependencies**: Story 4.1.1, Story 4.1.2

#### Story 4.5.2: Tests for Batch Processing

- **Priority**: P1
- **Story Points**: 3
- **Description**: Tests verifying batch processing works correctly: multiple images processed efficiently, responses mapped correctly, partial failures handled, performance improvement validated.
- **Acceptance Criteria**:
  - Batch API tested
  - Performance improvement measured
  - Error cases covered
  - Edge cases tested (empty batch, single item, etc.)
- **Dependencies**: Story 4.2.2

---

## Phase 5: Production Readiness (Week 9-10)

**Goal**: Production deployment, documentation, security, and code cleanup

### 5.1 Code Cleanup and Optimization

#### Story 5.1.1: Remove Model Code from Worker Service

- **Priority**: P0
- **Story Points**: 3
- **Description**: Complete removal of all ML/model code from Worker service now that ModelMora handles it. Remove torch, open-clip dependencies if not needed elsewhere. Clean up imports and dead code.
- **Acceptance Criteria**:
  - All model code removed from worker
  - Dependencies cleaned up in pyproject.toml
  - Worker image size reduced
  - Tests still pass
  - No regressions
- **Dependencies**: Story 3.4.1

#### Story 5.1.2: Code Review and Refactoring

- **Priority**: P1
- **Story Points**: 5
- **Description**: Comprehensive code review of ModelMora: identify duplication, improve naming, add missing docstrings, enforce type hints, apply design patterns consistently, improve error messages.
- **Acceptance Criteria**:
  - Code review completed
  - Major issues addressed
  - Consistent code style
  - Type hints complete
  - Docstrings on all public APIs
- **Dependencies**: None

#### Story 5.1.3: Performance Optimization

- **Priority**: P1
- **Story Points**: 5
- **Description**: Profile ModelMora under load, identify bottlenecks, optimize hot paths. Focus on: image preprocessing, model inference, memory allocation, serialization/deserialization.
- **Acceptance Criteria**:
  - Profiling completed
  - Bottlenecks identified
  - Optimizations implemented
  - Performance improvement measured
  - No functionality regressions
- **Dependencies**: Story 3.5.3

### 5.2 Security Hardening

#### Story 5.2.1: Implement Authentication for gRPC

- **Priority**: P1
- **Story Points**: 5
- **Description**: Add JWT token validation to gRPC endpoints. Integrate with Keycloak for token verification. Extract user context from tokens. Support service-to-service authentication.
- **Acceptance Criteria**:
  - gRPC requires valid JWT token
  - Tokens validated against Keycloak
  - User context extracted
  - Service tokens supported
  - Unauthorized requests rejected
- **Dependencies**: None

#### Story 5.2.2: Implement Authorization and Resource Quotas

- **Priority**: P2
- **Story Points**: 5
- **Description**: Add role-based access control: admin role for model management, user role for inference. Implement per-user rate limiting and resource quotas to prevent abuse.
- **Acceptance Criteria**:
  - Roles checked on endpoints
  - Admin operations protected
  - Rate limiting per user/org
  - Quotas enforced
  - Exceeded limits return proper errors
- **Dependencies**: Story 5.2.1

#### Story 5.2.3: Enable TLS for gRPC

- **Priority**: P1
- **Story Points**: 3
- **Description**: Configure TLS for gRPC in production. Generate/configure certificates, update server to use secure channel, update clients to connect securely. Support both mutual TLS and server-only TLS.
- **Acceptance Criteria**:
  - TLS enabled for gRPC
  - Certificates configured
  - Clients connect securely
  - Insecure connections rejected in production
  - Certificate rotation supported
- **Dependencies**: None

#### Story 5.2.4: Security Audit and Penetration Testing

- **Priority**: P2
- **Story Points**: 5
- **Description**: Conduct security review: dependency scanning for vulnerabilities, code analysis for security issues, test authentication/authorization, validate input sanitization, review error messages for information leakage.
- **Acceptance Criteria**:
  - Security scan completed
  - Vulnerabilities documented
  - High/critical issues fixed
  - Security report generated
  - Remediation plan for remaining issues
- **Dependencies**: Story 5.2.1, Story 5.2.3

### 5.3 Documentation

#### Story 5.3.1: Write API Documentation

- **Priority**: P1
- **Story Points**: 3
- **Description**: Complete documentation for gRPC and REST APIs: endpoint descriptions, request/response schemas, error codes, authentication requirements, rate limits, examples in multiple languages.
- **Acceptance Criteria**:
  - All endpoints documented
  - Examples provided
  - Error codes explained
  - Authentication described
  - Published and accessible
- **Dependencies**: None

#### Story 5.3.2: Write Deployment Guide

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create deployment documentation: Docker setup, Kubernetes manifests, configuration guide, environment variables reference, scaling guidelines, resource requirements.
- **Acceptance Criteria**:
  - Deployment steps clear
  - Configuration documented
  - Examples provided
  - Troubleshooting section
  - Tested by following guide
- **Dependencies**: None

#### Story 5.3.3: Write Operations Runbook

- **Priority**: P1
- **Story Points**: 3
- **Description**: Create operations documentation: monitoring setup, common issues and solutions, log analysis guide, performance tuning, backup/restore (if applicable), disaster recovery.
- **Acceptance Criteria**:
  - Common operations documented
  - Troubleshooting steps clear
  - Monitoring setup explained
  - Alert responses defined
  - Runbook tested
- **Dependencies**: Story 4.4.3

#### Story 5.3.4: Write Developer Guide

- **Priority**: P2
- **Story Points**: 3
- **Description**: Documentation for developers: architecture overview, adding new model providers, extending functionality, testing guide, contribution guidelines, code style.
- **Acceptance Criteria**:
  - Architecture explained
  - Extension points documented
  - Examples provided
  - Testing guide complete
  - Contribution process clear
- **Dependencies**: None

### 5.4 Production Deployment Configuration

#### Story 5.4.1: Create Kubernetes Manifests

- **Priority**: P1
- **Story Points**: 5
- **Description**: Create production-ready Kubernetes manifests: deployment, service (headless for gRPC), configmap, secrets, horizontal pod autoscaler, pod disruption budget, resource limits/requests.
- **Acceptance Criteria**:
  - Manifests deploy successfully
  - Both gRPC and REST services exposed
  - ConfigMaps for configuration
  - Secrets for sensitive data
  - HPA configured
  - Resource limits tuned
- **Dependencies**: Story 5.1.3

#### Story 5.4.2: Configure Horizontal Auto-Scaling

- **Priority**: P1
- **Story Points**: 3
- **Description**: Configure HPA based on CPU, memory, and custom metrics (request queue depth). Define scaling thresholds, min/max replicas, cooldown periods. Test scaling behavior.
- **Acceptance Criteria**:
  - HPA configured
  - Scales up under load
  - Scales down when idle
  - Thresholds appropriate
  - Behavior validated
- **Dependencies**: Story 5.4.1

#### Story 5.4.3: Set Up Continuous Deployment Pipeline

- **Priority**: P1
- **Story Points**: 5
- **Description**: Create CI/CD pipeline: build Docker image, run tests, push to registry, deploy to staging, run smoke tests, deploy to production (with approval). Use GitHub Actions or similar.
- **Acceptance Criteria**:
  - Pipeline builds on commit
  - Tests run automatically
  - Image pushed to registry
  - Staging deployment automated
  - Production deployment with approval
  - Rollback capability
- **Dependencies**: Story 5.4.1

#### Story 5.4.4: Configure Production Monitoring

- **Priority**: P1
- **Story Points**: 3
- **Description**: Set up production monitoring: Prometheus scraping ModelMora, Grafana dashboards, log aggregation (ELK/Loki), alerting configured, on-call rotation defined.
- **Acceptance Criteria**:
  - Prometheus collecting metrics
  - Grafana dashboards deployed
  - Logs centralized
  - Alerts firing correctly
  - On-call notified
- **Dependencies**: Story 4.4.1, Story 4.4.3

### 5.5 Production Testing

#### Story 5.5.1: Staging Environment Validation

- **Priority**: P0
- **Story Points**: 5
- **Description**: Deploy to staging and run comprehensive tests: functional tests, integration tests with other services, load tests, chaos testing (kill pods, network issues), verify monitoring and alerting.
- **Acceptance Criteria**:
  - Deployed to staging
  - All tests pass
  - Monitoring working
  - Alerts tested
  - Chaos scenarios handled
  - Performance acceptable
- **Dependencies**: Story 5.4.3

#### Story 5.5.2: Production Smoke Tests

- **Priority**: P0
- **Story Points**: 2
- **Description**: Create automated smoke tests to run after production deployment: health checks pass, can generate embedding, can load/unload model, Kafka integration working, metrics available.
- **Acceptance Criteria**:
  - Smoke test suite created
  - Tests cover critical paths
  - Fast execution (<2 minutes)
  - Integrated into CD pipeline
  - Failures block deployment
- **Dependencies**: Story 5.4.3

#### Story 5.5.3: Canary Deployment Test

- **Priority**: P2
- **Story Points**: 3
- **Description**: Test canary deployment: deploy new version to small percentage of traffic, monitor metrics, gradually increase traffic, rollback if issues detected.
- **Acceptance Criteria**:
  - Canary deployment configured
  - Traffic routing works
  - Metrics compared between versions
  - Automatic rollback on errors
  - Process documented
- **Dependencies**: Story 5.4.3

---

## Phase 6: Optimization (Ongoing)

**Goal**: Continuous improvement, advanced features, cost optimization

### 6.1 GPU Support (Optional)

#### Story 6.1.1: Add GPU Detection and Configuration

- **Priority**: P3
- **Story Points**: 3
- **Description**: Implement GPU detection on startup, configure torch to use GPU when available, add GPU-specific configuration options, handle graceful fallback to CPU.
- **Acceptance Criteria**:
  - GPU detected if available
  - Torch uses GPU for inference
  - Configuration supports GPU selection
  - Fallback to CPU works
  - Logs GPU usage
- **Dependencies**: None

#### Story 6.1.2: Update Providers for GPU Acceleration

- **Priority**: P3
- **Story Points**: 5
- **Description**: Modify providers to leverage GPU when available. Handle data transfer to/from GPU, optimize batch sizes for GPU memory, benchmark GPU vs CPU performance.
- **Acceptance Criteria**:
  - Providers use GPU
  - Data transfer optimized
  - Performance improvement measured
  - GPU memory managed
  - Works on both GPU and CPU
- **Dependencies**: Story 6.1.1

#### Story 6.1.3: GPU Resource Management

- **Priority**: P3
- **Story Points**: 5
- **Description**: Implement GPU memory tracking similar to CPU memory management. Handle GPU OOM errors, implement eviction for GPU models, monitor GPU utilization.
- **Acceptance Criteria**:
  - GPU memory tracked
  - Eviction works for GPU models
  - GPU OOM handled gracefully
  - Metrics include GPU usage
  - Multi-GPU support (optional)
- **Dependencies**: Story 6.1.2

### 6.2 Model Quantization

#### Story 6.2.1: Implement Model Quantization Support

- **Priority**: P3
- **Story Points**: 8
- **Description**: Add support for quantized models (INT8, FP16) to reduce memory usage. Update providers to load quantized weights, benchmark accuracy vs memory tradeoff, provide configuration options.
- **Acceptance Criteria**:
  - Quantized models load successfully
  - Memory reduction measured (50%+ expected)
  - Accuracy impact acceptable (<2% degradation)
  - Configuration per model
  - Documentation on tradeoffs
- **Dependencies**: None

#### Story 6.2.2: Automatic Model Quantization Pipeline

- **Priority**: P3
- **Story Points**: 8
- **Description**: Create pipeline to automatically quantize models: calibration data collection, quantization process, accuracy validation, storage of quantized weights. Integrate into model registry.
- **Acceptance Criteria**:
  - Pipeline quantizes models automatically
  - Calibration dataset configurable
  - Validation against original model
  - Quantized models in registry
  - Process documented
- **Dependencies**: Story 6.2.1

### 6.3 Advanced Caching

#### Story 6.3.1: Implement Embedding Cache

- **Priority**: P3
- **Story Points**: 5
- **Description**: Add Redis cache for generated embeddings. Cache by image hash to avoid regenerating for duplicate images. Configure TTL and eviction policy. Monitor cache hit rate.
- **Acceptance Criteria**:
  - Redis integrated
  - Embeddings cached by hash
  - Cache hits skip inference
  - TTL configurable
  - Metrics for cache hit rate
  - Significant performance improvement
- **Dependencies**: None

#### Story 6.3.2: Implement Model Weight Caching

- **Priority**: P3
- **Story Points**: 5
- **Description**: Cache downloaded model weights on persistent volume to speed up container restarts. Implement checksum validation, version management, cleanup of old weights.
- **Acceptance Criteria**:
  - Weights cached on volume
  - Faster restarts (no redownload)
  - Checksums validated
  - Old versions cleaned up
  - Storage limits enforced
- **Dependencies**: None

### 6.4 A/B Testing Framework

#### Story 6.4.1: Implement Model Comparison API

- **Priority**: P3
- **Story Points**: 5
- **Description**: Add endpoint to generate embeddings from multiple models simultaneously for comparison. Return embeddings from all models with metadata for analysis.
- **Acceptance Criteria**:
  - API accepts multiple model names
  - Generates embeddings in parallel
  - Returns all results
  - Performance acceptable
  - Used for model evaluation
- **Dependencies**: Story 4.2.2

#### Story 6.4.2: Implement A/B Testing Configuration

- **Priority**: P3
- **Story Points**: 8
- **Description**: Add framework for A/B testing models: traffic splitting configuration, experiment tracking, metrics collection per variant, statistical significance calculation, automatic winner selection.
- **Acceptance Criteria**:
  - Traffic split between models
  - Metrics tracked per variant
  - Statistical analysis performed
  - Winner can be selected
  - Experiment configuration external
  - Results dashboard
- **Dependencies**: Story 6.4.1

### 6.5 Cost Optimization

#### Story 6.5.1: Implement Inference Cost Tracking

- **Priority**: P3
- **Story Points**: 3
- **Description**: Track costs per inference: compute time, memory used, GPU time (if applicable). Aggregate by user, organization, model. Expose cost metrics.
- **Acceptance Criteria**:
  - Cost calculated per inference
  - Aggregated by dimensions
  - Metrics exposed
  - Cost reports generated
  - Used for optimization decisions
- **Dependencies**: Story 2.3.2

#### Story 6.5.2: Implement Intelligent Model Selection

- **Priority**: P3
- **Story Points**: 8
- **Description**: Automatically select optimal model based on image characteristics, accuracy requirements, latency constraints, and cost targets. Machine learning-based selection or rule-based heuristics.
- **Acceptance Criteria**:
  - Selection algorithm implemented
  - Considers multiple factors
  - Improves cost efficiency
  - Maintains accuracy requirements
  - Configurable policies
  - Metrics on selection quality
- **Dependencies**: Story 6.5.1

---

## Appendix: Summary by Phase

### Phase 1 (Foundation)

- **Total Stories**: 19
- **Total Story Points**: 47
- **Expected Duration**: 2 weeks
- **Key Deliverables**: Working service with gRPC/REST, single CLIP model, basic testing

### Phase 2 (Core Functionality)

- **Total Stories**: 15
- **Total Story Points**: 56
- **Expected Duration**: 2 weeks
- **Key Deliverables**: ModelManager with LRU cache, memory management, monitoring

### Phase 3 (Integration)

- **Total Stories**: 12
- **Total Story Points**: 48
- **Expected Duration**: 2 weeks
- **Key Deliverables**: Full integration with API and Worker services

### Phase 4 (Enhancement)

- **Total Stories**: 13
- **Total Story Points**: 56
- **Expected Duration**: 2 weeks
- **Key Deliverables**: Multiple models, batch processing, advanced monitoring

### Phase 5 (Production Readiness)

- **Total Stories**: 19
- **Total Story Points**: 70
- **Expected Duration**: 2 weeks
- **Key Deliverables**: Production deployment, security, documentation

### Phase 6 (Optimization)

- **Total Stories**: 11
- **Total Story Points**: 67
- **Expected Duration**: Ongoing
- **Key Deliverables**: GPU support, quantization, caching, A/B testing

---

## Total Project Scope

- **Total Stories**: 89
- **Total Story Points**: 344
- **Estimated Duration**: 10+ weeks
- **Team Size**: 2-3 developers recommended

---

**Note**: Story points and durations are estimates. Actual velocity will vary based on team experience, unforeseen issues, and requirement changes. Review and adjust after each sprint.
