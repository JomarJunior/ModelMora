# ModelMora Microservice Architecture Plan

> **ModelMora**: A dedicated neural network model management and inference serving microservice for the MiraVeja ecosystem.
>
> **Etymology**: Combining "Model" (neural network models) and "Mora" (Spanish/Portuguese for "to dwell/reside"), suggesting a place where models reside and are managed.

---

## 1. Core Technology Stack

### Runtime & Framework

- **Python 3.11** (consistent with existing services)
- **FastAPI** for REST/HTTP endpoints (health checks, model management)
- **gRPC** for high-throughput inference requests
- **Uvicorn** as ASGI server

### Dependencies

- **miraveja-core** (shared domain models, infrastructure patterns)
- **PyTorch 2.5.0+** (CPU/GPU configurable)
- **OpenCLIP, Transformers, etc.** (model-specific libraries)
- **grpcio + grpcio-tools** for gRPC communication
- **aiokafka** for async event consumption/production

### Rationale

- Maintains consistency with existing services
- FastAPI for lightweight admin/health endpoints
- gRPC for efficient, low-latency inference (better than REST for high-frequency embedding requests)
- Kafka for async, decoupled event-driven tasks

---

## 2. Communication Protocols

### Primary: gRPC for Inference ⭐

**Protocol Buffer Definition:**

```protobuf
// modemora.proto
service ModelInference {
  // Synchronous inference (when caller needs immediate result)
  rpc GenerateEmbedding(EmbeddingRequest) returns (EmbeddingResponse);

  // Batch inference for efficiency
  rpc GenerateBatch(BatchRequest) returns (stream EmbeddingResponse);

  // Stream for real-time progress updates
  rpc GenerateEmbeddingStream(EmbeddingRequest) returns (stream InferenceStatus);
}

message EmbeddingRequest {
  string request_id = 1;
  bytes image_data = 2;  // or S3 URI
  string model_name = 3;
  map<string, string> options = 4;
}

message EmbeddingResponse {
  string request_id = 1;
  repeated float vector = 2;
  string model_name = 3;
  int32 dimension = 4;
  int64 inference_time_ms = 5;
}
```

**Advantages:**

- **Binary protocol**: 2-10x smaller payloads than JSON
- **HTTP/2**: Multiplexing, streaming, bidirectional communication
- **Strong typing**: Protocol buffers prevent serialization errors
- **Performance**: Critical for embedding generation (can process 100s/second)

### Secondary: Kafka for Async Processing ⭐

**Event Definitions:**

```python
# Subscribe to events
@event_registry.register_event("image.metadata.registered", 1)
class ImageMetadataRegisteredEvent(DomainEvent):
    image_metadata_id: str
    owner_id: str
    image_url: str
    content_type: str

# Publish completion events
@event_registry.register_event("embedding.generated", 1)
class EmbeddingGeneratedEvent(DomainEvent):
    image_metadata_id: str
    vector_id: str
    model_name: str
    dimension: int
    generation_time_ms: int

@event_registry.register_event("model.loaded", 1)
class ModelLoadedEvent(DomainEvent):
    model_name: str
    memory_mb: int
    warmup_time_ms: int
```

**Use Cases:**

- Background embedding generation (fire-and-forget)
- Model retraining triggers
- Batch processing queues
- Event notifications (embedding complete, model loaded, etc.)

**Advantages:**

- **Decoupling**: Services don't need to know ModelMora's location
- **Resilience**: Kafka handles failures, retries automatically
- **Scalability**: Scale ModelMora independently based on queue depth
- **Consistency**: Matches existing MiraVeja architecture

### Tertiary: REST API for Management

**Endpoints:**

```python
# FastAPI endpoints for admin/monitoring
@router.get("/health")
async def health() -> dict:
    """Health check with loaded models status"""
    return {
        "status": "healthy",
        "models_loaded": [...],
        "memory_usage_mb": 4096,
        "uptime_seconds": 3600
    }

@router.get("/models")
async def list_models() -> List[ModelInfo]:
    """List available models from registry"""

@router.get("/models/{model_name}")
async def get_model_info(model_name: str) -> ModelInfo:
    """Get detailed info about a specific model"""

@router.post("/models/{model_name}/load")
async def load_model(model_name: str) -> dict:
    """Load model into memory"""

@router.delete("/models/{model_name}")
async def unload_model(model_name: str) -> dict:
    """Unload model and free memory"""

@router.post("/models/{model_name}/warmup")
async def warmup_model(model_name: str, samples: int = 10) -> dict:
    """Warm up model with test inferences"""

@router.get("/metrics")
async def metrics() -> dict:
    """Prometheus-compatible metrics"""
```

**Rationale:**

- Simple, human-readable for DevOps tasks
- Low frequency operations (not performance-critical)
- Easy integration with monitoring tools (Prometheus, Grafana)

---

## 3. Service Architecture

### Directory Structure

```bash
modemora/
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── README.md
├── src/
│   └── ModelMora/
│       ├── __init__.py
│       ├── main.py                    # FastAPI + gRPC server startup
│       ├── Dependencies.py            # DI container registration
│       │
│       ├── Configuration/             # Pydantic settings
│       │   ├── __init__.py
│       │   ├── ModeModaConfig.py
│       │   └── ModelRegistry.yaml     # Model definitions
│       │
│       ├── Domain/                    # Business logic
│       │   ├── __init__.py
│       │   ├── Models/
│       │   │   ├── __init__.py
│       │   │   ├── IModelProvider.py  # Interface
│       │   │   ├── ModelMetadata.py   # Value objects
│       │   │   └── ModelRegistry.py   # Registry domain model
│       │   └── Services/
│       │       ├── __init__.py
│       │       └── ModelManager.py    # Load/unload logic
│       │
│       ├── Application/               # Use cases
│       │   ├── __init__.py
│       │   ├── Commands/
│       │   │   ├── __init__.py
│       │   │   ├── GenerateEmbedding.py
│       │   │   ├── LoadModel.py
│       │   │   └── UnloadModel.py
│       │   └── Subscribers/           # Kafka event handlers
│       │       ├── __init__.py
│       │       └── HandleImageRegistered.py
│       │
│       ├── Infrastructure/
│       │   ├── __init__.py
│       │   ├── GRPC/
│       │   │   ├── __init__.py
│       │   │   ├── protos/
│       │   │   │   └── modemora.proto
│       │   │   ├── Server.py
│       │   │   └── Services.py        # gRPC service implementation
│       │   │
│       │   ├── Providers/             # Model implementations
│       │   │   ├── __init__.py
│       │   │   ├── BaseProvider.py
│       │   │   ├── ClipProvider.py
│       │   │   ├── DINOv2Provider.py
│       │   │   └── SAMProvider.py
│       │   │
│       │   └── Cache/
│       │       ├── __init__.py
│       │       └── ModelCache.py      # LRU cache for models
│       │
│       └── API/                       # REST endpoints
│           ├── __init__.py
│           ├── Health.py
│           └── Models.py
│
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_model_manager.py
    │   └── test_providers.py
    └── integration/
        ├── __init__.py
        └── test_grpc_server.py
```

### Key Design Decisions

1. **Separate gRPC and REST servers**: Run on different ports (gRPC: 50051, REST: 8080)
2. **Model registry**: YAML file defining available models, their configs, resource requirements
3. **Lazy loading**: Models loaded on first use, not startup (prevents OOM)
4. **LRU cache**: Automatically unload least-used models when memory constrained
5. **Provider pattern**: Easy to add new model types (extend `IModelProvider`)

---

## 4. Model Management Strategy

### Model Registry Configuration

```yaml
# Configuration/ModelRegistry.yaml
models:
  - name: "clip-vit-g-14"
    type: "embedding"
    provider: "ClipProvider"
    config:
      pretrained: "laion2b_s12b_b42k"
      architecture: "ViT-g-14"
    resources:
      memory_mb: 2048
      gpu_required: false
      cpu_threads: 4
    warmup:
      enabled: true
      samples: 10
    default: true  # Load on startup

  - name: "dinov2-vitl"
    type: "embedding"
    provider: "DINOv2Provider"
    config:
      pretrained: "dinov2_vitl14"
      architecture: "vit_large"
    resources:
      memory_mb: 1200
      gpu_required: false
      cpu_threads: 4
    warmup:
      enabled: true
      samples: 5

  - name: "sam-vit-h"
    type: "segmentation"
    provider: "SAMProvider"
    config:
      checkpoint: "sam_vit_h_4b8939.pth"
      architecture: "vit_h"
    resources:
      memory_mb: 4096
      gpu_required: true  # Prefers GPU but can run on CPU
      cpu_threads: 8
    warmup:
      enabled: false  # Too slow for warmup
```

### Dynamic Loading with Memory Management

```python
class ModelManager:
    """Manages model lifecycle with LRU eviction"""

    def __init__(self, max_memory_mb: int = 8192):
        self._loaded_models: Dict[str, IModelProvider] = {}
        self._lru_cache: OrderedDict = OrderedDict()
        self._max_memory = max_memory_mb
        self._current_memory = 0
        self._registry: ModelRegistry = None
        self._lock = asyncio.Lock()

    async def get_model(self, model_name: str) -> IModelProvider:
        """Load model if needed, evict LRU if memory constrained"""
        async with self._lock:
            if model_name in self._loaded_models:
                # Move to end (most recently used)
                self._lru_cache.move_to_end(model_name)
                return self._loaded_models[model_name]

            # Check if we have memory
            model_config = self._registry.get_model(model_name)
            await self._ensure_memory(model_config.resources.memory_mb)

            # Load model
            provider = await self._load_provider(model_config)
            self._loaded_models[model_name] = provider
            self._lru_cache[model_name] = model_config.resources.memory_mb
            self._current_memory += model_config.resources.memory_mb

            logger.info(
                f"Loaded model: {model_name} "
                f"(Memory: {self._current_memory}/{self._max_memory} MB)"
            )

            return provider

    async def _ensure_memory(self, required_mb: int):
        """Evict LRU models until we have enough memory"""
        while self._current_memory + required_mb > self._max_memory:
            if not self._lru_cache:
                raise OutOfMemoryError(
                    f"Cannot allocate {required_mb}MB. "
                    f"Max: {self._max_memory}MB, Current: {self._current_memory}MB"
                )

            # Evict least recently used
            lru_name, lru_memory = self._lru_cache.popitem(last=False)
            await self._unload_model(lru_name)
            logger.info(f"Evicted model: {lru_name} (freed {lru_memory}MB)")

    async def _load_provider(self, model_config: ModelConfig) -> IModelProvider:
        """Factory method to instantiate provider"""
        provider_class = self._get_provider_class(model_config.provider)
        provider = provider_class(model_config)
        await provider.initialize()
        return provider

    async def _unload_model(self, model_name: str):
        """Unload model and free resources"""
        if model_name in self._loaded_models:
            provider = self._loaded_models[model_name]
            await provider.cleanup()
            del self._loaded_models[model_name]
            memory = self._lru_cache.pop(model_name)
            self._current_memory -= memory
```

### Warm-up Strategy

```python
async def warm_up_models(self):
    """Pre-load default models and run test inference"""
    default_models = self._registry.get_default_models()

    for model_config in default_models:
        if not model_config.warmup.enabled:
            continue

        try:
            model = await self.get_model(model_config.name)

            # Run test inference to load weights fully
            start_time = time.time()
            for i in range(model_config.warmup.samples):
                await model.generate_test_embedding()
                logger.debug(f"Warmup {i+1}/{model_config.warmup.samples}")

            warmup_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"Warmed up model: {model_config.name} "
                f"({warmup_time}ms for {model_config.warmup.samples} samples)"
            )

            # Publish event
            await self._event_publisher.publish(
                ModelLoadedEvent(
                    model_name=model_config.name,
                    memory_mb=model_config.resources.memory_mb,
                    warmup_time_ms=warmup_time
                )
            )
        except Exception as e:
            logger.error(f"Failed to warm up {model_config.name}: {e}")
```

---

## 5. Integration with Existing Services

### API Service → ModelMora (Synchronous via gRPC)

```python
# api/src/MiravejaApi/Gallery/Application/Commands/UploadImage.py
class UploadImageHandler:
    def __init__(
        self,
        modemora_client: ModelMoraGrpcClient,  # gRPC stub
        kafka_producer: IKafkaProducer,        # Fallback to async
        minio_service: IMinioService,
        qdrant_service: IQdrantService,
        database_manager_factory: IDatabaseManagerFactory,
    ):
        self._modemora_client = modemora_client
        self._kafka_producer = kafka_producer
        self._minio_service = minio_service
        self._qdrant_service = qdrant_service
        self._database_manager_factory = database_manager_factory

    async def handle(self, command: UploadImageCommand) -> HandlerResponse:
        # Save image to MinIO
        image_url = await self._minio_service.upload(command.file)

        # Register metadata in database
        with self._database_manager_factory.create() as db_manager:
            repository = db_manager.get_repository(IImageMetadataRepository)
            metadata = ImageMetadata.register(
                owner_id=command.owner_id,
                title=command.title,
                image_url=image_url
            )
            repository.save(metadata)

        # Try synchronous embedding generation (if user needs immediate search)
        try:
            embedding = await self._modemora_client.generate_embedding(
                image_data=command.file.read(),
                model_name="clip-vit-g-14",
                timeout=5.0  # 5 second timeout
            )

            # Store embedding immediately
            vector_id = await self._qdrant_service.store(
                collection="image_embeddings",
                vector=embedding.vector,
                payload={"image_metadata_id": metadata.id}
            )

            # Update metadata with vector ID
            metadata.assign_vector_id(vector_id)
            repository.save(metadata)

            logger.info(f"Generated embedding synchronously: {metadata.id}")

        except (TimeoutError, grpc.RpcError) as e:
            # Fallback to async processing if ModelMora is slow/unavailable
            logger.warning(f"Sync embedding failed, falling back to async: {e}")
            await self._kafka_producer.publish(
                ImageMetadataRegisteredEvent(
                    image_metadata_id=metadata.id,
                    owner_id=metadata.owner_id,
                    image_url=image_url,
                    content_type=command.file.content_type
                )
            )

        return {"image_metadata_id": str(metadata.id)}
```

### Worker Service → ModelMora (Asynchronous via Kafka)

```python
# worker/src/MiravejaWorker/Subscribers/GenerateImageVector.py
class GenerateImageVector(IEventSubscriber[ImageMetadataRegisteredEvent]):
    """Handles async embedding generation triggered by Kafka events"""

    def __init__(
        self,
        modemora_client: ModelMoraGrpcClient,
        minio_service: IMinioService,
        qdrant_service: IQdrantService,
        kafka_producer: IKafkaProducer,
        database_manager_factory: IDatabaseManagerFactory,
    ):
        self._modemora_client = modemora_client
        self._minio_service = minio_service
        self._qdrant_service = qdrant_service
        self._kafka_producer = kafka_producer
        self._database_manager_factory = database_manager_factory

    async def handle(self, event: ImageMetadataRegisteredEvent):
        try:
            # Download image from MinIO
            image_data = await self._minio_service.download(event.image_url)

            # Request embedding via gRPC (no timeout - background task)
            embedding = await self._modemora_client.generate_embedding(
                image_data=image_data,
                model_name="clip-vit-g-14"
            )

            # Store in Qdrant
            vector_id = await self._qdrant_service.store(
                collection="image_embeddings",
                vector=embedding.vector,
                payload={
                    "image_metadata_id": event.image_metadata_id,
                    "owner_id": event.owner_id,
                    "model_name": embedding.model_name,
                    "dimension": embedding.dimension
                }
            )

            # Update database with vector ID
            with self._database_manager_factory.create() as db_manager:
                repository = db_manager.get_repository(IImageMetadataRepository)
                metadata = repository.find_by_id(event.image_metadata_id)
                metadata.assign_vector_id(vector_id)
                repository.save(metadata)

            # Publish completion event
            await self._kafka_producer.publish(
                EmbeddingGeneratedEvent(
                    image_metadata_id=event.image_metadata_id,
                    vector_id=vector_id,
                    model_name=embedding.model_name,
                    dimension=embedding.dimension,
                    generation_time_ms=embedding.inference_time_ms
                )
            )

            logger.info(
                f"Generated embedding for {event.image_metadata_id} "
                f"in {embedding.inference_time_ms}ms"
            )

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Could publish failure event here for monitoring
```

---

## 6. Docker Configuration

### Dockerfile

```dockerfile
# modemora/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.0.1
RUN poetry config virtualenvs.create false

# Copy and install core package first
COPY ./core /core
RUN cd /core && poetry install --only main

# Copy Poetry configuration files
COPY ./modemora/pyproject.toml ./modemora/poetry.lock ./modemora/README.md ./

# Install dependencies
RUN poetry install --only main --no-root

# Copy application source
COPY ./modemora/src ./src

# Install the modemora package itself
RUN poetry install --only-root

# Expose ports
EXPOSE 8080  # REST API
EXPOSE 50051 # gRPC

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start both servers
CMD ["poetry", "run", "python", "-m", "ModelMora.main"]
```

### Docker Compose Integration

```yaml
# docker-compose.yml (additions)
services:
  modemora:
    build:
      context: .
      dockerfile: ./modemora/Dockerfile
    container_name: miraveja-modemora
    env_file:
      - .env
    environment:
      - MODEMORA_REST_PORT=8080
      - MODEMORA_GRPC_PORT=50051
      - MODEMORA_MAX_MODEL_MEMORY_MB=8192
      - MODEMORA_AUTO_WARMUP=true
      - MODEMORA_GPU_ENABLED=false
    ports:
      - "${MODEMORA_REST_PORT:-8080}:8080"
      - "${MODEMORA_GRPC_PORT:-50051}:50051"
    volumes:
      - ./models:/models              # Model cache (shared with worker if needed)
      - ./modemora/src:/app/src       # Hot reload (dev only)
      - ./modemora/Configuration:/app/Configuration  # Model registry
      - ./core:/core                  # Shared library
      - ./schemas:/schemas            # Event schemas
    networks:
      - miraveja
    depends_on:
      - kafka
      - minio
    deploy:
      resources:
        limits:
          memory: 10G  # Adjust based on models
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.modemora.rule=Host(`${APPLICATION_HOST}`) && PathPrefix(`/modemora`)"
      - "traefik.http.routers.modemora.entrypoints=web,websecure"
      - "traefik.http.services.modemora.loadbalancer.server.port=8080"

networks:
  miraveja:
    driver: bridge

volumes:
  models:
```

---

## 7. Configuration

### pyproject.toml

```toml
[tool.poetry]
name = "ModelMora"
version = "0.1.0"
description = "Neural network model management and inference serving microservice for MiraVeja"
authors = ["Jomar Júnior de Souza Pereira <jomarjunior@poli.ufrj.br>"]
readme = "README.md"
license = "MIT"
packages = [
    { include = "ModelMora", from = "src" }
]

[tool.poetry.dependencies]
python = "^3.11"
# Local core package dependency
miraveja-core = {path = "../core", develop = true}
# Web frameworks
fastapi = ">=0.117.1"
uvicorn = {extras = ["standard"], version = ">=0.37.0"}
# gRPC
grpcio = "^1.60.0"
grpcio-tools = "^1.60.0"
grpcio-reflection = "^1.60.0"
# Configuration
pydantic-settings = ">=2.7.1"
python-dotenv = ">=1.1.1"
pyyaml = ">=6.0"
# Event streaming
aiokafka = ">=0.11.0"
# ML/AI (inherited from core but explicitly listed for clarity)
torch = {version = "^2.5.0", source = "torch_cpu"}
torchvision = {version = "^0.20.0", source = "torch_cpu"}
open-clip-torch = "^2.26.1"
# Monitoring
prometheus-client = "^0.19.0"

[[tool.poetry.source]]
name = "torch_cpu"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"

[tool.poetry.group.dev.dependencies]
pytest = ">=8.2.0"
pytest-asyncio = ">=0.21.0"
pytest-grpc = "^0.8.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 120
target-version = ["py311"]

[tool.pylint.master]
module-naming-style = "snake_case"
const-naming-style = "UPPER_CASE"
class-naming-style = "PascalCase"
function-naming-style = "snake_case"
method-naming-style = "snake_case"
attr-naming-style = "snake_case"
argument-naming-style = "snake_case"
variable-naming-style = "snake_case"

[tool.isort]
profile = "black"
line_length = 120
known_first_party = ["ModelMora", "MiravejaCore"]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=src --cov-report=term-missing --asyncio-mode=auto"
testpaths = ["tests"]
```

### Application Configuration

```python
# ModelMora/Configuration/modemora_config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List

class ModemoraConfig(BaseSettings):
    """ModelMora service configuration"""

    # Service Identity
    service_name: str = "ModelMora"
    service_version: str = "0.1.0"

    # Server Ports
    rest_port: int = Field(default=8080, env="MODEMORA_REST_PORT")
    grpc_port: int = Field(default=50051, env="MODEMORA_GRPC_PORT")

    # Performance
    max_concurrent_requests: int = Field(default=100, env="MODEMORA_MAX_CONCURRENT_REQUESTS")
    request_timeout_sec: int = Field(default=30, env="MODEMORA_REQUEST_TIMEOUT_SEC")
    batch_size: int = Field(default=32, env="MODEMORA_BATCH_SIZE")
    num_workers: int = Field(default=4, env="MODEMORA_NUM_WORKERS")

    # Model Management
    model_registry_path: str = Field(
        default="/app/Configuration/ModelRegistry.yaml",
        env="MODEMORA_MODEL_REGISTRY_PATH"
    )
    model_cache_dir: str = Field(default="/models", env="MODEMORA_MODEL_CACHE_DIR")
    max_model_memory_mb: int = Field(default=8192, env="MODEMORA_MAX_MODEL_MEMORY_MB")
    auto_warmup: bool = Field(default=True, env="MODEMORA_AUTO_WARMUP")

    # GPU Configuration
    gpu_enabled: bool = Field(default=False, env="MODEMORA_GPU_ENABLED")
    gpu_device_id: int = Field(default=0, env="MODEMORA_GPU_DEVICE_ID")

    # Kafka Integration
    kafka_config: "KafkaConfig"  # Imported from miraveja-core
    subscribe_to_events: List[str] = Field(
        default=[
            "image.metadata.registered.v1",
            "model.retrain.requested.v1"
        ],
        env="MODEMORA_SUBSCRIBE_TO_EVENTS"
    )

    # Monitoring
    prometheus_port: int = Field(default=9090, env="MODEMORA_PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="MODEMORA_LOG_LEVEL")
    enable_tracing: bool = Field(default=False, env="MODEMORA_ENABLE_TRACING")

    class Config:
        env_prefix = "MODEMORA_"
        env_file = ".env"
        case_sensitive = False
```

---

## 8. Key Advantages

✅ **Performance**: gRPC reduces latency by 40-60% vs REST for inference
✅ **Scalability**: Horizontal scaling with load balancing (gRPC + Kafka)
✅ **Resilience**: LRU cache prevents OOM, Kafka ensures no dropped requests
✅ **Flexibility**: Supports both sync (gRPC) and async (Kafka) patterns
✅ **Maintainability**: Follows existing MiraVeja patterns (DI, repositories, events)
✅ **Observability**: Prometheus metrics, structured logging, health checks
✅ **Resource Efficiency**: Dynamic model loading, memory management
✅ **Extensibility**: Easy to add new models via provider pattern
✅ **Decoupling**: Services don't directly depend on model implementations

---

## 9. Migration Path

### Phase 1: Foundation (Week 1-2)

- Create ModelMora service structure
- Implement basic gRPC server with health checks
- Set up model registry and configuration
- Implement single model provider (CLIP)
- Add REST API for model management
- Docker integration

### Phase 2: Core Functionality (Week 3-4)

- Implement ModelManager with LRU cache
- Add memory management and eviction logic
- Implement model warmup strategy
- Add comprehensive logging and metrics
- Unit tests for core components

### Phase 3: Integration (Week 5-6)

- Create gRPC client for API service
- Migrate worker's embedding generation to ModelMora
- Add Kafka event subscribers
- Integration tests with existing services
- Performance testing and optimization

### Phase 4: Enhancement (Week 7-8)

- Add multiple model providers (DINOv2, SAM, etc.)
- Implement batch processing
- Add streaming inference support
- Advanced metrics and monitoring
- Load testing and tuning

### Phase 5: Production Readiness (Week 9-10)

- Remove model code from worker service
- Complete API migration to ModelMora for sync requests
- Production deployment configuration
- Documentation and runbooks
- Security hardening

### Phase 6: Optimization (Ongoing)

- GPU support (optional)
- Model quantization for memory reduction
- Advanced caching strategies
- A/B testing framework for model comparison
- Auto-scaling policies based on queue depth

---

## 10. Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
inference_requests_total = Counter(
    'modemora_inference_requests_total',
    'Total inference requests',
    ['model_name', 'status']
)

inference_duration_seconds = Histogram(
    'modemora_inference_duration_seconds',
    'Inference request duration',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Model metrics
models_loaded = Gauge(
    'modemora_models_loaded',
    'Number of models currently loaded'
)

model_memory_usage_bytes = Gauge(
    'modemora_model_memory_usage_bytes',
    'Memory used by loaded models',
    ['model_name']
)

model_evictions_total = Counter(
    'modemora_model_evictions_total',
    'Total number of model evictions due to memory pressure'
)

# System metrics
system_memory_available_bytes = Gauge(
    'modemora_system_memory_available_bytes',
    'Available system memory'
)

# Service info
service_info = Info(
    'modemora_service',
    'ModelMora service information'
)
service_info.info({
    'version': '0.1.0',
    'python_version': '3.11',
    'gpu_enabled': 'false'
})
```

### Health Check

```python
@router.get("/health")
async def health(model_manager: ModelManager) -> dict:
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": {
            "name": "ModelMora",
            "version": "0.1.0",
            "uptime_seconds": get_uptime()
        },
        "models": {
            "loaded": list(model_manager.get_loaded_models().keys()),
            "count": len(model_manager.get_loaded_models()),
            "memory_used_mb": model_manager.get_current_memory_usage(),
            "memory_max_mb": model_manager.get_max_memory()
        },
        "resources": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "dependencies": {
            "kafka": await check_kafka_connection(),
            "minio": await check_minio_connection()
        }
    }
```

### Logging

```python
# Use MiravejaCore logging infrastructure
from MiravejaCore.Shared.Logger import ILogger

logger: ILogger = container.Get(ILogger)

# Structured logging
logger.info(
    "Model inference completed",
    extra={
        "model_name": "clip-vit-g-14",
        "inference_time_ms": 245,
        "request_id": request_id,
        "image_size_bytes": len(image_data)
    }
)
```

---

## 11. Security Considerations

### Authentication

- ModelMora should validate JWT tokens from Keycloak for REST endpoints
- gRPC should use TLS with mutual authentication in production
- Internal services (worker) can use service-to-service auth

### Authorization

- Model management endpoints require admin role
- Inference endpoints require authenticated user
- Resource quotas per user/organization

### Network Security

- ModelMora on internal network only (not publicly exposed)
- API gateway (Traefik) handles external access with auth
- gRPC with TLS in production
- Rate limiting on inference endpoints

### Data Security

- Image data encrypted in transit (TLS)
- No persistent storage of image data in ModelMora
- Embeddings treated as non-sensitive metadata
- Audit logging for model changes

---

## 12. Open Questions & Future Considerations

### Performance

- [ ] Should we support GPU acceleration? (Would require nvidia-docker)
- [ ] Is LRU the best eviction strategy, or should we use LFU/adaptive?
- [ ] Should we implement request queuing with priorities?

### Architecture

- [ ] Should ModelMora have its own database for model metadata/history?
- [ ] Do we need versioning for models (v1, v2, etc.)?
- [ ] Should we support model ensembles (multiple models for same task)?

### Operations

- [ ] How to handle model updates without downtime?
- [ ] Should we support blue/green deployment for models?
- [ ] Do we need a model registry service (separate from ModeMora)?

### Advanced Features

- [ ] Model fine-tuning/retraining capabilities?
- [ ] A/B testing framework for comparing models?
- [ ] Automatic model selection based on image characteristics?
- [ ] Cost optimization (track inference costs, optimize model selection)?

---

## Conclusion

ModelMora represents a significant architectural improvement for MiraVeja by:

1. **Centralizing** neural network operations
2. **Optimizing** resource usage with dynamic loading
3. **Improving** scalability with gRPC and Kafka
4. **Maintaining** consistency with existing patterns
5. **Enabling** future AI/ML feature expansion

The phased migration path ensures minimal disruption to existing services while progressively realizing the benefits of the new architecture.

---

**Document Version**: 1.0
**Last Updated**: November 19, 2025
**Author**: GitHub Copilot (Claude Sonnet 4.5)
**Status**: Planning / Not Implemented
