# import asyncio
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from miraveja_auth import FastAPIAuthenticator
from miraveja_di import DIContainer
from miraveja_di.infrastructure.fastapi_integration import ScopedContainerMiddleware
from miraveja_log import IAsyncLogger, ILogger, LoggerConfig, LoggerFactory

from modelmora import ModelMoraDependencies
from modelmora.configuration import AppConfig
from modelmora.registry.infrastructure import RegistryController, RegistryRoutes
from modelmora.registry.infrastructure.dependencies import RegistryDependencies
from modelmora.shared import DomainException, ErrorMiddleware
from modelmora.shared.infrastructure.fastapi_integration.middleware import LoggerMiddleware

# Load environment variables from .env file
dotenv.load_dotenv("/app/.env")

# Create DI container
container: DIContainer = DIContainer()

# Register dependencies
ModelMoraDependencies.register_dependencies(container=container)
RegistryDependencies.register_dependencies(container=container)

app_config: AppConfig = container.resolve(AppConfig)


# Create global loggers
global_logger: ILogger = container.resolve(ILogger)
global_a_logger: IAsyncLogger = container.resolve(IAsyncLogger)


@asynccontextmanager
async def lifespan(application: FastAPI):  # pylint: disable=unused-argument
    # Start gRPC server when FastAPI starts.
    await global_a_logger.info("Starting gRPC server...")
    # asyncio.create_task(start_grpc_server())
    # Initialize and start your gRPC server here (e.g., create asyncio task)
    await global_a_logger.info("gRPC server started on port 50051")
    try:
        yield
    finally:
        # Clean up/shutdown gRPC server here
        await global_a_logger.info("Shutting down gRPC server...")


# Initialize FastAPI app
app = FastAPI(
    title="ModelMora API",
    description="A modular AI model serving platform.",
    version=app_config.app_version,
    lifespan=lifespan,
    redirect_slashes=False,
    root_path="/modelmora/api",
)

# Middlewares

app.add_middleware(
    ErrorMiddleware,
)
app.add_middleware(
    LoggerMiddleware,
)
app.add_middleware(
    ScopedContainerMiddleware,  # type: ignore
    container=container,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Important: Handle preflight requests properly
    expose_headers=["Content-Type", "X-Content-Type-Options"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Setup routers for API versioning
apiV1Router: APIRouter = APIRouter(prefix="/v1")

fastapi_authenticator = container.resolve(FastAPIAuthenticator)

# Define API endpoints
RegistryRoutes.register_routes(router=apiV1Router, controller=container.resolve(RegistryController))


@apiV1Router.get("/health")
async def health_check():
    return {"status": "healthy"}


@apiV1Router.get("/error")
async def trigger_error():
    raise DomainException(
        message="This is a test domain exception.", detail="Additional details about the domain exception.", code=404
    )


# include the API router in the main app
app.include_router(apiV1Router)
if __name__ == "__main__":
    uvicorn.run(
        "ModelMora.main:app",
        host=app_config.connection_config.http_host,
        port=app_config.connection_config.http_port,
        reload=True,
        reload_dirs=["/app"],
    )
