# import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):  # pylint: disable=unused-argument
    # Start gRPC server when FastAPI starts.
    logger.info("Starting gRPC server...")
    # asyncio.create_task(start_grpc_server())
    # Initialize and start your gRPC server here (e.g., create asyncio task)
    logger.info("gRPC server started on port 50051")
    try:
        yield
    finally:
        # Clean up/shutdown gRPC server here
        logger.info("Shutting down gRPC server...")


app = FastAPI(
    title="ModelMora",
    description="A modular AI model serving platform.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("ModelMora.main:app", host="0.0.0.0", port=8080, reload=True, reload_dirs=["src"])
