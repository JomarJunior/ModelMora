import asyncio

from miraveja_log import IAsyncLogger
from pydantic import BaseModel, Field

from modelmora.registry.domain import (
    IModelProvider,
    IModelRegistry,
    ModelMetadata,
    ModelProvider,
    ModelResources,
    ModelType,
)


class DownloadModelFromProviderCommand(BaseModel):
    pretrained: str = Field(..., description="The pretrained configuration of the model to download", min_length=1)
    architecture: str = Field(..., description="The architecture of the model to download", min_length=1)
    model_type: ModelType = Field(..., description="The type of the model to download")
    is_gpu_required: bool = Field(..., description="Whether GPU is required for the model to download")
    threads: int = Field(..., description="The number of threads for the model to download")
    is_warmup_enabled: bool = Field(..., description="Whether to enable warmup for the model to download")
    warmup_samples: int = Field(..., description="The number of warmup samples for the model to download")
    set_defaults: bool = Field(default=False, description="Whether to set default values for unspecified parameters")


class DownloadModelFromProviderHandler:
    def __init__(
        self,
        model_provider: IModelProvider,
        model_registry: IModelRegistry,
        logger: IAsyncLogger,
    ) -> None:
        self.model_provider = model_provider
        self.model_registry = model_registry
        self.logger = logger

    async def handle(self, command: DownloadModelFromProviderCommand) -> None:
        await self.logger.info(
            "Requesting model '%s/%s' from provider",
            command.pretrained,
            command.architecture,
        )

        thread = asyncio.to_thread(  # Send to thread to ensure memory used by provider is released after download
            self.model_provider.download_model_from_pretrained,
            architecture=command.architecture,
            pretrained=command.pretrained,
        )
        ram_memory_mB, vram_memory_mB = await thread

        await self.logger.info(
            "Model '%s' downloaded from provider, ram memory required: %d MB, vram memory required: %d MB",
            command.pretrained,
            ram_memory_mB,
            vram_memory_mB,
        )
        await self.logger.info("Saving model '%s' to registry", command.pretrained)

        model = ModelMetadata(
            pretrained=command.pretrained,
            architecture=command.architecture,
            model_type=command.model_type,
            provider=ModelProvider(str(self.model_provider.name)),
            resources=ModelResources.model_validate(
                {
                    "is_gpu_required": command.is_gpu_required,
                    "threads": command.threads,
                    "ram_memory_mB": ram_memory_mB,
                    "vram_memory_mB": vram_memory_mB,
                }
            ),
            warmup={
                "is_enabled": command.is_warmup_enabled,
                "samples": command.warmup_samples,
            },
        )

        self.model_registry.save(model)

        if command.set_defaults:
            await self.logger.info(
                "Setting model '%s' as default for type '%s'", command.pretrained, command.model_type
            )
            self.model_registry.save_default_model(command.model_type, model.id)

        self.model_registry.commit()
        await self.logger.info("Model '%s' downloaded and saved to registry", command.pretrained)
