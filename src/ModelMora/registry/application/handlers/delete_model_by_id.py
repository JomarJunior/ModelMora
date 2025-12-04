from miraveja_log import IAsyncLogger
from pydantic import BaseModel, Field

from modelmora.registry.domain import IModelProvider, IModelRegistry


class DeleteModelByIdCommand(BaseModel):
    id: str = Field(..., description="The ID of the AI model to delete.")
    remove_files: bool = Field(
        default=False, description="Whether to remove the model files from storage when deleting the model."
    )


class DeleteModelByIdHandler:
    def __init__(self, model_registry: IModelRegistry, model_provider: IModelProvider, logger: IAsyncLogger) -> None:
        self.model_registry = model_registry
        self.model_provider = model_provider
        self.logger = logger

    async def handle(self, command: DeleteModelByIdCommand) -> None:
        await self.logger.info("Deleting model by ID: %s", command.id)

        model = self.model_registry.find_by_id(command.id)
        if model is None:
            await self.logger.error("Model '%s' not found for deletion", command.id)
            return

        self.model_registry.delete(command.id)

        if command.remove_files:
            await self.logger.info("Removing files for model '%s'", command.id)
            self.model_provider.remove_model(
                architecture=model.architecture,
                pretrained=model.pretrained,
            )
            await self.logger.info("Files for model '%s' removed", command.id)

        self.model_registry.commit()
        await self.logger.info("Model with ID '%s' deleted", command.id)
