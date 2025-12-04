from miraveja_log import IAsyncLogger
from pydantic import BaseModel, Field

from modelmora.registry.domain import IModelRegistry, ModelType


class SetDefaultModelForTypeCommand(BaseModel):
    model_type: ModelType = Field(..., description="The type of the model to set the default for.")
    model_id: str = Field(..., description="The ID of the model to set as default for the specified type.")


class SetDefaultModelForTypeHandler:
    def __init__(self, model_registry: IModelRegistry, logger: IAsyncLogger) -> None:
        self.model_registry = model_registry
        self.logger = logger

    async def handle(self, command: SetDefaultModelForTypeCommand) -> None:
        self.model_registry.save_default_model(model_type=command.model_type, id=command.model_id)
        self.model_registry.commit()
        await self.logger.info(f"Set default model for type {command.model_type} to model ID {command.model_id}.")
