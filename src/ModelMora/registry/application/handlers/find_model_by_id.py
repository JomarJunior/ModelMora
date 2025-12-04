from typing import Any, Dict

from miraveja_log import IAsyncLogger
from pydantic import BaseModel, Field

from modelmora.registry.domain import IModelRegistry
from modelmora.registry.domain.exceptions import ModelNotFoundException


class FindModelByIdCommand(BaseModel):
    id: str = Field(..., description="The unique identifier of the AI model to find.")


class FindModelByIdHandler:
    def __init__(self, model_registry: IModelRegistry, logger: IAsyncLogger) -> None:
        self.model_registry = model_registry
        self.logger = logger

    async def handle(self, command: FindModelByIdCommand) -> Dict[str, Any]:
        await self.logger.info("Finding model by id: %s", command.id)

        model = self.model_registry.find_by_id(command.id)
        if model is None:
            raise ModelNotFoundException(command.id)

        await self.logger.info("Model with id '%s' found", command.id)
        return model.model_dump()
