from typing import Any, Dict, List, Optional

from miraveja_log import IAsyncLogger
from pydantic import BaseModel, Field

from modelmora.registry.domain import IModelRegistry


class ListAllAvailableModelsCommand(BaseModel):
    cursor: Optional[str] = Field(default=None, description="Pagination cursor for fetching the next set of models")
    limit: Optional[int] = Field(default=None, description="Maximum number of models to return")


class ListAllAvailableModelsHandler:
    def __init__(
        self,
        model_registry: IModelRegistry,
        logger: IAsyncLogger,
    ) -> None:
        self.model_registry = model_registry
        self.logger = logger

    async def handle(self, command: ListAllAvailableModelsCommand) -> List[Dict[str, Any]]:
        await self.logger.info("Listing all available models with command:\n%s", command.model_dump_json())

        models = self.model_registry.list_models(
            cursor=command.cursor,
            limit=command.limit,
        )
        models = list(models)

        await self.logger.info("Retrieved %d models", len(models))
        return [model.model_dump() for model in models]
