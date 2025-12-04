from fastapi import APIRouter, Response

from modelmora.registry.application import (
    DeleteModelByIdCommand,
    DownloadModelFromProviderCommand,
    FindModelByIdCommand,
    ListAllAvailableModelsCommand,
    ListModelsByTypeCommand,
    SetDefaultModelForTypeCommand,
)
from modelmora.registry.domain import ModelType
from modelmora.registry.infrastructure.http.controller import RegistryController


class RegistryRoutes:
    @staticmethod
    def register_routes(router: APIRouter, controller: RegistryController) -> None:
        @router.delete("/models/{model_id}")
        async def delete_model_by_id(model_id: str) -> Response:
            command = DeleteModelByIdCommand(id=model_id)
            return await controller.delete_model_by_id(command)

        @router.post("/models/download")
        async def download_model_from_provider(command: DownloadModelFromProviderCommand) -> Response:
            return await controller.download_model_from_provider(command)

        @router.get("/models/{model_id}")
        async def find_model_by_id(model_id: str) -> Response:
            command = FindModelByIdCommand(id=model_id)
            return await controller.find_model_by_id(command)

        @router.get("/models/")
        async def list_all_available_models() -> Response:
            command = ListAllAvailableModelsCommand()
            return await controller.list_all_available_models(command)

        @router.get("/models/types/{model_type}")
        async def list_models_by_type(model_type: str) -> Response:
            command = ListModelsByTypeCommand(model_type=ModelType(model_type))
            return await controller.list_models_by_type(command)

        @router.post("/models/types/{model_type}/default")
        async def set_default_model_for_type(model_type: str, model_id: str) -> Response:
            command = SetDefaultModelForTypeCommand(
                model_type=ModelType(model_type),
                model_id=model_id,
            )
            return await controller.set_default_model_for_type(command)
