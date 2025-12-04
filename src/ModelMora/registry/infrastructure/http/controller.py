import json

from fastapi import Response

from modelmora.registry.application import (
    DeleteModelByIdCommand,
    DeleteModelByIdHandler,
    DownloadModelFromProviderCommand,
    DownloadModelFromProviderHandler,
    FindModelByIdCommand,
    FindModelByIdHandler,
    ListAllAvailableModelsCommand,
    ListAllAvailableModelsHandler,
    ListModelsByTypeCommand,
    ListModelsByTypeHandler,
    SetDefaultModelForTypeCommand,
    SetDefaultModelForTypeHandler,
)


class RegistryController:
    def __init__(
        self,
        delete_model_by_id_handler: DeleteModelByIdHandler,
        download_model_from_provider_handler: DownloadModelFromProviderHandler,
        find_model_by_id_handler: FindModelByIdHandler,
        list_all_available_models_handler: ListAllAvailableModelsHandler,
        list_models_by_type_handler: ListModelsByTypeHandler,
        set_default_model_for_type_handler: SetDefaultModelForTypeHandler,
    ) -> None:
        self._delete_model_by_id_handler = delete_model_by_id_handler
        self._download_model_from_provider_handler = download_model_from_provider_handler
        self._find_model_by_id_handler = find_model_by_id_handler
        self._list_all_available_models_handler = list_all_available_models_handler
        self._list_models_by_type_handler = list_models_by_type_handler
        self._set_default_model_for_type_handler = set_default_model_for_type_handler

    async def delete_model_by_id(self, command: DeleteModelByIdCommand) -> Response:
        await self._delete_model_by_id_handler.handle(command)
        return Response(status_code=204)

    async def download_model_from_provider(self, command: DownloadModelFromProviderCommand) -> Response:
        model_path = await self._download_model_from_provider_handler.handle(command)
        return Response(content=json.dumps({"model_path": model_path}))

    async def find_model_by_id(self, command: FindModelByIdCommand) -> Response:
        model_info = await self._find_model_by_id_handler.handle(command)
        return Response(content=json.dumps(model_info))

    async def list_all_available_models(self, command: ListAllAvailableModelsCommand) -> Response:
        models_list = await self._list_all_available_models_handler.handle(command)
        return Response(content=json.dumps(models_list))

    async def list_models_by_type(self, command: ListModelsByTypeCommand) -> Response:
        models_list = await self._list_models_by_type_handler.handle(command)
        return Response(content=json.dumps(models_list))

    async def set_default_model_for_type(self, command: SetDefaultModelForTypeCommand) -> Response:
        await self._set_default_model_for_type_handler.handle(command)
        return Response(status_code=204)
