from typing import Any, Dict, Iterable, List, Optional

import yaml

from modelmora.configuration import AppConfig
from modelmora.registry.domain import IModelRegistry, ModelMetadata


class YamlModelRegistry(IModelRegistry):
    def __init__(self, app_config: AppConfig) -> None:
        self._app_config = app_config
        self._models: Optional[Dict[str, ModelMetadata]] = None
        self._models_by_type: Optional[Dict[str, List[str]]] = None  # model_type -> list of model_names
        self._default_models: Dict[str, str] = {}  # model_type -> model_name

    def _parse_model(self, model_data: Dict[str, Any]) -> ModelMetadata:
        return ModelMetadata.model_validate(model_data)

    def _load_models_from_yaml(self) -> None:
        if self._models is not None:
            return

        with open(self._app_config.model_management_config.model_registry_path, "r", encoding="utf-8") as file:
            yaml_content = yaml.safe_load(file)
            if yaml_content is None:
                yaml_content = {}

        models: Dict[str, ModelMetadata] = {}
        models_by_type: Dict[str, List[str]] = {}
        default_models: Dict[str, str] = {}
        default_models_ids: Dict[str, str] = yaml_content.get("default_models", {})
        for model_entry in yaml_content.get("models", []) or []:
            # Parse each model entry
            model = self._parse_model(model_entry)

            # Add to the cache indexed by model id
            models[model.id] = model

            # Additionaly index by model type
            if models_by_type.get(model.model_type) is None:
                models_by_type[model.model_type] = []
            models_by_type[model.model_type].append(model.id)

            # Store default models for quick access
            if model.id == default_models_ids.get(str(model.model_type)):
                default_models[model.model_type] = model.id

        self._models = models
        self._models_by_type = models_by_type
        self._default_models = default_models

    def commit(self) -> None:
        """Write the current models back to the YAML file."""
        if self._models is None:
            return

        models_list = [model.model_dump() for model in self._models.values()]
        default_models_ids = dict(self._default_models)

        yaml_content = {
            "default_models": default_models_ids,
            "models": models_list,
        }

        with open(self._app_config.model_management_config.model_registry_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(yaml_content, file)

    def count_models(self) -> int:
        self._load_models_from_yaml()
        if self._models is None:
            return 0
        return len(self._models)

    def count_models_by_type(self, model_type: str) -> int:
        self._load_models_from_yaml()
        if self._models_by_type is None:
            return 0
        return len(self._models_by_type.get(model_type, []))

    def find_default_model_by_type(self, model_type: str) -> Optional[ModelMetadata]:
        self._load_models_from_yaml()
        if self._models is None:
            return None
        default_model_name = self._default_models.get(model_type)
        if default_model_name is None:
            return None
        return self._models.get(default_model_name)

    def save_default_model(self, model_type: str, id: str) -> None:
        self._load_models_from_yaml()
        if self._models is None:
            raise ValueError("No models loaded in the registry.")
        if id not in self._models:
            raise ValueError(f"Model '{id}' does not exist in the registry.")
        self._default_models[model_type] = id

    def find_by_id(self, id: str) -> Optional[ModelMetadata]:
        self._load_models_from_yaml()
        if self._models is None:
            return None
        return self._models.get(id)

    def model_exists(self, id: str) -> bool:
        self._load_models_from_yaml()
        if self._models is None:
            return False
        return id in self._models

    def save(self, model_metadata: ModelMetadata) -> None:
        self._load_models_from_yaml()
        if self._models is None or self._models_by_type is None:
            raise ValueError("No models loaded in the registry.")

        # Save or update the model metadata
        self._models[model_metadata.id] = model_metadata

        # Update the models by type index
        if model_metadata.model_type not in self._models_by_type:
            self._models_by_type[model_metadata.model_type] = []
        if model_metadata.id not in self._models_by_type[model_metadata.model_type]:
            self._models_by_type[model_metadata.model_type].append(model_metadata.id)

    def delete(self, id: str) -> None:
        self._load_models_from_yaml()
        if self._models is None or self._models_by_type is None:
            raise ValueError("No models loaded in the registry.")
        model = self._models.get(id)
        if model is None:
            raise ValueError(f"Model '{id}' does not exist in the registry.")

        # Remove from models dictionary
        del self._models[id]

        # Remove from models by type index
        if model.model_type in self._models_by_type:
            if id in self._models_by_type[model.model_type]:
                self._models_by_type[model.model_type].remove(id)

        # If it was the default model, remove it from default models
        if self._default_models.get(model.model_type) == id:
            del self._default_models[model.model_type]

    def list_models(self, cursor: Optional[str] = None, limit: Optional[int] = 10) -> Iterable[ModelMetadata]:
        self._load_models_from_yaml()
        if self._models is None:
            return

        model_names = sorted(self._models.keys())
        start_index = 0
        if cursor:
            try:
                start_index = model_names.index(cursor) + 1
            except ValueError:
                start_index = 0

        if limit is None:
            limit = len(model_names) - start_index

        selected_model_names = model_names[start_index : start_index + limit]
        for model_name in selected_model_names:
            yield self._models[model_name]

    def list_models_by_type(
        self, model_type: str, cursor: Optional[str] = None, limit: Optional[int] = 10
    ) -> Iterable[ModelMetadata]:
        self._load_models_from_yaml()
        if self._models is None or self._models_by_type is None:
            return

        model_names = sorted(self._models_by_type.get(model_type, []))
        start_index = 0
        if cursor:
            try:
                start_index = model_names.index(cursor) + 1
            except ValueError:
                start_index = 0

        if limit is None:
            limit = len(model_names) - start_index

        selected_model_names = model_names[start_index : start_index + limit]
        for model_name in selected_model_names:
            yield self._models[model_name]
