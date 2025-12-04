from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Optional, Tuple

from modelmora.configuration import AppConfig
from modelmora.registry.domain import ModelProvider
from modelmora.registry.domain.enums import ModelType
from modelmora.registry.domain.models import ModelMetadata


class IModelRegistry(ABC):
    @abstractmethod
    def list_models(self, cursor: Optional[str] = None, limit: Optional[int] = None) -> Iterable[ModelMetadata]:
        """Lists available models in the registry.

        Args:
            cursor (Optional[str]): Cursor for pagination.
            limit (Optional[int]): Maximum number of models to return.

        Returns:
            Iterable[ModelMetadata]: An iterable containing the list of models.
        """

    @abstractmethod
    def list_models_by_type(
        self, model_type: ModelType, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterable[ModelMetadata]:
        """Lists models of a specific type in the registry.

        Args:
            model_type (ModelType): The type of models to list.
            cursor (Optional[str]): Cursor for pagination.
            limit (Optional[int]): Maximum number of models to return.

        Returns:
            Iterable[ModelMetadata]: An iterable containing the list of models of the specified type.
        """

    @abstractmethod
    def count_models(self) -> int:
        """Counts the total number of models in the registry.

        Returns:
            int: The total number of models.
        """

    @abstractmethod
    def count_models_by_type(self, model_type: ModelType) -> int:
        """Counts the number of models of a specific type in the registry.

        Args:
            model_type (ModelType): The type of models to count.

        Returns:
            int: The number of models of the specified type.
        """

    @abstractmethod
    def model_exists(self, id: str) -> bool:
        """Checks if a model exists in the registry.

        Args:
            id (str): The unique identifier of the model.
        Returns:
            bool: True if the model exists, False otherwise.
        """

    @abstractmethod
    def find_by_id(self, id: str) -> Optional[ModelMetadata]:
        """Finds a model by its unique identifier.

        Args:
            id (str): The unique identifier of the model.
        Returns:
            Optional[ModelMetadata]: The model details if found, otherwise None.
        """

    @abstractmethod
    def find_default_model_by_type(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """Finds the default model in the registry.

        Returns:
            Optional[ModelMetadata]: The default model details if set, otherwise None.
        """

    @abstractmethod
    def save_default_model(self, model_type: ModelType, id: str) -> None:
        """Sets the default model for a specific type.

        Args:
            model_type (ModelType): The type of the model.
            id (str): The unique identifier of the model to set as default.
        """

    @abstractmethod
    def save(self, model_metadata: ModelMetadata) -> None:
        """Saves or updates model metadata in the registry.

        Args:
            model_metadata (ModelMetadata): The model metadata to save.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Deletes a model from the registry by its unique identifier.

        Args:
            id (str): The unique identifier of the model to delete.
        """

    @abstractmethod
    def commit(self) -> None:
        """Commits any pending changes to the registry."""


class IModelProvider(ABC):
    model_type: ClassVar[ModelType]
    provider_type: ClassVar[ModelProvider]
    cache_dir: str

    def __init__(self, app_config: AppConfig) -> None:
        """Initializes the model provider with the application configuration."""
        self.cache_dir = app_config.model_management_config.model_cache_dir

    @property
    def name(self) -> str:
        """The name of the model provider."""
        return str(self.__class__.provider_type)

    @abstractmethod
    def download_model_from_pretrained(
        self, architecture: str, pretrained: str, create_dir: bool = True, overwrite: bool = False, **kwargs
    ) -> Tuple[int, int]:
        """Downloads a pretrained model and returns the total amount of memory, in MB, required to load the model.

        Args:
            architecture (str): The architecture of the model.
            pretrained (str): The name of the pretrained model.
            create_dir (bool): Whether to create the model directory if it does not exist.
             Defaults to True.
            overwrite (bool): Whether to overwrite the model if it already exists.
             Defaults to False.
            **kwargs: Additional keyword arguments for model downloading.
        Returns:
            Tuple[int, int]: The total amount of RAM and VRAM memory, in MB, required to load the model.
        Raises:
            ErrorFetchingModel: If it was not possible to fetch the model by the given parameters.
            DirectoryNotFoundException: If the model directory does not exist before downloading.
            CannotCreateDirectoryException: If the model directory cannot be created.
            CannotWriteToDirectoryException: If the model directory is not writable.
            ModelAlreadyDownloadedException: If the model is already downloaded.
        """

    @abstractmethod
    def remove_model(self, architecture: str, pretrained: str) -> None:
        """Removes a downloaded model from the local cache.

        Args:
            architecture (str): The architecture of the model.
            pretrained (str): The name of the pretrained model.
        """
