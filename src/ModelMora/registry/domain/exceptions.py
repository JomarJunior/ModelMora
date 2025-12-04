from modelmora.shared import DomainException


class ModelNotFoundException(DomainException):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model with name '{model_name}' not found.",
            detail=f"The model '{model_name}' does not exist in the registry.",
        )


class AttributeNotFoundException(DomainException):
    def __init__(self, attribute_name: str) -> None:
        super().__init__(
            f"Attribute '{attribute_name}' not found.",
            detail=f"The attribute '{attribute_name}' does not exist in the model metadata.",
        )


class DirectoryNotFoundException(DomainException):
    def __init__(self, directory_path: str) -> None:
        super().__init__(
            f"Directory '{directory_path}' not found.",
            detail=f"The directory '{directory_path}' does not exist.",
        )


class CannotCreateDirectoryException(DomainException):
    def __init__(self, directory_path: str) -> None:
        super().__init__(
            f"Cannot create directory '{directory_path}'.",
            detail=f"The directory '{directory_path}' could not be created due to permission issues or invalid path.",
        )


class CannotWriteToDirectoryException(DomainException):
    def __init__(self, directory_path: str) -> None:
        super().__init__(
            f"Cannot write to directory '{directory_path}'.",
            detail=f"The directory '{directory_path}' is not writable due to permission issues.",
        )


class ModelAlreadyDownloadedException(DomainException):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' is already downloaded.",
            detail=f"The model '{model_name}' already exists in the cache directory.",
        )


class CannotFetchModelException(DomainException):
    def __init__(self, architecture: str, pretrained: str) -> None:
        super().__init__(
            f"Cannot fetch model '{pretrained}' with architecture '{architecture}'.",
            detail=f"The model '{pretrained}' with architecture '{architecture}' could not be fetched from the provider.",
        )
