from enum import Enum


class ModelType(str, Enum):
    """This class represents the different types of deep learning models.
    These names are arbitrary for the ModelMora context, do not necessarily reflect standard industry terminology.
    They are used to categorize models based on their primary function or use case.
    """

    EMBEDDING = "embedding"
    GENERATION = "generation"
    CAPTIONING = "captioning"
    SAFETY = "safety"

    def __str__(self) -> str:
        return self.value


class ModelProvider(str, Enum):
    """This class represents the different providers of deep learning models.
    Each provider may offer models with varying architectures, capabilities, and performance characteristics.
    Also, the choice of provider can influence how models are loaded, used, and integrated into applications.
    Each value should have a corresponding IModelProvider implementation.
    """

    OPEN_CLIP = "OpenCLIP"

    def __str__(self) -> str:
        return self.value
