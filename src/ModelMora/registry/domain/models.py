from typing import Annotated, Any, ClassVar, Dict, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from typing_extensions import TypedDict

from modelmora.registry.domain.enums import ModelProvider, ModelType
from modelmora.registry.domain.exceptions import AttributeNotFoundException


class ModelResources(BaseModel):
    """This class represents the resource requirements for running a deep learning model.

    Attributes:
        ram_memory_mB: (int) The approximate CPU memory requirement of the model in megabytes.
        vram_memory_mB: (int | None) The approximate GPU memory requirement of the model in megabytes.
        threads: (int) The number of threads the model is allowed to use during inference.
        is_gpu_required: (bool) Indicates whether the model requires a GPU for inference.
    """

    ram_memory_mB: int = Field(..., description="The RAM memory requirement of the model in megabytes.")
    vram_memory_mB: Optional[int] = Field(
        default=None, description="The VRAM memory requirement of the model in megabytes, if applicable."
    )
    threads: int = Field(..., description="The number of threads to use for model inference.")
    is_gpu_required: bool = Field(..., description="Indicates whether the model requires a GPU for inference.")

    @model_validator(mode="after")
    @classmethod
    def validate_vram_memory(cls, values: Annotated[Any, "ModelResources"]) -> Any:
        """Validates that if the model requires a GPU, the VRAM memory is specified.

        Raises:
            ValueError: If the model requires a GPU but VRAM memory is not specified.
        """
        is_gpu_required = values.is_gpu_required
        vram_memory_mB = values.vram_memory_mB

        if is_gpu_required and vram_memory_mB is None:
            raise ValueError("VRAM memory must be specified if the model requires a GPU.")

        return values


class ModelWarmup(TypedDict):
    """This class represents the warmup settings for a deep learning model.

    Attributes:
        is_enabled: (bool) Indicates whether warmup is enabled for the model.
        samples: (int) The number of samples to use for warmup if enabled.
    """

    is_enabled: bool
    samples: int


class ModelMetadata(BaseModel):
    """This class represents the information about a deep learning model stored, or to be stored, in the registry.
    It includes fields for identifying the model, the task it is designed for,
    performance constraints and usage details. It is the aggregate root for model metadata management.

    Attributes:
        id: (str) A unique identifier for the model.
        pretrained: (str) The name of the pretrained model (e.g., 'resnet50', 'bert-base-uncased', 'laion2b_s12b_b42k').
        architecture: (str) The architecture type of the model (e.g., 'resnet', 'bert', 'ViT-g-14').
        model_type: (ModelType) The type of model, or its intended use case.
        provider: (str) The provider of the model (e.g., 'HuggingFace', 'OpenAI', 'LAION').
        resources: (ModelResources) The resource requirements for running the model.
        warmup: (ModelWarmup) The warmup settings for the model.

    Methods:
        update(**kwargs) -> None:
            Updates the model metadata with the provided keyword arguments.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()), description="The unique identifier of the AI model.")
    pretrained: str = Field(..., description="The name of the pretrained model.")
    architecture: str = Field(..., description="The architecture type of the AI model.")
    model_type: ModelType = Field(..., description="The type of the AI model.")
    provider: ModelProvider = Field(..., description="The provider of the AI model.")
    resources: ModelResources = Field(..., description="Resource requirements for the AI model.")
    warmup: ModelWarmup = Field(..., description="Warmup settings for the AI model.")

    @model_serializer
    def serialize(self) -> Dict[str, Any]:
        """Serializes the model metadata to a dictionary.

        Returns:
            Dict[str, Any]: The serialized model metadata.
        """
        return {
            "id": self.id,
            "pretrained": self.pretrained,
            "architecture": self.architecture,
            "model_type": str(self.model_type),
            "provider": str(self.provider),
            "resources": self.resources.model_dump(),
            "warmup": self.warmup,
        }

    def to_yaml(self) -> str:
        """Converts the model metadata to a YAML string representation.

        Returns:
            str: The YAML string representation of the model metadata.
        """

        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelMetadata":
        """Creates a ModelMetadata instance from a YAML string representation.

        Args:
            yaml_str (str): The YAML string representation of the model metadata.
        Returns:
            ModelMetadata: The ModelMetadata instance created from the YAML string.
        """
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)
