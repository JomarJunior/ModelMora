import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import RepositoryNotFoundError


@dataclass
class RelevantModelInfo:
    id: str
    library_name: str
    pipeline_tag: Optional[str]

    def __init__(self, id: str, library_name: str, pipeline_tag: str, **kwargs):  # pylint: disable=unused-argument
        self.id = id
        self.library_name = library_name
        self.pipeline_tag = pipeline_tag


def list_first_n_models(n: int):
    api = HfApi()
    models: List[ModelInfo] = list(api.list_models(limit=n, model_name="laion/CLIP-ViT-g-14-laion2B-s12B-b42K"))
    for model in models:
        re_model = RelevantModelInfo(**asdict(model))
        print(json.dumps(asdict(re_model), indent=4, sort_keys=True, default=str))


def fetch_single_model(model_id: str):
    api = HfApi()
    try:
        model: ModelInfo = api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model with id '{model_id}' not found.")
        return
    re_model = RelevantModelInfo(**asdict(model))
    print(json.dumps(asdict(re_model), indent=4, sort_keys=True, default=str))


def list_repo_tags(model_id: str):
    api = HfApi()
    try:
        tags = api.list_repo_refs(model_id)
    except RepositoryNotFoundError:
        print(f"Model with id '{model_id}' not found.")
        return
    print(f"Tags for model '{model_id}': {tags}")


if __name__ == "__main__":
    MODEL_IDS = [
        "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "google/flan-t5-xxl",
        "facebook/opt-6.7b",
    ]
    MODEL_ID = MODEL_IDS[2]
    print("Listing first 5 models:\n")
    list_first_n_models(5)
    print("\nFetching single model info:\n")
    fetch_single_model(MODEL_ID)
    print("\nListing repository tags:\n")
    list_repo_tags(MODEL_ID)
