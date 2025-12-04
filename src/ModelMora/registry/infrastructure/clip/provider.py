import gc
import os
import shutil
from typing import ClassVar, Tuple
from uuid import uuid4

import open_clip
import psutil
import torch

from modelmora.configuration import AppConfig
from modelmora.registry.domain import IModelProvider, ModelProvider, ModelType
from modelmora.registry.domain.exceptions import (
    CannotCreateDirectoryException,
    CannotFetchModelException,
    DirectoryNotFoundException,
    ModelAlreadyDownloadedException,
)


class OpenClipModelProvider(IModelProvider):
    model_type: ClassVar[ModelType] = ModelType.EMBEDDING
    provider_type: ClassVar[ModelProvider] = ModelProvider.OPEN_CLIP

    def __init__(self, app_config: AppConfig) -> None:
        super().__init__(app_config)

        gpu_enabled = app_config.gpu_config.gpu_enabled and torch.cuda.is_available()
        self.use_device = "cuda" if gpu_enabled else "cpu"
        self.device_id = app_config.gpu_config.gpu_device_id

        self.device = torch.device(f"cuda:{self.device_id}" if self.use_device == "cuda" else "cpu")

    # ----------------------------
    # Internal Helpers
    # ----------------------------

    def _ensure_dir(self, path: str, create: bool) -> None:
        if os.path.exists(path):
            return

        if not create:
            raise DirectoryNotFoundException(path)

        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise CannotCreateDirectoryException(path) from e

    def _ensure_parent_dir(self, full_path: str, create: bool) -> None:
        parent = os.path.dirname(full_path)
        self._ensure_dir(parent, create)

    def _remove_model_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)

    def _model_cache_dir(self, architecture: str, pretrained: str) -> str:
        return os.path.join(self.cache_dir, architecture, pretrained)

    # ----------------------------
    # Public API
    # ----------------------------

    def download_model_from_pretrained(
        self,
        architecture: str,
        pretrained: str,
        create_dir: bool = True,
        overwrite: bool = False,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Downloads ONLY the model weights into the Miraveja model registry directory.
        Returns: (RAM_used_MB, VRAM_used_MB)
        """

        # ----------------------------
        # Prepare memory measurement
        # ----------------------------
        if self.use_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss

        # ----------------------------
        # Prepare final destination
        # ----------------------------
        final_dir = self._model_cache_dir(architecture, pretrained)
        model_exists = os.path.isdir(final_dir) and any(os.scandir(final_dir))  # Folder exists AND contains files

        if model_exists and not overwrite:
            raise ModelAlreadyDownloadedException(final_dir)

        # Ensure parent dirs exist
        self._ensure_parent_dir(final_dir, create_dir)

        # Temporary isolated download directory
        temp_name = f"modelmora_tmp_{uuid4()}"
        temp_dir = os.path.join(self.cache_dir, "tmp", temp_name)

        # ----------------------------
        # Download from OpenCLIP
        # ----------------------------
        try:
            model, _ = open_clip.create_model_from_pretrained(  # type: ignore
                architecture,
                pretrained,
                cache_dir=temp_dir,
                precision="fp16",
                device=self.device,
            )

            # Overwrite existing final directory if needed
            if overwrite:
                self._remove_model_directory(final_dir)

            # Move temp directory -> final directory
            shutil.move(temp_dir, final_dir)

        except Exception as e:
            self._remove_model_directory(temp_dir)
            raise CannotFetchModelException(architecture, pretrained) from e

        # ----------------------------
        # Memory measurement
        # ----------------------------
        rss_after = process.memory_info().rss
        ram_used_MB = int((rss_after - rss_before) / (1024**2))

        vram_used_MB = 0
        if self.use_device == "cuda":
            torch.cuda.synchronize()
            vram_used_MB = int(torch.cuda.max_memory_allocated() / (1024**2))

        # ----------------------------
        # Cleanup model from RAM/VRAM
        # ----------------------------
        del model
        gc.collect()

        if self.use_device == "cuda":
            torch.cuda.empty_cache()

        return ram_used_MB, vram_used_MB

    # ----------------------------
    # Public removal API
    # ----------------------------

    def remove_model(self, architecture: str, pretrained: str) -> None:
        dir_path = self._model_cache_dir(architecture, pretrained)
        self._remove_model_directory(dir_path)
