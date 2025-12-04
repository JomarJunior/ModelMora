from miraveja_di import DIContainer

from modelmora.registry.domain import IModelProvider, IModelRegistry
from modelmora.registry.infrastructure.clip import OpenClipModelProvider
from modelmora.registry.infrastructure.yaml_registry import YamlModelRegistry


class RegistryDependencies:
    @staticmethod
    def register_dependencies(container: DIContainer):
        container.register_singletons(
            {
                IModelRegistry: lambda container: container.resolve(YamlModelRegistry),
            }
        )

        container.register_scoped(
            {
                IModelProvider: lambda container: container.resolve(OpenClipModelProvider),
            }
        )
