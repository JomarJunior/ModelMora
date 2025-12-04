from miraveja_auth import (
    IOAuth2Provider,
    IOIDCDiscoveryService,
    OAuth2Configuration,
    OAuth2Provider,
    OIDCDiscoveryService,
)
from miraveja_auth.domain import IClaimsParser
from miraveja_auth.infrastructure.providers import KeycloakClaimsParser
from miraveja_di import DIContainer
from miraveja_log import IAsyncLogger, ILogger, LoggerConfig, LoggerFactory
from miraveja_log.infrastructure import AsyncPythonLoggerAdapter, PythonLoggerAdapter

from modelmora.configuration import AppConfig


class ModelMoraDependencies:
    @staticmethod
    def register_dependencies(container: DIContainer) -> None:
        # register singletons
        container.register_singletons(
            {
                AppConfig: lambda container: AppConfig.from_env(),
                # Logger
                LoggerConfig: lambda container: LoggerConfig.from_env(),
                LoggerFactory: lambda container: LoggerFactory(
                    logger_implementation=PythonLoggerAdapter,
                    async_logger_implementation=AsyncPythonLoggerAdapter,
                ),
                ILogger: lambda container: container.resolve(LoggerFactory).get_or_create_logger(
                    container.resolve(LoggerConfig)
                ),
                IAsyncLogger: lambda container: container.resolve(LoggerFactory).get_or_create_async_logger(
                    container.resolve(LoggerConfig)
                ),
                # OAuth2 Configuration
                OAuth2Configuration: lambda container: OAuth2Configuration.from_env(),
                IOIDCDiscoveryService: lambda container: container.resolve(OIDCDiscoveryService),
                IClaimsParser: lambda container: container.resolve(KeycloakClaimsParser),
                IOAuth2Provider: lambda container: container.resolve(OAuth2Provider),
            }
        )
