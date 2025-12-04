# FastAPI middleware for handling errors in ModelMora

import json
import traceback
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from fastapi import Request, Response
from miraveja_auth import OAuth2Provider, User
from miraveja_di import DIContainer
from miraveja_log import IAsyncLogger, ILogger, LoggerConfig, LoggerFactory
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from modelmora.shared.error import DomainException


class ErrorMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.a_logger: Optional[IAsyncLogger] = None

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Middleware to handle errors in ModelMora application.

        It catches exceptions raised during request processing and returns appropriate HTTP responses.
        """
        try:
            if not hasattr(request.state, "di_container"):
                raise RuntimeError("DI container not found in request state.")
            scoped_container: Optional[DIContainer] = request.state.di_container
            if scoped_container is None:
                raise RuntimeError("Scoped DI container not found in request state.")

            self.a_logger = scoped_container.resolve(IAsyncLogger)

            if self.a_logger is None:
                # Fallback to a default logger if not found in scoped container
                logger_factory: LoggerFactory = scoped_container.resolve(LoggerFactory)
                logger_config: LoggerConfig = LoggerConfig.from_env()
                self.a_logger = logger_factory.get_or_create_async_logger(logger_config)

            response = await call_next(request)
            return response
        except DomainException as domain_exception:
            await self._log_traceback()
            return domain_exception.to_http_response()
        except Exception as e:
            await self._log_traceback()
            return Response(
                content=str(e),
                status_code=500,
            )

    async def _log_traceback(self) -> None:
        """Logs the current traceback with a unique error ID."""
        if self.a_logger is None:
            return

        error_id: str = str(uuid4())
        await self.a_logger.error(f"#{error_id} - An error occurred while processing the request.")
        await self.a_logger.error(traceback.format_exc())


class LoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not hasattr(request.state, "di_container"):
            raise RuntimeError("DI container not found in request state.")
        scoped_container: Optional[DIContainer] = request.state.di_container
        if scoped_container is None:
            raise RuntimeError("Scoped DI container not found in request state.")

        oauth2_provider: OAuth2Provider = scoped_container.resolve(OAuth2Provider)
        logger_config: LoggerConfig = scoped_container.resolve(LoggerConfig)

        token = request.headers.get("Authorization", None)
        if token:
            token = token.replace("Bearer ", "")

            user: User = await oauth2_provider.validate_token(token)

            username = user.username if user and user.username else "anonymous"
            logger_config.filename = f"{username}.log"
            logger_config.name = f"{username}_logger"

        a_logger: IAsyncLogger = scoped_container.resolve(IAsyncLogger)
        await a_logger.info(f"Incoming request:\n{await self._format_incoming_request_text(request)}")

        response = await call_next(request)
        await a_logger.info("End of request processing...\n" + "=" * 100)

        return response

    async def _format_incoming_request_text(self, request: Request) -> str:
        first_block = (
            "=" * 100 + "\n" + f"[{request.method}] {request.url}\n"
            f"Headers:\n{json.dumps(dict(request.headers), indent=2)}\n"
        )
        ending_block = "-" * 100

        json_body = await request.body()
        if not json_body:
            return first_block + ending_block

        second_block = f"Body:\n{json_body.decode('utf-8')}\n"

        return first_block + "-" * 100 + "\n" + second_block + ending_block
