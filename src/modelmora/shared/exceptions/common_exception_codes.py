from enum import Enum


class CommonExceptionCodes(str, Enum):
    VALIDATION_EXCEPTION = "validation_exception"
    UNAUTHORIZED_EXCEPTION = "unauthorized_exception"
    FORBIDDEN_EXCEPTION = "forbidden_exception"
    INTERNAL_ERROR_EXCEPTION = "internal_error_exception"
    SERVICE_UNAVAILABLE_EXCEPTION = "service_unavailable_exception"
    TIMEOUT_EXCEPTION = "timeout_exception"
