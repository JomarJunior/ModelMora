import json
from typing import Any, Dict, Optional
from uuid import UUID

from modelmora.shared.custom_types import ShortString


class DomainException(Exception):
    code: ShortString
    message: ShortString
    details: Optional[Dict[str, Any]]
    trace_id: Optional[UUID]

    def __init__(
        self,
        code: ShortString,
        message: ShortString,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[UUID] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details
        self.trace_id = trace_id
        super().__init__(message)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "trace_id": str(self.trace_id) if self.trace_id else None,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())
