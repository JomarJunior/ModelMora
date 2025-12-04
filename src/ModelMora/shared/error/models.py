import json

from starlette.responses import Response


class DomainException(Exception):
    def __init__(self, message: str, detail: str, code: int = 400):
        self.message = message
        self.detail = detail
        self.code = code
        super().__init__(self.message)

    def to_http_response(self) -> Response:
        json_response = {"error": {"message": self.message, "detail": self.detail, "code": self.code}}
        return Response(
            content=json.dumps(json_response), status_code=self.code, headers={"Content-Type": "application/json"}
        )
