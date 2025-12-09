from typing import Annotated

from pydantic import Field

BigString = Annotated[
    str,
    Field(
        min_length=1,
        max_length=131072,
    ),
]
