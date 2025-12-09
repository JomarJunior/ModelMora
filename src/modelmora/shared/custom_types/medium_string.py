from typing import Annotated

from pydantic import Field

MediumString = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1024,
    ),
]
