from typing import Annotated

from pydantic import BaseModel, Field


class Dataset(BaseModel):
    classes: dict[str, str]
    max_samples_per_class: Annotated[int, Field(..., gt=0)]
