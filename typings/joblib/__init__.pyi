"""Used to shut Pylance up"""

from os import PathLike
from typing import Any

def load(
    filename: str | PathLike[str],
    mmap_mode: str | None = None,
) -> Any: ...
def dump(
    value: Any,
    filename: str | PathLike[str],
    compress: int | bool | tuple[str, int] = 0,
) -> list[str]: ...
