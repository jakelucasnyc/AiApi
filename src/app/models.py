from pydantic import BaseModel
from typing import Union, List

class Prompt(BaseModel):
    user: List[str]
    system: str = ''
    temp: float = 0.2

