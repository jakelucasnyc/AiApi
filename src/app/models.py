from pydantic import BaseModel
from typing import Union

class Prompt(BaseModel):
    user: str 
    system: str = ''
    temp: float = 0.2

