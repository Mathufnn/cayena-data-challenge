from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    col_0: float
    col_1: float
    col_2: float
    col_3: float
    col_4: float
    col_5: float
    col_6: float
    col_7: float
    total_quantity: int
    augmented_quantity: int
    FU: str
    City: str
    CEP: int
    date_time_login: str
    date_time_confirm: str

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]