from pydantic import BaseModel
from typing import List, Optional, Any

class ReactAIvateInput(BaseModel):
    smiles: str
    atom_mapped: List[int]
    # Optional fields if you want to send other metadata later

class ReactAIvateOutput(BaseModel):
    predicted_class: int
    confidence: Optional[float] = None
    message: str = "Prediction successful"
