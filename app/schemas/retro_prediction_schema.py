from pydantic import BaseModel

class RetroRequest(BaseModel):
    product: str

class RetroResponse(BaseModel):
    predicted_reactants: str
