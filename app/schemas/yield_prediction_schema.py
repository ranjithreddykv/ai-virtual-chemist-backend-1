from pydantic import BaseModel

class YieldRequest(BaseModel):
    reactants: str
    reagents: str
    product: str

class YieldResponse(BaseModel):
    predicted_yield: float
