from pydantic import BaseModel

class ForwardRequest(BaseModel):
    reactants: str
    reagents: str

class ForwardResponse(BaseModel):
    predicted_product: str
