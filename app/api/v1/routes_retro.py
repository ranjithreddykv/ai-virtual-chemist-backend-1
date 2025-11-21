from fastapi import APIRouter, HTTPException
from app.schemas.retro_prediction_schema import RetroRequest
from app.ml.sagawa_retro import predict_reactants
from app.utils.mechanism_explainer import explain_retro_reaction
from app.utils.mechanism_visualizer_forward import generate_3d_molblock

router = APIRouter(prefix="/retro", tags=["Retrosynthesis Prediction"])


@router.post("/predict")
async def predict_retro(request: RetroRequest):
    """
    Basic retrosynthesis: returns predicted reactants as SMILES.
    """
    try:
        reactants = predict_reactants(request.product)
        return {"predicted_reactants": reactants}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_full")
async def predict_retro_full(request: RetroRequest):
    """
    Full retrosynthesis:
    - Predict reactants from product
    - Create 3D molblocks for all reactants + product
    - Generate LLM explanation (professor-style)
    """

    try:
        # ML Model
        reactants = predict_reactants(request.product)

        # Create 3D models
        reactants_list = reactants.split(".")  # model returns reactant1.reactant2...
        reactants_3d = [generate_3d_molblock(r) for r in reactants_list]
        product_3d = generate_3d_molblock(request.product)

        # LLM Explanation
        explanation = explain_retro_reaction(
            request.product,
            reactants_list
        )

        return {
            "product": request.product,
            "predicted_reactants": reactants,
            "reactants_list": reactants_list,

            "reactants_3d": reactants_3d,
            "product_3d": product_3d,

            "teacher_explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
