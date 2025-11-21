# routes_forward.py
from fastapi import APIRouter, HTTPException
from app.schemas.forward_predection_schema import ForwardRequest, ForwardResponse
from app.ml.sagawa_forward import predict_product
from app.utils.mechanism_visualizer_forward import generate_forward_visual , generate_3d_molblock
from app.utils.mechanism_explainer import explain_forward_reaction
router = APIRouter(prefix="/forward", tags=["Reaction Forward Prediction"])

@router.post("/predict", response_model=ForwardResponse)
async def predict_forward(request: ForwardRequest):
    """
    Predicts the reaction product given reactants and reagents using sagawa/ReactionT5v2-forward.
    """
    try:
        result = predict_product(request.reactants, request.reagents)
        return ForwardResponse(predicted_product=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_full")
async def predict_forward_full(request: ForwardRequest):
    try:
        product = predict_product(request.reactants, request.reagents)

        # 2D
        svg = generate_forward_visual(
            request.reactants,
            request.reagents,
            product
        )

        # 3D
        react3d = [
            generate_3d_molblock(s)
            for s in request.reactants.split('.')
        ]
        prod3d = generate_3d_molblock(product)

        # LLM
        explanation = explain_forward_reaction(
            request.reactants,
            request.reagents,
            product
        )

        return {
            "reactants": request.reactants,
            "reagents": request.reagents,
            "predicted_product": product,

            "visual_svg": svg,
            "reactants_3d": react3d,
            "product_3d": prod3d,

            "teacher_explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))